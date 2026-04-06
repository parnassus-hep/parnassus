"""PyTorch implementation of Delphes ParticlePropagator module.

Propagates charged and neutral particles from a given vertex to a cylindrical
detector surface defined by its radius and half-length, centered at (0,0,0)
with the axis along z.

This module:
1. Propagates neutral particles in straight lines to the detector surface
2. Propagates charged particles in helical paths through the magnetic field
3. Computes position-based EtaOuter and PhiOuter at detector intersection
4. Separates output into five branches: ParticleAfterProp, NeutralParticle,
   ChargedHadron, Electron, Muon

Units:
    - Positions are stored in mm (converted from m during calculations)
    - Momentum is in GeV
    - Time T is stored as propagation time * c * 1e3 (mm/c units)
    - Magnetic field Bz is in Tesla
    - Detector dimensions (radius, half_length) are in meters

Reference:
    C++ Delphes: modules/ParticlePropagator.cc
"""

import torch
from torch import nn

from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes import pdg_filters


class ParticlePropagator(nn.Module):
    """PyTorch implementation of Delphes ParticlePropagator module.

    Propagates particles from production vertex to detector surface (cylinder).
    Handles three cases:

    1. **Neutral particles**: Straight-line propagation to cylinder surface
    2. **Charged particles**: Helical propagation through magnetic field
    3. **Already outside**: Particles starting outside tracker pass through unchanged

    The module computes position-based quantities at the detector surface:

    - **EtaOuter**: asinh(Z / sqrt(X² + Y²)) at detector intersection
    - **PhiOuter**: Azimuthal angle at point of closest approach to z-axis

    For charged particles, the momentum direction (Px, Py, Phi) is updated to
    reflect the direction at closest approach to the z-axis, matching C++ Delphes.

    Attributes
    ----------
    radius: float
        Detector cylinder radius in meters
    half_length: float
        Detector cylinder half-length in meters
    bz: float
        Magnetic field strength in Tesla (along z-axis)
    radius_max: float | None
        Maximum radius for valid particle origin
    half_length_max: float | None
        Maximum |z| for valid particle origin
    c_light: float
        Speed of light constant (m/s)

    Examples
    --------
    >>> propagator = ParticlePropagator(radius=1.29, half_length=3.0, bz=3.8)
    >>> particles_out, neutrals, charged_hadrons, electrons, muons = propagator(particles)
    """

    def __init__(
        self,
        radius: float = 1.29,  # Detector radius in meters
        half_length: float = 3.0,  # Detector half-length in meters
        bz: float = 3.8,  # Magnetic field in Tesla
        radius_max: float | None = None,  # Max radius for initial position check
        half_length_max: float | None = None,  # Max half-length for initial position check
    ) -> None:
        """Init.

        Parameters
        ----------
        radius: float
            Detector cylinder radius in meters (default: 1.29m for CMS)
        half_length: float
            Detector cylinder half-length in meters (default: 3.0m)
        bz: float
            Magnetic field strength in Tesla (default: 3.8T for CMS)
        radius_max: float | None
            Maximum radius for particle origin (default: same as radius)
        half_length_max: float | None
            Maximum half-length for particle origin (default: same as half_length)
        """
        super().__init__()
        self.radius = radius
        self.radius2 = radius * radius
        self.half_length = half_length
        self.bz = torch.tensor(bz, dtype=torch.float64)
        self.radius_max = radius_max if radius_max is not None else radius
        self.half_length_max = half_length_max if half_length_max is not None else half_length

        # Physical constant
        self.c_light = 2.99792458e8  # Speed of light in m/s

    def forward(
        self, particles: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate particles to the detector surface.

        Processes all input particles and returns five separate tensors for
        different particle categories. Each output tensor contains only the
        particles that passed propagation for that category.

        Parameters
        ----------
        particles: torch.Tensor
            Tensor of shape (N, N_FEATURES) containing generator-level
            particles. Must be flattened (not batched by event). Key columns:
            - PID (col 0): PDG particle ID
            - CHARGE (col 2): Electric charge
            - E (col 3): Energy in GeV
            - PX, PY, PZ (cols 4-6): Momentum components in GeV
            - PT (col 7): Transverse momentum in GeV
            - X, Y, Z (cols 11-13): Production vertex in mm
            - T (col 10): Production time in mm/c

        Returns
        -------
        particles_propagated, neutrals, charged_hadrons, electrons, muons: torch.Tensor
            Tuple of 5 tensors, each of shape (N_i, N_FEATURES):

            - **particles_propagated**: All particles that passed propagation,
                with updated positions (X, Y, Z, T) at detector surface
            - **neutrals**: Neutral particles (photons, neutral hadrons, etc.)
            - **charged_hadrons**: Charged hadrons (pions, kaons, protons, etc.)
            - **electrons**: Electrons and positrons (|PDG| = 11)
            - **muons**: Muons and antimuons (|PDG| = 13)

            For the individual particle branches (neutrals, charged_hadrons,
            electrons, muons), the X, Y, Z, T values are reset to the
            production vertex for consistency with C++ Delphes Trac output.

        Notes
        -----
        Charged particles that fail to reach the detector (helix doesn't
        intersect cylinder) are filtered out and not included in any output.
        """
        # TODO: Remove after debugging
        particles_before_prop = particles.clone()

        # Compute PT and Phi from raw momentum components (for all particles)
        px = particles[:, ColumnMap.PX]  # Momentum components (GeV)
        py = particles[:, ColumnMap.PY]
        pz = particles[:, ColumnMap.PZ]
        pt = particles[:, ColumnMap.PT]

        # NOTE: EtaOuter and PhiOuter will be computed as POSITION-BASED eta after propagation
        # in _propagate_neutral and _propagate_charged methods

        # Extract positions (stored in cm, convert to m for calculations)
        x_cm = particles[:, ColumnMap.X]  # X position in cm
        y_cm = particles[:, ColumnMap.Y]  # Y position in cm
        z_cm = particles[:, ColumnMap.Z]  # Z position in cm
        x = x_cm * 1.0e-3  # Convert mm to m
        y = y_cm * 1.0e-3
        z = z_cm * 1.0e-3
        e = particles[:, ColumnMap.E]  # Energy in GeV
        q = particles[:, ColumnMap.CHARGE]  # Charge

        # Check if particles are within detector volume
        r = torch.sqrt(x**2 + y**2)
        inside_volume = (r <= self.radius_max) & (torch.abs(z) <= self.half_length_max)

        # Check minimum PT
        valid_pt = pt**2 >= 1.0e-9

        # Update mask: filter out particles that fail these checks
        mask = inside_volume & valid_pt

        # ==================== HANDLE "ALREADY OUTSIDE TRACKER" CASE ====================
        # C++ Delphes: if(r > fRadius || |z| > fHalfLength) → pass through without propagation
        inside_tracker = (r <= self.radius) & (torch.abs(z) <= self.half_length)
        needs_propagation = inside_tracker & mask
        already_outside_tracker = (~inside_tracker) & mask

        # Separate neutral and charged particles (among those needing propagation)
        no_bfield = 1 if torch.abs(self.bz) < 1.0e-9 else 0
        neutral_mask = (
            no_bfield * (torch.ones_like(q, dtype=torch.bool))
            + (1 - no_bfield) * (torch.abs(q) < 1.0e-9)
        ) & needs_propagation
        charged_mask = (~(torch.abs(q) < 1.0e-9)) & needs_propagation

        # ==================== NEUTRAL PARTICLE PROPAGATION ====================
        particles = self._propagate_neutral(particles, neutral_mask, x, y, z, px, py, pz, pt, e)

        # ==================== CHARGED PARTICLE PROPAGATION ====================
        particles = self._propagate_charged(particles, charged_mask, x, y, z, px, py, pz, pt, e, q)

        # ==================== COMPUTE ETA FOR "ALREADY OUTSIDE" PARTICLES ====================
        # Particles that were already outside tracker don't go through propagation
        # but still need position-based Eta computed at their current position
        x_out = particles[already_outside_tracker, ColumnMap.X] * 1.0e-3  # mm to m
        y_out = particles[already_outside_tracker, ColumnMap.Y] * 1.0e-3  # mm to m
        z_out = particles[already_outside_tracker, ColumnMap.Z] * 1.0e-3  # mm to m
        r_xy_out = torch.sqrt(x_out**2 + y_out**2)
        eta_out = torch.asinh(z_out / (r_xy_out + 1e-10))
        particles[already_outside_tracker, ColumnMap.ETA_OUTER] = eta_out

        # ==================== FILTER CHARGED PARTICLES THAT FAILED PROPAGATION ====================
        # C++ Delphes: for charged particles, only add to output if r_t > 0.0 (line 338)
        # For neutral particles, always add to output (no check after propagation)
        # The check is: did the charged particle successfully reach the detector?
        # We detect failure by checking if r_t is very small
        # (position set to ~zero indicates failure)
        final_x = particles[:, ColumnMap.X] * 1.0e-3  # mm to m
        final_y = particles[:, ColumnMap.Y] * 1.0e-3
        final_r = torch.sqrt(final_x**2 + final_y**2)

        # Charged particles with r_t ≈ 0 failed propagation (helix doesn't reach detector)
        # Neutral particles always succeed (straight line always reaches somewhere)
        # Particles already outside always succeed (pass through)
        is_charged = torch.abs(particles[:, ColumnMap.CHARGE]) > 1.0e-9
        charged_failed = is_charged & (final_r < 1.0e-6) & needs_propagation

        # Update mask: remove charged particles that failed propagation
        mask = mask & (~charged_failed)

        # Update the mask column
        particles[:, ColumnMap.PASS_PROP] = mask.float()

        # Collect the 4 branches/outputs
        # NOTE: Since we will use both the particles object and the specific branch objects,
        # we instantiate the branches as clones
        charged_hadron_pid_mask = mask * pdg_filters.charged_hadron_filter(particles)
        charged_hadrons = particles[charged_hadron_pid_mask > 0.5].to(torch.float32).clone()
        charged_hadrons_before_prop = particles_before_prop[charged_hadron_pid_mask > 0.5].to(
            torch.float32
        )

        electron_pid_mask = mask * pdg_filters.electron_filter(particles)
        electrons = particles[electron_pid_mask > 0.5].to(torch.float32).clone()
        electrons_before_prop = particles_before_prop[electron_pid_mask > 0.5].to(torch.float32)

        muon_pid_mask = mask * pdg_filters.muon_filter(particles)
        muons = particles[muon_pid_mask > 0.5].to(torch.float32).clone()
        muons_before_prop = particles_before_prop[muon_pid_mask > 0.5].to(torch.float32)

        neutral_pid_mask = mask * pdg_filters.neutral_filter(particles)
        neutrals = particles[neutral_pid_mask > 0.5].to(torch.float32).clone()
        neutrals_before_prop = particles_before_prop[neutral_pid_mask > 0.5].to(torch.float32)

        # NOTE: We purposely/manually leave their positions unchanged
        # (i.e. leave it as production vertex)
        # This is for consistency with C++ logic in order to help debugging
        # TODO: Remove this after debugging. Unnecessary and memory-intensive.
        for var in [ColumnMap.X, ColumnMap.Y, ColumnMap.Z, ColumnMap.T]:
            charged_hadrons[:, var] = charged_hadrons_before_prop[:, var].clone()
            electrons[:, var] = electrons_before_prop[:, var].clone()
            muons[:, var] = muons_before_prop[:, var].clone()
            neutrals[:, var] = neutrals_before_prop[:, var].clone()

        valid_particles_mask = mask
        particles = particles[valid_particles_mask].to(torch.float32).clone()

        return particles, neutrals, charged_hadrons, electrons, muons

    def _propagate_neutral(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        pz: torch.Tensor,
        pt: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate neutral particles in straight lines to detector surface.

        For neutral particles, trajectories are straight lines. The intersection
        with the detector cylinder is found by solving the quadratic equation
        for the time to reach either the cylindrical surface or the endcaps.

        The time T is updated as: T_new = T_old + t * E * 1e3
        where t is the propagation time and E is the energy (speed = c for neutrals).

        Parameters
        ----------
        particles: torch.Tensor
            Full particle tensor to update in-place
        mask: torch.Tensor
            Boolean mask indicating which particles to propagate
        x, y, z: torch.Tensor
            Initial positions in meters
        px, py, pz: torch.Tensor
            Momentum components in GeV
        pt: torch.Tensor
            Transverse momentum in GeV
        e: torch.Tensor
            Energy in GeV

        Returns
        -------
        particles: torch.Tensor
            Updated particles tensor with propagated positions for masked particles
        """
        # Convert mask to boolean if needed (it might be int64 from operations)
        mask = mask.bool()

        # Extract neutral particle data
        x_n = x[mask]
        y_n = y[mask]
        z_n = z[mask]
        px_n = px[mask]
        py_n = py[mask]
        pz_n = pz[mask]
        pt_n = pt[mask]
        e_n = e[mask]

        pt2_n = pt_n**2

        # Time to reach cylinder sides (solve quadratic)
        tmp = px_n * y_n - py_n * x_n
        discriminant = pt2_n * self.radius2 - tmp**2
        discriminant = torch.clamp(discriminant, min=0.0)  # Ensure non-negative

        t_r = (torch.sqrt(discriminant) - px_n * x_n - py_n * y_n) / (pt2_n + 1e-10)

        # Time to reach cylinder ends
        t_z = torch.where(
            torch.abs(pz_n) > 1e-10,
            (torch.sign(pz_n) * self.half_length - z_n) / pz_n,
            torch.full_like(pz_n, 1.0e99),
        )

        # Take minimum time
        t = torch.min(t_r, t_z)

        # Compute final position
        x_t = x_n + px_n * t
        y_t = y_n + py_n * t
        z_t = z_n + pz_n * t

        # Compute position-based eta at final position (EtaOuter)
        r_t_xy = torch.sqrt(x_t**2 + y_t**2)
        eta_outer = torch.asinh(z_t / (r_t_xy + 1e-10))

        # Update positions and EtaOuter in particles (convert m back to mm)
        # We need to map from masked indices to full particle array
        mask_indices = torch.where(mask)[0]

        particles[mask_indices, ColumnMap.ETA_OUTER] = (
            eta_outer  # EtaOuter (position eta at final position)
        )
        particles[mask_indices, ColumnMap.PHI_OUTER] = particles[
            mask_indices, ColumnMap.PHI
        ]  # PhiOuter (same as momentum phi for neutral)
        particles[mask_indices, ColumnMap.X] = x_t * 1.0e3  # X
        particles[mask_indices, ColumnMap.Y] = y_t * 1.0e3  # Y
        particles[mask_indices, ColumnMap.Z] = z_t * 1.0e3  # Z
        particles[mask_indices, ColumnMap.T] = (
            particles[mask_indices, ColumnMap.T] + t * e_n * 1.0e3
        )  # T (time in mm/c)

        return particles

    def _propagate_charged(
        self,
        particles: torch.Tensor,
        mask: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        pz: torch.Tensor,
        pt: torch.Tensor,
        e: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate charged particles in helical paths through magnetic field.

        In a uniform magnetic field Bz along the z-axis, charged particles follow
        helical trajectories. The helix parameters are computed from the particle's
        momentum and charge, and the intersection with the detector cylinder is found.

        Key physics:

        - Helix radius: r = pt / (q * Bz) * c
        - Gyration frequency: ω = q * Bz / (ym)
        - z-velocity: vz = pz * c / E (constant along helix)

        The momentum direction (Px, Py, Phi) is updated to the value at the point
        of closest approach to the z-axis, matching C++ Delphes behavior.

        The time T is updated as: T_new = T_old + t * c_light * 1e3

        Parameters
        ----------
        particles: torch.Tensor
            Full particle tensor to update in-place
        mask: torch.Tensor
            Boolean mask indicating which particles to propagate
        x, y, z: torch.Tensor
            Initial positions in meters
        px, py, pz: torch.Tensor
            Momentum components in GeV
        pt: torch.Tensor
            Transverse momentum in GeV
        e: torch.Tensor
            Energy in GeV
        q: Electric charge

        Returns
        -------
        particles: torch.Tensor
            Updated particles tensor. Particles that fail to reach the detector
            (helix doesn't intersect cylinder) have their positions set to zero.
        """
        # Convert mask to boolean if needed (it might be int64 from operations)
        mask = mask.bool()

        # Extract charged particle data
        x_c = x[mask]
        y_c = y[mask]
        z_c = z[mask]
        px_c = px[mask]
        py_c = py[mask]
        pz_c = pz[mask]
        pt_c = pt[mask]
        e_c = e[mask]
        q_c = q[mask]

        # 1. Calculate helix parameters
        # gammam = E / c^2 (in eV/c^2)
        gammam = e_c * 1.0e9 / (self.c_light * self.c_light)

        # Gyration frequency: omega = q * Bz / gammam (in rad/s)
        omega = q_c * self.bz / gammam

        # Helix radius: r = pt / (q * Bz) * c (in meters)
        r = pt_c / (q_c * self.bz) * 1.0e9 / self.c_light

        # Initial phi angle
        phi_0 = torch.atan2(py_c, px_c)

        # 2. Helix center coordinates
        x_center = x_c + r * torch.sin(phi_0)
        y_center = y_c - r * torch.cos(phi_0)
        r_center = torch.sqrt(x_center**2 + y_center**2)

        # 3. Calculate propagation time
        # Velocity along z
        vz = pz_c * self.c_light / e_c

        # Time to reach z boundaries
        t_z = torch.where(
            torch.abs(vz) > 1e-10,
            (torch.sign(pz_c) * self.half_length - z_c) / vz,
            torch.full_like(vz, 1.0e99),
        )

        # Check if helix crosses cylinder sides
        crosses_sides = (r_center + torch.abs(r)) >= self.radius

        # Time to reach cylinder sides (for helices that cross)
        # Use law of cosines to find angle
        cos_arg = (r**2 + r_center**2 - self.radius**2) / (2.0 * torch.abs(r) * r_center + 1e-10)
        cos_arg = torch.clamp(cos_arg, -1.0, 1.0)
        alpha = torch.acos(cos_arg)

        # Time of closest approach
        td = (phi_0 + torch.atan2(x_center, y_center)) / omega

        # Remove modulo pi ambiguities using modulo arithmetic (matching C++ while loop)
        pio = torch.abs(torch.pi / omega)
        # Wrap td to range [-0.5*pio, 0.5*pio]
        td = ((td + 0.5 * pio) % pio) - 0.5 * pio

        t_r = td + torch.abs(alpha / omega)

        # Choose minimum time
        t = torch.where(crosses_sides, torch.min(t_r, t_z), t_z)

        # 4. Calculate final position
        phi_t = phi_0 - omega * t
        x_t = x_center - r * torch.sin(phi_t)
        y_t = y_center + r * torch.cos(phi_t)
        z_t = z_c + vz * t
        r_t = torch.sqrt(x_t**2 + y_t**2)

        # Only update particles that successfully reached detector
        valid = r_t > 0.0

        # Calculate track parameters at closest approach
        # IMPORTANT: Momentum is updated at closest approach (phid), NOT at final position (phi_t)
        # This matches C++ Delphes behavior (line 295-296 in ParticlePropagator.cc)
        phid = phi_0 - omega * td

        # Update momentum direction at CLOSEST APPROACH (not final position)
        # Use the raw phid angle to compute Px, Py (matching C++ behavior)
        px_d = pt_c * torch.cos(phid)
        py_d = pt_c * torch.sin(phid)
        # pz unchanged

        # Compute the final phi angle from Px and Py to ensure perfect consistency
        # This matches what ROOT's SetPtEtaPhiE does internally and handles ±pi wrapping correctly
        phi_final = torch.atan2(py_d, px_d)

        # Compute position-based eta at final position (EtaOuter)
        r_t_xy = torch.sqrt(x_t[valid] ** 2 + y_t[valid] ** 2)
        eta_outer = torch.asinh(z_t[valid] / (r_t_xy + 1e-10))

        # Update particles for valid propagations
        # We need to map from masked indices to full particle array
        mask_indices = torch.where(mask)[0]
        valid_indices = mask_indices[valid]

        particles[valid_indices, ColumnMap.PX] = px_d[valid]  # Px (at closest approach)
        particles[valid_indices, ColumnMap.PY] = py_d[valid]  # Py (at closest approach)
        particles[valid_indices, ColumnMap.PHI] = phi_final[
            valid
        ]  # Phi = atan2(Py, Px) for perfect consistency
        particles[valid_indices, ColumnMap.ETA_OUTER] = (
            eta_outer  # EtaOuter (position eta at final position)
        )
        particles[valid_indices, ColumnMap.PHI_OUTER] = phi_final[
            valid
        ]  # PhiOuter (same as momentum phi)
        particles[valid_indices, ColumnMap.X] = x_t[valid] * 1.0e3  # X (m to mm) - final position
        particles[valid_indices, ColumnMap.Y] = y_t[valid] * 1.0e3  # Y (m to mm) - final position
        particles[valid_indices, ColumnMap.Z] = z_t[valid] * 1.0e3  # Z (m to mm) - final position
        particles[valid_indices, ColumnMap.T] = (
            particles[valid_indices, ColumnMap.T] + t[valid] * self.c_light * 1.0e3
        )  # T

        # Mark invalid particles by setting position to zero
        invalid_indices = mask_indices[~valid]
        particles[invalid_indices, ColumnMap.X] = 0.0
        particles[invalid_indices, ColumnMap.Y] = 0.0
        particles[invalid_indices, ColumnMap.Z] = 0.0

        return particles

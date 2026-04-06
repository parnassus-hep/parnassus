"""Low-level particle → tensor conversion utilities.

This module is a dependency-free leaf: it imports only standard libraries,
numpy, torch, and pyhepmc. It intentionally has no parnassus imports so that
both ``parnassus.data`` and ``parnassus.torch_delphes`` can import from it
without creating circular dependencies.

Provides
--------
- ``ColumnMap`` — IntEnum of column indices in the particle tensor
- ``N_FEATURES`` — total number of features per particle
- ``hepmc_particles_to_tensor`` — convert a list of pyhepmc particles to a tensor
- ``pythia_particles_to_tensor`` — convert a Pythia8 event to a tensor
- PDG lookup utilities: ``get_charge_from_pdg_id``, ``get_mass_from_pdg_id``
"""

from enum import IntEnum
from typing import Any

import numpy as np
import particle
import torch

# ==================== COLUMN MAP ====================


class ColumnMap(IntEnum):
    """Enum for column indices in the particle tensor."""

    PID = 0
    STATUS = 1
    CHARGE = 2
    E = 3
    PX = 4
    PY = 5
    PZ = 6
    PT = 7
    ETA = 8
    PHI = 9
    T = 10
    X = 11
    Y = 12
    Z = 13
    MASS = 14
    ETA_OUTER = 15
    PHI_OUTER = 16
    EVENT_NUMBER = 17
    PASS_PROP = 18
    TRACK_RESOLUTION = 19


# Number of features per particle
N_FEATURES = len(ColumnMap)

# ==================== CONSTANTS ====================
PT_MIN = 1e-10
CHARGED_PION_MASS = 0.13957

_MASS_DICT = {
    # Leptons
    11: 0.000511,  # electron
    13: 0.10566,  # muon
    15: 1.77686,  # tau
    # Neutrinos
    12: 0.0,  # electron neutrino
    14: 0.0,  # muon neutrino
    16: 0.0,  # tau neutrino
    # Photon
    22: 0.0,
    # Pions
    111: 0.13498,  # pi0
    211: 0.13957,  # pi+/-
    # Kaons
    130: 0.49761,  # K_L0
    310: 0.49761,  # K_S0
    311: 0.49761,  # K0
    321: 0.49368,  # K+/-
    # Nucleons
    2212: 0.93827,  # proton
    2112: 0.93957,  # neutron
    # Lambda
    3122: 1.11568,
    # Sigma baryons
    3222: 1.18937,  # Sigma+
    3212: 1.19264,  # Sigma0
    3112: 1.19745,  # Sigma-
    # Xi baryons
    3322: 1.31486,  # Xi0
    3312: 1.32171,  # Xi-
    # Omega
    3334: 1.67245,
    # D mesons
    411: 1.86966,  # D+
    421: 1.86484,  # D0
    431: 1.96835,  # Ds+
    # B mesons
    521: 5.27934,  # B+
    511: 5.27965,  # B0
    531: 5.36688,  # Bs0
}

# ==================== PDG ID UTILITIES (VECTORIZED) ====================

# Cache for PDG lookups to avoid repeated particle library queries
_PDG_CHARGE_CACHE: dict[int, float] = {}


def _get_pdg_charge(pid: int) -> float:
    """Get charge for a single PDG ID using the particle library.

    Matches C++ Delphes behavior:
    - Returns 0 for quarks and diquarks (fractional charge particles)
    - Returns -999 for unknown particles.

    Parameters
    ----------
    pid : int
        PDG ID of the particle

    Returns
    -------
    charge : float
        Electric charge in units of e, or -999 for unknown particles
    """
    if pid in _PDG_CHARGE_CACHE:
        return _PDG_CHARGE_CACHE[pid]

    abs_pid = abs(pid)

    # Quarks (1-6) have fractional charges — C++ Delphes returns 0
    if 1 <= abs_pid <= 6:
        _PDG_CHARGE_CACHE[pid] = 0
        return 0

    # Diquarks: 4-digit codes with format n1 n2 n3 n4
    if 1000 <= abs_pid <= 9999:
        n4 = abs_pid % 10
        n3 = (abs_pid // 10) % 10
        n2 = (abs_pid // 100) % 10
        n1 = (abs_pid // 1000) % 10
        if n4 in {1, 3} and n3 == 0 and 1 <= n1 <= 6 and 1 <= n2 <= 6:
            _PDG_CHARGE_CACHE[pid] = 0
            return 0

    try:
        p = particle.Particle.from_pdgid(pid)
        charge = round(p.charge) if p.charge is not None else -999
    except (particle.ParticleNotFound, particle.InvalidParticle):
        charge = -999
    _PDG_CHARGE_CACHE[pid] = charge
    return charge


def get_charge_from_pdg_id(pids: np.ndarray) -> np.ndarray:
    """Get electric charges for an array of PDG IDs (vectorized).

    Uses the particle library for comprehensive PDG coverage.
    Returns -999 for unknown particles (matching C++ Delphes behavior).

    Parameters
    ----------
    pids : np.ndarray
        NumPy array of PDG IDs

    Returns
    -------
    charges : np.ndarray
        NumPy array of electric charges
    """
    unique_pids = np.unique(pids)
    charge_map = {pid: _get_pdg_charge(int(pid)) for pid in unique_pids}
    return np.array([charge_map[pid] for pid in pids], dtype=np.float64)


def get_mass_from_pdg_id(pids: np.ndarray) -> np.ndarray:
    """Get particle masses for an array of PDG IDs (vectorized).

    Parameters
    ----------
    pids : np.ndarray
        NumPy array of PDG IDs

    Returns
    -------
    masses : np.ndarray
        NumPy array of masses in GeV
    """
    abs_pids = np.abs(pids)

    # Default to pion mass for unknown particles
    masses = np.full_like(pids, CHARGED_PION_MASS, dtype=np.float64)
    unique_abs_pids = np.unique(abs_pids)
    for pid in unique_abs_pids:
        if pid in _MASS_DICT:
            masses[abs_pids == pid] = _MASS_DICT[pid]

    return masses


# ==================== PARTICLE CONVERSION ====================


def hepmc_particles_to_tensor(
    particles_list: list,
    event_number: int,
    use_hepmc_mass: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a list of HepMC particles to a tensor.

    Reads particle properties directly from HepMC to match C++ Delphes behavior:
    - Mass: read from HepMC ``generated_mass`` (what the generator produced)
    - Charge: looked up from PDG ID (HepMC doesn't store charge)
    - Eta: use ±999.9 for particles with zero transverse momentum

    Parameters
    ----------
    particles_list : list[pyhepmc.GenParticle]
        List of pyhepmc particle objects
    event_number : int
        Event number assigned to all particles in the output tensor
    use_hepmc_mass : bool, optional
        If True, read mass from HepMC ``generated_mass``.
        If False, use PDG mass lookup. Default True to match C++ Delphes.
    dtype : torch.dtype, optional
        Desired data type of the output tensor. Default is torch.float64 for precision.

    Returns
    -------
    particles_tensor : torch.Tensor
        Tensor of shape ``(n_particles, N_FEATURES)`` with particle features.
    """
    n_particles = len(particles_list)
    if n_particles == 0:
        return torch.zeros((0, N_FEATURES), dtype=dtype)

    # ===== VECTORIZED EXTRACTION =====
    pids = np.array([p.pid for p in particles_list], dtype=np.int64)
    statuses = np.array([p.status for p in particles_list], dtype=np.int64)

    momenta = np.array(
        [[p.momentum.e, p.momentum.px, p.momentum.py, p.momentum.pz] for p in particles_list],
        dtype=np.float64,
    )

    vertices = np.array(
        [
            [
                p.production_vertex.position.x,
                p.production_vertex.position.y,
                p.production_vertex.position.z,
                p.production_vertex.position.t,
            ]
            if p.production_vertex
            else [0.0, 0.0, 0.0, 0.0]
            for p in particles_list
        ],
        dtype=np.float64,
    )

    if use_hepmc_mass:
        masses = np.array([p.generated_mass for p in particles_list], dtype=np.float64)
    else:
        masses = get_mass_from_pdg_id(pids)

    # ===== VECTORIZED COMPUTATIONS =====
    e = momenta[:, 0]
    px = momenta[:, 1]
    py = momenta[:, 2]
    pz = momenta[:, 3]

    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)

    # Match C++ Delphes: use ±999.9 for zero-pt particles
    eta = np.where(
        pt < PT_MIN,
        np.sign(pz) * 999.9,
        np.arcsinh(pz / pt),
    )
    eta = np.where((pt < PT_MIN) & (pz == 0), 0.0, eta)

    charges = get_charge_from_pdg_id(pids)

    # ===== BUILD TENSOR =====
    particles = np.zeros((n_particles, N_FEATURES), dtype=np.float64)
    particles[:, ColumnMap.PID] = pids
    particles[:, ColumnMap.STATUS] = statuses
    particles[:, ColumnMap.CHARGE] = charges
    particles[:, ColumnMap.E] = e
    particles[:, ColumnMap.PX] = px
    particles[:, ColumnMap.PY] = py
    particles[:, ColumnMap.PZ] = pz
    particles[:, ColumnMap.PT] = pt
    particles[:, ColumnMap.ETA] = eta
    particles[:, ColumnMap.PHI] = phi
    particles[:, ColumnMap.T] = vertices[:, 3]
    particles[:, ColumnMap.X] = vertices[:, 0]
    particles[:, ColumnMap.Y] = vertices[:, 1]
    particles[:, ColumnMap.Z] = vertices[:, 2]
    particles[:, ColumnMap.MASS] = masses
    particles[:, ColumnMap.EVENT_NUMBER] = event_number
    # ETA_OUTER, PHI_OUTER computed by ParticlePropagator

    return torch.from_numpy(particles).to(dtype)


def pythia_particles_to_tensor(
    event: Any,
    event_idx: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a Pythia8 event to a ColumnMap tensor.

    Stores **all** particles in the event (including intermediates), mirroring
    what a Pythia→HepMC write would produce.  Final-state particles receive
    ``STATUS = 1``; all others keep their Pythia status code.

    Parameters
    ----------
    event : pythia8mc event object
        The ``pythia.event`` iterable for a single generated event.
    event_idx : int
        Event index assigned to all particles in the output tensor.
    dtype : torch.dtype, optional
        Output tensor dtype.  Default ``torch.float64`` matches
        :func:`particles_to_tensor`.

    Returns
    -------
    torch.Tensor
        Shape ``(n_particles, N_FEATURES)``.
    """
    if event.size() == 0:
        return torch.zeros((0, N_FEATURES), dtype=dtype)

    rows = np.zeros((event.size(), N_FEATURES), dtype=np.float64)

    for i, part in enumerate(event):
        pid = int(part.id())
        status = 1 if part.isFinal() else int(part.status())
        charge = float(part.charge())
        e = float(part.e())
        px = float(part.px())
        py = float(part.py())
        pz = float(part.pz())
        pt = float(part.pT())
        phi = float(part.phi())

        # Match C++ Delphes: use ±999.9 for zero-pt particles
        # np.sign(0.0) == 0.0, so eta is naturally 0.0 when pz==0
        eta = float(np.sign(pz) * 999.9) if pt < PT_MIN else float(part.eta())

        t = float(part.tProd())
        x = float(part.xProd())
        y = float(part.yProd())
        z = float(part.zProd())
        mass = float(part.m())

        rows[i, ColumnMap.PID] = float(pid)
        rows[i, ColumnMap.STATUS] = float(status)
        rows[i, ColumnMap.CHARGE] = charge
        rows[i, ColumnMap.E] = e
        rows[i, ColumnMap.PX] = px
        rows[i, ColumnMap.PY] = py
        rows[i, ColumnMap.PZ] = pz
        rows[i, ColumnMap.PT] = pt
        rows[i, ColumnMap.ETA] = eta
        rows[i, ColumnMap.PHI] = phi
        rows[i, ColumnMap.T] = t
        rows[i, ColumnMap.X] = x
        rows[i, ColumnMap.Y] = y
        rows[i, ColumnMap.Z] = z
        rows[i, ColumnMap.MASS] = mass
        rows[i, ColumnMap.EVENT_NUMBER] = float(event_idx)
        # ETA_OUTER, PHI_OUTER, PASS_PROP, TRACK_RESOLUTION left as 0.0

    return torch.from_numpy(rows).to(dtype)

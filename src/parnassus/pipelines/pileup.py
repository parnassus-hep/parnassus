"""Delphes-style pile-up merger for the parametric pipeline.

Merges pile-up (PU) particles into a hard-scatter (HS) batch tensor using
fully vectorized PyTorch operations suitable for GPU execution.

The merger operates on the flat ``(N, N_FEATURES)`` batch tensor used by the
parametric pipeline, where multiple events are concatenated together and the
``EVENT_NUMBER`` column identifies which event each particle belongs to.
"""

from __future__ import annotations

import math
from typing import Self

import torch

from parnassus.configs.pileup import DelphesPileUpConfig
from parnassus.data.particle_io import ColumnMap
from parnassus.data.pileup_io import read_pileup_file

__all__ = ["DelphesPileUpMerger"]

# Speed of light in m/s
C_LIGHT = 2.99792458e8


class DelphesPileUpMerger:
    """Merge Delphes-style pile-up into the parametric pipeline batch tensor.

    All random sampling uses a dedicated ``torch.Generator`` on CPU for
    reproducibility.  Tensor transforms (vertex smearing, phi rotation) are
    applied as batched operations so the merger can run efficiently on GPU.

    Parameters
    ----------
    config : DelphesPileUpConfig
        Pile-up configuration (file path, mean pileup, smearing parameters).
    seed : int | None
        Optional random seed for the dedicated generator.
    """

    def __init__(self, config: DelphesPileUpConfig, seed: int | None = None) -> None:
        self.config = config

        # Load MinBias events from pileup file and convert to tensors
        raw_events = read_pileup_file(config.file_path)
        self._minbias_events: list[torch.Tensor] = [torch.from_numpy(arr) for arr in raw_events]
        # Store particle counts per MinBias event for efficient gathering
        self._minbias_counts = torch.tensor(
            [ev.shape[0] for ev in self._minbias_events], dtype=torch.long
        )

        # Create a dedicated CPU generator for reproducibility
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)

    @property
    def n_minbias_events(self) -> int:
        """Number of MinBias events loaded from the pileup file."""
        return len(self._minbias_events)

    def to(self, device: torch.device) -> Self:
        """Move MinBias data to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device.

        Returns
        -------
        Self
            This merger instance (for chaining).
        """
        self._minbias_events = [ev.to(device) for ev in self._minbias_events]
        self._minbias_counts = self._minbias_counts.to(device)
        return self

    def _sample_truncated_gaussian(
        self,
        n: int,
        sigma: float,
        max_spread: float,
    ) -> torch.Tensor:
        """Sample from a truncated Gaussian (CPU, using the dedicated generator).

        Parameters
        ----------
        n : int
            Number of samples.
        sigma : float
            Standard deviation of the Gaussian.
        max_spread : float
            Truncation bound (symmetric: [-max_spread, +max_spread]).

        Returns
        -------
        torch.Tensor
            Shape ``(n,)`` with dtype float64.
        """
        return (
            torch
            .empty(n, dtype=torch.float64)
            .normal_(0, sigma, generator=self._rng)
            .clamp_(-max_spread, max_spread)
        )

    def merge(self, stable_particles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge PU particles into the HS batch tensor.

        Parameters
        ----------
        stable_particles : torch.Tensor
            Shape ``(N_hs, N_FEATURES)`` — concatenated HS particles for all
            events in the batch, with ``EVENT_NUMBER`` identifying each event.

        Returns
        -------
        merged : torch.Tensor
            Shape ``(N_hs + N_pu, N_FEATURES)`` — HS + PU particles.
        truth : torch.Tensor
            Shape ``(N_hs, N_FEATURES)`` — HS particles after vertex smearing
            but before PU addition (cloned).
        """
        device = stable_particles.device

        # Work on a copy so the caller's tensor is never mutated.
        stable_particles = stable_particles.clone()

        # ----------------------------------------------------------------
        # 1. HS vertex smearing
        # ----------------------------------------------------------------
        if self.config.smear_hs_vertex:
            self._smear_hs_vertices(stable_particles)

        # Clone truth AFTER HS vertex smearing, BEFORE PU addition
        truth = stable_particles.clone()

        # ----------------------------------------------------------------
        # 2. Determine unique events and sample PU counts
        # ----------------------------------------------------------------
        event_col = stable_particles[:, ColumnMap.EVENT_NUMBER]
        unique_events, _inverse_idx = torch.unique(event_col, return_inverse=True)
        n_events = unique_events.shape[0]

        if n_events == 0 or self.config.mean_pileup <= 0:
            return stable_particles, truth

        # Sample number of PU interactions per event (Poisson)
        n_pu_per_event = torch.poisson(
            torch.full((n_events,), self.config.mean_pileup, dtype=torch.float64),
            generator=self._rng,
        ).long()

        total_pu_interactions = int(n_pu_per_event.sum().item())
        if total_pu_interactions == 0:
            return stable_particles, truth

        # ----------------------------------------------------------------
        # 3. Sample MinBias events and gather particles
        # ----------------------------------------------------------------
        mb_indices = torch.randint(
            0,
            self.n_minbias_events,
            (total_pu_interactions,),
            generator=self._rng,
        )

        # Gather all sampled MinBias particles (the ONE allowed loop)
        pu_particles = torch.cat([self._minbias_events[int(idx)] for idx in mb_indices], dim=0)

        # Number of particles per PU interaction
        counts_per_interaction = self._minbias_counts[mb_indices]  # (total_pu_interactions,)

        # ----------------------------------------------------------------
        # 4. Per-PU-interaction vertex smearing and phi rotation
        # ----------------------------------------------------------------
        # Sample dz, dt, dphi for each PU interaction
        dz = self._sample_truncated_gaussian(
            total_pu_interactions, self.config.sigma_z, self.config.max_z_spread
        )
        dt = self._sample_truncated_gaussian(
            total_pu_interactions, self.config.sigma_t, self.config.max_t_spread
        )
        dphi = torch.empty(total_pu_interactions, dtype=torch.float64).uniform_(
            -math.pi, math.pi, generator=self._rng
        )

        # Convert units: meters -> mm, seconds -> mm/c
        dz_mm = dz * 1e3
        dt_mmc = dt * C_LIGHT * 1e3

        # Map each PU interaction to an HS event number
        # n_pu_per_event tells us how many interactions per event;
        # repeat_interleave gives one event number per interaction
        event_number_per_interaction = torch.repeat_interleave(
            unique_events.cpu(), n_pu_per_event.cpu()
        )

        # Expand per-interaction values to per-particle
        dz_mm_per_particle = torch.repeat_interleave(dz_mm, counts_per_interaction.cpu())
        dt_mmc_per_particle = torch.repeat_interleave(dt_mmc, counts_per_interaction.cpu())
        dphi_per_particle = torch.repeat_interleave(dphi, counts_per_interaction.cpu())
        evnum_per_particle = torch.repeat_interleave(
            event_number_per_interaction, counts_per_interaction.cpu()
        )

        # Move to device
        pu_particles = pu_particles.to(device)
        dz_mm_per_particle = dz_mm_per_particle.to(device)
        dt_mmc_per_particle = dt_mmc_per_particle.to(device)
        dphi_per_particle = dphi_per_particle.to(device)
        evnum_per_particle = evnum_per_particle.to(device)

        # Apply vertex smearing
        pu_particles[:, ColumnMap.Z] = pu_particles[:, ColumnMap.Z] + dz_mm_per_particle
        pu_particles[:, ColumnMap.T] = pu_particles[:, ColumnMap.T] + dt_mmc_per_particle

        # Apply phi rotation
        cos_dphi = torch.cos(dphi_per_particle)
        sin_dphi = torch.sin(dphi_per_particle)

        # Rotate momentum (PX, PY)
        px = pu_particles[:, ColumnMap.PX].clone()
        py = pu_particles[:, ColumnMap.PY].clone()
        pu_particles[:, ColumnMap.PX] = px * cos_dphi - py * sin_dphi
        pu_particles[:, ColumnMap.PY] = px * sin_dphi + py * cos_dphi
        # PT is invariant under rotation — no update needed
        # Update PHI = atan2(PY', PX')
        pu_particles[:, ColumnMap.PHI] = torch.atan2(
            pu_particles[:, ColumnMap.PY], pu_particles[:, ColumnMap.PX]
        )

        # Rotate position (X, Y)
        x = pu_particles[:, ColumnMap.X].clone()
        y = pu_particles[:, ColumnMap.Y].clone()
        pu_particles[:, ColumnMap.X] = x * cos_dphi - y * sin_dphi
        pu_particles[:, ColumnMap.Y] = x * sin_dphi + y * cos_dphi

        # Stamp event numbers
        pu_particles[:, ColumnMap.EVENT_NUMBER] = evnum_per_particle

        # ----------------------------------------------------------------
        # 5. Concatenate HS + PU
        # ----------------------------------------------------------------
        merged = torch.cat([stable_particles, pu_particles], dim=0)
        return merged, truth

    def _smear_hs_vertices(self, particles: torch.Tensor) -> torch.Tensor:
        """Apply vertex smearing to hard-scatter particles.

        For each event, finds the first particle's (Z, T) as a reference,
        subtracts it from all particles in the event, then adds a random
        (dz, dt) drawn from truncated Gaussians.

        Parameters
        ----------
        particles : torch.Tensor
            Shape ``(N, N_FEATURES)`` — the HS batch tensor (modified in-place
            and returned).

        Returns
        -------
        torch.Tensor
            The same tensor, with Z and T columns smeared.
        """
        if particles.shape[0] == 0:
            return particles

        device = particles.device
        event_col = particles[:, ColumnMap.EVENT_NUMBER]

        # Find unique events and the mapping from particles to event index
        unique_events, inverse_idx = torch.unique(event_col, return_inverse=True)
        n_events = unique_events.shape[0]

        # Find the first particle per event using scatter_reduce_ to find
        # the minimum original index for each event
        arange = torch.arange(particles.shape[0], device=device, dtype=torch.float64)
        # Initialize with a large value so scatter_reduce(min) works
        first_idx = torch.full((n_events,), particles.shape[0], device=device, dtype=torch.float64)
        first_idx.scatter_reduce_(0, inverse_idx, arange, reduce="amin", include_self=False)
        first_idx = first_idx.long()

        # Extract reference (z0, t0) for each event
        z0_per_event = particles[first_idx, ColumnMap.Z]  # (n_events,)
        t0_per_event = particles[first_idx, ColumnMap.T]  # (n_events,)

        # Sample dz, dt for each event
        dz = self._sample_truncated_gaussian(
            n_events, self.config.sigma_z, self.config.max_z_spread
        )
        dt = self._sample_truncated_gaussian(
            n_events, self.config.sigma_t, self.config.max_t_spread
        )

        # Convert units
        dz_mm = (dz * 1e3).to(device)
        dt_mmc = (dt * C_LIGHT * 1e3).to(device)

        # Broadcast per-event values to per-particle via inverse_idx
        z0_per_particle = z0_per_event[inverse_idx]
        t0_per_particle = t0_per_event[inverse_idx]
        dz_mm_per_particle = dz_mm[inverse_idx]
        dt_mmc_per_particle = dt_mmc[inverse_idx]

        # Shift Z and T: subtract reference, add sampled offset
        particles[:, ColumnMap.Z] = particles[:, ColumnMap.Z] - z0_per_particle + dz_mm_per_particle
        particles[:, ColumnMap.T] = (
            particles[:, ColumnMap.T] - t0_per_particle + dt_mmc_per_particle
        )

        return particles

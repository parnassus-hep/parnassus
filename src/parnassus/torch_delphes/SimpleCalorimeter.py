"""PyTorch implementation of Delphes SimpleCalorimeter module.

Fills calorimeter towers, performs energy resolution smearing,
and creates energy flow objects (tracks and photons).

This module:
1. Bins particles into (η, φ) towers based on position
2. Accumulates energy per tower with PDG-dependent fractions
3. Applies log-normal energy resolution smearing
4. Computes energy flow by comparing calorimeter and track energies

TODO: Add Eem/Ehad fields to tower tensor representation
Currently, Eem (electromagnetic energy) and Ehad (hadronic energy) are computed
during ROOT file writing based on PID. A cleaner approach would be to:
- Add dedicated columns for Eem/Ehad in the tensor
- Set Eem=E, Ehad=0 for ECal towers (IsEcal=true)
- Set Eem=0, Ehad=E for HCal towers (IsEcal=false)
This would match C++ SimpleCalorimeter.cc:462-463 behavior more directly.
See EFlowMerger.md for details.
"""

from collections.abc import Callable

import torch
from torch import nn

from parnassus.data.particle_io import ColumnMap

N_FEATURES = len(ColumnMap)


class SimpleCalorimeter(nn.Module):
    """PyTorch implementation of Delphes SimpleCalorimeter module.

    Input:
        particles: (N_particles, N_FEATURES) - stable particles after propagation
        tracks: (N_tracks, N_FEATURES) - merged tracks after momentum smearing

    Output:
        towers: (N_towers, N_FEATURES) - calorimeter towers with smeared energy
        eflow_tracks: (N_eflow_tracks, N_FEATURES) - tracks for particle flow
        eflow_excess_neutrals: (N_eflow_excess_neutrals, N_FEATURES) - neutral excess towers
    """

    def __init__(
        self,
        eta_bins: list[float],  # From TCL EtaPhiBins
        phi_bins: list[list[float]],  # Phi bins per eta bin - len must equal len(eta_bins)
        energy_fractions: dict[int, float],  # PDG → fraction
        resolution_formula: str | Callable = "ecal_cms",
        energy_min: float = 0.5,
        energy_sig_min: float = 2.0,
        is_ecal: bool = True,
        smear_tower_center: bool = True,  # If True, smear eta/phi uniformly within bin
        compute_phi_bins_fn: Callable | None = None,  # Custom phi bin computation function
    ) -> None:
        super().__init__()

        # Store configuration
        self.energy_min = energy_min
        self.energy_sig_min = energy_sig_min
        self.is_ecal = is_ecal
        self.smear_tower_center = smear_tower_center

        # Store eta bins as tensor
        self.eta_bins = nn.Buffer(torch.tensor(eta_bins, dtype=torch.float64))  # (n_eta_bins,)

        # Store phi bins as padded 2D tensor for vectorized operations
        # phi_bins_2d[eta_bin_idx, :] gives phi bin edges for that eta bin
        # Padded with +inf so searchsorted returns max index for out-of-range values
        self._phi_bins_2d: nn.Buffer  # (n_eta_bins, max_n_phi_bins) padded with +inf
        self._n_phi_bins_per_eta: nn.Buffer  # (n_eta_bins,) number of valid phi bins per eta bin
        self._setup_phi_bins_2d(phi_bins)

        # Store energy fractions as lookup table
        self.energy_fractions = energy_fractions
        self.default_fraction = energy_fractions.get(0, 0.0)
        self._fraction_lut: nn.Buffer  # (max_pdg_id + 1,) lookup table for energy fractions
        self._setup_fraction_lookup()

        # Resolution formula
        if resolution_formula == "ecal_cms":
            self.resolution_func = self._ecal_cms_resolution
        elif resolution_formula == "hcal_cms":
            self.resolution_func = self._hcal_cms_resolution
        elif resolution_formula == "ecal_atlas":
            self.resolution_func = self._ecal_atlas_resolution
        elif resolution_formula == "hcal_atlas":
            self.resolution_func = self._hcal_atlas_resolution
        elif callable(resolution_formula):
            self.resolution_func = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")

        # Optional custom phi bin computation function
        self._custom_compute_phi_bins_fn = compute_phi_bins_fn

    def _setup_phi_bins_2d(self, phi_bins: list[list[float]]) -> None:
        """Setup phi bins as a padded 2D tensor for vectorized lookups.

        Creates:
        - _phi_bins_2d: (n_eta_bins, max_n_phi_bins) tensor, padded with +inf
        - _n_phi_bins_per_eta: (n_eta_bins,) tensor with count of valid phi bins per eta

        The +inf padding ensures searchsorted returns the last valid index for
        any phi value when that eta bin has fewer phi bins than the maximum.
        """
        n_eta_bins = len(phi_bins)
        max_n_phi_bins = max(len(pb) for pb in phi_bins)

        # Initialize with +inf (ensures searchsorted returns max index for padded region)
        phi_bins_2d = torch.full((n_eta_bins, max_n_phi_bins), float("inf"), dtype=torch.float64)
        n_phi_bins_per_eta = torch.zeros(n_eta_bins, dtype=torch.long)

        for i, pb in enumerate(phi_bins):
            n_bins = len(pb)
            phi_bins_2d[i, :n_bins] = torch.tensor(pb, dtype=torch.float64)
            n_phi_bins_per_eta[i] = n_bins

        self._phi_bins_2d = nn.Buffer(phi_bins_2d)
        self._n_phi_bins_per_eta = nn.Buffer(n_phi_bins_per_eta)
        self._max_n_phi_bins = max_n_phi_bins

    def forward(
        self,
        particles: torch.Tensor,  # (N_particles, N_FEATURES) - can span multiple events
        tracks: torch.Tensor,  # (N_tracks, N_FEATURES) - can span multiple events
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process particles and tracks from one or more events.

        Supports batched processing: particles and tracks can come from multiple events,
        identified by their EVENT_NUMBER column. Towers are aggregated per-event using
        event-indexed tower keys to prevent cross-event mixing.

        Parameters
        ----------
        particles: torch.Tensor
            Tensor of shape (N_particles, N_FEATURES) containing stable particles after propagation.
        tracks: torch.Tensor
            Tensor of shape (N_tracks, N_FEATURES) containing merged tracks after momentum smearing.

        Returns
        -------
        eflow_tracks: torch.Tensor
            (N_eflow_tracks, N_FEATURES) - tracks for particle flow
        towers: torch.Tensor
            (N_towers, N_FEATURES) - calorimeter towers
        eflow_excess_neutrals: torch.Tensor
            (N_eflow_excess_neutrals, N_FEATURES) - neutral excess towers
        """
        # 0. Extract Event Indices ########
        # Get event numbers and map to compact indices [0, n_events)
        particle_event_num = particles[:, ColumnMap.EVENT_NUMBER]
        track_event_num = tracks[:, ColumnMap.EVENT_NUMBER]

        # Find unique events across both particles and tracks
        all_event_nums = torch.cat([particle_event_num, track_event_num])
        unique_event_nums, inverse_indices = torch.unique(all_event_nums, return_inverse=True)
        n_events = len(unique_event_nums)

        # Map event numbers to compact indices
        n_particles = particles.shape[0]
        particle_event_idx = inverse_indices[:n_particles]
        track_event_idx = inverse_indices[n_particles:]

        # 1. Compute Energy Fractions ########
        # Get PDG IDs
        particle_pids = particles[:, ColumnMap.PID]
        track_pids = tracks[:, ColumnMap.PID]

        # Compute energy fractions based on PDG ID
        # This matches fTowerFractions and fTrackFractions in C++
        particle_energy_fractions = self._compute_energy_fractions(particle_pids)
        track_energy_fractions = self._compute_energy_fractions(track_pids)

        # 2. Bin particles into Towers ########
        # C++: if(fraction < 1.0E-9) continue;  // particles with zero fraction are skipped
        # Get particle positions (eta, phi from Position, not Momentum)
        # Use .contiguous() to avoid performance warning from searchsorted
        particle_eta = particles[:, ColumnMap.ETA_OUTER].contiguous()  # Position-based eta
        particle_phi = particles[:, ColumnMap.PHI_OUTER].contiguous()  # Position-based phi

        # Find eta bin for each particle
        particle_eta_bin = torch.searchsorted(self.eta_bins, particle_eta)

        # Find phi bin for each particle - variable per eta bin
        particle_phi_bin, particle_valid = self._compute_phi_bins(particle_phi, particle_eta_bin)

        # Particles: filter by energy fraction (C++: if(fraction < 1.0E-9) continue)
        particle_valid = particle_valid & (particle_energy_fractions > 1e-9)

        # 3. Bin tracks into Towers ########
        # C++: tracks are NOT filtered by fraction before binning
        # Get track positions (eta and phi from outer position, set by ParticlePropagator)
        # C++ uses track->Position.Eta() and track->Position.Phi()
        # Use .contiguous() to avoid performance warning from searchsorted
        track_eta = tracks[:, ColumnMap.ETA_OUTER].contiguous()  # Position-based eta
        track_phi = tracks[:, ColumnMap.PHI_OUTER].contiguous()  # Position-based phi

        # Find eta bin for each track
        track_eta_bin = torch.searchsorted(self.eta_bins, track_eta)

        # Find phi bin for each track - variable per eta bin
        # Tracks: do NOT filter by energy fraction (C++ doesn't skip tracks with fraction < 1e-9)
        track_phi_bin, track_valid = self._compute_phi_bins(track_phi, track_eta_bin)

        # 4. Aggregate Energies per Tower ########
        # C++ Delphes:
        #   - Sorts all hits by (etaBin, phiBin)
        #   - For each tower (unique etaBin, phiBin):
        #       fTowerEnergy += momentum.E() * fTowerFractions[number] (particles)
        #       fTrackEnergy += momentum.E() * fTrackFractions[number] (tracks with fraction > 1e-9)
        #   - Time weighting is also done but we handle that separately

        # Get energies
        particle_energy = particles[:, ColumnMap.E]
        track_energy = tracks[:, ColumnMap.E]

        # Compute weighted energies (energy x fraction)
        particle_weighted_energy = particle_energy * particle_energy_fractions
        track_weighted_energy = track_energy * track_energy_fractions

        # Create unique tower index from (event_idx, eta_bin, phi_bin)
        # Using event-indexed keys to prevent cross-event mixing in batched processing
        max_phi_bins = self._max_n_phi_bins
        n_eta_bins = len(self.eta_bins)
        n_towers_per_event = n_eta_bins * max_phi_bins

        # Global tower index = event_idx * n_towers_per_event + eta_bin * max_phi_bins + phi_bin
        particle_local_tower_idx = particle_eta_bin * max_phi_bins + particle_phi_bin
        track_local_tower_idx = track_eta_bin * max_phi_bins + track_phi_bin

        particle_tower_idx = particle_event_idx * n_towers_per_event + particle_local_tower_idx
        track_tower_idx = track_event_idx * n_towers_per_event + track_local_tower_idx

        # For tracks, only those with fraction > 1e-9 contribute to fTrackEnergy
        # C++: if(fTrackFractions[number] > 1.0E-9) { fTrackEnergy += energy; ... }
        track_has_fraction = track_energy_fractions > 1e-9

        # Find unique towers from valid particles AND valid tracks
        # NOTE: In C++, particles are filtered by fraction BEFORE creating tower hits,
        #       but tracks are NOT filtered by fraction for tower creation.
        #       Tracks create tower hits regardless of fraction; the fraction check
        #       only affects whether they contribute to fTrackEnergy.
        valid_particle_tower_idx = torch.where(
            particle_valid, particle_tower_idx, torch.full_like(particle_tower_idx, -1)
        )
        valid_track_tower_idx = torch.where(
            track_valid,  # NOT filtered by fraction for tower creation
            track_tower_idx,
            torch.full_like(track_tower_idx, -1),
        )

        # Combine all valid tower indices to find unique towers
        all_tower_idx = torch.cat([
            valid_particle_tower_idx[particle_valid],
            valid_track_tower_idx[track_valid],  # NOT filtered by fraction
        ])
        unique_tower_idx = torch.unique(all_tower_idx[all_tower_idx >= 0])
        n_towers = len(unique_tower_idx)

        # Create mapping from global tower index to compact tower index [0, n_towers)
        # Size must accommodate all events: n_events * n_towers_per_event
        max_global_tower_idx = n_events * n_towers_per_event
        tower_idx_map = torch.full(
            (max_global_tower_idx,), -1, dtype=torch.long, device=particles.device
        )
        tower_idx_map[unique_tower_idx] = torch.arange(n_towers, device=particles.device)

        # Map particles and tracks to compact tower indices
        # Clamp tower indices to valid range before lookup
        # (invalid particles have valid=False anyway)
        max_idx = max_global_tower_idx - 1
        particle_tower_idx_clamped = particle_tower_idx.clamp(0, max_idx)
        track_tower_idx_clamped = track_tower_idx.clamp(0, max_idx)

        particle_compact_idx = torch.where(
            particle_valid,
            tower_idx_map[particle_tower_idx_clamped],
            torch.full_like(particle_tower_idx, -1),
        )
        track_compact_idx = torch.where(
            track_valid & track_has_fraction,
            tower_idx_map[track_tower_idx_clamped],
            torch.full_like(track_tower_idx, -1),
        )

        # Aggregate particle energies per tower using scatter_add
        tower_energy = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        valid_particle_mask = particle_compact_idx >= 0
        tower_energy.scatter_add_(
            0,
            particle_compact_idx[valid_particle_mask],
            particle_weighted_energy[valid_particle_mask],
        )

        # Aggregate track energies per tower
        tower_track_energy = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        valid_track_mask = track_compact_idx >= 0
        tower_track_energy.scatter_add_(
            0, track_compact_idx[valid_track_mask], track_weighted_energy[valid_track_mask]
        )

        # 4b. Compute Time-Weighted Average per Tower ########
        # C++: fTowerTime += energy * energy * position.T();  // sigma_t ~ 1/E
        #      fTowerTimeWeight += energy * energy;
        #      time = (fTowerTimeWeight < 1.0E-09) ? 0.0 : fTowerTime / fTowerTimeWeight;
        # Note: The "energy" here is the weighted energy (E * fraction)

        # Get particle times
        particle_time = particles[:, ColumnMap.T]

        # Compute E^2 weights for particles (weighted_energy^2)
        particle_time_weight = particle_weighted_energy**2

        # Compute E^2 * T for particles
        particle_time_weighted = particle_time_weight * particle_time

        # Aggregate time-weighted sum per tower
        tower_time_weighted = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        tower_time_weighted.scatter_add_(
            0,
            particle_compact_idx[valid_particle_mask],
            particle_time_weighted[valid_particle_mask],
        )

        # Aggregate time weights per tower
        tower_time_weight = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        tower_time_weight.scatter_add_(
            0, particle_compact_idx[valid_particle_mask], particle_time_weight[valid_particle_mask]
        )

        # Compute final tower time: time = time_weighted_sum / weight_sum
        # Set to 0 if weight is too small (matching C++ check: fTowerTimeWeight < 1.0E-09)
        tower_time = torch.where(
            tower_time_weight > 1e-9,
            tower_time_weighted / tower_time_weight,
            torch.zeros_like(tower_time_weighted),
        )

        # Extract event_idx, eta_bin, and phi_bin from global tower index
        # Global tower index = event_idx * n_towers_per_event + eta_bin * max_phi_bins + phi_bin
        tower_event_idx = unique_tower_idx // n_towers_per_event
        tower_local_idx = unique_tower_idx % n_towers_per_event
        tower_eta_bin = tower_local_idx // max_phi_bins
        tower_phi_bin = tower_local_idx % max_phi_bins

        # Map tower event indices back to original event numbers for output
        tower_event_num = unique_event_nums[tower_event_idx]

        # 5. Compute Tower Centers ########
        # C++:
        #   fTowerEta = 0.5 * (fEtaBins[etaBin - 1] + fEtaBins[etaBin]);
        #   fTowerPhi = 0.5 * ((*phiBins)[phiBin - 1] + (*phiBins)[phiBin]);
        #   fTowerEdges[0] = fEtaBins[etaBin - 1];  // eta_lo
        #   fTowerEdges[1] = fEtaBins[etaBin];       // eta_hi
        #   fTowerEdges[2] = (*phiBins)[phiBin - 1]; // phi_lo
        #   fTowerEdges[3] = (*phiBins)[phiBin];     // phi_hi

        # Compute tower eta centers and edges
        # tower_eta_bin is in range [1, n_eta_bins-1] for valid towers
        tower_eta_lo = self.eta_bins[tower_eta_bin - 1]
        tower_eta_hi = self.eta_bins[tower_eta_bin]

        # Compute tower phi centers and edges using vectorized 2D gather
        # Gather the phi bins row for each tower's eta bin: (n_towers, max_n_phi_bins)
        tower_phi_bins = self._phi_bins_2d[tower_eta_bin]  # (n_towers, max_n_phi_bins)

        # Gather phi_lo = phi_bins[phi_bin - 1], phi_hi = phi_bins[phi_bin]
        # Use gather to select the specific phi bin edges for each tower
        tower_phi_lo = tower_phi_bins.gather(1, (tower_phi_bin - 1).unsqueeze(1)).squeeze(1)
        tower_phi_hi = tower_phi_bins.gather(1, tower_phi_bin.unsqueeze(1)).squeeze(1)

        # Compute bin centers for resolution calculation
        # C++ uses bin centers (fTowerEta, fTowerPhi) for resolution, NOT smeared positions
        tower_eta_center = 0.5 * (tower_eta_lo + tower_eta_hi)
        tower_phi_center = 0.5 * (tower_phi_lo + tower_phi_hi)

        # 6. Apply Resolution Smearing ########
        # C++:
        #   sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, fTowerEnergy);
        #   energy = LogNormal(fTowerEnergy, sigma);
        #   sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, energy); // recompute with smeared
        #   if(energy < fEnergyMin || energy < fEnergySignificanceMin * sigma) energy = 0.0;
        #
        # IMPORTANT: C++ uses bin CENTER (fTowerEta) for resolution, then smears position AFTER

        # Compute sigma before smearing - using bin CENTER
        sigma_before = self.resolution_func(tower_eta_center, tower_energy)

        # Apply LogNormal smearing
        tower_energy_smeared = self._log_normal_smear(tower_energy, sigma_before)

        # Recompute sigma with smeared energy - still using bin CENTER
        sigma_after = self.resolution_func(tower_eta_center, tower_energy_smeared)

        # Apply energy thresholds
        # Tower energy is zeroed if below minimum or below significance threshold
        below_min = tower_energy_smeared < self.energy_min
        below_sig = tower_energy_smeared < self.energy_sig_min * sigma_after
        tower_energy_final = torch.where(
            below_min | below_sig, torch.zeros_like(tower_energy_smeared), tower_energy_smeared
        )

        # 6b. Smear Tower Position (AFTER resolution smearing) ########
        # C++: Position smearing happens AFTER energy smearing/thresholds
        #   if(fSmearTowerCenter) {
        #       eta = gRandom->Uniform(fTowerEdges[0], fTowerEdges[1]);
        #       phi = gRandom->Uniform(fTowerEdges[2], fTowerEdges[3]);
        #   } else {
        #       eta = fTowerEta;  phi = fTowerPhi;
        #   }
        if self.smear_tower_center:
            tower_eta = tower_eta_lo + torch.rand(
                n_towers, dtype=torch.float64, device=particles.device
            ) * (tower_eta_hi - tower_eta_lo)
            tower_phi = tower_phi_lo + torch.rand(
                n_towers, dtype=torch.float64, device=particles.device
            ) * (tower_phi_hi - tower_phi_lo)
        else:
            tower_eta = tower_eta_center
            tower_phi = tower_phi_center

        # 7. Compute track sigma per Tower ########
        # C++:
        #   sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, momentum.E());
        #   if(sigma / momentum.E() < track->TrackResolution)
        #       energyGuess = energy;  // energy = momentum.E() * fraction
        #   else
        #       energyGuess = momentum.E();
        #   fTrackSigma += (
        #       (track->TrackResolution * energyGuess) * (track->TrackResolution * energyGuess);
        #   )
        #   ...
        #   fTrackSigma = TMath::Sqrt(fTrackSigma);  // in FinalizeTower
        #
        # NOTE: C++ uses fTowerEta (bin center) for resolution, not smeared position

        # Get track resolution from MomentumSmearing (stored in TRACK_RESOLUTION column)
        track_momentum_resolution = tracks[:, ColumnMap.TRACK_RESOLUTION]
        n_tracks = tracks.shape[0]

        # For valid tracks, get the tower eta CENTER using the track's tower assignment
        # Only tracks with fraction > 1e-9 contribute to fTrackSigma
        track_tower_eta_center = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        track_calo_sigma = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        track_energy_guess = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        track_sigma_sq = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)

        # Mask for tracks that contribute to track sigma (valid AND has fraction)
        track_sigma_valid = track_valid & track_has_fraction

        # Get tower eta CENTER for each valid track from the tower mapping
        # Use bin center (tower_eta_center) for resolution, not smeared position
        # For tracks with valid tower assignment, directly index into tower_eta_center
        valid_tower_mask = track_compact_idx >= 0
        track_tower_eta_center[valid_tower_mask] = tower_eta_center[
            track_compact_idx[valid_tower_mask]
        ]

        # Compute calorimeter sigma at tower eta CENTER using track energy
        # C++: sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, momentum.E())
        track_calo_sigma[track_sigma_valid] = self.resolution_func(
            track_tower_eta_center[track_sigma_valid], track_energy[track_sigma_valid]
        )

        # Determine energy_guess based on resolution comparison
        # C++:  if(sigma / momentum.E() < track->TrackResolution)
        #           energyGuess = energy;
        #       else energyGuess = momentum.E()
        calo_relative_sigma = track_calo_sigma / (track_energy + 1e-30)  # Avoid div by zero
        use_weighted_energy = calo_relative_sigma < track_momentum_resolution

        # energy_guess = fraction * track_energy if calo resolution better, else track_energy
        track_energy_guess = torch.where(
            use_weighted_energy & track_sigma_valid,
            track_weighted_energy,  # energy = momentum.E() * fraction
            torch.where(track_sigma_valid, track_energy, torch.zeros_like(track_energy)),
        )

        # Compute per-track sigma squared: (track_resolution * energy_guess)^2
        track_sigma_sq = (track_momentum_resolution * track_energy_guess) ** 2

        # Aggregate track_sigma_sq per tower using scatter_add
        tower_track_sigma_sq = torch.zeros(n_towers, dtype=torch.float64, device=tracks.device)
        tower_track_sigma_sq.scatter_add_(
            0, track_compact_idx[track_sigma_valid], track_sigma_sq[track_sigma_valid]
        )

        # Final tower track sigma = sqrt(sum of squares)
        tower_track_sigma = torch.sqrt(tower_track_sigma_sq)

        # 8. Identify Neutral Excess and Create eflow objects ########
        # C++:
        #   neutralEnergy = max((energy - fTrackEnergy), 0.0);
        #   neutralSigma = neutralEnergy / TMath::Sqrt(fTrackSigma * fTrackSigma + sigma * sigma);
        #
        #   if(neutralEnergy > fEnergyMin && neutralSigma > fEnergySignificanceMin) {
        #       // Create EFlowTower with neutralEnergy
        #       // Clone tracks to EFlowTrack unchanged
        #   } else if(fTrackEnergy > 0.0) {
        #       // Rescale tracks based on weighted average of calo and track measurements
        #       weightTrack = 1 / (fTrackSigma^2)
        #       weightCalo = 1 / (sigma^2)
        #       bestEnergyEstimate =
        #           (weightTrack * fTrackEnergy + weightCalo * energy) / (weightTrack + weightCalo)
        #       rescaleFactor = bestEnergyEstimate / fTrackEnergy
        #       // Clone tracks to EFlowTrack with rescaled pT
        #   }

        # Use final (thresholded) tower energy and sigma_after for this computation
        energy = tower_energy_final  # After smearing and thresholds
        sigma = sigma_after

        # Compute neutral energy per tower
        neutral_energy = torch.clamp(energy - tower_track_energy, min=0.0)

        # Compute neutral sigma per tower
        # neutralSigma = neutralEnergy / sqrt(trackSigma² + sigma²)
        denominator = torch.sqrt(tower_track_sigma**2 + sigma**2)
        neutral_sigma = torch.where(
            denominator > 0, neutral_energy / denominator, torch.zeros_like(neutral_energy)
        )

        # Case A: Neutral excess is significant
        # Condition: neutralEnergy > EnergyMin AND neutralSigma > EnergySignificanceMin
        significant_neutral = (neutral_energy > self.energy_min) & (
            neutral_sigma > self.energy_sig_min
        )

        # Case B: Neutral excess is NOT significant but has track energy
        # Condition: NOT significant_neutral AND tower_track_energy > 0
        rescale_tracks = (~significant_neutral) & (tower_track_energy > 0)

        # Compute rescale factor for Case B towers
        # weightTrack = 1 / (trackSigma^2), weightCalo = 1 / (sigma^2)
        # bestEnergyEstimate =
        # (weightTrack * trackEnergy + weightCalo * energy) / (weightTrack + weightCalo)
        weight_track = torch.where(
            tower_track_sigma > 0, 1.0 / (tower_track_sigma**2), torch.zeros_like(tower_track_sigma)
        )
        weight_calo = torch.where(sigma > 0, 1.0 / (sigma**2), torch.zeros_like(sigma))

        total_weight = weight_track + weight_calo
        best_energy_estimate = torch.where(
            total_weight > 0,
            (weight_track * tower_track_energy + weight_calo * energy) / total_weight,
            tower_track_energy,  # Fallback to track energy if no weights
        )

        rescale_factor = torch.where(
            tower_track_energy > 0,
            best_energy_estimate / tower_track_energy,
            torch.ones_like(tower_track_energy),
        )

        # ===== Create Tower output =====
        # Towers with energy > 0 after thresholds
        tower_has_energy = tower_energy_final > 0

        # Tower output tensor: [PT, Eta, Phi, E, Eem, Ehad, T, Edges...]
        # For ECal: Eem = energy, Ehad = 0
        tower_pt = tower_energy_final / torch.cosh(tower_eta)

        # Build tower output (towers with energy > 0)
        n_valid_towers = int(tower_has_energy.sum())

        # ===== Create EFlowTower output (neutral excess) =====
        # Only for towers with significant neutral excess
        n_eflow_towers = int(significant_neutral.sum())

        eflow_tower_energy = neutral_energy[significant_neutral]
        eflow_tower_eta = tower_eta[significant_neutral]
        eflow_tower_phi = tower_phi[significant_neutral]
        eflow_tower_pt = eflow_tower_energy / torch.cosh(eflow_tower_eta)
        eflow_tower_time = tower_time[significant_neutral]

        # ===== Create EFlowTrack output =====
        # Tracks are output in two cases:
        # 1. Significant neutral: clone track unchanged
        # 2. Rescale: apply rescale factor to track momentum

        # For each track, determine which case applies based on its tower
        # track_compact_idx maps tracks to tower indices

        # Get the tower status for each track
        # Only tracks with valid sigma (track_sigma_valid) are in towers
        track_in_significant_tower = torch.zeros(n_tracks, dtype=torch.bool, device=tracks.device)
        track_in_rescale_tower = torch.zeros(n_tracks, dtype=torch.bool, device=tracks.device)
        track_rescale_factor = torch.ones(n_tracks, dtype=torch.float64, device=tracks.device)

        # For tracks with valid tower assignment, directly gather tower properties
        valid_sigma_tower_mask = (track_compact_idx >= 0) & track_sigma_valid
        valid_tower_indices = track_compact_idx[valid_sigma_tower_mask]

        # Gather tower properties for valid tracks
        tower_is_significant = significant_neutral[valid_tower_indices]
        tower_is_rescale = rescale_tracks[valid_tower_indices]
        tower_rescale_values = rescale_factor[valid_tower_indices]

        # Assign to track arrays
        # Track is in significant tower if its tower has significant neutral excess
        track_in_significant_tower[valid_sigma_tower_mask] = tower_is_significant

        # Track is in rescale tower if its tower needs rescaling AND is not significant
        track_in_rescale_tower[valid_sigma_tower_mask] = tower_is_rescale & ~tower_is_significant

        # Assign rescale factors (only matters for tracks in rescale towers)
        track_rescale_factor[valid_sigma_tower_mask] = torch.where(
            tower_is_rescale & ~tower_is_significant,
            tower_rescale_values,
            torch.ones_like(tower_rescale_values),
        )

        # Tracks that become EFlowTracks: either in significant tower OR in rescale tower
        track_is_eflow = track_in_significant_tower | track_in_rescale_tower

        # Also include tracks with fraction < 1e-9 (they go directly to EFlowTrack in C++)
        # These are tracks that are valid but don't contribute to tower energy
        track_no_fraction = track_valid & (~track_has_fraction)
        track_is_eflow = track_is_eflow | track_no_fraction

        # Create EFlowTrack tensor
        # Clone the track and apply rescale factor if applicable
        eflow_tracks = tracks.clone()

        # Apply rescale factor to tracks in rescale towers
        # PT is rescaled, then PX, PY, PZ, E are recomputed
        original_pt = eflow_tracks[:, ColumnMap.PT]
        rescaled_pt = original_pt * track_rescale_factor

        # Only apply to tracks in rescale towers
        eflow_tracks[:, ColumnMap.PT] = torch.where(
            track_in_rescale_tower, rescaled_pt, original_pt
        )

        # Recompute PX, PY from rescaled PT
        eta = eflow_tracks[:, ColumnMap.ETA]
        phi = eflow_tracks[:, ColumnMap.PHI]
        mass = eflow_tracks[:, ColumnMap.MASS]

        eflow_tracks[:, ColumnMap.PX] = torch.where(
            track_in_rescale_tower, rescaled_pt * torch.cos(phi), eflow_tracks[:, ColumnMap.PX]
        )
        eflow_tracks[:, ColumnMap.PY] = torch.where(
            track_in_rescale_tower, rescaled_pt * torch.sin(phi), eflow_tracks[:, ColumnMap.PY]
        )
        eflow_tracks[:, ColumnMap.PZ] = torch.where(
            track_in_rescale_tower, rescaled_pt * torch.sinh(eta), eflow_tracks[:, ColumnMap.PZ]
        )

        # Recompute E from P and mass
        p_sq = (
            eflow_tracks[:, ColumnMap.PX] ** 2
            + eflow_tracks[:, ColumnMap.PY] ** 2
            + eflow_tracks[:, ColumnMap.PZ] ** 2
        )
        eflow_tracks[:, ColumnMap.E] = torch.where(
            track_in_rescale_tower, torch.sqrt(p_sq + mass**2), eflow_tracks[:, ColumnMap.E]
        )

        # Filter to only EFlow tracks
        eflow_tracks = eflow_tracks[track_is_eflow]

        # ===== Create Tower Tensor with COLUMN_MAP format =====
        # Tower tensor: (n_valid_towers, N_FEATURES)
        r_calo = 1.29  # meters, CMS ECAL radius
        towers = torch.zeros(
            n_valid_towers, N_FEATURES, dtype=torch.float64, device=particles.device
        )

        if n_valid_towers > 0:
            valid_tower_energy = tower_energy_final[tower_has_energy]
            valid_tower_eta = tower_eta[tower_has_energy]
            valid_tower_phi = tower_phi[tower_has_energy]
            valid_tower_pt = tower_pt[tower_has_energy]
            valid_tower_event_num = tower_event_num[tower_has_energy]

            # Set tower properties
            towers[:, ColumnMap.PID] = 22.0  # Photon for ECAL towers
            towers[:, ColumnMap.STATUS] = 1.0
            towers[:, ColumnMap.CHARGE] = 0.0
            towers[:, ColumnMap.E] = valid_tower_energy
            towers[:, ColumnMap.PT] = valid_tower_pt
            towers[:, ColumnMap.ETA] = valid_tower_eta
            towers[:, ColumnMap.PHI] = valid_tower_phi

            # Compute PX, PY, PZ from PT, ETA, PHI (massless)
            cos_valid_tower_phi = torch.cos(valid_tower_phi)
            sin_valid_tower_phi = torch.sin(valid_tower_phi)
            sinh_valid_tower_eta = torch.sinh(valid_tower_eta)
            towers[:, ColumnMap.PX] = valid_tower_pt * cos_valid_tower_phi
            towers[:, ColumnMap.PY] = valid_tower_pt * sin_valid_tower_phi
            towers[:, ColumnMap.PZ] = valid_tower_pt * sinh_valid_tower_eta

            # Position (approximate at calorimeter surface)
            towers[:, ColumnMap.X] = r_calo * cos_valid_tower_phi * 1000  # mm
            towers[:, ColumnMap.Y] = r_calo * sin_valid_tower_phi * 1000  # mm
            towers[:, ColumnMap.Z] = r_calo * sinh_valid_tower_eta * 1000  # mm
            towers[:, ColumnMap.T] = tower_time[tower_has_energy]  # Time-weighted average
            towers[:, ColumnMap.MASS] = 0.0

            # Outer position same as momentum direction
            towers[:, ColumnMap.ETA_OUTER] = valid_tower_eta
            towers[:, ColumnMap.PHI_OUTER] = valid_tower_phi

            # Set EVENT_NUMBER from tower's event (supports batched multi-event processing)
            towers[:, ColumnMap.EVENT_NUMBER] = valid_tower_event_num

        # ===== Create EFlowPhoton Tensor with COLUMN_MAP format =====
        # EFlowPhoton tensor: (n_eflow_towers, N_FEATURES)
        # These are towers with significant neutral excess
        eflow_excess_neutrals = torch.zeros(
            n_eflow_towers, N_FEATURES, dtype=torch.float64, device=particles.device
        )

        if n_eflow_towers > 0:
            # Get event numbers for eflow towers
            eflow_tower_event_num = tower_event_num[significant_neutral]

            # Set eflow photon properties
            eflow_excess_neutrals[:, ColumnMap.PID] = 22.0  # Photon
            eflow_excess_neutrals[:, ColumnMap.STATUS] = 1.0
            eflow_excess_neutrals[:, ColumnMap.CHARGE] = 0.0
            eflow_excess_neutrals[:, ColumnMap.E] = eflow_tower_energy
            eflow_excess_neutrals[:, ColumnMap.PT] = eflow_tower_pt
            eflow_excess_neutrals[:, ColumnMap.ETA] = eflow_tower_eta
            eflow_excess_neutrals[:, ColumnMap.PHI] = eflow_tower_phi

            # Compute PX, PY, PZ from PT, ETA, PHI (massless)
            cos_eflow_tower_phi = torch.cos(eflow_tower_phi)
            sin_eflow_tower_phi = torch.sin(eflow_tower_phi)
            sinh_eflow_tower_eta = torch.sinh(eflow_tower_eta)
            eflow_excess_neutrals[:, ColumnMap.PX] = eflow_tower_pt * cos_eflow_tower_phi
            eflow_excess_neutrals[:, ColumnMap.PY] = eflow_tower_pt * sin_eflow_tower_phi
            eflow_excess_neutrals[:, ColumnMap.PZ] = eflow_tower_pt * sinh_eflow_tower_eta

            # Position (approximate at calorimeter surface)
            eflow_excess_neutrals[:, ColumnMap.X] = r_calo * cos_eflow_tower_phi * 1000  # mm
            eflow_excess_neutrals[:, ColumnMap.Y] = r_calo * sin_eflow_tower_phi * 1000  # mm
            eflow_excess_neutrals[:, ColumnMap.Z] = r_calo * sinh_eflow_tower_eta * 1000  # mm
            eflow_excess_neutrals[:, ColumnMap.T] = eflow_tower_time  # Time-weighted average
            eflow_excess_neutrals[:, ColumnMap.MASS] = 0.0

            # Outer position same as momentum direction
            eflow_excess_neutrals[:, ColumnMap.ETA_OUTER] = eflow_tower_eta
            eflow_excess_neutrals[:, ColumnMap.PHI_OUTER] = eflow_tower_phi

            # Set masks

            # Set EVENT_NUMBER from tower's event (supports batched multi-event processing)
            eflow_excess_neutrals[:, ColumnMap.EVENT_NUMBER] = eflow_tower_event_num

        # Return results
        return eflow_tracks, towers, eflow_excess_neutrals

    def _compute_phi_bins(
        self,
        phi: torch.Tensor,  # (N,) phi values
        eta_bin: torch.Tensor,  # (N,) eta bin indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute phi bin for each particle/track using variable phi binning per eta.

        Uses vectorized operations with padded 2D phi bin tensor.
        Supports custom override via compute_phi_bins_fn in __init__.

        In C++ Delphes:
            phiBins = fPhiBins[etaBin];
            itPhiBin = lower_bound(phiBins->begin(), phiBins->end(), position.Phi());
            if(itPhiBin == phiBins->begin() || itPhiBin == phiBins->end()) continue;
            phiBin = distance(phiBins->begin(), itPhiBin);

        Parameters
        ----------
        phi: torch.Tensor
            (N,) tensor of phi values for particles/tracks
        eta_bin: torch.Tensor
            (N,) tensor of eta bin indices

        Returns
        -------
        phi_bin: torch.Tensor
            (N,) phi bin indices
        valid: torch.Tensor
            (N,) boolean mask for valid bins (eta and phi both in range)
        """
        # Allow custom override
        if self._custom_compute_phi_bins_fn is not None:
            return self._custom_compute_phi_bins_fn(phi, eta_bin, self)

        # Filter by valid eta bin first: [1, len(eta_bins) - 1]
        n_eta_bins = len(self.eta_bins)
        valid_eta = (eta_bin > 0) & (eta_bin < n_eta_bins)

        # Clamp eta_bin to valid range for safe indexing (invalid will be filtered by valid_eta)
        eta_bin_clamped = eta_bin.clamp(0, n_eta_bins - 1)

        # Gather phi bins for each particle's eta bin: (N, max_n_phi_bins)
        # This is the key vectorization - we get all phi bins for all particles at once
        phi_bins_per_particle = self._phi_bins_2d[eta_bin_clamped]  # (N, max_n_phi_bins)

        # Vectorized searchsorted along the last dimension
        # For each particle, find which phi bin its phi value falls into
        phi_expanded = phi.unsqueeze(1)  # (N, 1)

        # Compare phi against all bin edges: phi_bins_per_particle is (N, max_n_phi_bins)
        # We want: for each particle i, find smallest j such that phi_bins[i,j] > phi[i]
        # This is equivalent to searchsorted behavior
        phi_bin = torch.searchsorted(
            phi_bins_per_particle.contiguous(), phi_expanded.contiguous()
        ).squeeze(1)  # (N,)

        # Get the number of valid phi bins for each particle's eta bin
        n_phi_bins = self._n_phi_bins_per_eta[eta_bin_clamped]  # (N,)

        # Valid phi bin range: [1, n_phi_bins - 1]
        # (not at begin or end, matching C++ continue conditions)
        valid_phi = (phi_bin > 0) & (phi_bin < n_phi_bins)

        # Combine eta and phi validity
        valid = valid_eta & valid_phi

        return phi_bin, valid

    def _setup_fraction_lookup(self) -> None:
        """Setup efficient PDG → energy fraction lookup using a fully dense LUT.

        Creates a single dense tensor that covers all PDG IDs up to the maximum
        configured PDG ID (e.g., 1000045 for neutralinos). This enables:
        - Pure tensor operations with no conditional branches
        - Full compatibility with batched processing
        - Clean gradient flow for differentiability

        Memory usage is ~8MB for max PDG ID of 1000045, which is negligible.
        """
        # Find the maximum PDG ID we need to handle
        max_pdg_id = max(abs(pid) for pid in self.energy_fractions)

        # Build fully dense LUT: lut[abs_pid] = fraction
        # All unspecified PDG IDs get the default fraction
        lut = torch.full((max_pdg_id + 1,), self.default_fraction, dtype=torch.float64)

        for pid, frac in self.energy_fractions.items():
            lut[abs(pid)] = frac

        self._fraction_lut = nn.Buffer(lut)
        self._max_pdg_id = max_pdg_id

    def _compute_energy_fractions(self, pids: torch.Tensor) -> torch.Tensor:
        """Compute energy fractions for particles based on their PDG IDs.

        Uses fully dense LUT lookup - pure tensor operation with no branches.
        Fully compatible with batching and differentiable.

        This matches the C++ Delphes behavior:
        1. Look up |PDG ID| in the fraction map
        2. If not found, use the default fraction (PDG=0)

        Parameters
        ----------
        pids: torch.Tensor
            (N,) tensor of PDG IDs

        Returns
        -------
        fractions: torch.Tensor
            (N,) tensor of energy fractions
        """
        abs_pids = torch.abs(pids).to(torch.int64)

        # Clamp to LUT range - any PDG ID beyond max gets default fraction
        clamped_pids = abs_pids.clamp(max=self._max_pdg_id)

        # Pure tensor lookup - no branches, fully differentiable
        fractions = self._fraction_lut[clamped_pids]

        return fractions

    @staticmethod
    def _ecal_cms_resolution(eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """ECAL resolution formula from delphes_card_CMS_5_0.tcl.

        Formula:
            |eta| <= 1.5: (1 + 0.64*eta^2) * sqrt(E^2*0.008^2 + E*0.11^2 + 0.40^2)
            1.5 < |eta| <= 2.5: (2.16 + 5.6*(|eta|-2)^2) * sqrt(E^2*0.008^2 + E*0.11^2 + 0.40^2)
            2.5 < |eta| <= 5.0: sqrt(E^2*0.107^2 + E*2.08^2)

        Parameters
        ----------
        eta: torch.Tensor
            (N,) tensor of eta values (tower centers)
        energy: torch.Tensor
            (N,) tensor of energy values (tower energies)

        Returns
        -------
        sigma: torch.Tensor
            (N,) tensor of energy resolution sigma values
        """
        abs_eta = torch.abs(eta)

        # Common term for barrel and endcap
        common_term = torch.sqrt(energy**2 * 0.008**2 + energy * 0.11**2 + 0.40**2)

        # Barrel: |eta| <= 1.5
        barrel_factor = 1.0 + 0.64 * eta**2
        barrel_sigma = barrel_factor * common_term

        # Endcap: 1.5 < |eta| <= 2.5
        endcap_factor = 2.16 + 5.6 * (abs_eta - 2.0) ** 2
        endcap_sigma = endcap_factor * common_term

        # Forward: 2.5 < |eta| <= 5.0
        forward_sigma = torch.sqrt(energy**2 * 0.107**2 + energy * 2.08**2)

        # Select based on eta region
        sigma = torch.where(
            abs_eta <= 1.5, barrel_sigma, torch.where(abs_eta <= 2.5, endcap_sigma, forward_sigma)
        )
        # TODO: Should it be zero if eta>5.0, i.e. the below?
        # See delphes_cards/delphes_card_CMS_6_1.tcl, lines 267->269
        # sigma = torch.where(abs_eta <= 1.5, barrel_sigma,
        #             torch.where(abs_eta <= 2.5, endcap_sigma,
        #                     torch.where(abs_eta <= 5.0, forward_sigma, 0)))

        return sigma

    @staticmethod
    def _ecal_atlas_resolution(eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """ECAL resolution formula from delphes_card_ATLAS.tcl.

        Formula:
            |eta| <= 3.2: sqrt(E^2*0.0017^2 + E*0.101^2)
            3.2 < |eta| <= 4.9: sqrt(E^2*0.0350^2 + E*0.285^2)

        Parameters
        ----------
        eta: torch.Tensor
            (N,) tensor of eta values (tower centers)
        energy: torch.Tensor
            (N,) tensor of energy values (tower energies)

        Returns
        -------
        sigma: torch.Tensor
            (N,) tensor of energy resolution sigma values
        """
        energy = energy.to(torch.float64)
        abs_eta = torch.abs(eta)

        # Barrel + Endcap: |eta| <= 3.2
        barrel_endcap_sigma = torch.sqrt((energy**2) * (0.0017**2) + energy * (0.101**2))

        # Forward: 2.5 < |eta| <= 5.0
        forward_sigma = torch.sqrt((energy**2) * (0.0035**2) + energy * (0.285**2))

        # Select based on eta region
        sigma = torch.where(abs_eta <= 3.2, barrel_endcap_sigma, forward_sigma)

        # TODO: Should it be zero if eta>4.9, i.e. the below?
        # See delphes_cards/delphes_card_ATLAS_6_1.tcl, lines 260->261
        # sigma = torch.where(abs_eta <= 3.2, barrel_endcap_sigma,
        #             torch.where(abs_eta <= 4.9, forward_sigma, 0))

        return sigma

    @staticmethod
    def _hcal_cms_resolution(eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """HCAL resolution formula from delphes_card_CMS_5_1.tcl.

        Formula:
            |eta| <= 3.0: sqrt(E^2*0.050^2 + E*1.50^2)
            3.0 < |eta| <= 5.0: sqrt(E^2*0.130^2 + E*2.70^2)

        Parameters
        ----------
        eta: torch.Tensor
            (N,) tensor of eta values (tower centers)
        energy: torch.Tensor
            (N,) tensor of energy values (tower energies)

        Returns
        -------
        sigma: torch.Tensor
            (N,) tensor of energy resolution sigma values
        """
        abs_eta = torch.abs(eta)

        # Central: |eta| <= 3.0
        central_sigma = torch.sqrt(energy**2 * 0.050**2 + energy * 1.50**2)

        # Forward: 3.0 < |eta| <= 5.0
        forward_sigma = torch.sqrt(energy**2 * 0.130**2 + energy * 2.70**2)

        # Select based on eta region
        sigma = torch.where(abs_eta <= 3.0, central_sigma, forward_sigma)

        # TODO: Should it be zero if eta>5.0, i.e. the below?
        # See delphes_cards/delphes_card_CMS_6_1.tcl, lines 347->348
        # sigma = torch.where(abs_eta <= 3.0, central_sigma,
        #             torch.where(abs_eta <= 5.0, forward_sigma, 0))

        return sigma.to(torch.float64)

    @staticmethod
    def _hcal_atlas_resolution(eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """HCAL resolution formula from delphes_card_CMS_5_1.tcl.

        Formula:
            |eta| <= 1.7: sqrt(E^2*0.0302^2 + E*0.5205^2 + 1.59^2)
            1.7 < |eta| <= 3.2: sqrt(E^2*0.0500^2 + E*0.706^2)
            3.2 < |eta| <= 4.9: sqrt(E^2*0.09420^2 + E*1.00^2)

        Parameters
        ----------
        eta: torch.Tensor
            (N,) tensor of eta values (tower centers)
        energy: torch.Tensor
            (N,) tensor of energy values (tower energies)

        Returns
        -------
        sigma: torch.Tensor
            (N,) tensor of energy resolution sigma values
        """
        abs_eta = torch.abs(eta)

        # Inner: |eta| <= 1.7
        inner_sigma = torch.sqrt((energy**2) * (0.0302**2) + energy * (0.5205**2) + (1.59**2))

        # Central: 1.7 < |eta| <= 3.2
        central_sigma = torch.sqrt((energy**2) * (0.0500**2) + energy * (0.706**2))

        # Forward: 3.0 < |eta| <= 4.9
        forward_sigma = torch.sqrt((energy**2) * (0.09420**2) + energy * (1.00**2))

        # Select based on eta region
        sigma = torch.where(
            abs_eta <= 1.7, inner_sigma, torch.where(abs_eta <= 3.2, central_sigma, forward_sigma)
        )

        # TODO: Should it be zero if eta>4.9, i.e. the below?
        # See delphes_cards/delphes_card_ATLAS_6_1.tcl, lines 334->336
        # sigma = torch.where(abs_eta <= 1.7, inner_sigma,
        #             torch.where(abs_eta <= 3.2, central_sigma,
        #                 torch.where(abs_eta <= 4.9, forward_sigma, 0)))

        return sigma.to(torch.float64)

    @staticmethod
    def _log_normal_smear(mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply log-normal smearing to energy values.

        C++ implementation:
            if(mean > 0.0) {
                b = sqrt(log((1.0 + (sigma * sigma) / (mean * mean))));
                a = log(mean) - 0.5 * b * b;
                return exp(a + b * gRandom->Gaus(0.0, 1.0));
            } else {
                return 0.0;
            }

        Parameters
        ----------
        mean: torch.Tensor
            Tower energy (before smearing)
        sigma: torch.Tensor
            Resolution sigma

        Returns
        -------
        smeared: torch.Tensor
            Smeared energy values
        """
        # For mean > 0, apply log-normal
        # For mean <= 0, return 0

        # Avoid division by zero
        safe_mean = torch.where(mean > 0, mean, torch.ones_like(mean))

        # b = sqrt(log(1 + sigma^2/mean^2))
        b = torch.sqrt(torch.log(1.0 + (sigma * sigma) / (safe_mean * safe_mean)))

        # a = log(mean) - 0.5 * b^2
        a = torch.log(safe_mean) - 0.5 * b * b

        # Sample from standard normal
        z = torch.randn_like(mean)

        # exp(a + b * z)
        smeared = torch.exp(a + b * z)

        # Zero out where mean <= 0
        smeared = torch.where(mean > 0, smeared, torch.zeros_like(smeared))

        return smeared

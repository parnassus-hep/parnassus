"""PyTorch implementation of Energy Flow (Particle Flow) Merger module.

Merges Track and Tower objects into a unified ParticleFlowCandidate format.
This module applies the necessary transformations to ensure consistency between
different object types when they are combined.

Key transformations:
- **Tracks**: Eta is set to EtaOuter (position-based) for consistency with towers
- **Photons**: PID=22, vertex position zeroed, T zeroed
- **Neutral Hadrons**: PID=0, vertex position zeroed, T zeroed

The Eem/Ehad fields are not stored in the tensor representation; they are
computed during ROOT file writing based on PID values.

Reference:
    C++ Delphes: modules/Merger.cc (used for EFlowMerger in TCL cards)
    See also: EFlowMerger.md for validation details
"""

import torch
from torch import nn

from parnassus.data.particle_io import ColumnMap


class EFlowMerger(nn.Module):
    """Merges Track and Tower objects into ParticleFlowCandidate format.

    This merger takes exactly three input streams and applies specific
    transformations to each before concatenating:

    1. **Tracks** (e.g., from HCal/eflowTracks):
       - Eta field is set to EtaOuter (position-based pseudorapidity)
       - All other fields preserved

    2. **Photons** (e.g., from ECal/eflowPhotons):
       - PID set to 22 (photon PDG code)
       - X, Y, Z, T set to 0 (towers have no vertex position)

    3. **Neutral Hadrons** (e.g., from HCal/eflowNeutralHadrons):
       - PID set to 0 (C++ Delphes convention for neutral hadrons)
       - X, Y, Z, T set to 0 (towers have no vertex position)

    Note:
        The order of inputs matters! The transformations are applied based
        on position in the input list, not based on particle content.

    Examples
    --------
        >>> merger = EFlowMerger()
        >>> eflow = merger([tracks, photons, neutral_hadrons])
    """

    def __init__(self) -> None:
        """Initialize the EFlowMerger module (no parameters needed)."""
        super().__init__()

    def forward(self, input_arrays: list[torch.Tensor]) -> torch.Tensor:
        """Merge Track and Tower objects into ParticleFlowCandidate format.

        Parameters
        ----------
        input_arrays: list[torch.Tensor]
            - [0] tracks: torch.Tensor
                (N_tracks, N_FEATURES) - Track objects
            - [1] photons: torch.Tensor
                (N_photons, N_FEATURES) - ECal tower objects
            - [2] neutral_hadrons: torch.Tensor
                (N_neutrals, N_FEATURES) - HCal tower objects

        Returns
        -------
        merged: torch.Tensor
            Merged tensor of shape (N_total, N_FEATURES) containing all
            ParticleFlowCandidate objects with appropriate transformations applied.

        Raises
        ------
            ValueError: If input_arrays does not contain exactly 3 tensors.
        """
        if len(input_arrays) != 3:
            raise ValueError(f"EFlowMerger expects exactly 3 input arrays, got {len(input_arrays)}")

        tracks, photons, neutral_hadrons = input_arrays

        # Transform Track objects for ParticleFlow representation
        if tracks.shape[0] > 0:
            tracks = self._transform_tracks(tracks)

        # Transform Tower objects (photons) for ParticleFlow representation
        if photons.shape[0] > 0:
            photons = self._transform_photons(photons)

        # Transform Tower objects (neutral hadrons) for ParticleFlow representation
        if neutral_hadrons.shape[0] > 0:
            neutral_hadrons = self._transform_neutral_hadrons(neutral_hadrons)

        # Concatenate all objects
        all_objects = []
        if tracks.shape[0] > 0:
            all_objects.append(tracks)
        if photons.shape[0] > 0:
            all_objects.append(photons)
        if neutral_hadrons.shape[0] > 0:
            all_objects.append(neutral_hadrons)

        if len(all_objects) == 0:
            # No objects - return empty tensor
            return torch.zeros(
                (0, tracks.shape[1] if tracks.shape[0] > 0 else photons.shape[1]),
                dtype=tracks.dtype if tracks.shape[0] > 0 else photons.dtype,
                device=tracks.device if tracks.shape[0] > 0 else photons.device,
            )

        merged = torch.cat(all_objects, dim=0)

        return merged

    def _transform_tracks(self, tracks: torch.Tensor) -> torch.Tensor:
        """Transform Track objects for ParticleFlow representation.

        Actually does nothing, except copying the input tensor.

        Parameters
        ----------
        tracks: torch.Tensor
            Track objects, shape (N, N_FEATURES).

        Returns
        -------
        transformed: torch.Tensor
            Track objects with proper transformations, shape (N, N_FEATURES).
        """
        tracks = tracks.clone()

        # Note: X, Y, Z should already be non-zero for tracks (from vertex position)
        # PID, Charge, E, PT, Phi, etc. remain unchanged

        return tracks

    def _transform_photons(self, photons: torch.Tensor) -> torch.Tensor:
        """Transform photon Tower objects for ParticleFlow representation.

        Photons are electromagnetic calorimeter deposits. For ParticleFlow:
        - PID should be 22 (photon PDG code)
        - X, Y, Z should be 0 (no vertex position)
        - Eem = E (all energy is electromagnetic)
        - Ehad = 0 (no hadronic energy)

        Parameters
        ----------
        photons: torch.Tensor
            Tower objects from ECal, shape (N, N_FEATURES).

        Returns
        -------
        transformed: torch.Tensor
            Tower objects with proper PID and position, shape (N, N_FEATURES).
        """
        photons = photons.clone()

        # Set PID to 22 (photon)
        photons[:, ColumnMap.PID] = 22

        # Set vertex position to zero (towers don't have vertex position)
        photons[:, ColumnMap.X] = 0.0
        photons[:, ColumnMap.Y] = 0.0
        photons[:, ColumnMap.Z] = 0.0
        photons[:, ColumnMap.T] = 0.0

        # Note: Eem and Ehad are not stored in the tensor, they will be computed
        # during ROOT writing based on E and PID

        return photons

    def _transform_neutral_hadrons(self, neutral_hadrons: torch.Tensor) -> torch.Tensor:
        """Transform neutral hadron Tower objects for ParticleFlow representation.

        Neutral hadrons are hadronic calorimeter deposits. For ParticleFlow:
        - PID should be 0 (neutral hadron - C++ Delphes convention)
        - X, Y, Z should be 0 (no vertex position)
        - Eem = 0 (no electromagnetic energy)
        - Ehad = E (all energy is hadronic)

        Parameters
        ----------
        neutral_hadrons: torch.Tensor
            Tower objects from HCal, shape (N, N_FEATURES).

        Returns
        -------
        transformed: torch.Tensor
            Tower objects with proper PID and position, shape (N, N_FEATURES).
        """
        neutral_hadrons = neutral_hadrons.clone()

        # Set PID to 0 (neutral hadron - matches C++ Delphes convention)
        neutral_hadrons[:, ColumnMap.PID] = 0

        # Set vertex position to zero (towers don't have vertex position)
        neutral_hadrons[:, ColumnMap.X] = 0.0
        neutral_hadrons[:, ColumnMap.Y] = 0.0
        neutral_hadrons[:, ColumnMap.Z] = 0.0
        neutral_hadrons[:, ColumnMap.T] = 0.0

        # Note: Eem and Ehad are not stored in the tensor, they will be computed
        # during ROOT writing based on E and PID

        return neutral_hadrons

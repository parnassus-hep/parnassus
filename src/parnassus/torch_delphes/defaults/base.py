from abc import ABC, abstractmethod

import torch
from torch import nn


class DelphesBaseCard(nn.Module, ABC):
    """PyTorch implementation of the default Delphes detector simulation."""

    def __init__(self, debug: bool = False) -> None:
        """Initialize the Delphes detector simulation.

        Parameters
        ----------
        debug: bool
            If True, return all intermediate processing stages
            for validation. If False, return only final objects.
        """
        super().__init__()
        self.debug = debug

    @abstractmethod
    def forward(self, stable_particles: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the Parametric detector simulation to input particles.

        Processes generator-level stable particles through the complete
        detector simulation chain: propagation, tracking, calorimetry,
        and particle flow reconstruction.

        Parameters
        ----------
        stable_particles: torch.Tensor
            Tensor of shape (N, N_FEATURES) containing
            generator-level stable particles. Should be flattened
            (not batched by event). Required columns include:

            - PID, CHARGE, E, PX, PY, PZ, PT, ETA, PHI
            - X, Y, Z, T (production vertex)
            - MASS, EVENT_NUMBER

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping branch names to tensors. Contents depend on
            debug mode:

            **Normal mode** (debug=False):

            - 'Track': Merged tracks after smearing
            - 'Tower': Merged calorimeter towers + muons
            - 'EFlowTrack': Tracks for particle flow
            - 'EFlowPhoton': Photon candidates from ECal
            - 'EFlowNeutralHadron': Neutral hadron candidates from HCal

            **Debug mode** (debug=True): All of the above plus:

            - 'ParticleBeforeProp', 'ParticleAfterProp'
            - 'ChargedHadron', 'Electron', 'Muon', 'NeutralParticle'
            - 'ChargedHadronEfficiency', 'ElectronEfficiency', 'MuonEfficiency'
            - 'ChargedHadronSmeared', 'ElectronSmeared', 'MuonSmeared'
            - 'ECal_EFlowTrack', 'ECalTower', 'HCalTower'
            - 'EFlowObject'
        """

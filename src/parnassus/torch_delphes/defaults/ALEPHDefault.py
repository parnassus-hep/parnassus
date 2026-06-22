"""PyTorch implementation of default ALEPH detector simulation.

Implements the ALEPH detector response chain from the Delphes TCL card
(delphes_card_ALEPH.tcl), producing energy flow objects suitable for
jet clustering and physics analysis.

Detector parameters:
- Tracker radius: 1.5 m
- Tracker half-length: 2.5 m
- Magnetic field: 0.435 T

Processing chain:
1. ParticlePropagator → propagate particles to tracker surface
2. Efficiency → apply tracking efficiency (charged hadrons, electrons, muons)
3. MomentumSmearing → smear track momenta
4. TrackMerger → combine all tracks
5. ECal/HCal → calorimeter simulation with energy flow
6. EFlowMerger → combine tracks and calorimeter objects

Reference:
    C++ Delphes card: cards/delphes_card_ALEPH.tcl
"""

import numpy as np
import torch

from parnassus.torch_delphes.Efficiency import Efficiency
from parnassus.torch_delphes.EFlowMerger import EFlowMerger
from parnassus.torch_delphes.Merger import Merger
from parnassus.torch_delphes.MomentumSmearing import MomentumSmearing
from parnassus.torch_delphes.ParticlePropagator import ParticlePropagator
from parnassus.torch_delphes.SimpleCalorimeter import SimpleCalorimeter

from .base import DelphesBaseCard


class ALEPHEnergyFlowDefault(DelphesBaseCard):
    """PyTorch implementation of the default ALEPH Delphes detector simulation.

    Simulates the full ALEPH detector response chain including:

    - **Tracking**: Particle propagation through 0.435 T magnetic field,
      tracking efficiency, and momentum smearing
    - **Calorimetry**: ECal and HCal simulation with energy deposits,
      tower clustering, and energy resolution smearing
    - **Particle Flow**: Energy flow reconstruction combining tracks
      and calorimeter deposits

    The module can operate in two modes controlled by the `debug` flag:

    - **Normal mode** (debug=False): Returns only final reconstructed objects
      (Track, Tower, EFlowTrack, EFlowPhoton, EFlowNeutralHadron)
    - **Debug mode** (debug=True): Returns all intermediate objects for
      validation against C++ Delphes

    Attributes
    ----------
    debug: bool
        If True, return all intermediate processing stages
    ParticlePropagator: ParticlePropagator
        Propagates particles to tracker surface
    ChargedHadronTrackingEfficiency: Efficiency
        Tracking efficiency for hadrons
    ElectronTrackingEfficiency: Efficiency
        Tracking efficiency for electrons
    MuonTrackingEfficiency: Efficiency
        Tracking efficiency for muons
    ChargedHadronMomentumSmearing: MomentumSmearing
        Momentum resolution for hadrons
    ElectronMomentumSmearing: MomentumSmearing
        Momentum resolution for electrons
    MuonMomentumSmearing: MomentumSmearing
        Momentum resolution for muons
    TrackMerger: Merger
        Combines all track types
    ECal: SimpleCalorimeter
        Electromagnetic calorimeter
    HCal: SimpleCalorimeter
        Hadronic calorimeter
    CalorimeterMerger: Merger
        Combines ECal and HCal towers
    EFlowMerger: EFlowMerger
        Creates particle flow candidates

    Examples
    --------
    >>> aleph = ALEPHEnergyFlowDefault(debug=False)
    >>> results = aleph(stable_particles)
    >>> tracks = results['Track']
    >>> eflow_tracks = results['EFlowTrack']
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the ALEPH detector simulation.

        Parameters
        ----------
        debug: bool
            If True, return all intermediate processing stages
            for validation. If False, return only final objects.
        """
        super().__init__()
        self.debug = debug

        # ParticlePropagator
        self.ParticlePropagator = ParticlePropagator(
            radius=1.5,
            half_length=2.5,
            bz=0.435,
        )

        # TrackingEfficiency
        self.ChargedHadronTrackingEfficiency = Efficiency(efficiency_formula="charged_hadron_aleph")
        self.ElectronTrackingEfficiency = Efficiency(efficiency_formula="electron_aleph")
        self.MuonTrackingEfficiency = Efficiency(efficiency_formula="muon_aleph")

        # MomentumSmearing
        self.ChargedHadronMomentumSmearing = MomentumSmearing(
            resolution_formula="charged_hadron_aleph"
        )
        self.ElectronMomentumSmearing = MomentumSmearing(resolution_formula="electron_aleph")
        self.MuonMomentumSmearing = MomentumSmearing(resolution_formula="muon_aleph")

        # TrackMerger
        self.TrackMerger = Merger()

        # ECal (Electromagnetic Calorimeter)
        self._setup_ECal()

        # HCal (Hadronic Calorimeter)
        self._setup_HCal()

        # CalorimeterMerger
        self.CalorimeterMerger = Merger()

        # EFlowMerger
        self.EFlowMerger = EFlowMerger()

    def forward(self, stable_particles: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the full ALEPH detector simulation to input particles.

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
            - 'Tower': Merged calorimeter towers
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
        _, n_dim = stable_particles.shape

        # ParticlePropagator
        particles = stable_particles.reshape(-1, n_dim)
        particles_before_prop = particles.clone() if self.debug else torch.empty(0)

        (
            particles_propagated,
            neutrals_propagated,
            charged_hadrons_propagated,
            electrons_propagated,
            muons_propagated,
        ) = self.ParticlePropagator(particles)

        # TrackingEfficiency
        charged_hadrons_eff = self.ChargedHadronTrackingEfficiency(charged_hadrons_propagated)
        electrons_eff = self.ElectronTrackingEfficiency(electrons_propagated)
        muons_eff = self.MuonTrackingEfficiency(muons_propagated)

        # MomentumSmearing
        charged_hadrons_smeared = self.ChargedHadronMomentumSmearing(charged_hadrons_eff)
        electrons_smeared = self.ElectronMomentumSmearing(electrons_eff)
        muons_smeared = self.MuonMomentumSmearing(muons_eff)

        # TrackMerger
        merged_tracks = self.TrackMerger([
            charged_hadrons_smeared,
            electrons_smeared,
            muons_smeared,
        ])

        # ECal
        ecal_tracks, ecal_towers, eflow_photons = self.ECal(particles_propagated, merged_tracks)

        # HCal
        hcal_tracks, hcal_towers, eflow_neutral_hadrons = self.HCal(
            particles_propagated, ecal_tracks
        )

        # CalorimeterMerger
        merged_towers = self.CalorimeterMerger([ecal_towers, hcal_towers])

        # EFlowMerger
        eflow_objects = self.EFlowMerger([hcal_tracks, eflow_photons, eflow_neutral_hadrons])

        if self.debug:
            return {
                "ParticleBeforeProp": particles_before_prop,
                "ParticleAfterProp": particles_propagated,
                "ChargedHadron": charged_hadrons_propagated,
                "Electron": electrons_propagated,
                "Muon": muons_propagated,
                "NeutralParticle": neutrals_propagated,
                "ChargedHadronEfficiency": charged_hadrons_eff,
                "ElectronEfficiency": electrons_eff,
                "MuonEfficiency": muons_eff,
                "ChargedHadronSmeared": charged_hadrons_smeared,
                "ElectronSmeared": electrons_smeared,
                "MuonSmeared": muons_smeared,
                "Track": merged_tracks,
                "ECal_EFlowTrack": ecal_tracks,
                "ECalTower": ecal_towers,
                "EFlowPhoton": eflow_photons,
                "EFlowTrack": hcal_tracks,
                "HCalTower": hcal_towers,
                "EFlowNeutralHadron": eflow_neutral_hadrons,
                "Tower": merged_towers,
                "EFlowObject": eflow_objects,
            }
        return {
            "Track": merged_tracks,
            "Tower": merged_towers,
            "EFlowTrack": hcal_tracks,
            "EFlowPhoton": eflow_photons,
            "EFlowNeutralHadron": eflow_neutral_hadrons,
            "EFlowObject": eflow_objects,
        }

    def _setup_ECal(self):
        energy_fractions = {
            0: 0.0,  # default (hadrons) - no ECAL response
            11: 1.0,  # electrons
            22: 1.0,  # photons
            111: 1.0,  # pi0
            12: 0.0,  # neutrino (electron)
            13: 0.0,  # muon
            14: 0.0,  # neutrino (muon)
            16: 0.0,  # neutrino (tau)
            1000022: 0.0,  # neutralino
            1000023: 0.0,  # neutralino
            1000025: 0.0,  # neutralino
            1000035: 0.0,  # neutralino
            1000045: 0.0,  # neutralino
            310: 0.3,  # K0short
            3122: 0.3,  # Lambda
        }

        eta_phi_map = {}  # eta -> set of phi bin edges

        phi_bins = [i * np.pi / 180.0 for i in range(-180, 181)]

        # 0.02 unit in eta up to eta = 3.0
        for i in range(-150, 151):
            eta = i * 0.02
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        self.ECal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=0.5,
            energy_sig_min=1.0,
            energy_fractions=energy_fractions,
            resolution_formula="ecal_aleph",
            is_ecal=True,
            smear_tower_center=True,
        )

    def _setup_HCal(self):
        energy_fractions = {
            0: 1.0,  # default (hadrons) - full HCAL response
            11: 0.0,  # electrons (no HCAL response - already absorbed by ECAL)
            22: 0.0,  # photons (no HCAL response)
            111: 0.0,  # pi0 (no HCAL response)
            12: 0.0,  # neutrino (electron)
            13: 0.0,  # muon
            14: 0.0,  # neutrino (muon)
            16: 0.0,  # neutrino (tau)
            1000022: 0.0,  # neutralino
            1000023: 0.0,  # neutralino
            1000025: 0.0,  # neutralino
            1000035: 0.0,  # neutralino
            1000045: 0.0,  # neutralino
            310: 0.7,  # K0short (70% HCAL)
            3122: 0.7,  # Lambda (70% HCAL)
        }

        eta_phi_map = {}  # eta -> set of phi bin edges

        phi_bins = [i * np.pi / 180.0 for i in range(-180, 181)]

        # 0.04 unit in eta up to eta = 3.0
        for i in range(-75, 76):
            eta = i * 0.04
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        self.HCal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=1.0,  # HCal has higher threshold
            energy_sig_min=1.0,  # HCal has lower significance threshold
            energy_fractions=energy_fractions,
            resolution_formula="hcal_aleph",
            is_ecal=False,
            smear_tower_center=True,
        )

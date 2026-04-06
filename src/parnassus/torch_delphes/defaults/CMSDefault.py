"""PyTorch implementation of default CMS detector simulation.

Implements the CMS detector response chain from the Delphes TCL card
(delphes_card_CMS_6_1.tcl), producing energy flow objects suitable for
jet clustering and physics analysis.

Detector parameters:
- Tracker radius: 1.29 m
- Tracker half-length: 3.0 m
- Magnetic field: 3.8 T

Processing chain:
1. ParticlePropagator → propagate particles to tracker surface
2. Efficiency → apply tracking efficiency (charged hadrons, electrons, muons)
3. MomentumSmearing → smear track momenta
4. TrackMerger → combine all tracks
5. ECal/HCal → calorimeter simulation with energy flow
6. EFlowMerger → combine tracks and calorimeter objects

Reference:
    C++ Delphes card: cards/delphes_card_CMS_6_1.tcl
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


class CMSEnergyFlowDefault(DelphesBaseCard):
    """PyTorch implementation of the default CMS Delphes detector simulation.

    Simulates the full CMS detector response chain including:

    - **Tracking**: Particle propagation through 3.8T magnetic field,
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
    >>> cms = CMSEnergyFlowDefault(debug=False)
    >>> results = cms(stable_particles)
    >>> tracks = results['Track']
    >>> eflow_tracks = results['EFlowTrack']
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the CMS detector simulation.

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
            radius=1.29,
            half_length=3.0,
            bz=3.8,
        )

        # TrackingEfficiency
        self.ChargedHadronTrackingEfficiency = Efficiency(efficiency_formula="charged_hadron_cms")
        self.ElectronTrackingEfficiency = Efficiency(efficiency_formula="electron_cms")
        self.MuonTrackingEfficiency = Efficiency(efficiency_formula="muon_cms")

        # MomentumSmearing
        self.ChargedHadronMomentumSmearing = MomentumSmearing(
            resolution_formula="charged_hadron_cms"
        )
        self.ElectronMomentumSmearing = MomentumSmearing(resolution_formula="electron_cms")
        self.MuonMomentumSmearing = MomentumSmearing(resolution_formula="muon_cms")

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
        """Apply the full CMS detector simulation to input particles.

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

        # Create eta and phi bins from CMS card (delphes_card_CMS_5_0.tcl)
        # The card builds a map: eta_value -> set of phi bins
        # We need to replicate this exactly

        # Fine phi bins for barrel and endcap (361 bins, -pi to pi in 1 degree steps)
        phi_bins_fine = [i * np.pi / 180.0 for i in range(-180, 181)]

        # Coarse phi bins for HF (37 bins, -pi to pi in 10 degree steps)
        phi_bins_coarse = [i * np.pi / 18.0 for i in range(-18, 19)]

        # Build the eta bins and corresponding phi bins exactly as C++ Delphes does
        # The C++ code uses a map<double, set<double>> which gets sorted by eta
        # Then converts to parallel vectors: fEtaBins and fPhiBins[etaBin]

        eta_phi_map = {}  # eta -> set of phi bin edges

        # Barrel: 0.02 unit in eta from -85*0.0174 to 86*0.0174
        for i in range(-85, 87):
            eta = i * 0.0174
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_fine)

        # Endcap negative: -2.958 + i*0.0174 for i in 1..84
        for i in range(1, 85):
            eta = -2.958 + i * 0.0174
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_fine)

        # Endcap positive: 1.4964 + i*0.0174 for i in 1..84
        for i in range(1, 85):
            eta = 1.4964 + i * 0.0174
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_fine)

        # HF: specific eta values with coarse phi binning
        hf_etas = [
            -5,
            -4.7,
            -4.525,
            -4.35,
            -4.175,
            -4,
            -3.825,
            -3.65,
            -3.475,
            -3.3,
            -3.125,
            -2.958,
            3.125,
            3.3,
            3.475,
            3.65,
            3.825,
            4,
            4.175,
            4.35,
            4.525,
            4.7,
            5,
        ]
        for eta in hf_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_coarse)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        self.ECal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=0.5,
            energy_sig_min=2.0,
            energy_fractions=energy_fractions,
            resolution_formula="ecal_cms",
            is_ecal=True,
            smear_tower_center=True,  # Match C++ Delphes: SmearTowerCenter true
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

        # 5 degrees towers (barrel): phi bins -36 to 36 in steps of pi/36
        phi_bins_5deg = [i * np.pi / 36.0 for i in range(-36, 37)]
        barrel_etas = [
            -1.566,
            -1.479,
            -1.392,
            -1.305,
            -1.218,
            -1.131,
            -1.044,
            -0.957,
            -0.87,
            -0.783,
            -0.696,
            -0.609,
            -0.522,
            -0.435,
            -0.348,
            -0.261,
            -0.174,
            -0.087,
            0,
            0.087,
            0.174,
            0.261,
            0.348,
            0.435,
            0.522,
            0.609,
            0.696,
            0.783,
            0.87,
            0.957,
            1.044,
            1.131,
            1.218,
            1.305,
            1.392,
            1.479,
            1.566,
            1.653,
        ]
        for eta in barrel_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_5deg)

        # 10 degrees towers (endcap): phi bins -18 to 18 in steps of pi/18
        phi_bins_10deg = [i * np.pi / 18.0 for i in range(-18, 19)]
        endcap_etas = [
            -4.35,
            -4.175,
            -4,
            -3.825,
            -3.65,
            -3.475,
            -3.3,
            -3.125,
            -2.95,
            -2.868,
            -2.65,
            -2.5,
            -2.322,
            -2.172,
            -2.043,
            -1.93,
            -1.83,
            -1.74,
            -1.653,
            1.74,
            1.83,
            1.93,
            2.043,
            2.172,
            2.322,
            2.5,
            2.65,
            2.868,
            2.95,
            3.125,
            3.3,
            3.475,
            3.65,
            3.825,
            4,
            4.175,
            4.35,
            4.525,
        ]
        for eta in endcap_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_10deg)

        # 20 degrees towers (forward): phi bins -9 to 9 in steps of pi/9
        phi_bins_20deg = [i * np.pi / 9.0 for i in range(-9, 10)]
        forward_etas = [-5, -4.7, -4.525, 4.7, 5]
        for eta in forward_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_20deg)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        self.HCal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=1.0,  # HCal has higher threshold
            energy_sig_min=1.0,  # HCal has lower significance threshold
            energy_fractions=energy_fractions,
            resolution_formula="hcal_cms",
            is_ecal=False,
            smear_tower_center=True,
        )

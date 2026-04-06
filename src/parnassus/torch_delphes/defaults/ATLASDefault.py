"""PyTorch implementation of default ATLAS detector simulation.

Implements the ATLAS detector response chain from the Delphes TCL card
(delphes_card_ATLAS_6_1.tcl), producing energy flow objects suitable for
jet clustering and physics analysis.

Detector parameters:
- Tracker radius: 1.15 m
- Tracker half-length: 3.51 m
- Magnetic field: 2.0 T

Key differences from CMS:
- ATLAS includes muons in the Tower branch (CalorimeterMerger)
- Different detector geometry and magnetic field strength
- Same efficiency/resolution formulas (functionally equivalent in TCL)

Processing chain:
1. ParticlePropagator → propagate particles to tracker surface
2. Efficiency → apply tracking efficiency (charged hadrons, electrons, muons)
3. MomentumSmearing → smear track momenta
4. TrackMerger → combine all tracks
5. ECal/HCal → calorimeter simulation with energy flow
6. CalorimeterMerger → combine ECal towers, HCal towers, AND muons
7. EFlowMerger → combine tracks and calorimeter objects

Reference:
    C++ Delphes card: cards/delphes_card_ATLAS_6_1.tcl
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


class ATLASEnergyFlowDefault(DelphesBaseCard):
    """PyTorch implementation of the default ATLAS Delphes detector simulation.

    Simulates the full ATLAS detector response chain including:

    - **Tracking**: Particle propagation through 2.0T magnetic field,
      tracking efficiency, and momentum smearing
    - **Calorimetry**: ECal and HCal simulation with energy deposits,
      tower clustering, and energy resolution smearing
    - **Particle Flow**: Energy flow reconstruction combining tracks
      and calorimeter deposits

    Key difference from CMS: The Tower branch includes muons in addition
    to ECal and HCal towers, matching the ATLAS TCL card configuration.

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
        Propagates particles to tracker surface (r=1.15m, z=3.51m)
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
        Combines ECal towers, HCal towers, AND muons
    EFlowMerger: EFlowMerger
        Creates particle flow candidates

    Examples
    --------
    >>> atlas = ATLASEnergyFlowDefault(debug=False)
    >>> results = atlas(stable_particles)
    >>> tracks = results['Track']
    >>> towers = results['Tower']  # Includes muons!
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the ATLAS detector simulation.

        Parameters
        ----------
        debug: bool
            If True, return all intermediate processing stages
            for validation. If False, return only final objects.
        """
        super().__init__()
        self.debug = debug

        # ParticlePropagator - ATLAS geometry
        self.ParticlePropagator = ParticlePropagator(
            radius=1.15,
            half_length=3.51,
            bz=2.0,
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
        """Apply the full ATLAS detector simulation to input particles.

        Processes generator-level stable particles through the complete
        detector simulation chain: propagation, tracking, calorimetry,
        and particle flow reconstruction.

        Note: Unlike CMS, the ATLAS Tower branch includes muons in addition
        to calorimeter towers.

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
        merged_towers = self.CalorimeterMerger([ecal_towers, hcal_towers, muons_smeared])

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
            resolution_formula="ecal_atlas",
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

        # 5 degrees towers (barrel+endcap): phi bins -36 to 36 in steps of pi/36
        phi_bins_10deg = [i * np.pi / 18.0 for i in range(-18, 19)]
        barrel_etas = [
            -3.2,
            -2.5,
            -2.4,
            -2.3,
            -2.2,
            -2.1,
            -2,
            -1.9,
            -1.8,
            -1.7,
            -1.6,
            -1.5,
            -1.4,
            -1.3,
            -1.2,
            -1.1,
            -1,
            -0.9,
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            3.3,
        ]
        for eta in barrel_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_10deg)

        # 20 degrees towers (forward): phi bins -18 to 18 in steps of pi/18
        phi_bins_20deg = [i * np.pi / 9.0 for i in range(-9, 10)]
        endcap_etas = [
            -4.9,
            -4.7,
            -4.5,
            -4.3,
            -4.1,
            -3.9,
            -3.7,
            -3.5,
            -3.3,
            -3,
            -2.8,
            -2.6,
            2.8,
            3,
            3.2,
            3.5,
            3.7,
            3.9,
            4.1,
            4.3,
            4.5,
            4.7,
            4.9,
        ]
        for eta in endcap_etas:
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
            energy_sig_min=2.0,  # HCal has lower significance threshold
            energy_fractions=energy_fractions,
            resolution_formula="hcal_atlas",
            is_ecal=False,
            smear_tower_center=True,
        )

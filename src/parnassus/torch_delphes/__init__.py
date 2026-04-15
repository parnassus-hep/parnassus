"""TorchDelphes: PyTorch implementation of Delphes detector simulation.

This package provides differentiable PyTorch modules that replicate the
functionality of the C++ Delphes fast detector simulation framework.
The implementation is validated against C++ Delphes output for both
CMS and ATLAS detector configurations.

Modules:
    ParticlePropagator: Propagates particles through magnetic field to detector
    Efficiency: Applies tracking efficiency based on kinematics
    MomentumSmearing: Smears track momenta with detector resolution
    SimpleCalorimeter: Simulates calorimeter response and energy flow
    Merger: Combines multiple particle collections
    EFlowMerger: Creates particle flow candidates from tracks and towers

Default configurations:
    CMSEnergyFlowDefault: Complete CMS detector simulation
    ATLASEnergyFlowDefault: Complete ATLAS detector simulation

Examples
--------
>>> from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
>>> detector = CMSEnergyFlowDefault(debug=False)
>>> results = detector(stable_particles)
>>> tracks = results['Track']
>>> eflow = results['EFlowTrack']
"""

# Learnable / Adam-optimizable variants. Imported lazily (after the static
# modules above) because ``learnable.py`` imports ``pdg_filters`` from this
# package at module load time.
from . import learnable
from .Efficiency import Efficiency
from .EFlowMerger import EFlowMerger
from .Merger import Merger
from .MomentumSmearing import MomentumSmearing
from .ParticlePropagator import ParticlePropagator
from .SimpleCalorimeter import SimpleCalorimeter

__all__ = [
    "EFlowMerger",
    "Efficiency",
    "Merger",
    "MomentumSmearing",
    "ParticlePropagator",
    "SimpleCalorimeter",
    "learnable",
]

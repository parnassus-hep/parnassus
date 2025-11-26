"""Pythia utilities for event generation and conversion.

This package provides classes for generating HepMC3 events and converting
Pythia8 events into HepMC3 format for use in the Parnassus fast simulation
pipeline.
"""

from .hepmc3_generator import HepMC3Generator
from .pythia8_to_hepmc3 import Pythia8ToHepMC3

__all__ = ["HepMC3Generator", "Pythia8ToHepMC3"]

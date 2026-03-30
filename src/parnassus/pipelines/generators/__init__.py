"""Event generator implementations."""

from .neural import NeuralEventGenerator
from .parametric import ParametricEventGenerator

__all__ = [
    "NeuralEventGenerator",
    "ParametricEventGenerator",
]

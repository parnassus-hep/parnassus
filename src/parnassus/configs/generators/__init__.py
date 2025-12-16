"""Generator configurations and registries."""

from .base import GeneratorConfig
from .neural import NEURAL_GENERATORS_REGISTRY, NeuralGeneratorConfig
from .parametric import PARAMETRIC_GENERATORS_REGISTRY, ParametricGeneratorConfig

# Unified registry combining all generator types
GENERATORS_REGISTRY: dict[str, GeneratorConfig] = {
    **NEURAL_GENERATORS_REGISTRY,
    **PARAMETRIC_GENERATORS_REGISTRY,
}

__all__ = [
    "GENERATORS_REGISTRY",
    "NEURAL_GENERATORS_REGISTRY",
    "PARAMETRIC_GENERATORS_REGISTRY",
    "GeneratorConfig",
    "NeuralGeneratorConfig",
    "ParametricGeneratorConfig",
]

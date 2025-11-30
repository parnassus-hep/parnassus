"""Generator configuration abstractions for different event generation backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GeneratorConfig(ABC):
    """Base configuration for event generators.

    This abstract base class defines the interface that all generator
    configurations must implement, regardless of the generation backend
    (neural networks, parametric models, Pythia8, etc.).
    """

    name: str
    type: str  # Generator type: "neural", "parametric", etc.

    @abstractmethod
    def get_output_vars(self) -> tuple[list[str], list[str]]:
        """Get output variable names for this generator.

        Returns
        -------
        tuple[list[str], list[str]]
            Tuple of (truth_output_vars, pflow_output_vars).
        """
        ...

    @abstractmethod
    def get_max_particles(self) -> int:
        """Get maximum number of particles per event.

        Returns
        -------
        int
            Maximum number of particles this generator produces.
        """
        ...

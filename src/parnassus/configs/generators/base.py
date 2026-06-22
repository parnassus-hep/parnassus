"""Generator configuration abstractions for different event generation backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class GeneratorConfig(ABC):
    """Base configuration for event generators.

    This abstract base class defines the interface that all generator
    configurations must implement, regardless of the generation backend
    (neural networks, parametric models, Pythia8, etc.).
    """

    name: str
    type: str  # Generator type: "neural", "parametric", etc.
    # Optional pt cut for truth particles (used by neural generators). Keyword-only with a
    # default so subclasses can declare non-default fields (e.g. ``card``) after it.
    truth_pt_cut: float | None = field(default=None, kw_only=True)

    @abstractmethod
    def get_max_particles(self) -> int:
        """Get maximum number of particles per event.

        Returns
        -------
        int
            Maximum number of particles this generator produces.
        """
        ...

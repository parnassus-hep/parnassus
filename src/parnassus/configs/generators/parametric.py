from dataclasses import dataclass, field
from pathlib import Path

from .base import GeneratorConfig


@dataclass(slots=True)
class ParametricGeneratorConfig(GeneratorConfig):
    """Configuration for parametric event generator.

    This configuration manages parametric physics-based event generation,
    routing input events through the torch_delphes fast detector simulation
    instead of a neural network.

    Parameters
    ----------
    name : str
        Name identifier for this generator configuration.
    max_particles : int
        Maximum number of particles per event.
    seed : int | None, optional
        Random seed for reproducible generation. Defaults to None.
    hepmc_file_path : Path | str | None, optional
        Path to the input HepMC file.  When set, ``build_dataset`` will use
        this path instead of the one in ``DatasetConfig``.  Defaults to None
        (uses ``DatasetConfig.file_path``).
    num_events : int | None, optional
        Maximum number of events to load from the HepMC file.
        ``None`` loads all events. Defaults to None.
    """

    type: str = field(default="parametric", init=False)
    max_particles: int = 100
    seed: int | None = None
    hepmc_file_path: Path | str | None = None
    num_events: int | None = None

    # Truth and pflow output variable names
    _truth_output_vars: list[str] = field(default_factory=lambda: ["pt", "eta", "phi", "class"])
    _pflow_output_vars: list[str] = field(
        default_factory=lambda: ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
    )

    def get_max_particles(self) -> int:
        """Get maximum particles for parametric generator.

        Returns
        -------
        int
            Maximum number of particles per event.
        """
        return self.max_particles

    @property
    def truth_output_vars(self) -> list[str]:
        """Truth-level output variable names (backward compatibility).

        Returns
        -------
        list[str]
            List of truth-level output variable names.
        """
        return self._truth_output_vars

    @property
    def pflow_output_vars(self) -> list[str]:
        """Particle flow output variable names (backward compatibility).

        Returns
        -------
        list[str]
            List of particle flow output variable names.
        """
        return self._pflow_output_vars


# Registry of available parametric generators
# Example placeholder - to be implemented
# >>> "simple_parametric_v1": ParametricGeneratorConfig(
# ...     name="simple_parametric_v1",
# ...     max_particles=100,
# ...     seed=42,
# ... ),
PARAMETRIC_GENERATORS_REGISTRY: dict[str, GeneratorConfig] = {}

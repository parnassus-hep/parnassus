from dataclasses import dataclass, field

from .base import GeneratorConfig


@dataclass(slots=True)
class ParametricGeneratorConfig(GeneratorConfig):
    """Configuration for parametric event generator.

    This configuration manages parametric physics-based event generation,
    using analytical models or lookup tables instead of neural networks.

    Parameters
    ----------
    name : str
        Name identifier for this generator configuration.
    max_particles : int
        Maximum number of particles per event.
    seed : int | None, optional
        Random seed for reproducible generation. Defaults to None.
    """

    type: str = field(default="parametric", init=False)
    max_particles: int = 100
    seed: int | None = None

    # Truth and pflow output variable names
    _truth_output_vars: list[str] = field(default_factory=lambda: ["pt", "eta", "phi", "class"])
    _pflow_output_vars: list[str] = field(
        default_factory=lambda: ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
    )

    def get_output_vars(self) -> tuple[list[str], list[str]]:
        """Get output variable names for parametric generator.

        Returns
        -------
        tuple[list[str], list[str]]
            Tuple of (truth_output_vars, pflow_output_vars).
        """
        return self._truth_output_vars, self._pflow_output_vars

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

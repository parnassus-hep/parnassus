from dataclasses import dataclass, field

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
    card : str
        Detector card name for the torch_delphes simulation. Supported values:
        ``"cms"`` (CMS detector) and ``"atlas"`` (ATLAS detector).
    seed : int | None, optional
        Random seed for reproducible generation. Defaults to None.
    """

    type: str = field(default="parametric", init=False)
    card: str  # "cms" or "atlas"
    max_particles: int = 100
    seed: int | None = None
    debug: bool = False

    def get_max_particles(self) -> int:
        """Get maximum particles for parametric generator.

        Returns
        -------
        int
            Maximum number of particles per event.
        """
        return self.max_particles


PARAMETRIC_GENERATORS_REGISTRY: dict[str, ParametricGeneratorConfig] = {
    "cms": ParametricGeneratorConfig(name="cms", card="cms"),
    "atlas": ParametricGeneratorConfig(name="atlas", card="atlas"),
}

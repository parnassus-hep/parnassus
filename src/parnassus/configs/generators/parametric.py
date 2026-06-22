from dataclasses import dataclass, field

from parnassus.configs.pileup import DelphesPileUpConfig

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
    pileup: DelphesPileUpConfig | None = None

    def update_from_dict(self, config_dict: dict) -> None:
        """Update common generator parameters from a dict.

        This method allows updating parameters like `seed` from a dict,
        which is useful when loading from YAML or other sources.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing generator parameters to update.
        """
        if "seed" in config_dict:
            self.seed = config_dict["seed"]
        if "debug" in config_dict:
            self.debug = config_dict["debug"]
        if config_dict.get("pileup") is not None:
            self.pileup = DelphesPileUpConfig.from_dict(config_dict["pileup"])

    def get_max_particles(self) -> int:
        """Get maximum particles for parametric generator.

        Returns
        -------
        int
            Maximum number of particles per event.
        """
        return self.max_particles


PARAMETRIC_GENERATORS_REGISTRY: dict[str, ParametricGeneratorConfig] = {
    "cms": ParametricGeneratorConfig(name="cms", card="cms", truth_pt_cut=None),
    "atlas": ParametricGeneratorConfig(name="atlas", card="atlas", truth_pt_cut=None),
    "aleph": ParametricGeneratorConfig(name="aleph", card="aleph", truth_pt_cut=None),
}

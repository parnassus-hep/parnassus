from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from parnassus.configs.variables import VariableRequirements

if TYPE_CHECKING:
    from parnassus.configs.generators import GeneratorConfig


@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for dataset loading and preprocessing.

    Parameters
    ----------
    file_path : Path | str
        Path to the input data file (HepMC or ROOT format).
    variable_requirements : VariableRequirements
        Variable configuration specifying which variables to load and process.
    max_particles : int
        Maximum number of particles per event (must match model training).
    num_events : int, optional
        Number of events to process. Defaults to 1.
    entry_start : int, optional
        Starting entry index in the file. Defaults to 0.
    batch_loading : bool, optional
        Whether to use batch loading. Defaults to False.
    batch_size : int | None, optional
        Batch size for loading. Required if batch_loading is True.
    """

    # Data loading configs
    file_path: Path | str
    variable_requirements: VariableRequirements
    max_particles: int

    num_events: int = 1
    entry_start: int = 0
    batch_loading: bool = False
    batch_size: int | None = None

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        if self.batch_loading and self.batch_size is None:
            raise ValueError("Asked for batch_loading, but batch_size is not provided")

    @property
    def truth_vars_to_load(self):
        """Truth variables to load from dataset."""
        return self.variable_requirements.truth_vars_to_load

    @property
    def ctxt_vars(self):
        """Context variables for conditioning."""
        return self.variable_requirements.ctxt_vars

    @property
    def ctxt_global_vars(self):
        """Global context variables for event-level conditioning."""
        return self.variable_requirements.ctxt_global_vars

    @classmethod
    def from_dict_and_model(
        cls, config: dict[str, Any], generator_config: "GeneratorConfig"
    ) -> Self:
        """Create DatasetConfig from dictionary and generator configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary containing dataset configuration (file_path, num_events, etc.).
        generator_config : GeneratorConfig
            Generator configuration to extract variable requirements from.

        Returns
        -------
        DatasetConfig
            A new DatasetConfig instance with variables from the generator.
        """
        # Extract variable requirements from generator
        variable_requirements = VariableRequirements.from_model_config(generator_config)

        # Use max_particles from generator config (must match training for neural generators)
        return cls(
            file_path=config["file_path"],
            variable_requirements=variable_requirements,
            max_particles=generator_config.get_max_particles(),
            num_events=config.get("num_events", 1),
            entry_start=config.get("entry_start", 0),
            batch_loading=config.get("batch_loading", False),
            batch_size=config.get("batch_size"),
        )

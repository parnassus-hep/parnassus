"""Generator configuration abstractions for different event generation backends."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import yaml

from parnassus.configs.variables import VariableRequirements
from parnassus.utils import TransformRegistry, VarTransformConfig

from .base import GeneratorConfig
from .model import ModelConfig, SamplerConfig


@dataclass(slots=True)
class NeuralGeneratorConfig(GeneratorConfig):
    """Configuration for neural network-based event generator.

    This configuration manages the complete NN-based generation pipeline,
    including event-level, particle-level, and optional impact parameter models.

    Parameters
    ----------
    name : str
        Name identifier for this generator configuration.
    max_particles : int
        Maximum number of particles per event (fixed at training time).
    transform_config_path : Path
        Path to YAML file containing variable transformation configurations.
    truth_vars_to_load : VarNameTuple
        Tuple of truth-level variable names to load from input data.
    event_model_config : ModelConfig
        Configuration for the event-level generative model.
    particle_model_config : ModelConfig
        Configuration for the particle-level generative model.
    impact_model_config : ModelConfig | None, optional
        Configuration for the impact parameter model. Defaults to None.
    """

    type: str = field(default="neural", init=False)
    max_particles: int
    transform_config_path: Path
    variable_requirements: VariableRequirements

    event_model_config: ModelConfig
    particle_model_config: ModelConfig
    impact_model_config: ModelConfig | None = None
    num_steps: int | None = None  # Override sampler steps at runtime (None = use model defaults)

    # Private field for lazy-loaded transform registry
    _transform_registry: TransformRegistry | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        """Post-initialization checks.

        Raises
        ------
        FileNotFoundError
            If the transform config file does not exist at the specified path.
        """
        if self.transform_config_path and not self.transform_config_path.exists():
            raise FileNotFoundError(
                f"Transform config file not found: {self.transform_config_path}"
            )

    def update_from_dict(self, config_dict: dict) -> None:
        """Update configuration parameters from a dictionary.

        This method allows updating specific parameters of the generator
        configuration at runtime, such as the number of sampling steps.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration parameters to update. Supported keys:
            - "num_steps": int, number of sampling steps for all models.
        """
        if "num_steps" in config_dict:
            self.set_num_steps(config_dict["num_steps"])

    def set_num_steps(self, num_steps: int) -> None:
        """Set the number of sampler steps for all models.

        Parameters
        ----------
        num_steps : int
            Number of sampling steps to use for event, particle, and impact models.
        """
        self.num_steps = num_steps
        self.event_model_config.sampler_config.num_steps = num_steps
        self.particle_model_config.sampler_config.num_steps = num_steps
        if self.impact_model_config is not None:
            self.impact_model_config.sampler_config.num_steps = num_steps

    @property
    def transform_registry(self) -> TransformRegistry:
        """Lazy-loaded registry of variable transformations.

        Returns
        -------
        TransformRegistry
            Registry containing all variable transformation configurations.
        """
        if self._transform_registry is None:
            self._transform_registry = TransformRegistry.from_yaml(self.transform_config_path)
        return self._transform_registry

    @property
    def var_transform_dict(self) -> dict[str, VarTransformConfig]:
        """Dictionary of variable transformation configurations.

        Returns
        -------
        dict[str, VarTransformConfig]
            Mapping of variable names to their transformation configurations.
        """
        return self.transform_registry.transforms

    def get_max_particles(self) -> int:
        """Get maximum particles for neural generator.

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
        return self.variable_requirements.ctxt_vars_stripped

    @property
    def pflow_output_vars(self) -> list[str]:
        """Particle flow output variable names (backward compatibility).

        Returns
        -------
        list[str]
            List of particle flow output variable names.
        """
        return [
            *self.particle_model_config.fs_vars_stripped,
            *(self.impact_model_config.fs_vars_stripped if self.impact_model_config else []),
        ]

    @classmethod
    def load_from_metadata(cls, metadata_path: Path) -> Self:
        """Load neural generator configuration from metadata file.

        Parameters
        ----------
        metadata_path : Path
            Path to the metadata YAML file.

        Returns
        -------
        NeuralGeneratorConfig
            Loaded generator configuration.

        Raises
        ------
        ValueError
            If required models (event, particle) are not defined in metadata.
        """
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        top_path = metadata_path.parent.absolute()

        event_model_config: ModelConfig | None = None
        particle_model_config: ModelConfig | None = None
        impact_model_config: ModelConfig | None = None

        for key, value in metadata["models"].items():
            sampler_config = SamplerConfig(**value.get("sampler", {}))
            if key == "event":
                event_model_config = ModelConfig(
                    name="event",
                    file_path=top_path / value["file_name"],
                    fs_vars=value["fs_vars"],
                    version=value["version"],
                    timestamp=value["timestamp"],
                    sampler_config=sampler_config,
                )
            elif key == "particle":
                particle_model_config = ModelConfig(
                    name="particle",
                    file_path=top_path / value["file_name"],
                    fs_vars=value["fs_vars"],
                    version=value["version"],
                    timestamp=value["timestamp"],
                    sampler_config=sampler_config,
                )
            elif key == "impact":
                impact_model_config = ModelConfig(
                    name="impact",
                    file_path=top_path / value["file_name"],
                    fs_vars=value["fs_vars"],
                    version=value["version"],
                    timestamp=value["timestamp"],
                    sampler_config=sampler_config,
                )
            else:
                raise ValueError(f"Unknown model type: {key}")

        if event_model_config is None or particle_model_config is None:
            raise ValueError("Both event and particle models must be defined in metadata.")

        variable_requirements = VariableRequirements.from_dict(metadata.get("variables", {}))
        return cls(
            name=metadata["name"],
            max_particles=metadata["max_particles"],
            transform_config_path=top_path / "var_transform.yaml",
            variable_requirements=variable_requirements,
            event_model_config=event_model_config,
            particle_model_config=particle_model_config,
            impact_model_config=impact_model_config,
        )


# Registry of available generators
NEURAL_GENERATORS_REGISTRY: dict[str, GeneratorConfig] = {
    "cms_2011_flow_v00": NeuralGeneratorConfig.load_from_metadata(
        Path(__file__).parent.parent.parent / "pretrained_models/cms_2011/metadata.yaml"
    )
}

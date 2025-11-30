from dataclasses import dataclass, field
from itertools import starmap
from pathlib import Path
from typing import Any, Self

import yaml

from parnassus.utils.logger import setup_logger

from .data import DatasetConfig
from .generators import GENERATORS_REGISTRY, GeneratorConfig
from .pipeline import GenPipelineConfig, get_pipeline_config
from .writer import WriterConfig

DEFAULT_GENERATOR = "cms_2011_flow_v00"


@dataclass(slots=True)
class Config:
    # Writer config
    writer_config: WriterConfig

    # Pipeline configs
    pipeline_configs: list[GenPipelineConfig]

    # Dataset config
    dataset_config: DatasetConfig

    # Event generator configuration
    generator_config: GeneratorConfig = field(init=False)
    generator_name: str = DEFAULT_GENERATOR

    # Execution parameters
    batch_size: int = 2000
    device: str = "mps"
    gpu_id: int = 0

    def __post_init__(self):
        # Validate generator name
        if self.generator_name not in GENERATORS_REGISTRY:
            available = ", ".join(GENERATORS_REGISTRY.keys())
            raise ValueError(
                f"Unknown generator '{self.generator_name}'. Available generators: {available}. "
                f"Currently only {DEFAULT_GENERATOR} is fully supported."
            )

        self.generator_config = GENERATORS_REGISTRY[self.generator_name]

    # Backward compatibility properties
    @property
    def generator(self) -> GeneratorConfig:
        """Backward compatibility: access generator_config as 'generator'.

        Returns
        -------
        GeneratorConfig
            The configured event generator.
        """
        return self.generator_config

    @property
    def model(self) -> GeneratorConfig:
        """Backward compatibility: access generator as 'model'.

        Returns
        -------
        GeneratorConfig
            The configured event generator.
        """
        return self.generator_config

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Create Config from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Dictionary containing configuration sections (output, pipelines, dataset, model).

        Returns
        -------
        Config
            A new Config instance with properly initialized components.

        Raises
        ------
        ValueError
            If the specified generator name is not found in GENERATORS_REGISTRY.
        """
        log = setup_logger()

        output_config_dict = config_dict["output"]
        pipeline_config_dict = config_dict["pipelines"]
        dataset_config_dict = config_dict["dataset"]
        gen_config_dict = config_dict["generator"]

        # Get generator config first to extract variable requirements
        gen_name = gen_config_dict["name"]
        gen_type = gen_config_dict.get("type", "neural")  # Default to neural for backward compat

        if gen_name not in GENERATORS_REGISTRY:
            available = ", ".join(GENERATORS_REGISTRY.keys())
            raise ValueError(f"Unknown generator '{gen_name}'. Available generators: {available}")

        # Get generator config from registry
        gen_config = GENERATORS_REGISTRY[gen_name]

        # Verify type matches if specified
        if gen_config.type != gen_type:
            raise ValueError(
                f"Generator '{gen_name}' has type '{gen_config.type}' "
                f"but config specifies '{gen_type}'"
            )

        # Override generator config parameters from YAML if provided (type-specific)
        from .generators import NeuralGeneratorConfig, ParametricGeneratorConfig

        if isinstance(gen_config, NeuralGeneratorConfig):
            num_steps = gen_config_dict.get("num_steps")
            if num_steps is not None:
                gen_config.set_num_steps(num_steps)
        elif isinstance(gen_config, ParametricGeneratorConfig):
            # Handle parametric-specific overrides here
            num_steps = gen_config_dict.get("num_steps")
            if num_steps is not None:
                log.warning(
                    f"'num_steps' parameter is ignored for parametric generator '{gen_name}'. "
                    "This parameter only applies to neural generators.",
                )
            seed = gen_config_dict.get("seed")
            if seed is not None:
                gen_config.seed = seed

        # Create dataset config with variables and max_particles from generator
        dataset_config = DatasetConfig.from_dict_and_model(dataset_config_dict, gen_config)

        return cls(
            pipeline_configs=list(starmap(get_pipeline_config, pipeline_config_dict.items())),
            dataset_config=dataset_config,
            generator_name=gen_name,
            batch_size=gen_config_dict.get("batch_size", 2000),
            device=gen_config_dict.get("device", "cpu"),
            writer_config=WriterConfig.from_dict(output_config_dict),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> Self:
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} doesn't exist!")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)

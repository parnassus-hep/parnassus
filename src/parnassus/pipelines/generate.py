"""Event generation pipeline and orchestration."""

from typing import final

import torch
from torch.utils.data import DataLoader

from parnassus.configs import Config
from parnassus.configs.accessors import Accessor
from parnassus.configs.generators import NeuralGeneratorConfig, ParametricGeneratorConfig
from parnassus.configs.scheme import GenEvent
from parnassus.data import build_dataset, parametric_collate_fn
from parnassus.data.adapters import ParametricAdapter
from parnassus.data.protocols import NeuralDataset, ParametricDataset
from parnassus.utils.logger import setup_logger

from .base import EventGenerator
from .generators import NeuralEventGenerator, ParametricEventGenerator


@final
class GenerationPipeline:
    """Orchestrates end-to-end event generation from dataset to GenEvent objects."""

    def __init__(self, config: Config):
        self.config = config
        self.generator: EventGenerator | None = None
        self._accessors: dict[str, list[Accessor]] | None = None

    def get_accessors(self) -> dict[str, list[Accessor]]:
        """Return copy of cached accessors from last run.

        Returns
        -------
        dict[str, list[Accessor]]
            Dictionary mapping accessor category names to lists of accessor instances.
        """
        return {key: list(value) for key, value in (self._accessors or {}).items()}

    def run(self) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
        """Generate events using configured generator and dataset.

        Returns
        -------
        tuple[list[GenEvent], dict[str, list[Accessor]]]
            Generated events and their accessor definitions.
        """
        log = setup_logger()

        log.info("[green]Starting loading input data...")
        dataset = self._build_dataset()
        dataloader = self._build_dataloader(dataset)
        log.info("[green]Data loading completed.")

        self.generator = self._init_generator()
        n_events = len(dataset)  # type: ignore[arg-type]
        with self.generator:
            self.generator.initialize(n_events, n_batches=len(dataloader))
            for batch in dataloader:
                self.generator.process_batch(batch)
            events = self.generator.get_events()
        log.info(f"[green]Generated {len(events)} events from requested {n_events}.")
        accessors = self.generator.get_accessors()
        self._accessors = accessors
        return events, accessors

    def _build_dataset(self) -> NeuralDataset | ParametricDataset:
        """Create dataset from configuration.

        Returns
        -------
        NeuralDataset | ParametricDataset
            Dataset instance based on configured file type and generator mode.
        """
        if isinstance(self.config.generator_config, NeuralGeneratorConfig):
            transform_registry = self.config.generator_config.transform_registry
            return build_dataset(self.config.dataset_config, transform_registry, mode="neural")
        if isinstance(self.config.generator_config, ParametricGeneratorConfig):
            return build_dataset(self.config.dataset_config, mode="parametric")
        raise TypeError(
            f"Unsupported generator type: {type(self.config.generator_config).__name__}"
        )

    def _build_dataloader(self, dataset: NeuralDataset | ParametricDataset) -> DataLoader:
        """Create dataloader for batching dataset.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for batch iteration.
        """
        if isinstance(dataset, ParametricAdapter):
            return DataLoader(
                dataset,  # type: ignore[arg-type]
                batch_size=self.config.batch_size,
                collate_fn=parametric_collate_fn,
                num_workers=0,
            )
        return DataLoader(dataset, batch_size=self.config.batch_size, num_workers=0)  # type: ignore[arg-type]

    def _init_generator(self) -> EventGenerator:
        """Initialise event generator based on configuration type.

        Returns
        -------
        EventGenerator
            Initialised event generator.

        Raises
        ------
        TypeError
            If generator configuration type is not supported.
        """
        log = setup_logger()
        device = torch.device(self.config.device)

        if isinstance(self.config.generator_config, NeuralGeneratorConfig):
            return NeuralEventGenerator(self.config.generator_config, log).to(device)
        if isinstance(self.config.generator_config, ParametricGeneratorConfig):
            return ParametricEventGenerator(self.config.generator_config, log).to(device)
        raise TypeError(
            f"Unsupported generator type: {type(self.config.generator_config).__name__}"
        )


def generate(config: Config) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
    """Legacy entrypoint for event generation. Prefer GenerationPipeline.run().

    Returns
    -------
    tuple[list[GenEvent], dict[str, list[Accessor]]]
        Generated events and their accessor definitions.
    """
    pipeline = GenerationPipeline(config)
    return pipeline.run()

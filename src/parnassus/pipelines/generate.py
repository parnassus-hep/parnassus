"""Event generation pipeline and orchestration."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import numpy as np
import torch
from torch.utils.data import DataLoader

from parnassus.configs import Config
from parnassus.configs.accessors import Accessor
from parnassus.configs.generators import NeuralGeneratorConfig, ParametricGeneratorConfig
from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.data import build_dataset
from parnassus.utils.logger import ProgressBar, setup_logger, update_task

from .base import EventGenerator, SourcePipeline
from .generators import NeuralEventGenerator

if TYPE_CHECKING:
    from parnassus.data.base import BaseDataset


@dataclass
class GenerationBuffers:
    """Storage for generated event data during batch processing."""

    truth_data: dict[str, np.ndarray]
    pflow_data: dict[str, np.ndarray]
    event_numbers: np.ndarray
    count: int = 0

    def trim(self) -> "GenerationBuffers":
        """Remove unused buffer space after generation completes.

        Returns
        -------
        GenerationBuffers
            Self with trimmed arrays.
        """
        for key in self.truth_data:
            self.truth_data[key] = self.truth_data[key][: self.count]
        for key in self.pflow_data:
            self.pflow_data[key] = self.pflow_data[key][: self.count]
        self.event_numbers = self.event_numbers[: self.count]
        return self


@final
class GenerationPipeline(SourcePipeline):
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
        buffers = self._run_sampling(dataloader)
        log.info(f"[green]Generated {buffers.count} events from requested {len(dataset)}.")

        event_list = self._build_events(buffers)
        accessors_dict = self._build_accessors()
        self._accessors = accessors_dict
        return event_list, accessors_dict

    def _build_dataset(self) -> "BaseDataset":
        """Create dataset from configuration.

        Returns
        -------
        BaseDataset
            Dataset instance based on configured file type.
        """
        # Extract transform_registry if available (NN generators have it)
        transform_registry = (
            self.config.generator_config.transform_registry
            if isinstance(self.config.generator_config, NeuralGeneratorConfig)
            else None
        )
        return build_dataset(self.config.dataset_config, transform_registry)

    def _build_dataloader(self, dataset: "BaseDataset") -> DataLoader:
        """Create dataloader for batching dataset.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for batch iteration.
        """
        return DataLoader(dataset, batch_size=self.config.batch_size, num_workers=0)

    def _init_generator(self) -> EventGenerator:
        """Initialize event generator based on configuration type.

        Returns
        -------
        EventGenerator
            Initialized event generator on configured device.

        Raises
        ------
        TypeError
            If generator configuration type is not supported.
        """
        log = setup_logger()
        device = torch.device(self.config.device)

        if isinstance(self.config.generator_config, NeuralGeneratorConfig):
            return NeuralEventGenerator(self.config.generator_config, log).to(device)
        # Placeholder for parametric generators
        if isinstance(self.config.generator_config, ParametricGeneratorConfig):
            raise NotImplementedError(
                f"Parametric generator '{self.config.generator_config.name}' "
                "is not yet implemented."
            )
        raise TypeError(
            f"Unsupported generator type: {type(self.config.generator_config).__name__}"
        )

    def _init_buffers(self, num_events: int) -> GenerationBuffers:
        """Allocate buffer storage for generation output.

        Returns
        -------
        GenerationBuffers
            Pre-allocated buffers for storing generated particle data.
        """
        assert self.generator is not None, "Generator must be initialized before buffers"

        truth_vars = [*self.generator.truth_output_vars, "ind"]
        pflow_vars = [*self.generator.pflow_output_vars, "ind"]
        max_particles = self.config.dataset_config.max_particles

        def zeros_for_vars(var_names: list[str]) -> dict[str, np.ndarray]:
            return {
                key.replace("ptrel", "pt"): np.zeros(
                    (num_events, max_particles),
                    dtype=np.float32,
                )
                for key in var_names
            }

        return GenerationBuffers(
            truth_data=zeros_for_vars(truth_vars),
            pflow_data=zeros_for_vars(pflow_vars),
            event_numbers=np.zeros(num_events, dtype=np.int32),
        )

    def _run_sampling(self, dataloader: DataLoader) -> GenerationBuffers:
        """Execute batch generation loop with progress tracking.

        Returns
        -------
        GenerationBuffers
            Buffers filled with generated event data, trimmed to actual count.
        """
        assert self.generator is not None, "Generator must be initialized before sampling"

        n_events = len(dataloader.dataset)  # pyright: ignore[reportArgumentType]
        buffers = self._init_buffers(n_events)
        if n_events == 0:
            return buffers

        with ProgressBar() as progress_bar:
            total_gen_task = progress_bar.add_task("[green]Generating data", total=len(dataloader))

            # Create progress tasks only if generator has sampler steps
            evt_sampler_task = None
            if self.generator.event_sampler_steps is not None:
                evt_sampler_task = progress_bar.add_task(
                    "[green]Sampling event data", total=self.generator.event_sampler_steps
                )

            part_sampler_task = None
            if self.generator.particle_sampler_steps is not None:
                part_sampler_task = progress_bar.add_task(
                    "[green]Sampling particle data", total=self.generator.particle_sampler_steps
                )

            impact_sampler_task = None
            if self.generator.impact_sampler_steps is not None:
                impact_sampler_task = progress_bar.add_task(
                    "[green]Sampling impact data",
                    total=self.generator.impact_sampler_steps,
                )

            for batch in dataloader:
                if evt_sampler_task is not None:
                    progress_bar.reset(evt_sampler_task)
                if part_sampler_task is not None:
                    progress_bar.reset(part_sampler_task)
                if impact_sampler_task is not None:
                    progress_bar.reset(impact_sampler_task)

                tr_data_dict, pf_data_dict, common_data_dict = self.generator.generate_batch(
                    batch,
                    event_callback=update_task(progress_bar, evt_sampler_task)
                    if evt_sampler_task is not None
                    else None,
                    particle_callback=update_task(progress_bar, part_sampler_task)
                    if part_sampler_task is not None
                    else None,
                    impact_callback=update_task(progress_bar, impact_sampler_task)
                    if impact_sampler_task is not None
                    else None,
                )
                event_number = common_data_dict["event_number"]
                tr_mask = common_data_dict["tr_mask"]
                pf_mask = common_data_dict["fs_mask"]
                gen_size = event_number.shape[0]
                start = buffers.count
                end = start + gen_size
                for var_name, tr_data in tr_data_dict.items():
                    buffers.truth_data[var_name][start:end] = tr_data
                for var_name, pf_data in pf_data_dict.items():
                    buffers.pflow_data[var_name][start:end] = pf_data
                buffers.truth_data["ind"][start:end] = tr_mask
                buffers.pflow_data["ind"][start:end] = pf_mask
                buffers.event_numbers[start:end] = event_number[..., 0]
                buffers.count = end
                progress_bar.update(total_gen_task, advance=1)

        return buffers.trim()

    def _build_events(self, buffers: GenerationBuffers) -> list[GenEvent]:
        """Convert buffer data into GenEvent objects.

        Returns
        -------
        list[GenEvent]
            List of GenEvent objects with truth and pflow particle collections.
        """
        assert self.generator is not None, "Generator must be initialized before building events"

        event_list: list[GenEvent] = []
        if buffers.count == 0:
            return event_list

        with ProgressBar() as progress:
            conv_task = progress.add_task("[green]Converting events.", total=buffers.count)
            for i in range(buffers.count):
                truth_ind = buffers.truth_data["ind"][i] > 0
                truth_particles = GenParticleCollection(
                    name="truth",
                    pt=buffers.truth_data["pt"][i][truth_ind],
                    eta=buffers.truth_data["eta"][i][truth_ind],
                    phi=buffers.truth_data["phi"][i][truth_ind],
                    vx=buffers.truth_data["vx"][i][truth_ind],
                    vy=buffers.truth_data["vy"][i][truth_ind],
                    vz=buffers.truth_data["vz"][i][truth_ind],
                    class_id=buffers.truth_data["class"][i][truth_ind].astype(np.int32),
                )
                pflow_ind = (buffers.pflow_data["ind"][i] > 0) & (buffers.pflow_data["pt"][i] > 1)
                impact_dict = {}
                if self.generator.has_impact_model:
                    impact_dict = {
                        "d0": buffers.pflow_data["d0"][i][pflow_ind],
                        "z0": buffers.pflow_data["z0"][i][pflow_ind],
                        "d0_error": buffers.pflow_data["d0Error"][i][pflow_ind],
                        "z0_error": buffers.pflow_data["z0Error"][i][pflow_ind],
                    }
                pflow_particles = GenParticleCollection(
                    name="pflow",
                    pt=buffers.pflow_data["pt"][i][pflow_ind],
                    eta=buffers.pflow_data["eta"][i][pflow_ind],
                    phi=buffers.pflow_data["phi"][i][pflow_ind],
                    vx=buffers.pflow_data["vx"][i][pflow_ind],
                    vy=buffers.pflow_data["vy"][i][pflow_ind],
                    vz=buffers.pflow_data["vz"][i][pflow_ind],
                    class_id=buffers.pflow_data["class"][i][pflow_ind].astype(np.int32),
                    **impact_dict,
                )
                event_list.append(
                    GenEvent(
                        event_number=buffers.event_numbers[i],
                        truth_particles=truth_particles,
                        pflow_particles=pflow_particles,
                    )
                )
                progress.update(conv_task, advance=1)
        return event_list

    def _build_accessors(self) -> dict[str, list[Accessor]]:
        """Build accessor definitions from generator output specification.

        Returns
        -------
        dict[str, list[Accessor]]
            Dictionary mapping categories (Truth, Pflow, Electrons, Muons) to accessor lists.
        """
        assert self.generator is not None, "Generator must be initialized before building accessors"

        # Get accessor partial constructors from generator
        accessor_partials = self.generator.get_accessors()

        # Build pflow accessors (all particle + impact if available)
        pflow_accessors: list[Accessor] = [
            accessor_partial(collection="pflow_particles") for accessor_partial in accessor_partials
        ]

        # Filter out impact accessors for truth (check partial keywords)
        impact_names = {"d0", "z0", "d0_error", "z0_error"}
        truth_accessor_partials = [
            ap for ap in accessor_partials if ap.keywords.get("name") not in impact_names
        ]

        # Lepton accessors only include pt, eta, phi
        lepton_names = {"pt", "eta", "phi"}
        lepton_accessor_partials = [
            ap for ap in accessor_partials if ap.keywords.get("name") in lepton_names
        ]

        return {
            "Truth": [
                accessor_partial(collection="truth_particles")
                for accessor_partial in truth_accessor_partials
            ],
            "Pflow": pflow_accessors,
            "Electrons": [
                accessor_partial(collection="electrons")
                for accessor_partial in lepton_accessor_partials
            ],
            "Muons": [
                accessor_partial(collection="muons")
                for accessor_partial in lepton_accessor_partials
            ],
        }


def generate(config: Config) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
    """Legacy entrypoint for event generation. Prefer GenerationPipeline.run().

    Returns
    -------
    tuple[list[GenEvent], dict[str, list[Accessor]]]
        Generated events and their accessor definitions.
    """
    pipeline = GenerationPipeline(config)
    return pipeline.run()

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Self, final

import numpy as np
import torch
from torch.utils.data import DataLoader

from parnassus.configs import Config
from parnassus.configs.accessors import Accessor, ParticleAccessor
from parnassus.configs.model import GenerativeModelConfig
from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.data import HepMCDataset, RootDataset
from parnassus.nn import ModelWrapper
from parnassus.utils import Unscaler
from parnassus.utils.logger import ProgressBar, setup_logger, update_task
from parnassus.utils.typing import TensorDict

from .base import SourcePipeline

if TYPE_CHECKING:
    from parnassus.configs.data import DatasetConfig
    from parnassus.data.base import BaseDataset

PARTICLE_ACCESSORS = [
    partial(ParticleAccessor, name=name, dtype="float32")
    for name in ["pt", "eta", "phi", "vx", "vy", "vz"]
] + [partial(ParticleAccessor, name=name, dtype="int32") for name in ["class_id", "pdg_id"]]

IMPACT_ACCESSORS = [
    partial(ParticleAccessor, name=name, dtype="float32")
    for name in ["d0", "z0", "d0_error", "z0_error"]
]

LEPTON_ACCESSORS = [
    partial(ParticleAccessor, name=name, dtype="float32") for name in ["pt", "eta", "phi"]
]


@dataclass
class GenerationBuffers:
    truth_data: dict[str, np.ndarray]
    pflow_data: dict[str, np.ndarray]
    event_numbers: np.ndarray
    count: int = 0

    def trim(self) -> "GenerationBuffers":
        for key in self.truth_data:
            self.truth_data[key] = self.truth_data[key][: self.count]
        for key in self.pflow_data:
            self.pflow_data[key] = self.pflow_data[key][: self.count]
        self.event_numbers = self.event_numbers[: self.count]
        return self


def build_dataset(
    dataset_config: "DatasetConfig",
    transform_registry,
    dataset_builders: Mapping[str, Callable[..., "BaseDataset"]] | None = None,
) -> "BaseDataset":
    input_file = dataset_config.file_path
    assert isinstance(input_file, Path)
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Trying to load file {input_file}, no file exist!")

    var_transform_dict = transform_registry.to_var_transform_dict()
    builders = dataset_builders or {".root": RootDataset, ".hepmc": HepMCDataset}
    suffix = input_file.suffix.lower()
    if suffix not in builders:
        raise ValueError(
            f"Only ROOT or HepMC files are supported as input, got {dataset_config.file_path}"
        )
    dataset_factory = builders[suffix]
    return dataset_factory(dataset_config, var_transform_dict=var_transform_dict)


def build_dataloader(dataset: "BaseDataset", batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


def init_generative_model(
    model_config: GenerativeModelConfig, log, device: torch.device
) -> "GenerativeModel":
    return GenerativeModel(model_config, log).to(device)


def init_generation_buffers(
    generative_model: "GenerativeModel",
    dataset_config: "DatasetConfig",
    num_events: int,
) -> GenerationBuffers:
    truth_vars = [*generative_model.config.truth_output_vars, "ind"]
    pflow_vars = [*generative_model.config.pflow_output_vars, "ind"]
    max_particles = dataset_config.max_particles

    def zeros_for_vars(var_names: list[str]) -> dict[str, np.ndarray]:
        return {
            key.replace("ptrel", "pt"): np.zeros(
                (
                    num_events,
                    max_particles,
                ),
                dtype=np.float32,
            )
            for key in var_names
        }

    return GenerationBuffers(
        truth_data=zeros_for_vars(truth_vars),
        pflow_data=zeros_for_vars(pflow_vars),
        event_numbers=np.zeros(num_events, dtype=np.int32),
    )


def run_sampling(
    dataloader: DataLoader, generative_model: "GenerativeModel", dataset_config: "DatasetConfig"
) -> GenerationBuffers:
    n_events = len(dataloader.dataset)  # pyright: ignore[reportArgumentType]
    buffers = init_generation_buffers(generative_model, dataset_config, n_events)
    if n_events == 0:
        return buffers

    with ProgressBar() as progress_bar:
        total_gen_task = progress_bar.add_task("[green]Generating data", total=len(dataloader))
        evt_sampler_task = progress_bar.add_task(
            "[green]Sampling event data", total=generative_model.event_model.sampler.n_steps
        )
        part_sampler_task = progress_bar.add_task(
            "[green]Sampling particle data", total=generative_model.particle_model.sampler.n_steps
        )
        impact_sampler_task = None
        if generative_model.impact_model is not None:
            impact_sampler_task = progress_bar.add_task(
                "[green]Sampling impact data",
                total=generative_model.impact_model.sampler.n_steps,
            )

        for batch in dataloader:
            progress_bar.reset(evt_sampler_task)
            progress_bar.reset(part_sampler_task)
            if impact_sampler_task is not None:
                progress_bar.reset(impact_sampler_task)

            tr_data_dict, pf_data_dict, common_data_dict = generative_model.generate_event(
                batch,
                event_callback=update_task(progress_bar, evt_sampler_task),
                particle_callback=update_task(progress_bar, part_sampler_task),
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


def build_events(buffers: GenerationBuffers, generative_model: "GenerativeModel") -> list[GenEvent]:
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
            if generative_model.impact_model is not None:
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


def build_accessors(generative_model: "GenerativeModel") -> dict[str, list[Accessor]]:
    pflow_accessors: list[Accessor] = [
        accessor(collection="pflow_particles") for accessor in PARTICLE_ACCESSORS
    ]
    if generative_model.impact_model is not None:
        pflow_accessors += [accessor(collection="pflow_particles") for accessor in IMPACT_ACCESSORS]
    return {
        "Truth": [accessor(collection="truth_particles") for accessor in PARTICLE_ACCESSORS],
        "Pflow": pflow_accessors,
        "Electrons": [accessor(collection="electrons") for accessor in LEPTON_ACCESSORS],
        "Muons": [accessor(collection="muons") for accessor in LEPTON_ACCESSORS],
    }


@final
class GenerationPipeline(SourcePipeline):
    def __init__(self, config: Config):
        self.config = config
        self._accessors: dict[str, list[Accessor]] | None = None

    def get_accessors(self) -> dict[str, list[Accessor]]:
        return {key: list(value) for key, value in (self._accessors or {}).items()}

    def run(self) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
        log = setup_logger()
        model_config = self.config.model
        dataset_config = self.config.dataset_config
        device = torch.device(self.config.device)

        log.info("[green]Starting loading input data...")
        dataset = build_dataset(dataset_config, model_config.transform_registry)
        dataloader = build_dataloader(dataset, self.config.batch_size)
        log.info("[green]Data loading completed.")
        generative_model = init_generative_model(model_config, log, device)

        buffers = run_sampling(dataloader, generative_model, dataset_config)
        log.info(f"[green]Generated {buffers.count} events from requested {len(dataset)}.")
        event_list = build_events(buffers, generative_model)
        accessors_dict = build_accessors(generative_model)
        self._accessors = accessors_dict
        return event_list, accessors_dict


@final
class GenerativeModel:
    def __init__(self, config: GenerativeModelConfig, log):
        self.config = config
        self.log = log
        self.device = torch.device("cpu")

        self.log.info("[green]Loading networks...")
        self.event_model = ModelWrapper(config.event_model_config)
        self.particle_model = ModelWrapper(config.particle_model_config)

        self.impact_model: ModelWrapper | None
        if config.impact_model_config is not None:
            self.impact_model = ModelWrapper(config.impact_model_config)
        else:
            self.impact_model = None
        self.log.info("[green]Networks loading completed.")

        # Use the transform registry to get VarTransform instances
        self.var_transform_dict = config.transform_registry.to_var_transform_dict()

        self.fs_npart_pos = config.event_model_config.variables_config.fs_vars.index("npflow")
        self.fs_ht_pos = config.event_model_config.variables_config.fs_vars.index("pflow_ht")

        self.ht_shift = self.var_transform_dict["ht"].shift
        self.ht_scale = self.var_transform_dict["ht"].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale
        self.max_particles = config.max_particles

        self.unscaler = Unscaler(
            transform_dict=self.var_transform_dict,
            ctxt_vars=config.particle_model_config.variables_config.ctxt_vars,
            fs_vars=config.particle_model_config.variables_config.fs_vars,
            ctxt_global_vars=config.particle_model_config.variables_config.ctxt_global_vars,
        )

    def to(self, device: torch.device) -> Self:
        self.event_model.to(device)
        self.particle_model.to(device)
        if self.impact_model is not None:
            self.impact_model.to(device)
        self.device = device
        return self

    def generate_event(
        self,
        data_dict: TensorDict,
        event_callback: Callable[[], None] | None = None,
        particle_callback: Callable[[], None] | None = None,
        impact_callback: Callable[[], None] | None = None,
    ) -> tuple[dict[str, np.ndarray], ...]:
        ctxt_data, ctxt_global_data, ctxt_mask, event_number = (
            data_dict["ctxt_data"].to(self.device),
            data_dict["ctxt_global_data"].to(self.device),
            data_dict["mask"].to(self.device),
            data_dict["event_number"].to(self.device),
        )
        batch_size = ctxt_data.shape[0]
        fs_mask = None
        fs_evt = self.event_model.sample(
            (batch_size,),
            mask=ctxt_mask,
            ctxt_data=ctxt_data,
            ctxt_global_data=ctxt_global_data,
            callback=event_callback,
            to_cpu=False,
        )
        fs_npart = self.var_transform_dict["npart"].inverse_transform(
            fs_evt[..., self.fs_npart_pos]
        )
        fs_ht = fs_evt[..., self.fs_ht_pos]
        good_evt_mask = (
            (fs_npart > 0) & (fs_npart <= self.max_particles) & (fs_ht > self.min_ht_scaled)
        )
        bad_idxs = torch.argwhere(~good_evt_mask).flatten().cpu().numpy()

        ctxt_data = ctxt_data[good_evt_mask]
        ctxt_global_data = ctxt_global_data[good_evt_mask]
        ctxt_mask = ctxt_mask[good_evt_mask]
        event_number = event_number[good_evt_mask]

        fs_npart = fs_npart[good_evt_mask]
        fs_evt = fs_evt[good_evt_mask]
        fs_mask = torch.arange(self.max_particles, device=fs_evt.device).expand(
            fs_evt.shape[0], self.max_particles
        ) < fs_npart.unsqueeze(1)

        # Concat event-level generated data to global context
        ctxt_global_data = torch.cat([ctxt_global_data, fs_evt], -1)
        particle_mask = torch.stack([ctxt_mask, fs_mask], -1)

        fs_part = self.particle_model.sample(
            (
                fs_evt.shape[0],
                self.max_particles,
            ),
            mask=particle_mask,
            ctxt_data=ctxt_data,
            ctxt_global_data=ctxt_global_data,
            callback=particle_callback,
            to_cpu=False,
        )
        pf_impact_data_dict = {}
        if self.impact_model is not None:
            pf_ctxt_data = fs_part
            fs_class = fs_part[..., -5:].argmax(
                -1
            )  # HACK: assume last 5 vars are class one-hot and last in the order
            particle_mask[..., -1] = (fs_class < 3) & (fs_mask)
            fs_impact_shape = (
                fs_part.shape[0],
                fs_part.shape[1],
            )
            fs_impact = self.impact_model.sample(
                fs_impact_shape,
                mask=particle_mask,
                ctxt_data=ctxt_data,
                ctxt_global_data=ctxt_global_data,
                pf_ctxt_data=pf_ctxt_data,
                callback=impact_callback,
                to_cpu=False,
            )
            pf_impact_data_dict = self.unscaler.unscale_impact_variables(fs_impact)

        tr_data_dict, pf_data_dict = self.unscaler.unscale_variables(
            fs_part, ctxt_data, ctxt_global_data
        )
        pf_data_dict.update(pf_impact_data_dict)
        return (
            tr_data_dict,
            pf_data_dict,
            {
                "bad_idxs": bad_idxs,
                "event_number": event_number.cpu().numpy(),
                "fs_mask": fs_mask.cpu().numpy(),
                "tr_mask": ctxt_mask.cpu().numpy(),
            },
        )


def generate(config: Config) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
    """Legacy entrypoint. Prefer GenerationPipeline.run()."""  # noqa: DOC201
    pipeline = GenerationPipeline(config)
    return pipeline.run()

"""Neural network-based event generator implementation."""

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import ClassVar, Self, final

import numpy as np
import torch
from rich.progress import Progress, TaskID
from torch import nn

from parnassus.configs.accessors import (
    Accessor,
    AccessorListBuilder,
    AccessorSpec,
    AccessorTemplates,
)
from parnassus.configs.generators import NeuralGeneratorConfig
from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.nn import ModelWrapper
from parnassus.utils import Unscaler
from parnassus.utils.logger import ProgressBar, update_task
from parnassus.utils.typing import TensorDict


@dataclass
class _GenerationBuffers:
    """Internal storage for accumulated batch outputs."""

    truth_data: dict[str, np.ndarray]
    pflow_data: dict[str, np.ndarray]
    event_numbers: np.ndarray
    count: int = 0

    def trim(self) -> "_GenerationBuffers":
        for key in self.truth_data:
            self.truth_data[key] = self.truth_data[key][: self.count]
        for key in self.pflow_data:
            self.pflow_data[key] = self.pflow_data[key][: self.count]
        self.event_numbers = self.event_numbers[: self.count]
        return self


@final
class NeuralEventGenerator:
    """Neural network-based event generator implementing EventGenerator protocol."""

    # Maps generation-buffer keys to GenParticleCollection kwargs. Buffers only
    # ever hold the keys a given model produces (derived from truth/pflow output
    # vars), so this lets get_events() assemble collections without assuming a
    # fixed schema (e.g. CMS vertices+errors vs. CLD significances, no vertices).
    _BUFFER_TO_FIELD: ClassVar[dict[str, str]] = {
        "pt": "pt",
        "eta": "eta",
        "phi": "phi",
        "vx": "vx",
        "vy": "vy",
        "vz": "vz",
        "d0": "d0",
        "z0": "z0",
        "d0Error": "d0_error",
        "z0Error": "z0_error",
        "d0Sig": "d0_sig",
        "z0Sig": "z0_sig",
        "class": "class_id",
        "charge": "charge",
    }
    _INT_FIELDS: ClassVar[frozenset[str]] = frozenset({"class_id", "charge"})

    def __init__(self, config: NeuralGeneratorConfig, log):
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

        self.fs_npart_pos = config.event_model_config.fs_vars.index("npflow")
        self.fs_ht_pos = config.event_model_config.fs_vars.index("pflow_ht")

        self.ht_shift = self.var_transform_dict["ht"].shift
        self.ht_scale = self.var_transform_dict["ht"].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale

        self.unscaler = Unscaler(
            transform_dict=self.var_transform_dict,
            ctxt_vars=config.variable_requirements.ctxt_vars,
            fs_vars=config.particle_model_config.fs_vars,
            ctxt_global_vars=config.variable_requirements.ctxt_global_vars,
        )

        # State managed across initialize / process_batch / get_events
        self._buffers: _GenerationBuffers | None = None
        self._exit_stack: ExitStack | None = None
        self._progress_bar: Progress | None = None
        self._total_gen_task: TaskID | None = None
        self._evt_sampler_task: TaskID | None = None
        self._part_sampler_task: TaskID | None = None
        self._impact_sampler_task: TaskID | None = None

    # ------------------------------------------------------------------ #
    # Internal-use properties (not part of the public protocol)           #
    # ------------------------------------------------------------------ #

    @property
    def has_impact_model(self) -> bool:
        return self.impact_model is not None

    @property
    def max_particles(self) -> int:
        return self.config.max_particles

    @property
    def _event_sampler_steps(self) -> int | None:
        return self.event_model.sampler.n_steps

    @property
    def _particle_sampler_steps(self) -> int | None:
        return self.particle_model.sampler.n_steps

    @property
    def _impact_sampler_steps(self) -> int | None:
        return self.impact_model.sampler.n_steps if self.impact_model else None

    # ------------------------------------------------------------------ #
    # Accessor management                                                  #
    # ------------------------------------------------------------------ #

    @property
    def _pflow_has_vertex(self) -> bool:
        """Whether the particle model emits production vertices (vx, vy, vz)."""
        return {"vx", "vy", "vz"}.issubset(self.config.pflow_output_vars)

    @property
    def _pflow_impact_specs(self) -> list[AccessorSpec]:
        """Impact accessors matching the pflow schema.

        Two schemas are supported:
          * error-based (CMS/ALEPH): d0/z0 + d0_error/z0_error, produced by a
            separate impact model (``has_impact_model``);
          * significance-based (CLD): d0/z0 + d0_sig/z0_sig, produced directly
            by the particle model as extra fs_vars (no separate impact model).
        """
        pflow_vars = set(self.config.pflow_output_vars)
        if {"d0Sig", "z0Sig"} & pflow_vars:
            return AccessorTemplates.IMPACT_SIGNIFICANCES
        if self.has_impact_model or ({"d0Error", "z0Error"} & pflow_vars):
            return AccessorTemplates.IMPACT_PARAMETERS
        return []

    def _get_accessors_builder(
        self, collection: str, specs: Sequence[AccessorSpec], use_impact: bool = False
    ) -> AccessorListBuilder:
        builder = AccessorListBuilder.for_particles(collection).add_from_specs(specs)
        if use_impact:
            builder.add_from_specs(self._pflow_impact_specs)
        return builder

    def get_accessors(self) -> dict[str, list[Accessor]]:
        """Return list of accessor constructors for neural network output.

        Returns
        -------
        dict[str, list[Accessor]]
            Dictionary mapping collection names to lists of accessors.
        """
        has_impact = bool(self._pflow_impact_specs)
        # PFlow specs adapt to the model schema: production vertices are only
        # emitted for schemas that produce them (e.g. CMS), while CLD reports
        # impact significances and no vertices.
        pflow_specs = [
            *AccessorTemplates.KINEMATICS,
            *(AccessorTemplates.VERTEX if self._pflow_has_vertex else []),
            *AccessorTemplates.CLASS_IDS,
        ]
        return {
            "Truth": self._get_accessors_builder(
                collection="truth_particles", specs=AccessorTemplates.FULL_PARTICLE
            ).build(),
            "PFlow": self._get_accessors_builder(
                collection="pflow_particles",
                specs=pflow_specs,
                use_impact=has_impact,
            ).build(),
            "Electrons": self._get_accessors_builder(
                collection="electrons",
                specs=AccessorTemplates.KINEMATICS,
                use_impact=has_impact,
            ).build(),
            "Muons": self._get_accessors_builder(
                collection="muons",
                specs=AccessorTemplates.KINEMATICS,
                use_impact=has_impact,
            ).build(),
        }

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def to(self, device: torch.device) -> Self:
        self.event_model.to(device)
        self.particle_model.to(device)
        if self.impact_model is not None:
            self.impact_model.to(device)
        self.device = device
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Close any open progress display (no-op if get_events() already did so)
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
            self._progress_bar = None
        # Move model weights back to CPU to free device memory
        self.to(torch.device("cpu"))

    # ------------------------------------------------------------------ #
    # EventGenerator protocol                                              #
    # ------------------------------------------------------------------ #

    def initialize(self, n_events: int, n_batches: int) -> None:
        """Pre-allocate buffers and set up progress tracking.

        Parameters
        ----------
        n_events : int
            Total number of events expected (for buffer pre-allocation).
        n_batches : int
            Total number of batches (for progress bar).
        """
        truth_vars = [*self.config.truth_output_vars, "ind"]
        pflow_vars = [*self.config.pflow_output_vars, "ind"]

        def _zeros(var_names: list[str]) -> dict[str, np.ndarray]:
            return {
                key.replace("ptrel", "pt"): np.zeros(
                    (n_events, self.max_particles), dtype=np.float32
                )
                for key in var_names
            }

        self._buffers = _GenerationBuffers(
            truth_data=_zeros(truth_vars),
            pflow_data=_zeros(pflow_vars),
            event_numbers=np.zeros(n_events, dtype=np.int32),
        )

        self._exit_stack = ExitStack()
        self._progress_bar = self._exit_stack.enter_context(ProgressBar())
        self._total_gen_task = self._progress_bar.add_task(
            "[green]Generating data", total=n_batches
        )
        self._evt_sampler_task = None
        if self._event_sampler_steps is not None:
            self._evt_sampler_task = self._progress_bar.add_task(
                "[green]Sampling event data", total=self._event_sampler_steps
            )
        self._part_sampler_task = None
        if self._particle_sampler_steps is not None:
            self._part_sampler_task = self._progress_bar.add_task(
                "[green]Sampling particle data", total=self._particle_sampler_steps
            )
        self._impact_sampler_task = None
        if self._impact_sampler_steps is not None:
            self._impact_sampler_task = self._progress_bar.add_task(
                "[green]Sampling impact data", total=self._impact_sampler_steps
            )

    def process_batch(self, batch: TensorDict) -> None:
        """Sample one batch, accumulate into internal buffers, and update progress.

        Parameters
        ----------
        batch : TensorDict
            Input batch from the dataloader.
        """
        assert self._buffers is not None, "Call initialize() before process_batch()"
        assert self._progress_bar is not None
        assert self._total_gen_task is not None

        if self._evt_sampler_task is not None:
            self._progress_bar.reset(self._evt_sampler_task)
        if self._part_sampler_task is not None:
            self._progress_bar.reset(self._part_sampler_task)
        if self._impact_sampler_task is not None:
            self._progress_bar.reset(self._impact_sampler_task)

        tr_data_dict, pf_data_dict, common_data_dict = self._sample_batch(
            batch,
            event_callback=update_task(self._progress_bar, self._evt_sampler_task)
            if self._evt_sampler_task is not None
            else None,
            particle_callback=update_task(self._progress_bar, self._part_sampler_task)
            if self._part_sampler_task is not None
            else None,
            impact_callback=update_task(self._progress_bar, self._impact_sampler_task)
            if self._impact_sampler_task is not None
            else None,
        )

        event_number = common_data_dict["event_number"]
        tr_mask = common_data_dict["tr_mask"]
        pf_mask = common_data_dict["fs_mask"]
        gen_size = event_number.shape[0]
        start = self._buffers.count
        end = start + gen_size
        for var_name, tr_data in tr_data_dict.items():
            self._buffers.truth_data[var_name][start:end] = tr_data
        for var_name, pf_data in pf_data_dict.items():
            self._buffers.pflow_data[var_name][start:end] = pf_data
        self._buffers.truth_data["ind"][start:end] = tr_mask
        self._buffers.pflow_data["ind"][start:end] = pf_mask
        self._buffers.event_numbers[start:end] = event_number[..., 0]
        self._buffers.count = end

        self._progress_bar.update(self._total_gen_task, advance=1)

    def get_events(self) -> list[GenEvent]:
        """Finalise, close progress bars, and convert buffers to GenEvent objects.

        Returns
        -------
        list[GenEvent]
            Generated events.
        """
        assert self._buffers is not None, "Call initialize() before get_events()"

        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
            self._progress_bar = None

        buffers = self._buffers.trim()

        event_list: list[GenEvent] = []
        if buffers.count == 0:
            return event_list

        for i in range(buffers.count):
            truth_ind = buffers.truth_data["ind"][i] > 0
            truth_particles = self._build_collection("truth", buffers.truth_data, i, truth_ind)
            pflow_ind = (buffers.pflow_data["ind"][i] > 0) & (buffers.pflow_data["pt"][i] > 1)
            pflow_particles = self._build_collection("pflow", buffers.pflow_data, i, pflow_ind)
            event_list.append(
                GenEvent(
                    event_number=buffers.event_numbers[i],
                    truth_particles=truth_particles,
                    pflow_particles=pflow_particles,
                )
            )

        return event_list

    def _build_collection(
        self, name: str, data: dict[str, np.ndarray], i: int, ind: np.ndarray
    ) -> GenParticleCollection:
        """Assemble a GenParticleCollection from the buffer keys that exist.

        Only keys the model actually produced are present in ``data``, so the
        resulting collection carries exactly the available fields (e.g. CLD has
        no vx/vy/vz and reports d0_sig/z0_sig instead of d0_error/z0_error).

        Returns
        -------
        GenParticleCollection
            Collection holding the mapped fields present in ``data``.
        """
        kwargs: dict[str, np.ndarray] = {}
        for key, arr in data.items():
            field = self._BUFFER_TO_FIELD.get(key)
            if field is None:  # e.g. "ind"
                continue
            values = arr[i][ind]
            if field in self._INT_FIELDS:
                values = values.astype(np.int32)
            kwargs[field] = values
        return GenParticleCollection(name=name, **kwargs)

    # ------------------------------------------------------------------ #
    # Internal sampling (renamed from generate_batch)                     #
    # ------------------------------------------------------------------ #

    def _sample_batch(
        self,
        data_dict: TensorDict,
        event_callback: Callable[[], None] | None = None,
        particle_callback: Callable[[], None] | None = None,
        impact_callback: Callable[[], None] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Run the neural sampling pipeline for one batch.

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]
            (truth_data_dict, pflow_data_dict, metadata_dict)
        """
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
        with nn.attention.sdpa_kernel([
            nn.attention.SDPBackend.CUDNN_ATTENTION,
            nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            nn.attention.SDPBackend.MATH,
        ]):
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

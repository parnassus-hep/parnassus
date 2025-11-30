"""Neural network-based event generator implementation."""

from collections.abc import Callable
from functools import partial
from typing import Self, final

import numpy as np
import torch

from parnassus.configs.accessors import ParticleAccessor
from parnassus.configs.generators import NeuralGeneratorConfig
from parnassus.nn import ModelWrapper
from parnassus.utils import Unscaler
from parnassus.utils.typing import TensorDict

# NN-specific accessor definitions
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


@final
class NeuralEventGenerator:
    """Neural network-based event generator implementing EventGenerator protocol."""

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

        self.fs_npart_pos = config.event_model_config.variables_config.fs_vars.index("npflow")
        self.fs_ht_pos = config.event_model_config.variables_config.fs_vars.index("pflow_ht")

        self.ht_shift = self.var_transform_dict["ht"].shift
        self.ht_scale = self.var_transform_dict["ht"].scale
        self.min_ht_scaled = -self.ht_shift / self.ht_scale

        self.unscaler = Unscaler(
            transform_dict=self.var_transform_dict,
            ctxt_vars=config.particle_model_config.variables_config.ctxt_vars,
            fs_vars=config.particle_model_config.variables_config.fs_vars,
            ctxt_global_vars=config.particle_model_config.variables_config.ctxt_global_vars,
        )

    # EventGenerator protocol properties
    @property
    def has_impact_model(self) -> bool:
        return self.impact_model is not None

    @property
    def max_particles(self) -> int:
        return self.config.max_particles

    @property
    def truth_output_vars(self) -> list[str]:
        return self.config.truth_output_vars

    @property
    def pflow_output_vars(self) -> list[str]:
        return self.config.pflow_output_vars

    @property
    def event_sampler_steps(self) -> int | None:
        return self.event_model.sampler.n_steps

    @property
    def particle_sampler_steps(self) -> int | None:
        return self.particle_model.sampler.n_steps

    @property
    def impact_sampler_steps(self) -> int | None:
        return self.impact_model.sampler.n_steps if self.impact_model else None

    def get_accessors(self) -> list:
        """Return list of accessor constructors for neural network output.

        Returns
        -------
        list[type[Accessor]]
            List of accessor constructors for the generated output.
        """
        accessors = PARTICLE_ACCESSORS.copy()
        if self.has_impact_model:
            accessors.extend(IMPACT_ACCESSORS)
        accessors.extend(LEPTON_ACCESSORS)
        return accessors

    def to(self, device: torch.device) -> Self:
        self.event_model.to(device)
        self.particle_model.to(device)
        if self.impact_model is not None:
            self.impact_model.to(device)
        self.device = device
        return self

    def generate_batch(
        self,
        data_dict: TensorDict,
        event_callback: Callable[[], None] | None = None,
        particle_callback: Callable[[], None] | None = None,
        impact_callback: Callable[[], None] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Generate truth and pflow particles for a batch using neural networks.

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

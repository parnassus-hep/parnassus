from collections.abc import Callable
from typing import final

from torch import Size, Tensor, nn
from torch.export import load

from parnassus.configs.generators.model import ModelConfig
from parnassus.nn.sampler import EulerSampler


@final
class ModelWrapper(nn.Module):
    """A wrapper class for a neural network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.num_fs_vars = len(config.fs_vars)
        if "pflow_phi" in config.fs_vars:
            self.num_fs_vars += 1  # phi expands to sin/cos (2 components), net +1 from original
        if "pflow_class" in config.fs_vars:
            # Add 4 for Class variables
            self.num_fs_vars += 4
        self.net: nn.Module = load(f=config.file_path).module()

        self.sampler = EulerSampler(
            n_steps=config.sampler_config.num_steps,
            reverse_time=config.sampler_config.reverse_time,
        )

    def forward(
        self,
        fs_data: Tensor,
        timestep: Tensor,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
        pf_ctxt_data: Tensor | None = None,
    ) -> Tensor:
        if pf_ctxt_data is None:
            return self.net(fs_data, timestep, mask, ctxt_data, ctxt_global_data)
        return self.net(fs_data, timestep, mask, ctxt_data, ctxt_global_data, pf_ctxt_data)

    def sample(
        self,
        shape: tuple[int, ...] | Size,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
        pf_ctxt_data: Tensor | None = None,
        callback: Callable[[], None] | None = None,
        to_cpu: bool = False,
    ) -> Tensor:
        return self.sampler.sample(
            self,
            (*shape, self.num_fs_vars),
            mask=mask,
            ctxt_data=ctxt_data,
            ctxt_global_data=ctxt_global_data,
            pf_ctxt_data=pf_ctxt_data,
            callback=callback,
            to_cpu=to_cpu,
        )

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final, override

import torch
from torch import Size, Tensor, nn
from torch.distributions import Normal

default_device = torch.device("cpu")


class Sampler(ABC):
    """Base class for samplers."""

    def __init__(
        self,
        n_steps: int,
        zero_init_padded: bool = True,
        random_seed: int | None = None,
        reverse_time=False,
    ):
        """Base class for sampler.
        Args:
        n_steps: Number of steps to take
        zero_init_padded: Whether to zero out the padded values
        random_seed: Random seed for reproducibility.
        reverse_time: Whether to reverse the time steps.
        """
        self.n_steps: int = n_steps
        self.zero_init_padded: bool = zero_init_padded
        self.reverse_time = reverse_time

        t_steps = torch.linspace(0, 1, n_steps)
        self.t_steps: Tensor = torch.cat([t_steps, torch.ones_like(t_steps)[:1]])

        if reverse_time:
            self.t_steps = torch.flip(self.t_steps, [0])

        if random_seed is not None:
            _ = torch.manual_seed(random_seed)
        self.random_seed: int | None = random_seed

        self.distribution: Normal = Normal(0, 1)

    def reset(self, random_seed: int | None = None):
        if random_seed is None:
            assert self.random_seed is not None, "Random seed not provided"
            random_seed = self.random_seed
        _ = torch.manual_seed(random_seed)

    def _init_fastsim(
        self,
        shape: tuple[int, ...] | Size,
        mask: Tensor | None = None,
        device: torch.device = default_device,
    ):
        fastsim = self.distribution.sample(shape).to(device)
        if mask is not None and self.zero_init_padded and mask.shape[-1] == 2:
            fastsim[~mask[..., 1]] = 0
        return fastsim

    @torch.no_grad()
    def _transfer(self, x_curr: Tensor, d_curr: Tensor, dt: Tensor) -> Tensor:
        """Do the integration step.

        Args:
            x_curr: Current state
            d_curr: Current derivative
            dt: Time step

        Returns
        -------
            x_next: Next state
        """
        return x_curr + d_curr * dt

    @torch.no_grad()
    def _step(
        self,
        model: nn.Module,
        fastsim: Tensor,
        timestep: Tensor,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
        dt: Tensor,
        pf_ctxt_data: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Do a single (euler) step of the model.

        Returns
        -------
            fastsim_next: The next state of the fastsim
            deriv: Model output at the current timestep (dx/dt)

        """
        deriv = model.forward(
            fastsim,
            timestep=timestep.expand(fastsim.shape[0], 1),
            mask=mask,
            ctxt_data=ctxt_data,
            ctxt_global_data=ctxt_global_data,
            pf_ctxt_data=pf_ctxt_data,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt)
        return fastsim_next, deriv

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        pflow_shape: tuple[int, ...] | Size,
        *,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
        pf_ctxt_data: Tensor | None = None,
        init_fastsim: Tensor | None = None,
        verbose: bool = True,
        callback: Callable[[], None] | None = None,
    ) -> Tensor:
        pass


@final
class EulerSampler(Sampler):
    """Euler sampler implementation."""

    @override
    def sample(
        self,
        model: nn.Module,
        pflow_shape: tuple[int, ...] | Size,
        *,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
        pf_ctxt_data: Tensor | None = None,
        init_fastsim: Tensor | None = None,
        verbose: bool = True,
        callback: Callable[[], None] | None = None,
        to_cpu: bool = False,
    ) -> Tensor:
        device = ctxt_data.device
        if init_fastsim is None:
            fastsim = self._init_fastsim(pflow_shape, mask, device)
        else:
            fastsim = init_fastsim
        t_steps = self.t_steps.to(device)
        for t_cur, t_next in itertools.pairwise(t_steps):
            fastsim, _ = self._step(
                model,
                fastsim,
                timestep=t_cur,
                mask=mask,
                dt=t_next - t_cur,
                ctxt_data=ctxt_data,
                ctxt_global_data=ctxt_global_data,
                pf_ctxt_data=pf_ctxt_data,
            )
            if verbose and callback is not None:
                callback()
        if to_cpu:
            return fastsim.cpu()
        return fastsim

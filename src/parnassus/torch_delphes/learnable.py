"""Learnable (Adam-optimizable) parameter modules for TorchDelphes.

This module provides ``nn.Module`` wrappers around the resolution, scale,
efficiency, and energy-fraction constants that are otherwise hardcoded in the
default Delphes-CMS / Delphes-ATLAS detector cards. Each wrapper exposes its
parameters as ``nn.Parameter``s so that they participate in
``model.parameters()`` and can be optimized with standard PyTorch optimizers
(Adam, SGD, etc.).

Design notes
------------
- **Defaults are preserved.** Every learnable module is initialized to the
  same numerical values used by the existing static formulas in
  ``MomentumSmearing.py`` and ``SimpleCalorimeter.py``. With no parameter
  updates, the learnable card produces output statistically identical to the
  default card (modulo Gumbel-ST noise on the efficiency mask).
- **Positivity / boundedness via reparameterization.** Strictly positive
  quantities (resolution coefficients, decay rates) are stored as raw
  parameters and exposed via ``softplus`` so that optimizer steps cannot
  drive them negative. Per-region multiplicative scales are stored as raw
  parameters and exposed as ``1 + 0.3 * tanh(raw)``, bounding them to
  [0.7, 1.3] and starting at exactly 1.0.
- **Geometry stays fixed.** Region boundaries (eta cuts at 0.5/1.5/2.5,
  pt cuts at 0.1/1/100/1000) are stored as buffers, not parameters. Per
  the design discussion, geometric quantities like detector dimensions and
  region edges are not optimized.
- **Efficiency uses a Gumbel-sigmoid straight-through mask.** The forward
  pass produces a hard 0/1 selection (so downstream physics is unchanged in
  expectation), while gradients flow through the soft sigmoid in the backward
  pass. Masked particles have their (PT, PX, PY, PZ, E) multiplied by the
  binary mask; their rows remain in the tensor so that the calorimeter still
  sees the original calorimeter deposit (efficiency only affects tracking,
  not calorimetry).

The classes here are designed to plug into the existing ``MomentumSmearing``
and ``SimpleCalorimeter`` modules via the ``resolution_formula`` callable
hook (and the new ``scale_fn`` / ``learnable_fractions`` hooks added in this
PR). See ``defaults/CMSDefault.py`` for how they are wired into a complete
detector card.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from parnassus.data.particle_io import ColumnMap

from . import pdg_filters

# Columns that carry 4-momentum information. The Gumbel-ST efficiency mask is
# multiplied into these (and only these) so that masked particles become
# "invisible" to downstream tracking-side code while leaving the calorimeter
# input (which uses the unfiltered particles tensor) untouched.
_MOMENTUM_COLS: tuple[int, ...] = (
    int(ColumnMap.PT),
    int(ColumnMap.PX),
    int(ColumnMap.PY),
    int(ColumnMap.PZ),
    int(ColumnMap.E),
)


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus, used to initialize raw parameters so that
    ``softplus(raw) == x``.

    softplus(y) = log(1 + exp(y))   =>   y = log(exp(x) - 1) = log(expm1(x))

    Returns
    -------
    torch.Tensor
        Raw parameter values ``raw`` such that ``softplus(raw) == x``.
    """
    # Clamp to avoid log(0) for very small x.
    return torch.log(torch.expm1(x.clamp(min=1e-12)))


def _safe_logit(p: float) -> float:
    """Compute logit(p) with clipping away from 0 and 1 to stay finite.

    Returns
    -------
    float
        ``log(p / (1 - p))`` after clamping ``p`` into ``[1e-6, 1-1e-6]``.
    """
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


# =============================================================================
# Track-side: momentum resolution, momentum scale
# =============================================================================


class LearnableMomentumResolution(nn.Module):
    """Per-species learnable momentum resolution **and** per-region scale.

    Implements the Delphes resolution formula
    ``sigma_rel(pt, eta) = sqrt(a_r^2 + (pt * b_r)^2)`` where ``r`` indexes
    the eta region (barrel / mid / endcap), and a multiplicative momentum
    scale ``s_r(eta)`` per region. Both are learnable.

    The pt cut at 0.1 GeV and the eta region boundaries are fixed buffers
    matching the existing C++ Delphes / TorchDelphes static formulas.

    This class implements two interfaces:

    - ``forward(pt, eta_outer) -> sigma_rel``: usable as the
      ``resolution_formula`` callable in ``MomentumSmearing``.
    - ``scale(eta_outer) -> scale``: usable as the new ``scale_fn`` callable
      in ``MomentumSmearing``.

    Parameters
    ----------
    a_init, b_init : list[float]
        Initial values for the constant and pt-dependent resolution terms,
        one per eta region. Must have the same length as ``boundaries``.
        (Each region has a 2-parameter resolution.)
    boundaries : tuple[float, ...]
        Upper edges of the absolute-eta regions. Defaults to
        ``(0.5, 1.5, 2.5)`` to match the default CMS formulas.
    """

    def __init__(
        self,
        a_init: list[float],
        b_init: list[float],
        boundaries: tuple[float, ...] = (0.5, 1.5, 2.5),
    ) -> None:
        super().__init__()
        if not (len(a_init) == len(b_init) == len(boundaries)):
            raise ValueError(
                f"a_init ({len(a_init)}), b_init ({len(b_init)}), and "
                f"boundaries ({len(boundaries)}) must all be the same length"
            )
        a = torch.tensor(a_init, dtype=torch.float64)
        b = torch.tensor(b_init, dtype=torch.float64)
        # Store as raw parameters; expose via softplus to guarantee positivity.
        self.a_raw = nn.Parameter(_softplus_inv(a))
        self.b_raw = nn.Parameter(_softplus_inv(b))
        # Per-region multiplicative scale: 1 + 0.3 * tanh(raw), default 1.0.
        self.scale_raw = nn.Parameter(torch.zeros(len(a_init), dtype=torch.float64))
        # Region boundaries are FIXED (treated as detector geometry).
        self.boundaries: nn.Buffer
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float64))
        # PT lower cut for resolution to be defined (matches default formulas).
        self.pt_min: nn.Buffer
        self.register_buffer("pt_min", torch.tensor(0.1, dtype=torch.float64))

    # ----- parameter accessors -----
    def get_a(self) -> torch.Tensor:
        return F.softplus(self.a_raw)

    def get_b(self) -> torch.Tensor:
        return F.softplus(self.b_raw)

    def get_scale(self) -> torch.Tensor:
        """Per-region scale, bounded to ``[0.7, 1.3]``, default 1.0.

        Returns
        -------
        torch.Tensor
            Length ``len(boundaries)`` tensor of scale values.
        """
        return 1.0 + 0.3 * torch.tanh(self.scale_raw)

    # ----- region indexing -----
    def _region_index(self, eta_outer: torch.Tensor) -> torch.Tensor:
        """Return the eta-region index for each particle.

        Indices in ``[0, len(boundaries) - 1]`` correspond to defined regions
        (``|eta| <= boundaries[idx]``). Index ``len(boundaries)`` means the
        particle is outside all defined regions and should not receive a
        resolution / scale (matches C++ Delphes "no formula match" -> 0).

        Returns
        -------
        torch.Tensor
            Integer tensor of eta-region indices, same shape as
            ``eta_outer``.
        """
        abs_eta = eta_outer.abs().to(self.boundaries.dtype)
        return torch.searchsorted(self.boundaries, abs_eta)

    # ----- callable interfaces -----
    def forward(self, pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """Relative momentum resolution. Drop-in for ``resolution_formula``.

        Returns
        -------
        torch.Tensor
            Per-particle relative resolution, same shape as ``pt``.
            Zero for particles outside the defined pt/eta ranges.
        """
        n = self.boundaries.numel()
        idx = self._region_index(eta_outer)
        in_range = idx < n
        idx_clamped = idx.clamp(max=n - 1)
        a_per = self.get_a()[idx_clamped]
        b_per = self.get_b()[idx_clamped]
        # Add a tiny constant inside sqrt to keep its gradient finite at 0
        # (sqrt has an infinite derivative at zero, which would corrupt
        # backward through the masked branch of the torch.where below).
        res = torch.sqrt(a_per * a_per + (pt * b_per) ** 2 + 1e-30)
        # Match C++ Delphes: undefined below pt_min and outside eta range.
        in_pt = pt > self.pt_min
        valid = in_range & in_pt
        return torch.where(valid, res, torch.zeros_like(res))

    def scale(self, eta_outer: torch.Tensor) -> torch.Tensor:
        """Per-particle momentum scale (1.0 outside the defined eta range).

        Returns
        -------
        torch.Tensor
            Per-particle scale factor, same shape as ``eta_outer``.
        """
        n = self.boundaries.numel()
        idx = self._region_index(eta_outer)
        in_range = idx < n
        idx_clamped = idx.clamp(max=n - 1)
        s_per = self.get_scale()[idx_clamped]
        return torch.where(in_range, s_per, torch.ones_like(s_per))


# Default initial values for the three CMS species, taken verbatim from
# ``MomentumSmearing._{species}_cms_momentum_resolution``.
CMS_TRACK_RES_DEFAULTS: dict[str, dict[str, list[float]]] = {
    "charged_hadron": {
        "a_init": [0.06, 0.10, 0.25],
        "b_init": [1.3e-3, 1.7e-3, 3.1e-3],
    },
    "electron": {
        "a_init": [0.03, 0.05, 0.15],
        "b_init": [1.3e-3, 1.7e-3, 3.1e-3],
    },
    "muon": {
        "a_init": [0.01, 0.02, 0.10],
        "b_init": [1.0e-3, 1.3e-3, 2.0e-3],
    },
}


def make_cms_track_resolution(species: str) -> LearnableMomentumResolution:
    """Build a learnable track-resolution module with default CMS values.

    Parameters
    ----------
    species : str
        One of ``{'charged_hadron', 'electron', 'muon'}``.

    Returns
    -------
    LearnableMomentumResolution
        A fresh module whose parameters are initialized to the static
        CMS defaults for ``species``.
    """
    if species not in CMS_TRACK_RES_DEFAULTS:
        raise ValueError(f"Unknown species: {species}")
    cfg = CMS_TRACK_RES_DEFAULTS[species]
    return LearnableMomentumResolution(a_init=cfg["a_init"], b_init=cfg["b_init"])


# =============================================================================
# Track-side: tracking efficiency with Gumbel-sigmoid straight-through
# =============================================================================


class _LearnableEfficiencyBase(nn.Module):
    """Common base for learnable tracking efficiencies.

    Subclasses implement ``compute_efficiency(pt, eta_outer)`` which returns a
    per-particle efficiency in ``[0, 1]``. The base class handles the
    Gumbel-sigmoid straight-through mask and applies it as a multiplier on the
    momentum columns of the input tensor.

    Parameters
    ----------
    pdg_filter_func : Callable[[torch.Tensor], torch.Tensor] | None
        Retained for parity with the legacy :class:`Efficiency` API. Not used
        internally because in the learnable card, particle splitting by
        species is already done by ``ParticlePropagator``.
    temperature : float
        Gumbel-sigmoid temperature ``tau``. Lower values (e.g. 0.1) give
        sharper Bernoulli-like behavior; higher values (e.g. 1.0) give
        smoother gradients but more sampling noise.
    """

    def __init__(
        self,
        pdg_filter_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.pdg_filter_func = pdg_filter_func
        self.temperature: float = float(temperature)

    # Subclasses must override.
    def compute_efficiency(self, pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ----- Gumbel-sigmoid sampling -----
    def _gumbel_sigmoid_st(self, eff: torch.Tensor) -> torch.Tensor:
        """Sample a hard 0/1 mask with straight-through gradient.

        Forward: ``mask`` is in ``{0, 1}``, drawn approximately as
        ``Bernoulli(eff)`` via the Gumbel-sigmoid (a.k.a. binary Concrete)
        trick. Backward: gradient flows as if the soft sigmoid were used.

        Returns
        -------
        torch.Tensor
            Per-particle mask, same shape as ``eff``. Hard 0/1 values in
            the forward pass, soft sigmoid gradient in backward.

        Notes
        -----
        - ``eff`` is clamped to ``(eps, 1 - eps)`` so the logit is finite.
        - Logistic noise ``L = log(U) - log(1-U)`` is the binary equivalent of
          Gumbel noise and is what makes ``sigmoid((logits + L) / tau)``
          converge to ``Bernoulli(eff)`` as ``tau -> 0``.
        """
        eps = 1e-6
        eff_c = eff.clamp(eps, 1.0 - eps)
        logits = torch.log(eff_c) - torch.log1p(-eff_c)
        u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        y_soft = torch.sigmoid((logits + logistic_noise) / self.temperature)
        y_hard = (y_soft > 0.5).to(y_soft.dtype)
        # Straight-through: hard value forward, soft gradient backward.
        return y_hard.detach() + (y_soft - y_soft.detach())

    # ----- forward: apply mask to momentum columns -----
    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        if particles.shape[0] == 0:
            return particles
        pt = particles[:, ColumnMap.PT]
        eta_outer = particles[:, ColumnMap.ETA_OUTER]
        eff = self.compute_efficiency(pt, eta_outer)
        mask = self._gumbel_sigmoid_st(eff)  # shape (N,)

        # Build a per-column multiplier: mask[i] for momentum columns, 1
        # everywhere else. Using torch.where keeps the operation fully
        # functional (no in-place writes) so autograd is straightforward.
        n_features = particles.shape[1]
        col_is_mom = torch.zeros(n_features, dtype=torch.bool, device=particles.device)
        for c in _MOMENTUM_COLS:
            col_is_mom[c] = True
        multiplier = torch.where(
            col_is_mom.unsqueeze(0),  # (1, F)
            mask.unsqueeze(1).to(particles.dtype),  # (N, 1)
            torch.ones((), dtype=particles.dtype, device=particles.device),
        )
        return particles * multiplier


class CMSChargedHadronLearnableEfficiency(_LearnableEfficiencyBase):
    """Learnable CMS charged-hadron tracking efficiency.

    Four piecewise-constant efficiency parameters covering the four
    (pt, |eta|) regions defined in
    :meth:`Efficiency._charged_hadron_cms_efficiency`. Each is stored as a
    logit so the sigmoid keeps it in ``(0, 1)``.
    """

    # (lowpt-barrel, highpt-barrel, lowpt-endcap, highpt-endcap)
    _DEFAULTS: tuple[float, ...] = (0.70, 0.95, 0.60, 0.85)

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__(pdg_filter_func=pdg_filters.charged_hadron_filter, temperature=temperature)
        logits = torch.tensor([_safe_logit(p) for p in self._DEFAULTS], dtype=torch.float64)
        self.eff_logits = nn.Parameter(logits)

    def get_efficiencies(self) -> torch.Tensor:
        return torch.sigmoid(self.eff_logits)

    def compute_efficiency(self, pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        abs_eta = eta_outer.abs()
        effs = self.get_efficiencies()

        in_pt_low = (pt > 0.1) & (pt <= 1.0)
        in_pt_high = pt > 1.0
        in_barrel = abs_eta <= 1.5
        in_endcap = (abs_eta > 1.5) & (abs_eta <= 2.5)

        eff = torch.zeros_like(pt)
        eff = torch.where(in_barrel & in_pt_low, effs[0].to(pt.dtype), eff)
        eff = torch.where(in_barrel & in_pt_high, effs[1].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_low, effs[2].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_high, effs[3].to(pt.dtype), eff)
        return eff


class CMSElectronLearnableEfficiency(_LearnableEfficiencyBase):
    """Learnable CMS electron tracking efficiency: 6 piecewise constants
    (3 pt bins x 2 eta bins).
    """

    # (low/mid/high)pt-barrel, then (low/mid/high)pt-endcap
    _DEFAULTS: tuple[float, ...] = (0.73, 0.95, 0.99, 0.50, 0.83, 0.90)

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__(pdg_filter_func=pdg_filters.electron_filter, temperature=temperature)
        logits = torch.tensor([_safe_logit(p) for p in self._DEFAULTS], dtype=torch.float64)
        self.eff_logits = nn.Parameter(logits)

    def get_efficiencies(self) -> torch.Tensor:
        return torch.sigmoid(self.eff_logits)

    def compute_efficiency(self, pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        abs_eta = eta_outer.abs()
        effs = self.get_efficiencies()

        in_pt_low = (pt > 0.1) & (pt <= 1.0)
        in_pt_mid = (pt > 1.0) & (pt <= 1.0e2)
        in_pt_high = pt > 1.0e2
        in_barrel = abs_eta <= 1.5
        in_endcap = (abs_eta > 1.5) & (abs_eta <= 2.5)

        eff = torch.zeros_like(pt)
        eff = torch.where(in_barrel & in_pt_low, effs[0].to(pt.dtype), eff)
        eff = torch.where(in_barrel & in_pt_mid, effs[1].to(pt.dtype), eff)
        eff = torch.where(in_barrel & in_pt_high, effs[2].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_low, effs[3].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_mid, effs[4].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_high, effs[5].to(pt.dtype), eff)
        return eff


class CMSMuonLearnableEfficiency(_LearnableEfficiencyBase):
    """Learnable CMS muon tracking efficiency: 6 piecewise constants plus
    2 high-pt exponential decay rates.

    The two high-pt regions (``pt > 1000`` GeV) follow
    ``eff(pt) = coeff * exp(0.5 - rate * pt)`` matching the static formula in
    :meth:`Efficiency._muon_cms_efficiency`.
    """

    # (low/mid/high-coeff)-barrel, then (low/mid/high-coeff)-endcap
    _DEFAULTS: tuple[float, ...] = (0.75, 0.99, 0.99, 0.70, 0.98, 0.98)
    _RATE_DEFAULTS: tuple[float, ...] = (5.0e-4, 5.0e-4)

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__(pdg_filter_func=pdg_filters.muon_filter, temperature=temperature)
        logits = torch.tensor([_safe_logit(p) for p in self._DEFAULTS], dtype=torch.float64)
        self.eff_logits = nn.Parameter(logits)
        rates = torch.tensor(self._RATE_DEFAULTS, dtype=torch.float64)
        # softplus reparameterization to keep rate > 0
        self.rate_raw = nn.Parameter(_softplus_inv(rates))

    def get_efficiencies(self) -> torch.Tensor:
        return torch.sigmoid(self.eff_logits)

    def get_rates(self) -> torch.Tensor:
        return F.softplus(self.rate_raw)

    def compute_efficiency(self, pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        abs_eta = eta_outer.abs()
        effs = self.get_efficiencies()
        rates = self.get_rates()

        in_pt_low = (pt > 0.1) & (pt <= 1.0)
        in_pt_mid = (pt > 1.0) & (pt <= 1.0e3)
        in_pt_high = pt > 1.0e3
        in_barrel = abs_eta <= 1.5
        in_endcap = (abs_eta > 1.5) & (abs_eta <= 2.5)

        # High-pt exponential roll-off.
        barrel_high = effs[2].to(pt.dtype) * torch.exp(0.5 - rates[0].to(pt.dtype) * pt)
        endcap_high = effs[5].to(pt.dtype) * torch.exp(0.5 - rates[1].to(pt.dtype) * pt)

        eff = torch.zeros_like(pt)
        eff = torch.where(in_barrel & in_pt_low, effs[0].to(pt.dtype), eff)
        eff = torch.where(in_barrel & in_pt_mid, effs[1].to(pt.dtype), eff)
        eff = torch.where(in_barrel & in_pt_high, barrel_high, eff)
        eff = torch.where(in_endcap & in_pt_low, effs[3].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_mid, effs[4].to(pt.dtype), eff)
        eff = torch.where(in_endcap & in_pt_high, endcap_high, eff)
        # Clamp because exp() can exceed 1 at the boundary pt = 1000 GeV
        # (where 0.5 - 5e-4 * 1000 = 0 ⇒ coeff * 1.0 = coeff, fine), but for
        # smaller rate values from the optimizer it could exceed 1.
        return eff.clamp(0.0, 1.0)


# =============================================================================
# Calorimeter-side: ECal/HCal resolution and per-region energy scale (CMS)
# =============================================================================


class LearnableEcalCMSResolution(nn.Module):
    """Learnable CMS ECal resolution (9 parameters).

    Reproduces the formula in
    :meth:`SimpleCalorimeter._ecal_cms_resolution`:

    .. code-block::

        |eta| <= 1.5 :  (a_b + b_b * eta^2)        * sqrt(E^2 c_E^2 + E c_S^2 + c_N^2)
        |eta| <= 2.5 :  (a_e + b_e * (|eta|-2)^2)  * sqrt(E^2 c_E^2 + E c_S^2 + c_N^2)
        else         :  sqrt(E^2 f_E^2 + E f_S^2)

    All nine coefficients are stored as raw parameters and exposed via
    softplus to enforce positivity.
    """

    def __init__(self) -> None:
        super().__init__()

        def _p(x: float) -> nn.Parameter:
            return nn.Parameter(_softplus_inv(torch.tensor(x, dtype=torch.float64)))

        # Common stochastic + noise terms (shared by barrel and endcap).
        self.common_c_E = _p(0.008)
        self.common_c_S = _p(0.11)
        self.common_c_N = _p(0.40)
        # Barrel pre-factor: barrel_a + barrel_b * eta^2  (default 1 + 0.64 eta^2)
        self.barrel_a = _p(1.0)
        self.barrel_b = _p(0.64)
        # Endcap pre-factor: endcap_a + endcap_b * (|eta| - 2)^2 (default 2.16 + 5.6*(...)^2)
        self.endcap_a = _p(2.16)
        self.endcap_b = _p(5.6)
        # Forward stochastic + noise terms.
        self.forward_c_E = _p(0.107)
        self.forward_c_S = _p(2.08)

    def forward(self, eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """Compute ECal sigma for each (eta, energy) tower center.

        Returns
        -------
        torch.Tensor
            Per-tower absolute sigma, same shape as ``energy``.
        """
        abs_eta = eta.abs()

        # Tiny constant inside sqrt() to keep gradients finite for energy=0
        # towers. ``torch.where`` masks the forward value but autograd still
        # evaluates the gradient on every element.
        eps = 1e-30

        c_e = F.softplus(self.common_c_E)
        c_s = F.softplus(self.common_c_S)
        c_n = F.softplus(self.common_c_N)
        common = torch.sqrt(energy * energy * c_e * c_e + energy * c_s * c_s + c_n * c_n + eps)

        b_a = F.softplus(self.barrel_a)
        b_b = F.softplus(self.barrel_b)
        barrel_factor = b_a + b_b * eta * eta
        barrel_sigma = barrel_factor * common

        e_a = F.softplus(self.endcap_a)
        e_b = F.softplus(self.endcap_b)
        endcap_factor = e_a + e_b * (abs_eta - 2.0) ** 2
        endcap_sigma = endcap_factor * common

        f_e = F.softplus(self.forward_c_E)
        f_s = F.softplus(self.forward_c_S)
        forward_sigma = torch.sqrt(energy * energy * f_e * f_e + energy * f_s * f_s + eps)

        return torch.where(
            abs_eta <= 1.5,
            barrel_sigma,
            torch.where(abs_eta <= 2.5, endcap_sigma, forward_sigma),
        )


class LearnableHcalCMSResolution(nn.Module):
    """Learnable CMS HCal resolution (4 parameters).

    Reproduces the formula in
    :meth:`SimpleCalorimeter._hcal_cms_resolution`:

    .. code-block::

        |eta| <= 3.0 :  sqrt(E^2 c_E^2 + E c_S^2)            (central)
        else         :  sqrt(E^2 f_E^2 + E f_S^2)            (forward)
    """

    def __init__(self) -> None:
        super().__init__()

        def _p(x: float) -> nn.Parameter:
            return nn.Parameter(_softplus_inv(torch.tensor(x, dtype=torch.float64)))

        self.central_c_E = _p(0.050)
        self.central_c_S = _p(1.50)
        self.forward_c_E = _p(0.130)
        self.forward_c_S = _p(2.70)

    def forward(self, eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """Compute HCal sigma for each (eta, energy) tower center.

        Returns
        -------
        torch.Tensor
            Per-tower absolute sigma, same shape as ``energy``.
        """
        abs_eta = eta.abs()
        eps = 1e-30  # keep sqrt gradients finite at energy=0
        c_e = F.softplus(self.central_c_E)
        c_s = F.softplus(self.central_c_S)
        central_sigma = torch.sqrt(energy * energy * c_e * c_e + energy * c_s * c_s + eps)

        f_e = F.softplus(self.forward_c_E)
        f_s = F.softplus(self.forward_c_S)
        forward_sigma = torch.sqrt(energy * energy * f_e * f_e + energy * f_s * f_s + eps)

        sigma = torch.where(abs_eta <= 3.0, central_sigma, forward_sigma)
        return sigma.to(torch.float64)


class LearnableCaloScale(nn.Module):
    """Per-region multiplicative energy scale for a calorimeter.

    Used by the new ``scale_fn`` hook in :class:`SimpleCalorimeter` to scale
    the tower energy passed into the log-normal smearing. Bounded to
    ``[0.7, 1.3]`` via ``1 + 0.3 * tanh(raw)``; defaults to 1.0 in every
    region.

    Parameters
    ----------
    boundaries : tuple[float, ...]
        Upper edges of the absolute-eta regions. The number of regions is
        ``len(boundaries) + 1`` (the last region is "everything above
        ``boundaries[-1]``").
    """

    def __init__(self, boundaries: tuple[float, ...]) -> None:
        super().__init__()
        n_regions = len(boundaries) + 1
        self.scale_raw = nn.Parameter(torch.zeros(n_regions, dtype=torch.float64))
        self.boundaries: nn.Buffer
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float64))

    def get_scales(self) -> torch.Tensor:
        return 1.0 + 0.3 * torch.tanh(self.scale_raw)

    def forward(self, eta: torch.Tensor) -> torch.Tensor:
        abs_eta = eta.abs().to(self.boundaries.dtype)
        idx = torch.searchsorted(self.boundaries, abs_eta)
        # idx is in [0, n_regions); no clamp needed because searchsorted
        # against an array of length n_regions-1 gives indices in [0, n_regions-1].
        scales = self.get_scales()
        return scales[idx].to(eta.dtype)


def make_cms_ecal_scale() -> LearnableCaloScale:
    """Build the default CMS ECal scale: 3 regions (barrel/endcap/forward).

    Returns
    -------
    LearnableCaloScale
        Fresh scale module with boundaries ``(1.5, 2.5)`` and all
        scale factors initialized to 1.0.
    """
    return LearnableCaloScale(boundaries=(1.5, 2.5))


def make_cms_hcal_scale() -> LearnableCaloScale:
    """Build the default CMS HCal scale: 2 regions (central/forward).

    Returns
    -------
    LearnableCaloScale
        Fresh scale module with boundary ``(3.0,)`` and both scale
        factors initialized to 1.0.
    """
    return LearnableCaloScale(boundaries=(3.0,))


# =============================================================================
# Hadronic energy fractions (the "pion / kaon leaves different amounts" knob)
# =============================================================================


class LearnableHadronFractions(nn.Module):
    """Shared learnable PDG → ECal/HCal energy fraction module.

    Three particle classes are tunable, each parameterized by a single ECal
    fraction logit (HCal fraction = 1 - ECal fraction):

    - **Charged hadron default** (PDG=0 in the Delphes fraction map):
      covers charged pions, kaons, protons, etc. Default 0.0/1.0.
    - **K-short** (PDG=310): default 0.3/0.7.
    - **Lambda** (PDG=3122): default 0.3/0.7.

    All other particle classes (electrons, photons, muons, neutrinos, etc.)
    have physics-fixed fractions and are *not* parameterized here.

    A single instance of this module is shared between the ECal and HCal
    submodules of the detector card so that the ECal and HCal fractions stay
    consistent (sum to 1).
    """

    # PDG IDs of the tunable particle classes.
    K0S_PDG: int = 310
    LAMBDA_PDG: int = 3122

    def __init__(self) -> None:
        super().__init__()
        # Defaults match the static dicts in CMSDefault._setup_ECal/_setup_HCal.
        self.chad_logit = nn.Parameter(torch.tensor(_safe_logit(0.0), dtype=torch.float64))
        self.k0s_logit = nn.Parameter(torch.tensor(_safe_logit(0.3), dtype=torch.float64))
        self.lambda_logit = nn.Parameter(torch.tensor(_safe_logit(0.3), dtype=torch.float64))

    # ----- per-class fraction accessors (ECal side; HCal = 1 - ECal) -----
    def chad_ecal_frac(self) -> torch.Tensor:
        return torch.sigmoid(self.chad_logit)

    def k0s_ecal_frac(self) -> torch.Tensor:
        return torch.sigmoid(self.k0s_logit)

    def lambda_ecal_frac(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_logit)

    def fraction_for(self, particle_class: str, is_ecal: bool) -> torch.Tensor:
        """Return the (scalar) fraction for one of the three tunable classes.

        Returns
        -------
        torch.Tensor
            Scalar fraction for the requested ``particle_class`` in the
            requested calorimeter.
        """
        ecal = {
            "chad": self.chad_ecal_frac,
            "k0s": self.k0s_ecal_frac,
            "lambda": self.lambda_ecal_frac,
        }[particle_class]()
        return ecal if is_ecal else (1.0 - ecal)

    # ----- override: rewrite a base fraction tensor for a batch of particles -----
    def apply_overrides(
        self,
        base_fractions: torch.Tensor,
        abs_pids: torch.Tensor,
        uses_default: torch.Tensor,
        is_ecal: bool,
    ) -> torch.Tensor:
        """Replace the entries in ``base_fractions`` that this module owns.

        Parameters
        ----------
        base_fractions : torch.Tensor
            Per-particle fractions from the static LUT lookup, shape ``(N,)``.
        abs_pids : torch.Tensor
            Absolute PDG IDs (long), shape ``(N,)``. Used to identify K0S and
            Lambda particles.
        uses_default : torch.Tensor
            Boolean mask of shape ``(N,)`` indicating which particles fall back
            to the "default" (PDG=0) fraction. These get overridden by
            ``chad_*_frac()``.
        is_ecal : bool
            Whether the calling :class:`SimpleCalorimeter` is the ECal or
            the HCal.

        Returns
        -------
        torch.Tensor
            ``base_fractions`` with the three tunable classes replaced by the
            current learnable values. Differentiable wrt the parameters of
            this module.
        """
        chad_frac = self.fraction_for("chad", is_ecal).to(base_fractions.dtype)
        k0s_frac = self.fraction_for("k0s", is_ecal).to(base_fractions.dtype)
        lam_frac = self.fraction_for("lambda", is_ecal).to(base_fractions.dtype)

        is_k0s = abs_pids == self.K0S_PDG
        is_lambda = abs_pids == self.LAMBDA_PDG

        out = torch.where(uses_default, chad_frac.expand_as(base_fractions), base_fractions)
        out = torch.where(is_k0s, k0s_frac.expand_as(out), out)
        out = torch.where(is_lambda, lam_frac.expand_as(out), out)
        return out

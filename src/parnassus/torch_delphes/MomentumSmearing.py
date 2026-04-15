"""PyTorch implementation of Delphes MomentumSmearing module.

Applies momentum resolution smearing to charged particle tracks using a
log-normal distribution. The resolution depends on particle kinematics
(pT, η) and detector region.

The log-normal distribution ensures smeared pT values remain positive,
which is physically required. The smearing preserves the particle direction
(η, φ) while only modifying the magnitude of the transverse momentum.

Reference:
    C++ Delphes: modules/MomentumSmearing.cc
    CMS tracking resolution: arXiv:1405.6569
"""

from collections.abc import Callable

import torch
from torch import nn

from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes.stochastic_utils import log_normal_sample


class MomentumSmearing(nn.Module):
    """PyTorch implementation of Delphes MomentumSmearing module.

    Applies momentum resolution smearing based on particle kinematics. The
    resolution formula takes (pT, η_outer) and returns a relative resolution
    (e.g., 0.06 = 6% resolution). The pT is then smeared using a log-normal
    distribution with this resolution.

    The module uses **position-based η (EtaOuter)** for the resolution formula,
    matching C++ Delphes where detector geometry determines resolution.
    However, the **momentum-based η** is preserved (not smeared).

    After smearing pT:

    - Px, Py, Pz are recomputed from smeared pT and original η, φ
    - E is recomputed from the new momentum and original mass
    - The track resolution is stored in TRACK_RESOLUTION column for use
      by SimpleCalorimeter in energy flow computation

    Predefined resolution formulas:

    - **charged_hadron_cms**: CMS charged hadron momentum resolution
    - **electron_cms**: CMS electron momentum resolution
    - **muon_cms**: CMS muon momentum resolution

    Attributes
    ----------
    resolution_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that computes resolution from (pt, eta_outer)

    Examples
    --------
    >>> smear = MomentumSmearing(resolution_formula='charged_hadron_cms')
    >>> smeared_tracks = smear(tracks)
    """

    def __init__(
        self,
        resolution_formula: str
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "charged_hadron_cms",
        scale_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the MomentumSmearing module.

        Parameters
        ----------
        resolution_formula: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
            Either a string naming a predefined formula
            ('charged_hadron_cms', 'electron_cms', 'muon_cms') or a
            callable that takes (pt, eta_outer) tensors and returns
            relative resolution values (e.g., 0.06 for 6%).
            If an ``nn.Module`` is passed, it is registered as a submodule
            so that its parameters appear in ``self.parameters()``. This
            is the hook used for differentiable-tuning support.
        scale_fn: Callable[[torch.Tensor], torch.Tensor] | None, optional
            Optional per-particle multiplicative momentum scale. Called as
            ``scale_fn(eta_outer)`` and expected to return a tensor of the
            same shape as ``eta_outer``. The scale is applied to both the
            mean and the standard deviation passed into the log-normal
            sampler so that ``E[smeared_pt] = scale * pt`` and
            ``Var[smeared_pt] = (scale * resolution * pt)^2``. Default
            ``None`` means scale = 1 everywhere (i.e. no scale, matching
            the legacy behavior). If an ``nn.Module`` is passed, it is
            registered as a submodule so that its parameters are visible.
        """
        super().__init__()

        self.resolution_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        # Load resolution formula
        if resolution_formula == "charged_hadron_cms":
            self.resolution_func = self._charged_hadron_cms_momentum_resolution
        elif resolution_formula == "electron_cms":
            self.resolution_func = self._electron_cms_momentum_resolution
        elif resolution_formula == "muon_cms":
            self.resolution_func = self._muon_cms_momentum_resolution
        elif isinstance(resolution_formula, nn.Module):
            # Register as a submodule so its nn.Parameters are exposed.
            self.resolution_module = resolution_formula
            self.resolution_func = resolution_formula
        elif callable(resolution_formula):
            self.resolution_func = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")

        # Optional per-particle momentum scale callable. Registered as a
        # submodule when an nn.Module is provided so its parameters show up
        # in self.parameters().
        self.scale_fn: Callable[[torch.Tensor], torch.Tensor] | None
        if scale_fn is None:
            self.scale_fn = None
        elif isinstance(scale_fn, nn.Module):
            self.scale_module = scale_fn
            self.scale_fn = scale_fn
        elif callable(scale_fn):
            self.scale_fn = scale_fn
        else:
            raise ValueError(f"scale_fn must be callable or None, got {type(scale_fn)}")

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """Apply momentum smearing to particles.

        Computes the momentum resolution for each particle, then smears the pT
        using a log-normal distribution. All momentum components and energy
        are updated to be consistent with the smeared pT.

        Parameters
        ----------
        particles: torch.Tensor
            Tensor of shape (N, N_FEATURES) containing tracks.
            Required columns:

            - PT (col 7): Transverse momentum in GeV
            - ETA (col 8): Momentum-based pseudorapidity
            - PHI (col 9): Azimuthal angle
            - MASS (col 14): Particle mass in GeV
            - ETA_OUTER (col 15): Position-based pseudorapidity

        Returns
        -------
        particles: torch.Tensor
            Updated tensor of shape (N, N_FEATURES) with smeared momentum.
            Modified columns: PT, PX, PY, PZ, E, TRACK_RESOLUTION.
            Original eta and phi are preserved.
        """
        pt = particles[:, ColumnMap.PT]  # Column 7: PT (transverse momentum)
        eta_outer = particles[
            :, ColumnMap.ETA_OUTER
        ]  # Column 8: Eta (for resolution formula - typically position-based)
        mass = particles[:, ColumnMap.MASS]  # Column 14: Mass

        eta = particles[:, ColumnMap.ETA]  # atanh(pz/p) = asinh(pz/pt)
        phi = particles[:, ColumnMap.PHI]

        # Compute resolution for each particle using eta_outer
        # Resolution is relative (dimensionless, like 0.06 = 6%)
        resolution = self.resolution_func(pt, eta_outer)
        resolution = torch.clamp(resolution, max=1.0)

        # Optional per-region momentum scale (default 1.0). Applied to both
        # the mean and the standard deviation of the log-normal so that
        # the *relative* resolution is unchanged.
        if self.scale_fn is not None:
            scale = self.scale_fn(eta_outer)
            mean_pt = scale * pt
        else:
            mean_pt = pt

        # Apply smearing using log-normal distribution
        # C++ does: LogNormal(pt, res * pt) where res is relative resolution
        # So sigma = resolution * pt (absolute resolution in GeV)
        smeared_pt = log_normal_sample(mean_pt, resolution * mean_pt)

        # Compute updated PX, PY, PZ, E from the smeared PT.
        smeared_px = smeared_pt * torch.cos(phi)
        smeared_py = smeared_pt * torch.sin(phi)
        smeared_pz = smeared_pt * torch.sinh(eta)
        p_squared = smeared_px**2 + smeared_py**2 + smeared_pz**2
        smeared_e = torch.sqrt(p_squared + mass**2)

        # Write all updated columns in a *single* index_put. Multiple
        # sequential in-place writes would trip autograd's version counter
        # when the smeared values carry gradient (learnable mode), because
        # the second write would invalidate slices the first write needed
        # for backward.
        out = particles.clone()
        cols_to_replace = torch.tensor(
            [
                int(ColumnMap.PT),
                int(ColumnMap.PX),
                int(ColumnMap.PY),
                int(ColumnMap.PZ),
                int(ColumnMap.E),
                int(ColumnMap.TRACK_RESOLUTION),
            ],
            dtype=torch.long,
            device=particles.device,
        )
        new_values = torch.stack(
            [
                smeared_pt.to(out.dtype),
                smeared_px.to(out.dtype),
                smeared_py.to(out.dtype),
                smeared_pz.to(out.dtype),
                smeared_e.to(out.dtype),
                resolution.to(out.dtype),
            ],
            dim=1,
        )
        out[:, cols_to_replace] = new_values

        return out

    @staticmethod
    def _charged_hadron_cms_momentum_resolution(
        pt: torch.Tensor, eta_outer: torch.Tensor
    ) -> torch.Tensor:
        """CMS charged hadron momentum resolution formula.
        Based on arXiv:1405.6569.

        Parameters
        ----------
        pt: torch.Tensor
            Transverse momentum (GeV)
        eta_outer: torch.Tensor
            Pseudorapidity

        Returns
        -------
        resolution: torch.Tensor
            Relative momentum resolution (dimensionless, e.g., 0.06 = 6%)
            To get absolute resolution in GeV, multiply by pt.
        """
        abs_eta_outer = torch.abs(eta_outer)

        # Initialize with zeros
        res = torch.zeros_like(pt)

        # Region 1: Central barrel (|eta_outer| <= 0.5, pt > 0.1)
        # Resolution = sqrt(0.06^2 + pt^2 * 1.3e-3^2)
        mask1 = (abs_eta_outer <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.06**2 + pt**2 * (1.3e-3) ** 2)
        res = torch.where(mask1, res1, res)

        # Region 2: Intermediate (0.5 < |eta_outer| <= 1.5, pt > 0.1)
        # Resolution = sqrt(0.10^2 + pt^2 * 1.7e-3^2)
        mask2 = (abs_eta_outer > 0.5) & (abs_eta_outer <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.10**2 + pt**2 * (1.7e-3) ** 2)
        res = torch.where(mask2, res2, res)

        # Region 3: Forward (1.5 < |eta_outer| <= 2.5, pt > 0.1)
        # Resolution = sqrt(0.25^2 + pt^2 * 3.1e-3^2)
        mask3 = (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.25**2 + pt**2 * (3.1e-3) ** 2)
        res = torch.where(mask3, res3, res)

        return res

    @staticmethod
    def _electron_cms_momentum_resolution(
        pt: torch.Tensor, eta_outer: torch.Tensor
    ) -> torch.Tensor:
        """CMS electron momentum resolution formula.
        Based on arXiv:1502.02701.

        Parameters
        ----------
        pt: torch.Tensor
            Transverse momentum (GeV)
        eta_outer: torch.Tensor
            Pseudorapidity

        Returns
        -------
        resolution: torch.Tensor
            Relative momentum resolution (dimensionless, e.g., 0.03 = 3%)
            To get absolute resolution in GeV, multiply by pt.
        """
        abs_eta_outer = torch.abs(eta_outer)
        res = torch.zeros_like(pt)

        # Central barrel
        mask1 = (abs_eta_outer <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.03**2 + pt**2 * (1.3e-3) ** 2)
        res = torch.where(mask1, res1, res)

        # Intermediate
        mask2 = (abs_eta_outer > 0.5) & (abs_eta_outer <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.05**2 + pt**2 * (1.7e-3) ** 2)
        res = torch.where(mask2, res2, res)

        # Forward
        mask3 = (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.15**2 + pt**2 * (3.1e-3) ** 2)
        res = torch.where(mask3, res3, res)

        return res

    @staticmethod
    def _muon_cms_momentum_resolution(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """CMS muon momentum resolution formula.
        Based on arXiv:1306.2016.

        Parameters
        ----------
        pt: torch.Tensor
            Transverse momentum (GeV)
        eta_outer: torch.Tensor
            Pseudorapidity

        Returns
        -------
        resolution: torch.Tensor
            Relative momentum resolution (dimensionless, e.g., 0.01 = 1%)
            To get absolute resolution in GeV, multiply by pt.
        """
        abs_eta_outer = torch.abs(eta_outer)
        res = torch.zeros_like(pt)

        # Central barrel
        mask1 = (abs_eta_outer <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.01**2 + pt**2 * (1.0e-3) ** 2)
        res = torch.where(mask1, res1, res)

        # Intermediate
        mask2 = (abs_eta_outer > 0.5) & (abs_eta_outer <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.02**2 + pt**2 * (1.3e-3) ** 2)
        res = torch.where(mask2, res2, res)

        # Forward
        mask3 = (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.10**2 + pt**2 * (2.0e-3) ** 2)
        res = torch.where(mask3, res3, res)

        return res

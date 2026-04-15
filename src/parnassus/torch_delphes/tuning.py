"""End-to-end differentiable tuning harness for TorchDelphes.

This module provides a small self-contained tuning loop for the
learnable :class:`CMSEnergyFlowDefault` detector card. It includes:

- :func:`soft_histogram` — a fully-differentiable sigmoid-based soft
  histogram. Each particle value contributes a smooth "bin indicator"
  (a difference of sigmoids) so gradients flow back through the bin
  counts into the learnable detector parameters.

- :func:`histogram_mse_loss` — a mean-squared-error loss between two
  soft histograms (normalized to densities), suitable for matching a
  trainee card's observable distributions to a target.

- :func:`make_synthetic_particles` — helper to generate a toy mixed-
  species particle batch when no HepMC input is available.

- :func:`tune_cms_to_target` — the actual training loop. Takes a frozen
  target card and a trainee card (both ``learnable=True`` so histograms
  are directly comparable) and runs Adam to minimize the MSE on a set
  of observable histograms.

- A ``__main__`` block that runs a self-contained demo: it constructs a
  "target" card whose charged-hadron momentum-resolution constants have
  been inflated by 50% and then tunes a fresh learnable card against
  it. The script prints the initial and final loss, plus the learned
  values of the parameters it moved most.

Why a toy target and not real data?
-----------------------------------
To close the loop without any additional infrastructure, the demo uses
a *re-simulated* target: the same detector simulation with deliberately
perturbed parameters. This proves that the gradient pipeline works
end-to-end and that Adam can recover known perturbations. For a real
tuning study one would replace :func:`make_synthetic_particles` with a
HepMC / Pythia event stream and :func:`_target_from_card` with either
full-sim Geant4 output, CMS open data, or the neural generator in
``parnassus.pipelines.generators``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.learnable import LearnableMomentumResolution

# =============================================================================
# Differentiable histogram and loss
# =============================================================================


def soft_histogram(
    values: torch.Tensor,
    bin_edges: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """Sigmoid-based soft histogram that is fully differentiable in
    ``values``.

    For each bin ``[lo, hi]`` and each value ``x``, the contribution is

    .. code-block::

        sigmoid((x - lo) / (beta * width)) - sigmoid((x - hi) / (beta * width))

    which approaches the hard indicator function as ``beta -> 0`` but
    retains a smooth gradient for any ``beta > 0``. The output is summed
    across all input values.

    Parameters
    ----------
    values : torch.Tensor
        1-D tensor of observable values, shape ``(N,)``. Values that
        fall outside ``[bin_edges[0], bin_edges[-1]]`` contribute very
        little to the histogram.
    bin_edges : torch.Tensor
        1-D tensor of ``n_bins + 1`` bin edges, strictly increasing.
    beta : float
        Relative softness, as a fraction of the per-bin width. Small
        values (0.01 to 0.05) give a near-hard histogram with still-
        well-conditioned gradients; larger values (0.1 to 0.3) give
        smoother gradients at the cost of bin overlap.

    Returns
    -------
    torch.Tensor
        A ``(n_bins,)`` tensor of soft counts. Not normalized; divide
        by ``.sum()`` to convert to a probability density.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {tuple(values.shape)}")
    if bin_edges.ndim != 1 or bin_edges.numel() < 2:
        raise ValueError("bin_edges must be 1-D with at least 2 entries")

    widths = (bin_edges[1:] - bin_edges[:-1]).clamp(min=1e-12)
    scale = beta * widths  # per-bin softness in the same units as `values`
    lo = bin_edges[:-1]  # (n_bins,)
    hi = bin_edges[1:]

    # values: (N,), lo/hi/scale: (n_bins,) -> broadcast to (N, n_bins)
    v = values.to(bin_edges.dtype).unsqueeze(1)  # (N, 1)
    scale_r = scale.unsqueeze(0)  # (1, n_bins)
    soft_lo = torch.sigmoid((v - lo.unsqueeze(0)) / scale_r)
    soft_hi = torch.sigmoid((v - hi.unsqueeze(0)) / scale_r)
    contrib = soft_lo - soft_hi  # (N, n_bins)
    return contrib.sum(dim=0)


def histogram_mse_loss(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    bin_edges: torch.Tensor,
    beta: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE between two soft histograms, both normalized to densities.

    Suitable as a training loss when ``pred_values`` are produced by a
    differentiable detector simulation and ``target_values`` are
    detached (a fixed target distribution).

    Returns
    -------
    torch.Tensor
        Scalar MSE loss between the two normalized histograms.
    """
    pred_hist = soft_histogram(pred_values, bin_edges, beta=beta)
    target_hist = soft_histogram(target_values.detach(), bin_edges, beta=beta)
    pred_norm = pred_hist / (pred_hist.sum() + eps)
    target_norm = target_hist / (target_hist.sum() + eps)
    return ((pred_norm - target_norm) ** 2).mean()


# =============================================================================
# Synthetic particle source
# =============================================================================


def make_synthetic_particles(
    n: int = 400,
    seed: int | None = None,
    pt_range: tuple[float, float] = (0.5, 60.0),
    eta_range: tuple[float, float] = (-3.0, 3.0),
) -> torch.Tensor:
    """Build a mixed-species particle batch for tuning demos.

    Species mix is chosen to activate every non-high-pt learnable
    parameter: charged pions/kaons/protons, electrons, muons, photons,
    K-short, Lambda, and neutrons in roughly equal proportions.

    Parameters
    ----------
    n : int
        Total number of particles.
    seed : int | None
        If given, seeds torch's RNG for reproducibility.
    pt_range, eta_range : tuple[float, float]
        Ranges for uniform-random pt and eta.

    Returns
    -------
    torch.Tensor
        Shape ``(n, N_FEATURES)`` particle tensor with all required
        columns populated.
    """
    if seed is not None:
        torch.manual_seed(seed)
    arr = torch.zeros(n, N_FEATURES, dtype=torch.float64)
    pt_lo, pt_hi = pt_range
    eta_lo, eta_hi = eta_range
    arr[:, ColumnMap.PT] = torch.rand(n) * (pt_hi - pt_lo) + pt_lo
    arr[:, ColumnMap.ETA] = torch.rand(n) * (eta_hi - eta_lo) + eta_lo
    arr[:, ColumnMap.PHI] = (torch.rand(n) - 0.5) * (2.0 * np.pi)
    arr[:, ColumnMap.PX] = arr[:, ColumnMap.PT] * torch.cos(arr[:, ColumnMap.PHI])
    arr[:, ColumnMap.PY] = arr[:, ColumnMap.PT] * torch.sin(arr[:, ColumnMap.PHI])
    arr[:, ColumnMap.PZ] = arr[:, ColumnMap.PT] * torch.sinh(arr[:, ColumnMap.ETA])
    arr[:, ColumnMap.MASS] = 0.14
    arr[:, ColumnMap.E] = torch.sqrt(
        arr[:, ColumnMap.PX] ** 2
        + arr[:, ColumnMap.PY] ** 2
        + arr[:, ColumnMap.PZ] ** 2
        + arr[:, ColumnMap.MASS] ** 2
    )
    species = [
        (211, 1.0),
        (321, 1.0),
        (2212, 1.0),
        (11, -1.0),
        (13, 1.0),
        (22, 0.0),
        (310, 0.0),
        (3122, 0.0),
        (2112, 0.0),
    ]
    for i, (pid, charge) in enumerate(species):
        lo_idx = (i * n) // len(species)
        hi_idx = ((i + 1) * n) // len(species)
        arr[lo_idx:hi_idx, ColumnMap.PID] = pid
        arr[lo_idx:hi_idx, ColumnMap.CHARGE] = charge
    arr[:, ColumnMap.STATUS] = 1
    return arr


# =============================================================================
# Tuning loop
# =============================================================================


def _collect_observables(
    card: CMSEnergyFlowDefault,
    particles: torch.Tensor,
    n_passes: int = 1,
) -> dict[str, torch.Tensor]:
    """Run ``card`` on ``particles`` ``n_passes`` times and collect the
    observables used in the loss.

    Averaging over multiple passes is how we beat down the stochastic
    noise from log-normal smearing and Gumbel-ST sampling without having
    to run extremely large batches.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from observable name to a 1-D tensor of all positive
        values collected across the ``n_passes`` forward passes.
    """
    track_pt_list = []
    tower_e_list = []
    eflow_pt_list = []
    for _ in range(n_passes):
        out = card(particles)
        # Keep only strictly positive values so the log-pt observable is
        # well-defined; the mask is boolean so no gradients are lost
        # through it.
        track_pt = out["Track"][:, ColumnMap.PT]
        track_pt_list.append(track_pt[track_pt > 0])
        tower_e = out["Tower"][:, ColumnMap.E]
        tower_e_list.append(tower_e[tower_e > 0])
        eflow_pt = out["EFlowObject"][:, ColumnMap.PT]
        eflow_pt_list.append(eflow_pt[eflow_pt > 0])
    return {
        "track_pt": torch.cat(track_pt_list) if track_pt_list else torch.empty(0),
        "tower_e": torch.cat(tower_e_list) if tower_e_list else torch.empty(0),
        "eflow_pt": torch.cat(eflow_pt_list) if eflow_pt_list else torch.empty(0),
    }


def _multi_observable_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Sum of per-observable histogram MSEs.

    Returns
    -------
    torch.Tensor
        Scalar loss: the sum of ``histogram_mse_loss`` over all
        observables for which both ``pred`` and ``target`` have at
        least one value.
    """
    total = torch.zeros((), dtype=torch.float64)
    for key, pred_vals in pred.items():
        if pred_vals.numel() == 0 or target[key].numel() == 0:
            continue
        total = total + histogram_mse_loss(pred_vals, target[key], bin_edges[key], beta=beta)
    return total


def tune_cms_to_target(
    target: CMSEnergyFlowDefault,
    trainee: CMSEnergyFlowDefault,
    particles: torch.Tensor,
    n_steps: int = 50,
    lr: float = 5e-2,
    n_passes_per_step: int = 4,
    beta: float = 0.1,
    log_every: int = 10,
    parameters_to_train: Iterable[nn.Parameter] | None = None,
) -> dict[str, list[float]]:
    """Run Adam to match ``trainee`` observable histograms to ``target``.

    Both cards must be constructed with ``learnable=True`` (otherwise
    ``target.parameters()`` is empty, which is fine, but the trainee
    needs parameters to optimize).

    Parameters
    ----------
    target : CMSEnergyFlowDefault
        Reference card. Its parameters are frozen and used only to
        generate the target observable distributions at each step.
    trainee : CMSEnergyFlowDefault
        Card whose parameters are being optimized.
    particles : torch.Tensor
        Input stable-particle batch; both cards run on the same input.
    n_steps : int
        Number of Adam optimization steps.
    lr : float
        Adam learning rate.
    n_passes_per_step : int
        How many forward passes to average over per step to reduce
        stochastic-smearing noise in the loss.
    beta : float
        Soft-histogram temperature (fraction of bin width).
    log_every : int
        Print loss every ``log_every`` steps.
    parameters_to_train : Iterable[nn.Parameter] | None
        Restrict optimization to this subset of parameters. If ``None``,
        optimize all of ``trainee.parameters()``.

    Returns
    -------
    dict[str, list[float]]
        ``{"loss": [...], "steps": [...]}`` time-series of the training
        loss, for plotting or assertion.
    """
    # Freeze target so its parameters don't drift and produce no gradient.
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()

    params = (
        list(parameters_to_train) if parameters_to_train is not None else list(trainee.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr)

    # Pre-compute bin edges once. Ranges chosen to cover the bulk of the
    # toy-batch observables without cutting too hard on the tails.
    bin_edges = {
        "track_pt": torch.linspace(0.0, 60.0, 25, dtype=torch.float64),
        "tower_e": torch.linspace(0.0, 80.0, 25, dtype=torch.float64),
        "eflow_pt": torch.linspace(0.0, 60.0, 25, dtype=torch.float64),
    }

    history: dict[str, list[float]] = {"loss": [], "steps": []}

    for step in range(n_steps):
        opt.zero_grad()
        pred = _collect_observables(trainee, particles, n_passes=n_passes_per_step)
        with torch.no_grad():
            tgt = _collect_observables(target, particles, n_passes=n_passes_per_step)
        loss = _multi_observable_loss(pred, tgt, bin_edges, beta=beta)
        loss.backward()
        opt.step()

        history["loss"].append(float(loss))
        history["steps"].append(step)
        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            print(f"  step {step:3d}/{n_steps}  loss = {float(loss):.4e}")

    return history


# =============================================================================
# __main__: end-to-end self-contained demo
# =============================================================================


def _perturb_target_scale_(card: CMSEnergyFlowDefault, target_scale: float = 1.2) -> None:
    """Set the charged-hadron momentum-scale parameters of a learnable
    card to ``target_scale`` in every eta region. This is a stronger
    signal than perturbing resolution alone: a multiplicative shift in
    pt directly moves the peak of reconstructed-pt histograms rather
    than just widening them.

    The scale is parameterized as ``1 + 0.3 * tanh(raw)`` so
    ``raw = atanh((s - 1) / 0.3)``.
    """
    # The ``resolution_module`` attribute only exists on learnable cards;
    # mypy can't see that through the ``MomentumSmearing`` base type.
    chad_res: LearnableMomentumResolution = (
        card.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr,assignment]
    )
    raw_value = float(np.arctanh((target_scale - 1.0) / 0.3))
    with torch.no_grad():
        chad_res.scale_raw.fill_(raw_value)


def main() -> None:
    """Run the self-contained CMS tuning demo.

    Constructs a "target" learnable card with inflated charged-hadron pT
    scales, a "trainee" at defaults, and runs a short Adam loop that
    should move the trainee's scale parameters toward the target values.
    Prints the loss trajectory and the before/after scale values.
    """
    torch.manual_seed(0)

    target_scale = 1.2

    print("=" * 60)
    print("TorchDelphes CMS tuning demo")
    print("=" * 60)
    print(f"Constructing target card (learnable, chad pT scale = {target_scale})...")
    target = CMSEnergyFlowDefault(debug=False, learnable=True)
    _perturb_target_scale_(target, target_scale=target_scale)

    print("Constructing trainee card (learnable, defaults)...")
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True)

    print("Generating synthetic particle batch...")
    particles = make_synthetic_particles(n=1500, seed=1)

    # Only train the charged-hadron scale parameters for a focused demo
    # (the full 66-parameter loop works too; this isolates the signal).
    trainee_chad_res: LearnableMomentumResolution = (
        trainee.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr,assignment]
    )
    scale_params: list[nn.Parameter] = [trainee_chad_res.scale_raw]

    # Record initial scales to see how far the optimizer moves them.
    def _scales(card: CMSEnergyFlowDefault) -> torch.Tensor:
        chad_res: LearnableMomentumResolution = (
            card.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr,assignment]
        )
        return (1.0 + 0.3 * torch.tanh(chad_res.scale_raw)).detach().clone()

    before = _scales(trainee)

    print("Running Adam tuning loop (charged-hadron scale only)...")
    history = tune_cms_to_target(
        target=target,
        trainee=trainee,
        particles=particles,
        n_steps=60,
        lr=2e-1,
        n_passes_per_step=4,
        beta=0.15,
        log_every=10,
        parameters_to_train=scale_params,
    )

    after = _scales(trainee)
    tgt = _scales(target)

    print()
    print("Charged-hadron momentum-scale parameters (3 eta regions):")
    print(f"  default : {before.tolist()}")
    print(f"  target  : {tgt.tolist()}")
    print(f"  trainee : {after.tolist()}")
    dist_before = float((before - tgt).abs().sum())
    dist_after = float((after - tgt).abs().sum())
    print(f"  L1 distance to target:  before={dist_before:.4g}  after={dist_after:.4g}")
    print()
    print(
        f"Loss trajectory: first={history['loss'][0]:.4e}  "
        f"last={history['loss'][-1]:.4e}  "
        f"min={min(history['loss']):.4e}"
    )


if __name__ == "__main__":
    main()

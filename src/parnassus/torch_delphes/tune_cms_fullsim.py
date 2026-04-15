r"""Tune the learnable CMS TorchDelphes card against CMS full-simulation reco.

This script fits the 66 learnable parameters of
:class:`parnassus.torch_delphes.defaults.CMSEnergyFlowDefault` so that its
reconstructed observable distributions match a CMS full-simulation sample.
It is designed to read the ROOT-file format used by the Parnassus paper
(arXiv:2406.01620) and distributed on
`Zenodo record 11389651 <https://zenodo.org/records/11389651>`_, which the
``parnassus-hep/cms-flow`` repository also consumes.

ROOT file schema
----------------
The script expects a ROOT file with a TTree named ``event_tree`` that
contains (at least) the following branches, each as a jagged per-event
array:

- ``truth_pt`` : GeV, generator-level stable-particle transverse momentum
- ``truth_eta`` : generator-level stable-particle pseudorapidity
- ``truth_phi`` : generator-level stable-particle azimuth
- ``truth_class`` : integer particle-type label
  (``0=charged hadron, 1=electron, 2=muon, 3=neutral hadron, 4=photon``)
- ``pflow_pt`` / ``pflow_eta`` / ``pflow_phi`` / ``pflow_class`` : the
  corresponding CMS particle-flow reconstructed objects that serve as
  the fitting target

Class-to-PDG mapping is done with
:func:`parnassus.utils.class_to_pid_vectorized`.

Usage
-----
.. code-block:: shell

    uv run python -m parnassus.torch_delphes.tune_cms_fullsim \
        --root-file /path/to/train_800_1000_filter.root \
        --n-events 2000 \
        --n-steps 100

If ``--root-file`` is omitted (or the path doesn't exist), the script
generates a small synthetic fixture with the same schema, fits against
it, and reports the loss trajectory. This is useful for sandbox
validation and for CI. The real sample lives on Zenodo and is too large
(and the host is often blocked in sandboxed environments) to download
inline.

The fit loop is exactly the one in :mod:`parnassus.torch_delphes.tuning`
adapted to a fixed *external* target: the reco observables are computed
once from the ROOT file and then re-used on every step, and each step
only re-runs the trainee on the truth input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from torch import nn

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.tuning import histogram_mse_loss
from parnassus.utils import class_to_pid_vectorized

# =============================================================================
# ROOT I/O
# =============================================================================

# Exact branch names we consume. Keeping them centralized here makes it trivial
# to retarget the script at a different reco format (e.g. a Delphes output
# tree) by passing an alternative schema dict.
TRUTH_BRANCHES: tuple[str, ...] = ("truth_pt", "truth_eta", "truth_phi", "truth_class")
PFLOW_BRANCHES: tuple[str, ...] = ("pflow_pt", "pflow_eta", "pflow_phi", "pflow_class")


def load_cms_flow_root(
    path: Path,
    n_events: int,
    tree_name: str = "event_tree",
) -> dict[str, np.ndarray]:
    """Load ``n_events`` from a cms-flow-format ROOT file into numpy arrays.

    Parameters
    ----------
    path : Path
        ROOT file path.
    n_events : int
        Number of leading events to read (``entry_stop=n_events``).
    tree_name : str
        Name of the TTree to read. Defaults to ``"event_tree"``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from branch name to a 1-D object array of jagged per-event
        arrays. Contains both truth and pflow branches.
    """
    with uproot.open(str(path)) as f:
        if tree_name not in f:
            raise KeyError(
                f"ROOT file {path} has no tree named {tree_name!r}. "
                f"Available keys: {list(f.keys())[:10]}"
            )
        tree = f[tree_name]
        arrays = tree.arrays(
            list(TRUTH_BRANCHES + PFLOW_BRANCHES),
            library="np",
            entry_stop=n_events,
        )
    return arrays


def truth_to_particle_tensor(
    arrays: dict[str, np.ndarray],
    n_events: int,
) -> torch.Tensor:
    """Convert per-event truth jagged arrays into a flat particle tensor.

    Each truth particle becomes one row of an ``(N_total, N_FEATURES)``
    tensor suitable to be fed directly into
    :class:`CMSEnergyFlowDefault.forward`. ``EVENT_NUMBER`` is set to the
    (0-indexed) event so that the calorimeter's per-event tower
    aggregation treats events independently.

    Particle class is mapped to PDG via
    :func:`parnassus.utils.class_to_pid_vectorized`. Electrons / muons
    are given small masses from the PDG table; everything else gets the
    charged-pion mass (0.14 GeV), which matches what the Parnassus
    pipeline does for minimum-bias-like particles.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N_total, N_FEATURES)`` ready for
        :class:`CMSEnergyFlowDefault.forward`.
    """
    rows_list: list[np.ndarray] = []
    for i in range(n_events):
        pt = np.asarray(arrays["truth_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["truth_eta"][i], dtype=np.float64)
        phi = np.asarray(arrays["truth_phi"][i], dtype=np.float64)
        cls = np.asarray(arrays["truth_class"][i], dtype=np.int64)
        n_p = pt.shape[0]
        if n_p == 0:
            continue
        pids = class_to_pid_vectorized(cls)
        # Rough masses: electrons/muons use their PDG masses, everything
        # else uses the charged-pion mass as a stand-in.
        abs_pid = np.abs(pids)
        mass = np.where(abs_pid == 11, 0.000511, np.where(abs_pid == 13, 0.10566, 0.13957)).astype(
            np.float64
        )
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        charge = np.where(
            abs_pid == 211,
            1.0,
            np.where(
                abs_pid == 11,
                -1.0,
                np.where(abs_pid == 13, -1.0, 0.0),
            ),
        ).astype(np.float64)
        # Sign the charge based on the sign of the underlying PDG (class->PDG
        # returns positive PIDs, so we conservatively keep |charge| = {0, 1}).
        row = np.zeros((n_p, N_FEATURES), dtype=np.float64)
        row[:, ColumnMap.PID] = pids
        row[:, ColumnMap.STATUS] = 1
        row[:, ColumnMap.CHARGE] = charge
        row[:, ColumnMap.E] = e
        row[:, ColumnMap.PX] = px
        row[:, ColumnMap.PY] = py
        row[:, ColumnMap.PZ] = pz
        row[:, ColumnMap.PT] = pt
        row[:, ColumnMap.ETA] = eta
        row[:, ColumnMap.PHI] = phi
        row[:, ColumnMap.MASS] = mass
        row[:, ColumnMap.EVENT_NUMBER] = i
        rows_list.append(row)
    if not rows_list:
        return torch.zeros((0, N_FEATURES), dtype=torch.float64)
    return torch.from_numpy(np.concatenate(rows_list, axis=0))


# =============================================================================
# Observables
# =============================================================================


def pflow_target_observables(
    arrays: dict[str, np.ndarray],
    n_events: int,
) -> dict[str, torch.Tensor]:
    """Build target observable tensors from the ``pflow_*`` branches.

    These are the fitting target. Because they come straight from the
    full-simulation CMS PF reco in the ROOT file, they do not depend on
    any learnable parameters and are computed once per call.

    Returns
    -------
    dict[str, torch.Tensor]
        ``"pt"`` and ``"eta"`` are per-particle 1-D tensors.
        ``"multiplicity"`` is per-event (``n_events``) and ``"ht"`` is
        the per-event scalar-pT sum.
    """
    all_pt: list[np.ndarray] = []
    all_eta: list[np.ndarray] = []
    per_event_mult = np.zeros(n_events, dtype=np.float64)
    per_event_ht = np.zeros(n_events, dtype=np.float64)
    for i in range(n_events):
        pt = np.asarray(arrays["pflow_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["pflow_eta"][i], dtype=np.float64)
        all_pt.append(pt)
        all_eta.append(eta)
        per_event_mult[i] = float(pt.shape[0])
        per_event_ht[i] = float(pt.sum())
    pt_cat = np.concatenate(all_pt) if all_pt else np.empty(0, dtype=np.float64)
    eta_cat = np.concatenate(all_eta) if all_eta else np.empty(0, dtype=np.float64)
    return {
        "pt": torch.from_numpy(pt_cat),
        "eta": torch.from_numpy(eta_cat),
        "multiplicity": torch.from_numpy(per_event_mult),
        "ht": torch.from_numpy(per_event_ht),
    }


def trainee_observables(
    card_out: dict[str, torch.Tensor],
    n_events: int,
    min_pt: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build the same observable dict from the trainee card's output.

    Operates on the ``EFlowObject`` branch of the card output, which is
    the PF-like final collection. Zero-pT ghost tracks (a byproduct of
    the Gumbel-ST efficiency mask; see ``learnable.py``) are filtered
    out before any statistic is computed; the filter is a hard boolean
    mask on values only, not on gradients, so it does not break
    backprop.

    Returns
    -------
    dict[str, torch.Tensor]
        Same keys as :func:`pflow_target_observables`. All tensors
        carry gradients back to the learnable parameters.
    """
    eflow = card_out["EFlowObject"]
    pt_all = eflow[:, ColumnMap.PT]
    eta_all = eflow[:, ColumnMap.ETA]
    event_all = eflow[:, ColumnMap.EVENT_NUMBER]

    valid = pt_all > min_pt
    pt_kept = pt_all[valid]
    eta_kept = eta_all[valid]
    ev_kept = event_all[valid].long()

    # Per-event multiplicity and HT via scatter-add. Using
    # torch.zeros(..., dtype=pt.dtype) on the same device as pt so the
    # accumulation stays in the autograd graph.
    device = pt_all.device
    dtype = pt_all.dtype
    mult = torch.zeros(n_events, dtype=dtype, device=device)
    ht = torch.zeros(n_events, dtype=dtype, device=device)
    if ev_kept.numel() > 0:
        mult.scatter_add_(0, ev_kept, torch.ones_like(pt_kept))
        ht.scatter_add_(0, ev_kept, pt_kept)

    return {
        "pt": pt_kept,
        "eta": eta_kept,
        "multiplicity": mult,
        "ht": ht,
    }


# Default bin edges for the four observables. Wide-enough to cover the bulk
# of both target and trainee distributions on a QCD-jet sample; users can
# override these via --bin-config when needed.
DEFAULT_BIN_EDGES: dict[str, torch.Tensor] = {
    "pt": torch.linspace(0.0, 200.0, 41, dtype=torch.float64),
    "eta": torch.linspace(-5.0, 5.0, 41, dtype=torch.float64),
    "multiplicity": torch.linspace(0.0, 400.0, 41, dtype=torch.float64),
    "ht": torch.linspace(0.0, 2000.0, 41, dtype=torch.float64),
}

# Default per-observable weights. The particle-level observables (pt, eta)
# are far less noisy than the per-event scalars (multiplicity, ht) because
# they have O(N_particles) rather than O(N_events) samples, so we upweight
# them and put a small tie-breaking weight on the per-event pair.
DEFAULT_OBS_WEIGHTS: dict[str, float] = {
    "pt": 1.0,
    "eta": 1.0,
    "multiplicity": 0.1,
    "ht": 0.1,
}


# =============================================================================
# Parameter groups for per-group learning rates
# =============================================================================
#
# The 66 learnable parameters live in parameter spaces with very different
# natural scales, so a single global Adam learning rate is a poor fit:
#
# - Resolution coefficients ``a_raw, b_raw, c_E, c_S, c_N`` are softplus-
#   wrapped positive scalars whose post-softplus values range from ~1e-3 to
#   ~3. An Adam step of 5e-2 on the raw parameter can blow up a tiny
#   coefficient by orders of magnitude in one step.
# - Region scales ``scale_raw`` sit at 0 in the raw space and are bounded
#   to [-1, +1] via ``tanh``; an Adam step of 5e-2 moves the bounded
#   output by at most ~1.5%, which is safe and responsive.
# - Efficiency logits ``eff_logits`` map to probabilities via sigmoid and
#   sit at finite values like ``logit(0.95)=2.94``. Stepping the logit by
#   5e-2 moves the probability by less than 0.01, which is the right
#   magnitude for PF efficiency fits.
# - Hadron-fraction logits behave like efficiency logits.
#
# The function below maps each named parameter in a learnable card to one
# of the four groups so that :func:`fit_card_to_fullsim` can build a
# ``torch.optim.Adam`` with per-group learning rates.

_SCALE_SUFFIXES = ("scale_raw",)
_EFFICIENCY_SUFFIXES = ("eff_logits", "rate_raw")
_FRACTION_FRAGMENTS = ("HadronFractions",)
# Everything else on a learnable card that has an ``nn.Parameter`` is a
# resolution coefficient (softplus-wrapped positive number).

_DEFAULT_LR_SCALES: float = 5e-2
_DEFAULT_LR_EFFICIENCY: float = 5e-2
_DEFAULT_LR_FRACTIONS: float = 5e-2
_DEFAULT_LR_RESOLUTION: float = 5e-3


def _classify_parameter(name: str) -> str:
    """Return the parameter-group key for a given fully-qualified name.

    Returns
    -------
    str
        One of ``"scales"``, ``"efficiency"``, ``"fractions"``,
        ``"resolution"``.
    """
    if any(name.endswith(s) for s in _SCALE_SUFFIXES):
        return "scales"
    if any(frag in name for frag in _FRACTION_FRAGMENTS):
        return "fractions"
    if any(name.endswith(s) for s in _EFFICIENCY_SUFFIXES):
        return "efficiency"
    return "resolution"


def build_parameter_groups(
    card: CMSEnergyFlowDefault,
    lr_scales: float = _DEFAULT_LR_SCALES,
    lr_resolution: float = _DEFAULT_LR_RESOLUTION,
    lr_efficiency: float = _DEFAULT_LR_EFFICIENCY,
    lr_fractions: float = _DEFAULT_LR_FRACTIONS,
) -> list[dict]:
    """Split ``card.parameters()`` into Adam parameter groups by type.

    Returns
    -------
    list[dict]
        A list of four ``{"params": [...], "lr": float, "name": str}``
        dicts, suitable to pass directly to ``torch.optim.Adam``.
        Empty groups are filtered out so the optimizer doesn't complain
        about an empty ``params`` list.
    """
    buckets: dict[str, list[nn.Parameter]] = {
        "scales": [],
        "resolution": [],
        "efficiency": [],
        "fractions": [],
    }
    for name, p in card.named_parameters():
        buckets[_classify_parameter(name)].append(p)
    lrs = {
        "scales": lr_scales,
        "resolution": lr_resolution,
        "efficiency": lr_efficiency,
        "fractions": lr_fractions,
    }
    groups: list[dict] = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append({"params": params, "lr": lrs[key], "name": key})
    return groups


def multi_observable_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor],
    beta: float = 0.15,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Sum of per-observable soft-histogram MSE losses, optionally weighted.

    Returns
    -------
    torch.Tensor
        Scalar loss summed over every observable for which both
        ``pred`` and ``target`` have at least one value.
    """
    total = torch.zeros((), dtype=torch.float64)
    for key, pred_vals in pred.items():
        tgt_vals = target[key]
        if pred_vals.numel() == 0 or tgt_vals.numel() == 0:
            continue
        edges = bin_edges[key]
        w = (weights or {}).get(key, 1.0)
        total = total + w * histogram_mse_loss(pred_vals, tgt_vals, edges, beta=beta)
    return total


# =============================================================================
# Fit loop
# =============================================================================


def fit_card_to_fullsim(
    card: CMSEnergyFlowDefault,
    truth_particles: torch.Tensor,
    target_observables: dict[str, torch.Tensor],
    n_events: int,
    n_steps: int = 100,
    lr: float | None = None,
    n_passes_per_step: int = 2,
    beta: float = 0.15,
    log_every: int = 10,
    parameters_to_train: list[nn.Parameter] | None = None,
    bin_edges: dict[str, torch.Tensor] | None = None,
    observable_weights: dict[str, float] | None = None,
    lr_scales: float = _DEFAULT_LR_SCALES,
    lr_resolution: float = _DEFAULT_LR_RESOLUTION,
    lr_efficiency: float = _DEFAULT_LR_EFFICIENCY,
    lr_fractions: float = _DEFAULT_LR_FRACTIONS,
    snapshot_parameters: bool = False,
) -> dict[str, list[float]]:
    """Run Adam on ``card`` to match ``target_observables``.

    The trainee is run ``n_passes_per_step`` times per step, with losses
    averaged, to beat down the stochastic-smearing / Gumbel-ST noise in
    the trainee observables. The target observables are read from the
    ROOT file once and re-used on every step.

    When ``parameters_to_train`` is ``None`` this function automatically
    builds four parameter groups (scales, resolution, efficiency,
    fractions) with independent learning rates, because the four
    groups have very different natural step sizes. Pass a non-None
    ``parameters_to_train`` list to get a single group at the global
    ``lr`` value (backwards-compatible with earlier callers that
    fit a small focused subset).

    Parameters
    ----------
    lr : float | None
        Only used when ``parameters_to_train`` is provided (single
        group). If ``None`` in that case, falls back to
        ``_DEFAULT_LR_SCALES``. When ``parameters_to_train`` is
        ``None``, the four ``lr_*`` arguments are used instead.
    snapshot_parameters : bool
        If True, the history dict will additionally contain a
        ``"parameters"`` list whose i-th entry is a ``{name: float}``
        dict recording every learnable parameter value after step i.
        Off by default because it is O(n_steps * 66) in memory and
        only needed for plotting parameter-drift trajectories.

    Returns
    -------
    dict[str, list]
        ``"step"`` and ``"loss"`` always present. If
        ``snapshot_parameters`` is True, also contains
        ``"parameters"`` (a list of ``dict[str, float]``) for offline
        plotting of the per-parameter trajectory.
    """
    if parameters_to_train is not None:
        param_groups: list[dict] = [
            {
                "params": list(parameters_to_train),
                "lr": lr if lr is not None else _DEFAULT_LR_SCALES,
                "name": "user",
            }
        ]
    else:
        param_groups = build_parameter_groups(
            card,
            lr_scales=lr_scales,
            lr_resolution=lr_resolution,
            lr_efficiency=lr_efficiency,
            lr_fractions=lr_fractions,
        )
    opt = torch.optim.Adam(param_groups)
    edges = bin_edges if bin_edges is not None else DEFAULT_BIN_EDGES
    weights = observable_weights if observable_weights is not None else DEFAULT_OBS_WEIGHTS
    history: dict[str, list] = {"step": [], "loss": []}
    if snapshot_parameters:
        history["parameters"] = []

    def _snapshot() -> dict[str, float]:
        """Record the current post-transform value of every parameter.

        We store the interpretable (post-softplus / post-tanh /
        post-sigmoid) value rather than the raw parameter, because
        that's what the plots and the JINST paper tables will show.

        Returns
        -------
        dict[str, float]
            ``{name_or_name[i]: value}`` mapping, one entry per scalar
            component of every ``nn.Parameter`` on the card.
        """
        snap: dict[str, float] = {}
        for name, p in card.named_parameters():
            if name.endswith(".scale_raw"):
                val = 1.0 + 0.3 * torch.tanh(p)
            elif name.endswith((".eff_logits", "_logit")):
                val = torch.sigmoid(p)
            elif name.endswith((".rate_raw", ".a_raw", ".b_raw")) or name.startswith((
                "ECal.resolution_func",
                "HCal.resolution_func",
            )):
                val = torch.nn.functional.softplus(p)
            else:
                val = p
            vflat = val.detach().flatten().tolist() if val.ndim else [float(val.detach())]
            for i, vv in enumerate(vflat):
                snap[f"{name}[{i}]" if val.ndim else name] = float(vv)
        return snap

    for step in range(n_steps):
        opt.zero_grad()
        loss_acc = torch.zeros((), dtype=torch.float64)
        for _ in range(n_passes_per_step):
            out = card(truth_particles)
            pred = trainee_observables(out, n_events=n_events)
            loss_acc = loss_acc + multi_observable_loss(
                pred, target_observables, edges, beta=beta, weights=weights
            )
        loss = loss_acc / max(n_passes_per_step, 1)
        loss.backward()
        opt.step()

        history["step"].append(step)
        history["loss"].append(float(loss))
        if snapshot_parameters:
            history["parameters"].append(_snapshot())
        if log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            print(f"  step {step:3d}/{n_steps}  loss = {float(loss):.4e}")

    return history


# =============================================================================
# Synthetic fixture (for sandbox validation / CI)
# =============================================================================


def write_synthetic_fixture(
    out_path: Path,
    n_events: int = 50,
    particles_per_event: int = 60,
    seed: int = 0,
) -> None:
    """Write a tiny ROOT file with the cms-flow schema for validation.

    The ``truth_*`` arrays are a plausible mixed-species QCD-like batch.
    The ``pflow_*`` arrays are generated from the same underlying truth
    by applying a *deliberately mis-scaled* version of the CMS default
    Delphes card: charged-hadron pT is multiplied by 1.2 and the ECal
    energy scale by 1.1. This mimics a "Delphes-with-wrong-scales" ->
    "CMS full sim" gap, so that fitting the fresh trainee card against
    the pflow branches should move those scale parameters back toward
    1.2 / 1.1.

    The fixture is NOT a substitute for the real Zenodo sample -- it's
    just a local stand-in so we can run and test the full pipeline.
    """
    rng = np.random.default_rng(seed)

    truth_pt_list: list[np.ndarray] = []
    truth_eta_list: list[np.ndarray] = []
    truth_phi_list: list[np.ndarray] = []
    truth_class_list: list[np.ndarray] = []

    # Underlying particle tensor we can feed into a card directly.
    truth_arrays_flat: list[np.ndarray] = []
    for i in range(n_events):
        n_p = int(max(10, rng.normal(particles_per_event, particles_per_event * 0.2)))
        pt = rng.uniform(0.5, 80.0, size=n_p).astype(np.float32)
        eta = rng.uniform(-3.0, 3.0, size=n_p).astype(np.float32)
        phi = rng.uniform(-np.pi, np.pi, size=n_p).astype(np.float32)
        cls = rng.integers(0, 5, size=n_p).astype(np.int32)
        truth_pt_list.append(pt)
        truth_eta_list.append(eta)
        truth_phi_list.append(phi)
        truth_class_list.append(cls)

        # Corresponding flat particle row for running through a card.
        pids = class_to_pid_vectorized(cls.astype(np.int64))
        abs_pid = np.abs(pids)
        mass = np.where(
            abs_pid == 11,
            0.000511,
            np.where(abs_pid == 13, 0.10566, 0.13957),
        ).astype(np.float64)
        pt_d = pt.astype(np.float64)
        eta_d = eta.astype(np.float64)
        phi_d = phi.astype(np.float64)
        px = pt_d * np.cos(phi_d)
        py = pt_d * np.sin(phi_d)
        pz = pt_d * np.sinh(eta_d)
        e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        charge = np.where(
            abs_pid == 211,
            1.0,
            np.where(
                abs_pid == 11,
                -1.0,
                np.where(abs_pid == 13, -1.0, 0.0),
            ),
        ).astype(np.float64)
        row = np.zeros((n_p, N_FEATURES), dtype=np.float64)
        row[:, ColumnMap.PID] = pids
        row[:, ColumnMap.STATUS] = 1
        row[:, ColumnMap.CHARGE] = charge
        row[:, ColumnMap.E] = e
        row[:, ColumnMap.PX] = px
        row[:, ColumnMap.PY] = py
        row[:, ColumnMap.PZ] = pz
        row[:, ColumnMap.PT] = pt_d
        row[:, ColumnMap.ETA] = eta_d
        row[:, ColumnMap.PHI] = phi_d
        row[:, ColumnMap.MASS] = mass
        row[:, ColumnMap.EVENT_NUMBER] = i
        truth_arrays_flat.append(row)

    truth_tensor = torch.from_numpy(np.concatenate(truth_arrays_flat, axis=0))

    # "Full-sim-like" reco is obtained by running a PERTURBED learnable card
    # on the truth particles and using its EFlowObject output as the
    # pflow_* truth. The perturbation is exactly what we'd like Adam to
    # recover.
    #
    # We make the perturbation fairly large (chad pT scale = 1.25, ECal
    # energy scale = 1.20 in every region) so the target distributions
    # are clearly distinguishable from the defaults, and we average the
    # target card's output over ``n_target_passes`` runs to beat down the
    # per-realization sampling noise in the saved pflow branches.
    torch.manual_seed(seed)
    target_card = CMSEnergyFlowDefault(debug=False, learnable=True)
    with torch.no_grad():
        chad_res = target_card.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
        # scale = 1 + 0.3*tanh(raw) -> raw = atanh((s-1)/0.3)
        chad_res.scale_raw.fill_(float(np.arctanh((1.25 - 1.0) / 0.3)))
        ecal_scale = target_card.ECal.scale_module  # type: ignore[union-attr]
        ecal_scale.scale_raw.fill_(float(np.arctanh((1.20 - 1.0) / 0.3)))
    for p in target_card.parameters():
        p.requires_grad_(False)
    target_card.eval()

    n_target_passes = 5
    with torch.no_grad():
        eflow_collected: list[torch.Tensor] = [
            target_card(truth_tensor)["EFlowObject"] for _ in range(n_target_passes)
        ]
    # Concatenate all passes into a single "pooled" EFlow tensor. The
    # corresponding event-number column is already correctly set because
    # each pass uses the same truth input, so we can sort by event and
    # then stack all passes' particles under each event.
    eflow = torch.cat(eflow_collected, dim=0)
    event_idx = eflow[:, ColumnMap.EVENT_NUMBER].long().cpu().numpy()
    pflow_pt_arr = eflow[:, ColumnMap.PT].cpu().numpy().astype(np.float32)
    pflow_eta_arr = eflow[:, ColumnMap.ETA].cpu().numpy().astype(np.float32)
    pflow_phi_arr = eflow[:, ColumnMap.PHI].cpu().numpy().astype(np.float32)
    pflow_pid = eflow[:, ColumnMap.PID].cpu().numpy().astype(np.int64)
    # Convert PID -> class for the fixture (matching the real schema).
    abs_pid = np.abs(pflow_pid)
    pflow_cls_arr = np.where(
        abs_pid == 11,
        1,
        np.where(
            abs_pid == 13,
            2,
            np.where(abs_pid == 22, 4, np.where(abs_pid == 211, 0, 3)),
        ),
    ).astype(np.int32)
    # Drop zero-pt ghost tracks from the fixture — no physical observable
    # depends on them and they'd pollute the target multiplicity.
    keep = pflow_pt_arr > 1e-6
    event_idx = event_idx[keep]
    pflow_pt_arr = pflow_pt_arr[keep]
    pflow_eta_arr = pflow_eta_arr[keep]
    pflow_phi_arr = pflow_phi_arr[keep]
    pflow_cls_arr = pflow_cls_arr[keep]

    pflow_pt_list = [pflow_pt_arr[event_idx == i] for i in range(n_events)]
    pflow_eta_list = [pflow_eta_arr[event_idx == i] for i in range(n_events)]
    pflow_phi_list = [pflow_phi_arr[event_idx == i] for i in range(n_events)]
    pflow_cls_list = [pflow_cls_arr[event_idx == i] for i in range(n_events)]

    with uproot.recreate(str(out_path)) as f:
        f["event_tree"] = {
            "truth_pt": ak.Array(truth_pt_list),
            "truth_eta": ak.Array(truth_eta_list),
            "truth_phi": ak.Array(truth_phi_list),
            "truth_class": ak.Array(truth_class_list),
            "pflow_pt": ak.Array(pflow_pt_list),
            "pflow_eta": ak.Array(pflow_eta_list),
            "pflow_phi": ak.Array(pflow_phi_list),
            "pflow_class": ak.Array(pflow_cls_list),
        }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.tune_cms_fullsim``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-file",
        type=Path,
        default=None,
        help=(
            "Path to a CMS full-simulation ROOT file with an event_tree "
            "following the cms-flow schema. If not provided, a synthetic "
            "fixture is generated under /tmp."
        ),
    )
    parser.add_argument("--n-events", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Global learning rate. Only used when --train-what is a strict "
            "subset (scale_only, resolution_only). With --train-what=all the "
            "four --lr-* flags are used instead, because the four parameter "
            "groups have very different natural step sizes."
        ),
    )
    parser.add_argument("--lr-scales", type=float, default=_DEFAULT_LR_SCALES)
    parser.add_argument("--lr-resolution", type=float, default=_DEFAULT_LR_RESOLUTION)
    parser.add_argument("--lr-efficiency", type=float, default=_DEFAULT_LR_EFFICIENCY)
    parser.add_argument("--lr-fractions", type=float, default=_DEFAULT_LR_FRACTIONS)
    parser.add_argument("--n-passes-per-step", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help=(
            "If set, write the full training history (loss trajectory and "
            "per-step parameter snapshots) to this path as JSON. Used by "
            "the plotting scripts and by the JINST-paper figures."
        ),
    )
    parser.add_argument(
        "--train-what",
        choices=("all", "scale_only", "resolution_only"),
        default="all",
        help="Which parameter subset to optimize.",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=Path("/tmp/tune_cms_fullsim_fixture.root"),
        help="Where to write (and load) the synthetic fallback fixture.",
    )
    args = parser.parse_args()

    # Resolve the input file: use the real file if it exists, otherwise
    # generate a synthetic fixture so the demo is always runnable.
    root_file: Path
    if args.root_file is not None and args.root_file.exists():
        root_file = args.root_file
        print(f"Loading full-simulation events from {root_file}")
    else:
        if args.root_file is not None:
            print(f"WARNING: {args.root_file} does not exist; falling back to fixture.")
        print(
            "No real ROOT file provided. Generating synthetic fixture at "
            f"{args.fixture_path} (same schema as Zenodo record 11389651)..."
        )
        write_synthetic_fixture(
            args.fixture_path,
            n_events=args.n_events,
            particles_per_event=60,
            seed=args.seed,
        )
        root_file = args.fixture_path
        print(
            "To fit against the real sample, download e.g. "
            "https://zenodo.org/records/11389651 and rerun with --root-file."
        )

    arrays = load_cms_flow_root(root_file, n_events=args.n_events)
    truth_tensor = truth_to_particle_tensor(arrays, n_events=args.n_events)
    target = pflow_target_observables(arrays, n_events=args.n_events)
    print(
        f"Loaded {args.n_events} events, {truth_tensor.shape[0]} truth particles, "
        f"{target['pt'].numel()} PFlow target particles."
    )

    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True)

    # Pick the parameter subset to train.
    if args.train_what == "all":
        params_to_train: list[nn.Parameter] | None = None
    else:
        chad_res = trainee.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
        ecal_scale = trainee.ECal.scale_module  # type: ignore[union-attr]
        if args.train_what == "scale_only":
            params_to_train = [chad_res.scale_raw, ecal_scale.scale_raw]
        else:  # resolution_only
            params_to_train = [chad_res.a_raw, chad_res.b_raw]

    print(
        f"Training {'all 66' if params_to_train is None else len(params_to_train)} "
        f"learnable params for {args.n_steps} Adam steps..."
    )

    def _averaged_loss(n: int = 6) -> float:
        """Average the soft-histogram loss over ``n`` fresh forward passes.

        Single-pass losses fluctuate by several tens of percent due to the
        log-normal smearing and the Gumbel-ST sampling, so a fair
        "before vs after" comparison has to average over samples.

        Returns
        -------
        float
            Mean loss value over the ``n`` passes.
        """
        acc = 0.0
        with torch.no_grad():
            for _ in range(n):
                out = trainee(truth_tensor)
                pred = trainee_observables(out, n_events=args.n_events)
                acc += float(
                    multi_observable_loss(
                        pred,
                        target,
                        DEFAULT_BIN_EDGES,
                        beta=args.beta,
                        weights=DEFAULT_OBS_WEIGHTS,
                    )
                )
        return acc / n

    loss_before = _averaged_loss()
    print(f"Averaged loss at init (6 passes): {loss_before:.4e}")

    history = fit_card_to_fullsim(
        trainee,
        truth_tensor,
        target,
        n_events=args.n_events,
        n_steps=args.n_steps,
        lr=args.lr,
        n_passes_per_step=args.n_passes_per_step,
        beta=args.beta,
        log_every=max(1, args.n_steps // 10),
        parameters_to_train=params_to_train,
        lr_scales=args.lr_scales,
        lr_resolution=args.lr_resolution,
        lr_efficiency=args.lr_efficiency,
        lr_fractions=args.lr_fractions,
        snapshot_parameters=args.history_path is not None,
    )

    if args.history_path is not None:
        args.history_path.parent.mkdir(parents=True, exist_ok=True)
        with args.history_path.open("w") as f:
            json.dump(
                {
                    "loss": history["loss"],
                    "step": history["step"],
                    "parameters": history.get("parameters", []),
                    "n_events": args.n_events,
                    "n_steps": args.n_steps,
                    "lr_scales": args.lr_scales,
                    "lr_resolution": args.lr_resolution,
                    "lr_efficiency": args.lr_efficiency,
                    "lr_fractions": args.lr_fractions,
                    "train_what": args.train_what,
                },
                f,
                indent=2,
            )
        print(f"Wrote training history to {args.history_path}")

    loss_after = _averaged_loss()
    print(f"Averaged loss after training (6 passes): {loss_after:.4e}")
    rel = 100.0 * (loss_before - loss_after) / max(loss_before, 1e-30)
    print(f"  relative improvement: {rel:+.1f}%")

    # Print the learned charged-hadron scale and ECal scale for a quick
    # sanity check. On the synthetic fixture the expected target is
    # chad_scale=1.2 and ecal_scale=1.1 in every region.
    chad_res = trainee.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
    chad_scales = (1.0 + 0.3 * torch.tanh(chad_res.scale_raw)).detach().tolist()
    ecal_scale_vals = (
        (
            1.0
            + 0.3
            * torch.tanh(
                trainee.ECal.scale_module.scale_raw  # type: ignore[union-attr]
            )
        )
        .detach()
        .tolist()
    )
    print()
    print(f"Final charged-hadron scale (3 eta regions): {chad_scales}")
    print(f"Final ECal scale            (3 eta regions): {ecal_scale_vals}")
    if root_file == args.fixture_path:
        print("(on synthetic fixture: target values are 1.25 and 1.20 respectively)")


if __name__ == "__main__":
    main()

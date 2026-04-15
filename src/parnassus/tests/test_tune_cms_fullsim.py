"""Tests for the ``tune_cms_fullsim`` harness.

These tests cover the full pipeline end-to-end using the synthetic
fixture writer (because the real Zenodo sample is not in the test
tree): generate a cms-flow-format ROOT file, load it, build the truth
particle tensor, collect target observables, run one forward + backward
step on the learnable CMS card, and confirm that every learnable
parameter received a finite gradient.

Real-sample smoke tests are the user's responsibility (download the
file and run ``python -m parnassus.torch_delphes.tune_cms_fullsim
--root-file ...``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim import (
    DEFAULT_BIN_EDGES,
    PFLOW_BRANCHES,
    TRUTH_BRANCHES,
    fit_card_to_fullsim,
    load_cms_flow_root,
    multi_observable_loss,
    pflow_target_observables,
    trainee_observables,
    truth_to_particle_tensor,
    write_synthetic_fixture,
)


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped synthetic ROOT fixture so we write it once per run."""
    path = tmp_path_factory.mktemp("fullsim_fixture") / "fixture.root"
    write_synthetic_fixture(path, n_events=30, particles_per_event=50, seed=0)
    return path


def test_fixture_has_expected_branches(fixture_root: Path):
    """The fixture is readable and contains every branch the script reads."""
    import uproot

    with uproot.open(str(fixture_root)) as f:
        tree = f["event_tree"]
        keys = set(tree.keys())
    for branch in TRUTH_BRANCHES + PFLOW_BRANCHES:
        assert branch in keys, f"missing branch {branch} in fixture"


def test_load_cms_flow_root_roundtrip(fixture_root: Path):
    """``load_cms_flow_root`` returns dense per-event numpy arrays."""
    arrays = load_cms_flow_root(fixture_root, n_events=30)
    for branch in TRUTH_BRANCHES + PFLOW_BRANCHES:
        assert branch in arrays
        assert len(arrays[branch]) == 30
    # Each per-event entry is a 1-D array.
    assert arrays["truth_pt"][0].ndim == 1
    assert arrays["pflow_pt"][0].ndim == 1


def test_truth_tensor_has_correct_columns(fixture_root: Path):
    """Conversion to particle tensor populates all mandatory columns."""
    arrays = load_cms_flow_root(fixture_root, n_events=10)
    t = truth_to_particle_tensor(arrays, n_events=10)
    assert t.ndim == 2
    assert t.shape[1] == N_FEATURES
    assert t.shape[0] > 0
    # Event numbers span 0..9 (with some possibly absent if an event was empty).
    event_numbers = set(int(e) for e in t[:, ColumnMap.EVENT_NUMBER].tolist())
    assert event_numbers.issubset(set(range(10)))
    # Mass is positive and pT is finite.
    assert float(t[:, ColumnMap.MASS].min()) > 0
    assert torch.isfinite(t[:, ColumnMap.PT]).all()
    # E^2 should be ~ p^2 + m^2 (up to roundoff).
    p_sq = t[:, ColumnMap.PX] ** 2 + t[:, ColumnMap.PY] ** 2 + t[:, ColumnMap.PZ] ** 2
    e_sq = t[:, ColumnMap.E] ** 2
    m_sq = t[:, ColumnMap.MASS] ** 2
    assert torch.allclose(e_sq, p_sq + m_sq, atol=1e-6)


def test_pflow_target_observables_shapes(fixture_root: Path):
    """Target observables have the right shapes and are all finite."""
    arrays = load_cms_flow_root(fixture_root, n_events=15)
    tgt = pflow_target_observables(arrays, n_events=15)
    # pt / eta are flat particle-level 1-D tensors.
    assert tgt["pt"].ndim == 1
    assert tgt["eta"].ndim == 1
    # multiplicity / ht are per-event, length 15.
    assert tgt["multiplicity"].shape == (15,)
    assert tgt["ht"].shape == (15,)
    for name, v in tgt.items():
        assert torch.isfinite(v).all(), f"non-finite in target '{name}'"


def test_trainee_observables_from_forward_pass(fixture_root: Path):
    """Running the learnable card and pulling observables out works and
    the kept particles all satisfy pt > min_pt."""
    arrays = load_cms_flow_root(fixture_root, n_events=12)
    t = truth_to_particle_tensor(arrays, n_events=12)
    torch.manual_seed(0)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    out = card(t)
    pred = trainee_observables(out, n_events=12)
    for v in pred.values():
        assert torch.isfinite(v).all()
    if pred["pt"].numel() > 0:
        assert float(pred["pt"].min()) > 0


def test_one_step_gradient_is_finite(fixture_root: Path):
    """A single forward + backward step produces finite gradients on
    every reachable learnable parameter."""
    arrays = load_cms_flow_root(fixture_root, n_events=20)
    t = truth_to_particle_tensor(arrays, n_events=20)
    target = pflow_target_observables(arrays, n_events=20)

    torch.manual_seed(7)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    out = card(t)
    pred = trainee_observables(out, n_events=20)
    loss = multi_observable_loss(pred, target, DEFAULT_BIN_EDGES, beta=0.15)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in card.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_fit_card_to_fullsim_runs(fixture_root: Path):
    """The fit loop runs through a handful of steps without errors and
    returns a history dict of the right shape. We do NOT assert
    convergence on the synthetic fixture because the loss is dominated
    by stochastic smearing/Gumbel noise at tiny batch sizes; the real
    convergence test is on the full Zenodo sample offline."""
    arrays = load_cms_flow_root(fixture_root, n_events=20)
    t = truth_to_particle_tensor(arrays, n_events=20)
    target = pflow_target_observables(arrays, n_events=20)

    torch.manual_seed(3)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    chad_scale = card.ChargedHadronMomentumSmearing.resolution_module.scale_raw  # type: ignore[union-attr]

    history = fit_card_to_fullsim(
        card,
        t,
        target,
        n_events=20,
        n_steps=5,
        lr=1e-1,
        n_passes_per_step=2,
        beta=0.2,
        log_every=0,
        parameters_to_train=[chad_scale],
    )
    assert len(history["step"]) == 5
    assert len(history["loss"]) == 5
    for loss in history["loss"]:
        assert loss == loss  # not NaN


# ---------------------------------------------------------------------------
# Real-Pythia pseudodata end-to-end test
# ---------------------------------------------------------------------------

PSEUDODATA_PATH = Path(__file__).parent / "benchmark_data" / "cms_pseudodata.root"


@pytest.mark.skipif(
    not PSEUDODATA_PATH.exists(),
    reason="committed pseudodata file not available",
)
def test_fit_against_committed_pseudodata():
    """End-to-end fit against the committed Pythia-generated pseudodata.

    The pseudodata file was generated by
    :mod:`parnassus.torch_delphes.generate_pseudodata` using Pythia8 QCD
    dijets and a deliberately-perturbed CMS card
    (charged-hadron pT scale 1.25, ECal energy scale 1.20). Fitting the
    ECal+chad scale parameters of a fresh learnable card should drive
    them meaningfully toward those perturbed values.

    To keep the test fast, we fit only the two scale parameter tensors
    for 20 Adam steps on a 150-event slice and assert that the
    ECal scale has moved substantially away from its default of 1.0
    toward the known-target 1.20. We do NOT assert full convergence --
    that requires longer runs and is covered by the CLI-based demo.
    """
    arrays = load_cms_flow_root(PSEUDODATA_PATH, n_events=150)
    t = truth_to_particle_tensor(arrays, n_events=150)
    target = pflow_target_observables(arrays, n_events=150)

    torch.manual_seed(11)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    chad_scale = card.ChargedHadronMomentumSmearing.resolution_module.scale_raw  # type: ignore[union-attr]
    ecal_scale = card.ECal.scale_module.scale_raw  # type: ignore[union-attr]
    params = [chad_scale, ecal_scale]

    before_ecal = (1.0 + 0.3 * torch.tanh(ecal_scale)).detach().clone()
    assert torch.allclose(before_ecal, torch.ones_like(before_ecal)), (
        "ECal scale was expected to start at 1.0 before the fit"
    )

    fit_card_to_fullsim(
        card,
        t,
        target,
        n_events=150,
        n_steps=20,
        lr=5e-2,
        n_passes_per_step=2,
        beta=0.15,
        log_every=0,
        parameters_to_train=params,
    )

    after_ecal = (1.0 + 0.3 * torch.tanh(ecal_scale)).detach().clone()
    # The pseudodata's ECal scale target is 1.20 in every region. After
    # 20 Adam steps on 150 events we expect the trainee to have moved
    # at least some regions a noticeable amount toward it -- at least
    # 5% of the way (absolute >= 0.01) in the region that sees the most
    # signal.
    max_shift = float((after_ecal - 1.0).abs().max())
    assert max_shift > 0.01, f"ECal scale barely moved from 1.0: after={after_ecal.tolist()}"
    # And none should have drifted the wrong way past 0.95 (which would
    # indicate divergence).
    assert float(after_ecal.min()) > 0.95, f"ECal scale drifted below 0.95: {after_ecal.tolist()}"

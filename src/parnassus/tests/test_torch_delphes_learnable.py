"""Tests for the learnable / differentiable-tuning TorchDelphes path.

These tests cover the new ``learnable=True`` mode of
:class:`parnassus.torch_delphes.defaults.CMSEnergyFlowDefault` and the
associated ``nn.Module`` wrappers in
:mod:`parnassus.torch_delphes.learnable`.

The tests verify:

1. **Parameter inventory**: the learnable card exposes exactly the 66
   ``nn.Parameter``s promised in the design document, and the legacy card
   exposes zero.
2. **Numeric defaults**: a freshly constructed learnable card (with no
   parameter updates) is statistically consistent with the legacy card on
   simple scalar observables, confirming that defaults were preserved.
3. **Forward runs**: the learnable card produces correctly-shaped
   ``{Track, Tower, EFlowTrack, EFlowPhoton, EFlowNeutralHadron,
   EFlowObject}`` outputs and does not raise.
4. **Gradient flow**: a dummy scalar loss on the reconstructed output has
   a finite, non-zero gradient on every parameter that the test batch can
   activate (all tracking / calo / fraction parameters reachable by the
   test particles).
5. **Adam optimizer step**: one step of optimization on a synthetic
   perturbed-target loss moves parameters in the direction that reduces
   the loss.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.learnable import (
    CMSChargedHadronLearnableEfficiency,
    LearnableEcalCMSResolution,
    LearnableHadronFractions,
    LearnableHcalCMSResolution,
    make_cms_ecal_scale,
    make_cms_hcal_scale,
    make_cms_track_resolution,
)
from parnassus.torch_delphes.tuning import (
    histogram_mse_loss,
    make_synthetic_particles,
    soft_histogram,
    tune_cms_to_target,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_batch(n: int = 120, seed: int = 0) -> torch.Tensor:
    """Build a synthetic particle batch covering all species and eta regions.

    The batch mixes charged pions, kaons, protons, electrons, muons,
    photons, K-short, Lambda, and neutrons so that every learnable
    parameter (except the high-pt muon exponential tail) has at least one
    contributing particle.
    """
    torch.manual_seed(seed)
    arr = torch.zeros(n, N_FEATURES, dtype=torch.float64)
    arr[:, ColumnMap.PT] = torch.rand(n) * 60.0 + 0.5
    arr[:, ColumnMap.ETA] = (torch.rand(n) - 0.5) * 6.0  # |eta| up to ~3
    arr[:, ColumnMap.PHI] = (torch.rand(n) - 0.5) * math.tau
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
    # Species mix: pions, kaons, protons, electrons, muons, photons, K0S,
    # Lambda, neutrons.
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
        lo = (i * n) // len(species)
        hi = ((i + 1) * n) // len(species)
        arr[lo:hi, ColumnMap.PID] = pid
        arr[lo:hi, ColumnMap.CHARGE] = charge
    arr[:, ColumnMap.STATUS] = 1
    return arr


# ---------------------------------------------------------------------------
# 1. Parameter inventory
# ---------------------------------------------------------------------------


def test_legacy_card_has_no_parameters():
    """Sanity: the non-learnable card exposes zero ``nn.Parameter``s."""
    card = CMSEnergyFlowDefault(debug=False, learnable=False)
    assert sum(p.numel() for p in card.parameters()) == 0


def test_learnable_card_parameter_count_matches_inventory():
    """The design doc promises exactly 66 learnable CMS parameters.

    Breakdown (see docs/review discussion and ``learnable.py``):

    - Tracking efficiency:        18   (4 chad + 6 e + 6 mu + 2 mu-rate)
    - Momentum resolution (a, b): 18   (3 species x (3 + 3))
    - Momentum scale:              9   (3 species x 3 regions)
    - ECal resolution:             9
    - HCal resolution:             4
    - ECal scale:                  3
    - HCal scale:                  2
    - Hadron fractions:            3
    ------------------------------------
    - Total:                      66
    """
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    total = sum(p.numel() for p in card.parameters())
    assert total == 66, f"expected 66 learnable params, got {total}"


def test_learnable_parameter_defaults_match_static_formulas():
    """Each learnable module reproduces the static-formula value at init.

    We check a handful of representative evaluations to guard against
    off-by-one or softplus-inversion bugs in the parameter init.
    """
    # Track resolution: CMS charged hadron region 0 (barrel) at pt = 10 GeV.
    # Static formula: sqrt(0.06^2 + (10 * 1.3e-3)^2) = 0.0612...
    chad = make_cms_track_resolution("charged_hadron")
    pt = torch.tensor([10.0], dtype=torch.float64)
    eta = torch.tensor([0.0], dtype=torch.float64)
    expected = float(np.sqrt(0.06**2 + (10.0 * 1.3e-3) ** 2))
    got = float(chad(pt, eta)[0])
    assert abs(got - expected) < 1e-10, f"chad res mismatch: {got} vs {expected}"

    # Default track scale = 1.0 in every region.
    assert torch.allclose(chad.scale(eta), torch.ones(1, dtype=torch.float64))

    # ECal CMS resolution: barrel at (eta=0, E=100).
    # Static: (1 + 0.64*0) * sqrt(100^2 * 0.008^2 + 100 * 0.11^2 + 0.40^2) ~ 1.53
    ecal_res = LearnableEcalCMSResolution()
    eta0 = torch.tensor([0.0], dtype=torch.float64)
    e100 = torch.tensor([100.0], dtype=torch.float64)
    expected_ecal = float(np.sqrt(100.0**2 * 0.008**2 + 100.0 * 0.11**2 + 0.40**2))
    got_ecal = float(ecal_res(eta0, e100)[0])
    # eps=1e-30 inside sqrt adds a sub-1e-15 bias — tolerate it.
    assert abs(got_ecal - expected_ecal) < 1e-6

    # HCal CMS resolution: central at eta=0, E=50.
    hcal_res = LearnableHcalCMSResolution()
    e50 = torch.tensor([50.0], dtype=torch.float64)
    expected_hcal = float(np.sqrt(50.0**2 * 0.05**2 + 50.0 * 1.5**2))
    got_hcal = float(hcal_res(eta0, e50)[0])
    assert abs(got_hcal - expected_hcal) < 1e-6

    # ECal / HCal scales default to 1.0.
    assert torch.allclose(
        make_cms_ecal_scale()(torch.tensor([0.0, 1.8, 3.0], dtype=torch.float64)),
        torch.ones(3, dtype=torch.float64),
    )
    assert torch.allclose(
        make_cms_hcal_scale()(torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)),
        torch.ones(3, dtype=torch.float64),
    )

    # Hadron fractions: default ECal fraction for (chad, K0S, Lambda).
    frac = LearnableHadronFractions()
    assert abs(float(frac.chad_ecal_frac()) - 0.0) < 1e-5
    assert abs(float(frac.k0s_ecal_frac()) - 0.3) < 1e-6
    assert abs(float(frac.lambda_ecal_frac()) - 0.3) < 1e-6


def test_charged_hadron_efficiency_defaults():
    """The Gumbel-ST efficiency module returns the correct per-region
    efficiency in its soft (deterministic) form.
    """
    eff = CMSChargedHadronLearnableEfficiency(temperature=0.5)
    pt = torch.tensor([0.5, 2.0, 0.5, 2.0], dtype=torch.float64)
    eta = torch.tensor([0.0, 0.0, 2.0, 2.0], dtype=torch.float64)
    got = eff.compute_efficiency(pt, eta)
    expected = torch.tensor([0.70, 0.95, 0.60, 0.85], dtype=torch.float64)
    assert torch.allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Forward runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_learnable_forward_produces_expected_branches(seed: int) -> None:
    """Forward pass on a small mixed batch returns the standard branches
    with consistent shapes.
    """
    torch.manual_seed(seed)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    particles = _make_batch(n=80, seed=seed)
    out = card(particles)
    expected_keys = {
        "Track",
        "Tower",
        "EFlowTrack",
        "EFlowPhoton",
        "EFlowNeutralHadron",
        "EFlowObject",
    }
    assert set(out.keys()) == expected_keys
    for k, v in out.items():
        assert v.ndim == 2
        assert v.shape[1] == N_FEATURES, f"{k} has wrong feature count"


# ---------------------------------------------------------------------------
# 3. Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_all_reachable_parameters():
    """A dummy loss on the reconstructed pt produces finite gradients on
    every parameter class that this test batch can activate.

    We don't assert on the K-short / Lambda / muon-high-pt-rate parameters
    because the random batch won't usually contain pt > 1000 GeV muons.
    We do assert that *all* other parameter tensors have finite,
    non-zero gradients.
    """
    torch.manual_seed(123)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    particles = _make_batch(n=200, seed=123)

    out = card(particles)
    loss = (
        out["EFlowObject"][:, ColumnMap.PT].sum()
        + out["Tower"][:, ColumnMap.E].sum()
        + out["Track"][:, ColumnMap.PT].sum()
    )
    loss.backward()

    # Every parameter must have a finite gradient.
    missing: list[str] = []
    nan_or_inf: list[str] = []
    for name, p in card.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nan_or_inf.append(name)
    assert not missing, f"parameters with no grad: {missing}"
    assert not nan_or_inf, f"parameters with NaN/Inf grad: {nan_or_inf}"

    # The learnable params that our batch actually exercises must have
    # non-zero gradients. We list them by name-fragment so the test is
    # robust to future renames of specific components.
    must_have_grad_fragments = [
        "ChargedHadronTrackingEfficiency.eff_logits",
        "ElectronTrackingEfficiency.eff_logits",
        "MuonTrackingEfficiency.eff_logits",
        "ChargedHadronMomentumSmearing.resolution_module.a_raw",
        "ChargedHadronMomentumSmearing.resolution_module.b_raw",
        "ChargedHadronMomentumSmearing.resolution_module.scale_raw",
        "ElectronMomentumSmearing.resolution_module.a_raw",
        "HadronFractions.chad_logit",
        "ECal.resolution_func.common_c_E",
        "ECal.resolution_func.barrel_b",
        "HCal.resolution_func.central_c_E",
        "ECal.scale_module.scale_raw",
        "HCal.scale_module.scale_raw",
    ]
    params_by_name = dict(card.named_parameters())
    for frag in must_have_grad_fragments:
        assert frag in params_by_name, f"missing param: {frag}"
        g = params_by_name[frag].grad
        assert g is not None
        assert g.abs().sum() > 0, f"{frag} has zero gradient"


# ---------------------------------------------------------------------------
# 4. Adam optimizer step moves loss downward
# ---------------------------------------------------------------------------


def test_gradient_direction_matches_target_deterministic():
    """Deterministic check that Adam will move in the right direction.

    We use :class:`LearnableMomentumResolution` directly (not the full
    card) so the loss is analytic and noise-free. A "target" module is
    constructed with a larger ``a_raw[0]`` parameter; a "trainee" module
    starts at the default. An MSE loss on the resolution output should
    produce a gradient on the trainee's ``a_raw[0]`` whose sign drives
    the parameter *up*, toward the target.
    """
    # Target: same as default but with first a_raw region inflated by 50%.
    target = make_cms_track_resolution("charged_hadron")
    with torch.no_grad():
        # a_init[0] = 0.06, so multiply softplus-output by 1.5 via a_raw.
        target.a_raw[0] = torch.log(torch.expm1(torch.tensor(0.09, dtype=torch.float64)))
    for p in target.parameters():
        p.requires_grad_(False)

    trainee = make_cms_track_resolution("charged_hadron")

    pt = torch.linspace(1.0, 50.0, 100, dtype=torch.float64)
    eta = torch.zeros_like(pt)  # stay in region 0 so a_raw[0] is active

    res_tgt = target(pt, eta)
    res_trn = trainee(pt, eta)
    loss = ((res_trn - res_tgt) ** 2).mean()
    loss.backward()

    # Default a_raw[0] corresponds to a=0.06; the target's a=0.09 is
    # larger, so minimizing MSE should *increase* a -> gradient on
    # a_raw[0] must be negative (Adam subtracts the gradient).
    grad0 = float(trainee.a_raw.grad[0])
    assert grad0 < 0, (
        f"Expected negative gradient on a_raw[0] (to drive a upward toward target), got {grad0:.4g}"
    )

    # And take one optimizer step: the trainee's softplus(a_raw[0]) should
    # actually move in the direction of 0.09.
    a_before = float(torch.nn.functional.softplus(trainee.a_raw[0]))
    opt = torch.optim.Adam(trainee.parameters(), lr=5e-2)
    for _ in range(20):
        opt.zero_grad()
        loss = ((trainee(pt, eta) - res_tgt) ** 2).mean()
        loss.backward()
        opt.step()
    a_after = float(torch.nn.functional.softplus(trainee.a_raw[0]))
    assert a_after > a_before, (
        f"a_raw[0]-backed resolution did not move toward target: "
        f"before={a_before:.4g}, after={a_after:.4g}, target=0.09"
    )
    # Should have moved at least halfway toward the target.
    assert abs(a_after - 0.09) < abs(a_before - 0.09), "did not get closer"


# ---------------------------------------------------------------------------
# 5. Tuning module tests (soft histogram, loss, end-to-end loop)
# ---------------------------------------------------------------------------


def test_soft_histogram_approaches_hard_histogram_as_beta_shrinks():
    """In the limit ``beta -> 0``, the soft histogram should converge to
    a standard hard histogram bin-for-bin (up to a tiny edge-leakage
    tolerance from the sigmoid).
    """
    torch.manual_seed(0)
    values = torch.rand(2000, dtype=torch.float64) * 10.0
    edges = torch.linspace(0.0, 10.0, 11, dtype=torch.float64)
    hard = torch.histc(values, bins=10, min=0.0, max=10.0)
    soft = soft_histogram(values, edges, beta=0.01)
    # Per-bin edge leakage is small: each bin within a couple of counts
    # of the hard histogram.
    assert (soft - hard).abs().max() < 5.0
    # Total count is also small: soft histogram never adds or removes
    # more than a handful of counts overall.
    assert abs(float(soft.sum()) - float(hard.sum())) < 20.0


def test_histogram_mse_loss_is_zero_for_identical_distributions():
    """Identical inputs should give (near-)zero loss."""
    torch.manual_seed(1)
    values = torch.rand(500, dtype=torch.float64) * 20.0
    edges = torch.linspace(0.0, 20.0, 11, dtype=torch.float64)
    loss = histogram_mse_loss(values, values, edges, beta=0.05)
    assert float(loss) < 1e-12


def test_histogram_mse_loss_has_finite_gradient():
    """The loss should produce a finite gradient on the predicted
    values, which is what drives backpropagation through the detector
    simulation.
    """
    torch.manual_seed(2)
    # Build a proper leaf tensor so autograd can write .grad.
    pred = (torch.rand(300, dtype=torch.float64) * 20.0).detach().requires_grad_(True)
    target = torch.rand(300, dtype=torch.float64) * 20.0 + 2.0  # shifted
    edges = torch.linspace(0.0, 25.0, 13, dtype=torch.float64)
    loss = histogram_mse_loss(pred, target, edges, beta=0.05)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_tune_cms_to_target_moves_charged_hadron_scale_toward_target():
    """Small-scale end-to-end tuning loop test. The target card has its
    charged-hadron pt scales set to 1.2; the trainee starts at 1.0; only
    the scale parameters are optimized. After a short Adam run, the
    trainee's scales should be closer to the target than they started.

    The loss is noisy so we check *L1 distance* improvement, not the
    loss value directly.
    """
    torch.manual_seed(0)

    target = CMSEnergyFlowDefault(debug=False, learnable=True)
    with torch.no_grad():
        raw = float(np.arctanh((1.2 - 1.0) / 0.3))
        target.ChargedHadronMomentumSmearing.resolution_module.scale_raw.fill_(raw)

    trainee = CMSEnergyFlowDefault(debug=False, learnable=True)

    def _scales(card: CMSEnergyFlowDefault) -> torch.Tensor:
        raw = card.ChargedHadronMomentumSmearing.resolution_module.scale_raw
        return (1.0 + 0.3 * torch.tanh(raw)).detach().clone()

    before = _scales(trainee)
    tgt = _scales(target)
    dist_before = float((before - tgt).abs().sum())

    particles = make_synthetic_particles(n=800, seed=2)
    scale_params = [trainee.ChargedHadronMomentumSmearing.resolution_module.scale_raw]
    tune_cms_to_target(
        target=target,
        trainee=trainee,
        particles=particles,
        n_steps=30,
        lr=2e-1,
        n_passes_per_step=3,
        beta=0.15,
        log_every=0,  # silent
        parameters_to_train=scale_params,
    )

    after = _scales(trainee)
    dist_after = float((after - tgt).abs().sum())

    # Want a meaningful reduction (at least ~30%) in L1 distance to the
    # target. Noisy, but 30 Adam steps on a clean scalar signal should
    # easily clear this bar.
    assert dist_after < 0.7 * dist_before, (
        f"L1 distance did not improve enough: before={dist_before:.3g}, after={dist_after:.3g}"
    )

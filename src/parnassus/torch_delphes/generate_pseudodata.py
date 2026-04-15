"""Generate CMS-full-sim-like pseudodata for the differentiable tuning harness.

This script produces a ROOT file in the cms-flow ``event_tree`` schema
(the same schema as Zenodo record 11389651) so that
``tune_cms_fullsim.py`` can be run end-to-end on reproducible input
without needing network access.

Event generation
----------------
1. Pythia8 generates hard QCD dijet events at ``sqrt(s) = 13 TeV`` with
   ``pTHatMin`` set by ``--pt-hat-min``. Hadronization is on; final-state
   radiation is on. This produces a realistic mix of charged hadrons,
   electrons, muons, photons, K-shorts, Lambdas, and neutrons -- the
   exact particle zoo that exercises every learnable parameter in
   ``CMSEnergyFlowDefault``.

2. Each event's stable particles are converted to the
   ``(N, N_FEATURES)`` particle tensor via
   :func:`parnassus.data.particle_io.pythia_particles_to_tensor`,
   with ``EVENT_NUMBER`` set to the (0-indexed) global event number so
   the calorimeter's per-event aggregation works correctly.

3. The stable particles are run through a
   :class:`CMSEnergyFlowDefault` card to produce a CMS-PF-like reco
   object collection. Two cards are used:

   - A **default** (non-learnable) card which gives us the truth
     particles themselves as the ``truth_*`` branches.
   - A **perturbed learnable** card which plays the role of "CMS full
     simulation" for the fitting target: its
     charged-hadron momentum scale is set to 1.25, its ECal energy
     scale is set to 1.20. This is exactly the ground-truth
     perturbation that Adam should recover when we fit the trainee
     against the ``pflow_*`` branches.

Output schema
-------------
The output ROOT file has a TTree named ``event_tree`` with jagged
per-event branches:

- ``truth_pt, truth_eta, truth_phi, truth_class``
- ``pflow_pt, pflow_eta, pflow_phi, pflow_class``

Class values follow :func:`parnassus.utils.pid_to_class` (0=charged
hadron, 1=electron, 2=muon, 3=neutral hadron, 4=photon), matching the
cms-flow loader.

Size targeting
--------------
The ``--target-size-mb`` flag controls how many events are generated.
The script measures the current file size after every 200 events and
stops once the target size is reached, so the output is always at
roughly the requested size regardless of event multiplicity.

Usage
-----
.. code-block:: shell

    # Generate ~20 MB of pseudodata at pT-hat > 100 GeV
    uv run python -m parnassus.torch_delphes.generate_pseudodata \
        --output src/parnassus/tests/benchmark_data/cms_pseudodata.root \
        --target-size-mb 20 \
        --pt-hat-min 100

By default, the output lives in ``src/parnassus/tests/benchmark_data/``
so it can be committed and consumed by
``test_tune_cms_fullsim_real_fit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pythia8mc
import torch
import uproot

from parnassus.data.particle_io import ColumnMap, pythia_particles_to_tensor
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.utils import pid_to_class

# Perturbations applied to the "target" (fake full-sim) card. These are the
# numbers we want Adam to recover when we run tune_cms_fullsim against the
# output of this script. The perturbation spans multiple *types* of
# parameters -- scales, resolutions, efficiencies, and fractions -- so that
# the harness is tested on a genuinely multi-knob identifiability problem
# rather than a single trivial offset.
TARGET_CHAD_SCALE: float = 1.25  # all 3 eta regions
TARGET_ECAL_SCALE: float = 1.20  # all 3 eta regions
TARGET_HCAL_SCALE: float = 0.90  # all 2 eta regions (a *downward* shift)

# Multiply the charged-hadron resolution constant-term `a` in the barrel
# region by this factor, so default 0.06 becomes 0.09 (50% worse barrel
# tracking resolution). Gives the fit a nonzero resolution-parameter target.
TARGET_CHAD_RES_A_BARREL_FACTOR: float = 1.5

# Shift the charged-hadron tracking efficiency in the barrel low-pT bin
# from the default 0.70 down to this value. Exercises the Gumbel-ST
# efficiency parameter gradient.
TARGET_CHAD_EFF_BARREL_LOWPT: float = 0.60

# Shift the K0S ECal fraction from the default 0.30 to this value. The
# fit's HadronFractions param should move toward it.
TARGET_K0S_ECAL_FRAC: float = 0.50


def build_pythia(pt_hat_min: float, seed: int) -> object:
    """Configure and initialise a Pythia8 instance for QCD dijets.

    Returns
    -------
    pythia8mc.Pythia
        An initialised Pythia8 instance ready for :meth:`next` calls.
    """
    pythia = pythia8mc.Pythia()
    settings = [
        "Beams:eCM = 13000.",
        "HardQCD:all = on",
        f"PhaseSpace:pTHatMin = {pt_hat_min}",
        f"Random:seed = {seed}",
        "Random:setSeed = on",
        "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0",
        "Next:numberShowEvent = 0",
        "Init:showProcesses = off",
        "Init:showMultipartonInteractions = off",
        "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off",
        "Stat:showProcessLevel = off",
        "Stat:showPartonLevel = off",
        "Stat:showErrors = off",
    ]
    for setting in settings:
        pythia.readString(setting)
    pythia.init()
    return pythia


def make_target_card(seed: int = 0) -> CMSEnergyFlowDefault:
    """Build a learnable CMS card with a *multi-knob* perturbation.

    Six distinct parameter types are perturbed, giving the fitting
    harness a non-trivial multi-knob identifiability problem:

    - ``chad`` track-pT scale: 1.0 -> :data:`TARGET_CHAD_SCALE` (all 3 eta regions)
    - ECal energy scale:       1.0 -> :data:`TARGET_ECAL_SCALE` (all 3 regions)
    - HCal energy scale:       1.0 -> :data:`TARGET_HCAL_SCALE` (both regions;
      note this is a *downward* shift, testing bidirectional gradients)
    - Chad momentum resolution barrel constant ``a``:
      default 0.06 -> 0.06 * :data:`TARGET_CHAD_RES_A_BARREL_FACTOR`
    - Chad tracking efficiency in the barrel low-pT bin:
      default 0.70 -> :data:`TARGET_CHAD_EFF_BARREL_LOWPT`
    - K0-short ECal/HCal fraction split (learnable hadron-fraction knob):
      default 0.30/0.70 -> :data:`TARGET_K0S_ECAL_FRAC`/(1-..)

    Returns
    -------
    CMSEnergyFlowDefault
        A frozen learnable card whose parameters match the published
        target values.
    """
    torch.manual_seed(seed)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)

    def raw_for_scale(s: float) -> float:
        """Inverse of the ``1 + 0.3 tanh(raw)`` scale parameterization.

        Returns
        -------
        float
            The raw parameter value whose post-transform is ``s``.
        """
        return float(np.arctanh((s - 1.0) / 0.3))

    def softplus_inv(x: float) -> float:
        """Inverse of ``softplus``, matching ``learnable._softplus_inv``.

        Returns
        -------
        float
            Raw scalar ``y`` satisfying ``softplus(y) == max(x, 1e-12)``.
        """
        return float(np.log(np.expm1(max(x, 1e-12))))

    def logit(p: float) -> float:
        """Logit of a probability, clipped away from the open interval ends.

        Returns
        -------
        float
            ``log(p_clipped / (1 - p_clipped))`` with
            ``p_clipped = clip(p, 1e-6, 1 - 1e-6)``.
        """
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    with torch.no_grad():
        # --- Scales ---
        chad_res = card.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
        chad_res.scale_raw.fill_(raw_for_scale(TARGET_CHAD_SCALE))

        ecal_scale = card.ECal.scale_module  # type: ignore[union-attr]
        ecal_scale.scale_raw.fill_(raw_for_scale(TARGET_ECAL_SCALE))

        hcal_scale = card.HCal.scale_module  # type: ignore[union-attr]
        hcal_scale.scale_raw.fill_(raw_for_scale(TARGET_HCAL_SCALE))

        # --- Resolution: barrel ``a`` for charged hadrons ---
        # a[0] (barrel) default is 0.06; lift it via softplus_inv.
        chad_res.a_raw[0] = softplus_inv(0.06 * TARGET_CHAD_RES_A_BARREL_FACTOR)

        # --- Efficiency: charged-hadron barrel low-pt logit ---
        chad_eff = card.ChargedHadronTrackingEfficiency  # type: ignore[union-attr]
        chad_eff.eff_logits[0] = logit(TARGET_CHAD_EFF_BARREL_LOWPT)

        # --- Hadron fraction: K0-short ECal split ---
        card.HadronFractions.k0s_logit.fill_(logit(TARGET_K0S_ECAL_FRAC))  # type: ignore[union-attr]

    for p in card.parameters():
        p.requires_grad_(False)
    card.eval()
    return card


def eflow_to_class_arrays(
    eflow: torch.Tensor,
    event_numbers: torch.Tensor,
    n_events: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split an EFlowObject tensor into per-event jagged arrays.

    Filters out zero-pT Gumbel-ST ghost tracks and converts PID -> class
    via :func:`parnassus.utils.pid_to_class` (0=chad, 1=e, 2=mu,
    3=neutral hadron, 4=photon).

    Returns
    -------
    tuple
        Four per-event jagged lists: pt, eta, phi, class.
    """
    pt_np = eflow[:, ColumnMap.PT].cpu().numpy().astype(np.float32)
    eta_np = eflow[:, ColumnMap.ETA].cpu().numpy().astype(np.float32)
    phi_np = eflow[:, ColumnMap.PHI].cpu().numpy().astype(np.float32)
    pid_np = eflow[:, ColumnMap.PID].cpu().numpy().astype(np.int64)
    ev_np = event_numbers.cpu().numpy().astype(np.int64)

    keep = pt_np > 1e-6
    pt_np = pt_np[keep]
    eta_np = eta_np[keep]
    phi_np = phi_np[keep]
    pid_np = pid_np[keep]
    ev_np = ev_np[keep]

    # Vectorize pid_to_class via a small lookup over |PID| mapped to
    # {11, 13, 22} -> {1, 2, 4} with charged/neutral hadrons filled in
    # by sign-of-charge considerations via pid_to_class itself.
    cls_np = np.fromiter(
        (pid_to_class(int(p)) for p in pid_np),
        dtype=np.int32,
        count=pid_np.shape[0],
    )

    pt_list = [pt_np[ev_np == i] for i in range(n_events)]
    eta_list = [eta_np[ev_np == i] for i in range(n_events)]
    phi_list = [phi_np[ev_np == i] for i in range(n_events)]
    cls_list = [cls_np[ev_np == i] for i in range(n_events)]
    return pt_list, eta_list, phi_list, cls_list


def truth_tensor_to_class_arrays(
    truth: torch.Tensor,
    n_events: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split a per-particle truth tensor into per-event jagged arrays.

    The tensor is the direct output of ``pythia_particles_to_tensor``,
    so it contains EVERY particle in the event (including intermediate
    / unstable ones). We filter to final-state particles
    (``STATUS == 1``) and drop neutrinos.

    Returns
    -------
    tuple
        Four per-event jagged lists: pt, eta, phi, class.
    """
    status = truth[:, ColumnMap.STATUS].cpu().numpy().astype(np.int64)
    pid = truth[:, ColumnMap.PID].cpu().numpy().astype(np.int64)
    abs_pid = np.abs(pid)
    # Keep status==1 and exclude electron/mu/tau neutrinos.
    keep = (status == 1) & ~np.isin(abs_pid, [12, 14, 16])

    pt_np = truth[:, ColumnMap.PT].cpu().numpy().astype(np.float32)[keep]
    eta_np = truth[:, ColumnMap.ETA].cpu().numpy().astype(np.float32)[keep]
    phi_np = truth[:, ColumnMap.PHI].cpu().numpy().astype(np.float32)[keep]
    pid_np = pid[keep]
    ev_np = truth[:, ColumnMap.EVENT_NUMBER].cpu().numpy().astype(np.int64)[keep]

    # Drop any particle with |eta| beyond something absurd (Pythia emits
    # ±999.9 for pT=0 particles). Those confuse the calorimeter and our
    # binning, and they carry no physics information.
    finite_eta = np.abs(eta_np) < 10.0
    pt_np = pt_np[finite_eta]
    eta_np = eta_np[finite_eta]
    phi_np = phi_np[finite_eta]
    pid_np = pid_np[finite_eta]
    ev_np = ev_np[finite_eta]

    cls_np = np.fromiter(
        (pid_to_class(int(p)) for p in pid_np),
        dtype=np.int32,
        count=pid_np.shape[0],
    )

    pt_list = [pt_np[ev_np == i] for i in range(n_events)]
    eta_list = [eta_np[ev_np == i] for i in range(n_events)]
    phi_list = [phi_np[ev_np == i] for i in range(n_events)]
    cls_list = [cls_np[ev_np == i] for i in range(n_events)]
    return pt_list, eta_list, phi_list, cls_list


def _flush_batch(
    pythia: object,
    target_card: CMSEnergyFlowDefault,
    start_idx: int,
    batch_size: int,
) -> tuple[dict[str, list[np.ndarray]], int]:
    """Generate one batch of events and return cms-flow-schema lists.

    Runs ``batch_size`` events through Pythia, converts them to a truth
    tensor, passes them through the perturbed target card, and splits
    both sides into per-event jagged arrays.

    Returns
    -------
    branches : dict
        Mapping from branch name (e.g. ``"truth_pt"``) to a list of
        per-event numpy arrays for the batch.
    n_generated : int
        Actual number of events generated (may be smaller than
        ``batch_size`` if Pythia skipped events).
    """
    del start_idx  # event numbers are shifted to [0, n_generated) locally
    batch_truth_rows: list[torch.Tensor] = []
    n_generated = 0
    for _ in range(batch_size):
        if not pythia.next():  # type: ignore[attr-defined]
            continue
        # Use a *local* event number in [0, n_generated) so the helpers
        # below can split jagged arrays with a straightforward
        # ``ev == i`` comparison. The absolute event index across
        # batches is tracked by the caller.
        row = pythia_particles_to_tensor(pythia.event, n_generated, dtype=torch.float64)  # type: ignore[attr-defined]
        batch_truth_rows.append(row)
        n_generated += 1

    if n_generated == 0:
        return {}, 0

    # Flatten all events in the batch into a single tensor for card.forward.
    batch_truth = torch.cat(batch_truth_rows, dim=0)

    # Only the final-state, finite-eta particles reach the detector; this
    # mirrors what the truth_tensor_to_class_arrays helper keeps for the
    # truth_* branches.
    status = batch_truth[:, ColumnMap.STATUS]
    pid = batch_truth[:, ColumnMap.PID].abs()
    eta = batch_truth[:, ColumnMap.ETA].abs()
    detector_mask = (
        (status == 1)
        & ~torch.isin(pid, torch.tensor([12.0, 14.0, 16.0], dtype=pid.dtype))
        & (eta < 10.0)
    )
    detector_input = batch_truth[detector_mask]

    with torch.no_grad():
        out = target_card(detector_input)
    eflow = out["EFlowObject"]

    truth_pt, truth_eta, truth_phi, truth_class = truth_tensor_to_class_arrays(
        batch_truth, n_events=n_generated
    )
    pflow_pt, pflow_eta, pflow_phi, pflow_class = eflow_to_class_arrays(
        eflow, eflow[:, ColumnMap.EVENT_NUMBER], n_events=n_generated
    )

    return (
        {
            "truth_pt": truth_pt,
            "truth_eta": truth_eta,
            "truth_phi": truth_phi,
            "truth_class": truth_class,
            "pflow_pt": pflow_pt,
            "pflow_eta": pflow_eta,
            "pflow_phi": pflow_phi,
            "pflow_class": pflow_class,
        },
        n_generated,
    )


def generate(
    output_path: Path,
    target_size_mb: float = 20.0,
    pt_hat_min: float = 100.0,
    batch_size: int = 200,
    max_events: int = 100_000,
    seed: int = 1,
) -> int:
    """Generate the pseudodataset and write it to ``output_path``.

    Returns
    -------
    int
        Number of events actually written to the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pythia = build_pythia(pt_hat_min=pt_hat_min, seed=seed)
    target_card = make_target_card(seed=seed)

    accumulated: dict[str, list[np.ndarray]] = {
        b: []
        for b in (
            "truth_pt",
            "truth_eta",
            "truth_phi",
            "truth_class",
            "pflow_pt",
            "pflow_eta",
            "pflow_phi",
            "pflow_class",
        )
    }

    total_written = 0
    while total_written < max_events:
        batch, n_gen = _flush_batch(pythia, target_card, total_written, batch_size)
        if n_gen == 0:
            break
        for key, val in batch.items():
            accumulated[key].extend(val)
        total_written += n_gen

        # Write a checkpoint so we can measure the file size on disk.
        with uproot.recreate(str(output_path)) as f:
            f["event_tree"] = {k: ak.Array(v) for k, v in accumulated.items()}
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(
            f"  generated {total_written} events, "
            f"file size = {size_mb:.2f} MB "
            f"(target {target_size_mb:.1f} MB)"
        )
        if size_mb >= target_size_mb:
            break

    return total_written


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.generate_pseudodata``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/parnassus/tests/benchmark_data/cms_pseudodata.root"),
    )
    parser.add_argument("--target-size-mb", type=float, default=20.0)
    parser.add_argument("--pt-hat-min", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-events", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(
        f"Generating pseudodata -> {args.output}\n"
        f"  pTHatMin={args.pt_hat_min}  target={args.target_size_mb} MB  seed={args.seed}"
    )
    n = generate(
        args.output,
        target_size_mb=args.target_size_mb,
        pt_hat_min=args.pt_hat_min,
        batch_size=args.batch_size,
        max_events=args.max_events,
        seed=args.seed,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {n} events, {size_mb:.2f} MB, to {args.output}")


if __name__ == "__main__":
    main()

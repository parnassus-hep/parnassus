import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# HepMC3 Python bindings
import pyhepmc
from joblib import Parallel, delayed

# FastJet (via pyjet)
from pyjet import DTYPE_PTEPM, cluster
from tqdm import tqdm

# DUT
from parnassus.pythia import HepMC3Generator

# Helper functions #############################
MZ = 91.1876
GAMMA_Z = 2.4952


def draw(event_dict, key, bins=60, range_=None, xlabel=None, fig_dir="figures/HZZ4l"):
    x = event_dict.get(key, [])
    print(f"{key}: {len(x)} entries")
    if len(x) == 0:
        print(f"[warn] no data for {key}")
        return
    plt.figure()
    plt.hist(x, bins=bins, range=range_, histtype="step")
    plt.xlabel(xlabel or key)
    plt.ylabel("Events")
    plt.title(key)
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / f"{key}.png")
    plt.close()


# ---------- Kinematics ----------
def pt(px, py):
    return math.hypot(px, py)


def eta(px, py, pz):
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return float("inf") if pz >= 0 else -float("inf")
    return 0.5 * math.log((p + pz) / (p - pz))


def phi(px, py):
    return math.atan2(py, px)


def inv_mass(p4s):
    E = sum(p[3] for p in p4s)
    px = sum(p[0] for p in p4s)
    py = sum(p[1] for p in p4s)
    pz = sum(p[2] for p in p4s)
    m2 = E * E - (px * px + py * py + pz * pz)
    return math.sqrt(max(m2, 0.0))


def deltaR(a, b):
    # a,b are (px,py,pz,E,eta,phi)
    dphi = (a[5] - b[5] + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(a[4] - b[4], dphi)


def as_p4(px, py, pz, E):
    return (px, py, pz, E, eta(px, py, pz), phi(px, py))


# ---------- Object ID ----------
NEUTRINOS = {12, 14, 16, -12, -14, -16}


def is_final(particle):
    # HepMC3 "status==1" is final state for Pythia
    return particle.status == 1


def is_lep(particle):
    return is_final(particle) and abs(particle.pid) in {11, 13}


def is_neutrino(particle):
    return is_final(particle) and particle.pid in NEUTRINOS


# ---------- Pairing (global chi^2 with BW scale) ----------
def best_ossf_pairs_min_chi2(leps_p4, leps_id):
    """leps_p4: list of 4-vectors (px,py,pz,E,eta,phi) for 4 leptons (selected & sorted)
    leps_id: matching PDG IDs (11,-11,13,-13)
    Returns: ((i,j),(k,l), m1, m2) with the pair closer to mZ listed first.
    """
    idx = [0, 1, 2, 3]
    best = None

    def mll(a, b):
        return inv_mass([
            (leps_p4[a][0], leps_p4[a][1], leps_p4[a][2], leps_p4[a][3]),
            (leps_p4[b][0], leps_p4[b][1], leps_p4[b][2], leps_p4[b][3]),
        ])

    for i in idx:
        for j in idx:
            if j <= i:
                continue
            if leps_id[i] != -leps_id[j]:  # OSSF
                continue
            rem = [k for k in idx if k not in {i, j}]
            if leps_id[rem[0]] != -leps_id[rem[1]]:
                continue
            mA, mB = mll(i, j), mll(rem[0], rem[1])
            # chi^2 with BW width as scale
            chi2 = ((mA - MZ) ** 2 + (mB - MZ) ** 2) / (GAMMA_Z**2)
            if abs(mB - MZ) < abs(mA - MZ):
                cand = (chi2, (rem[0], rem[1]), (i, j), mB, mA)
            else:
                cand = (chi2, (i, j), (rem[0], rem[1]), mA, mB)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        return None
    _, p1, p2, m1, m2 = best
    return p1, p2, m1, m2


# ---------- Jet clustering ----------
def build_jet_inputs(final_particles, exclude_p4, R=0.4, ptmin=30.0, etamax=4.5):
    """final_particles: iterable of HepMC particles (status==1)
    exclude_p4: list of lepton p4 to be excluded from clustering (ΔR>0.4)
    Returns list of jets, each as (px,py,pz,E,eta,phi)
    """
    # Keep visible particles: drop neutrinos; keep photons/hadrons
    cand = []
    for p in final_particles:
        if not is_final(p):
            continue
        if is_neutrino(p):  # drop nu, nubar
            continue
        p4 = as_p4(p.momentum.px, p.momentum.py, p.momentum.pz, p.momentum.e)
        # Overlap removal: don't feed the selected leptons to the jet finder
        if any(deltaR(p4, lp4) < R for lp4 in exclude_p4):
            continue
        # Feed all remaining to clustering; pT/eta cuts applied after clustering
        pt_i = pt(p4[0], p4[1])
        eta_i = p4[4]
        phi_i = p4[5]
        m_i = inv_mass([(p4[0], p4[1], p4[2], p4[3])])  # should be 0 or very small
        cand.append((pt_i, eta_i, phi_i, m_i))

    if not cand:
        return []

    a = np.array(cand, dtype=DTYPE_PTEPM)  # fields: px, py, pz, E
    sequence = cluster(a, R=R, p=-1)  # anti-kt (p=-1)
    jets = sequence.inclusive_jets()  # list of PseudoJets

    out = []
    for j in jets:
        px, py, pz, E = j.px, j.py, j.pz, j.e
        p4 = as_p4(px, py, pz, E)
        if pt(px, py) < ptmin:
            continue
        if abs(p4[4]) > etamax:
            continue
        out.append(p4)
    # sort by pT desc
    out.sort(key=lambda v: pt(v[0], v[1]), reverse=True)
    return out


def inspect_hepmc(fpath_hepmc, args):
    args = parse_args()

    plot_vars = [
        "m_4l",
        "mZ1",
        "mZ2",
        "j1_pt",
        "j2_pt",
        "j1_eta",
        "j2_eta",
        "mjj",
        "detajj",
        "phijj",
        "zeppenfeld_eta_star",
        "l1_pt",
        "l2_pt",
        "l3_pt",
        "l4_pt",
        "l1_eta",
        "l2_eta",
        "l3_eta",
        "l4_eta",
        "min_DR_ll",
    ]
    EVT_DICT = {k: np.full((args.max_events,), np.nan) for k in plot_vars}

    cut_vars = ["fail_nlep>=4", "fail_njets>=2", "fail_pairing", "fail_mZ2_window"]
    COUNTS = {k: np.zeros((args.max_events,)) for k in cut_vars}
    N_EVENTS = np.zeros((args.max_events,))

    def process_event(event_idx, event):
        N_EVENTS[event_idx] = 1
        final_parts = [p for p in event.particles if is_final(p)]

        # --- leptons (e, mu), baseline selection ---
        leptons = []
        for p in final_parts:
            if not is_lep(p):
                continue
            px, py, pz, E = p.momentum.px, p.momentum.py, p.momentum.pz, p.momentum.e
            p4 = as_p4(px, py, pz, E)
            if pt(px, py) < args.lep_pt:
                continue
            if abs(p4[4]) > args.lep_eta:
                continue
            leptons.append((p.pid, p4))

        if len(leptons) < 4:
            COUNTS["fail_nlep>=4"][event_idx] = 1
            return

        # sort leptons by pT
        leptons.sort(key=lambda t: pt(t[1][0], t[1][1]), reverse=True)
        # take 4 leading (analysis usually requires exactly 4 isolated; here we keep the top 4)
        leptons = leptons[:4]
        lep_ids = [x[0] for x in leptons]
        lep_p4s = [x[1] for x in leptons]

        # Pairing
        pair = best_ossf_pairs_min_chi2(lep_p4s, lep_ids)
        if pair is None:
            COUNTS["fail_pairing"][event_idx] = 1
            return
        _, _, mZ1, mZ2 = pair
        # Off-shell threshold (optional, classic 12 GeV)
        if mZ2 < args.min_mll2:
            COUNTS["fail_mZ2_window"][event_idx] = 1
            return

        # Higgs 4l
        p4_4l = tuple(np.sum(np.array([(p[0], p[1], p[2], p[3]) for p in lep_p4s]), axis=0))
        m_4l = inv_mass([(p[0], p[1], p[2], p[3]) for p in lep_p4s])

        # --- jets via anti-kt on visible final state, excluding the leptons ---
        jets = build_jet_inputs(
            final_parts, exclude_p4=lep_p4s, R=args.R, ptmin=args.jet_pt, etamax=args.jet_eta
        )
        if len(jets) < 2:
            COUNTS["fail_njets>=2"][event_idx] = 1
            # still fill purely leptonic histos
            EVT_DICT["m_4l"][event_idx] = m_4l
            EVT_DICT["mZ1"][event_idx] = mZ1
            EVT_DICT["mZ2"][event_idx] = mZ2
            for idx, p4 in enumerate(lep_p4s):
                EVT_DICT[f"l{idx + 1}_pt"][event_idx] = pt(p4[0], p4[1])
                EVT_DICT[f"l{idx + 1}_eta"][event_idx] = p4[4]
            EVT_DICT["min_DR_ll"][event_idx] = min(
                deltaR(lep_p4s[a], lep_p4s[b]) for a in range(4) for b in range(a + 1, 4)
            )
            return

        # Leading jets and VBF variables
        j1, j2 = jets[0], jets[1]
        mjj = inv_mass([(j1[0], j1[1], j1[2], j1[3]), (j2[0], j2[1], j2[2], j2[3])])
        detajj = abs(j1[4] - j2[4])
        dphijj = (j1[5] - j2[5] + math.pi) % (2 * math.pi) - math.pi
        etaH = eta(p4_4l[0], p4_4l[1], p4_4l[2])  # could use rapidity; eta is fine here
        zepp = etaH - 0.5 * (j1[4] + j2[4])

        # Fill histos
        EVT_DICT["m_4l"][event_idx] = m_4l
        EVT_DICT["mZ1"][event_idx] = mZ1
        EVT_DICT["mZ2"][event_idx] = mZ2
        EVT_DICT["j1_pt"][event_idx] = pt(j1[0], j1[1])
        EVT_DICT["j2_pt"][event_idx] = pt(j2[0], j2[1])
        EVT_DICT["j1_eta"][event_idx] = j1[4]
        EVT_DICT["j2_eta"][event_idx] = j2[4]
        EVT_DICT["mjj"][event_idx] = mjj
        EVT_DICT["detajj"][event_idx] = detajj
        EVT_DICT["phijj"][event_idx] = dphijj
        EVT_DICT["zeppenfeld_eta_star"][event_idx] = zepp
        for idx, p4 in enumerate(lep_p4s):
            EVT_DICT[f"l{idx + 1}_pt"][event_idx] = pt(p4[0], p4[1])
            EVT_DICT[f"l{idx + 1}_eta"][event_idx] = p4[4]
        EVT_DICT["min_DR_ll"][event_idx] = min(
            deltaR(lep_p4s[a], lep_p4s[b]) for a in range(4) for b in range(a + 1, 4)
        )

    # Reader handles plain or gz
    with pyhepmc.open(fpath_hepmc, "r") as f:
        if args.debug:
            subset = itertools.islice(enumerate(f), 0, 1000)
            Parallel(n_jobs=args.n_jobs, prefer="threads")(
                itertools.starmap(delayed(process_event), tqdm(subset, total=1000))
            )
        else:
            Parallel(n_jobs=args.n_jobs, prefer="threads")(
                itertools.starmap(delayed(process_event), tqdm(enumerate(f), total=args.max_events))
            )

    # Aggregate results
    for k in EVT_DICT:
        EVT_DICT[k] = EVT_DICT[k][~np.isnan(EVT_DICT[k])]  # trim to filled
    for k, v in COUNTS.items():
        COUNTS[k] = int(np.sum(v))
    N_EVENTS = int(np.sum(N_EVENTS))

    # ---------- Summary ----------
    print(f"Events total: {N_EVENTS}")
    n_skipped = 0
    for k, v in COUNTS.items():
        print(f"{k}: {v}")
        n_skipped += v
    print(f"\n\nEvents selected: {N_EVENTS - n_skipped}\n\n")

    fig_dir = "figures/HZZ4l"
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    # ---------- Plots ----------
    # Core Higgs/ZZ
    draw(EVT_DICT, "m_4l", bins=60, range_=(50, 200), xlabel="m(4ℓ) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "mZ1", bins=60, range_=(40, 120), xlabel="m(Z1) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "mZ2", bins=60, range_=(12, 120), xlabel="m(Z2) [GeV]", fig_dir=fig_dir)

    # Jets / VBF
    draw(EVT_DICT, "j1_pt", bins=60, range_=(0, 300), xlabel="pT(j1) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "j2_pt", bins=60, range_=(0, 200), xlabel="pT(j2) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "mjj", bins=60, range_=(0, 3000), xlabel="m(jj) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "detajj", bins=60, range_=(0, 10), xlabel="Δη(jj)", fig_dir=fig_dir)
    draw(EVT_DICT, "phijj", bins=60, range_=(-math.pi, math.pi), xlabel="Δφ(jj)", fig_dir=fig_dir)
    draw(EVT_DICT, "zeppenfeld_eta_star", bins=60, range_=(-5, 5), xlabel="η*(H)", fig_dir=fig_dir)

    # Leptons
    draw(EVT_DICT, "l1_pt", bins=60, range_=(0, 150), xlabel="pT(ℓ1) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "l2_pt", bins=60, range_=(0, 100), xlabel="pT(ℓ2) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "l3_pt", bins=60, range_=(0, 60), xlabel="pT(ℓ3) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "l4_pt", bins=60, range_=(0, 40), xlabel="pT(ℓ4) [GeV]", fig_dir=fig_dir)
    draw(EVT_DICT, "min_DR_ll", bins=60, range_=(0, 5), xlabel="min ΔR(ℓ,ℓ)", fig_dir=fig_dir)


# Testing function #################################
def parse_args():
    ap = argparse.ArgumentParser(description="Inspect HepMC H->ZZ->4l VBF-like events")
    ap.add_argument("--R", type=float, default=0.4, help="anti-kt jet radius")
    ap.add_argument("--jet-pt", type=float, default=30.0, help="jet pT min [GeV]")
    ap.add_argument("--jet-eta", type=float, default=4.5, help="|eta| max for jets")
    ap.add_argument("--lep-pt", type=float, default=10.0, help="lepton pT min [GeV]")
    ap.add_argument("--lep-eta", type=float, default=2.5, help="|eta| max for leptons")
    ap.add_argument(
        "--min-mll2", type=float, default=12.0, help="min mass for off-shell Z window [GeV]"
    )
    ap.add_argument("--max-events", type=int, default=100_000, help="max events to process")
    ap.add_argument("--n-jobs", type=int, default=32, help="number of parallel jobs")
    ap.add_argument("--debug", action="store_true", help="debug mode (only 1000 events)")
    return ap.parse_args()


def main():
    args = parse_args()
    generator = HepMC3Generator(
        cmnd_file="HZZ4l.cmnd", output_dir="data_out/HZZ4l", log_dir="logs/HZZ4l"
    )
    fpath_merged = generator.generate(n_events=args.max_events, max_workers=args.n_jobs)
    print(f"Wrote to file {fpath_merged}")

    print("Inspecting hepmc file and generating histograms")
    args = parse_args()
    inspect_hepmc(fpath_merged, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Fig. 3 for the PRD paper: SM vs BSM discriminating distributions.

Produces a 4-panel figure comparing H→ZZ→4l (SM) and H→aa→ggmumu (BSM)
at truth level using Pythia8 + FastJet. Panels:
  (a) Dijet invariant mass m_jj
  (b) Jet multiplicity
  (c) Leading jet p_T
  (d) H_T (scalar sum of all particle p_T)

Usage:
    python make_process_comparison.py [--nevents 1000] [-o figures/process_comparison_sm_bsm.pdf]
"""

from __future__ import annotations

import argparse

import fastjet
import matplotlib.pyplot as plt
import numpy as np
import pythia8mc as pythia8


def generate_events(pythia_settings: list[str], nevents: int):
    """Generate events with Pythia8 and cluster jets."""
    py = pythia8.Pythia("", False)
    for s in pythia_settings:
        py.readString(s)
    py.init()

    all_jet_pts = []
    all_jet_multiplicities = []
    all_mjj = []
    all_ht = []
    all_leading_jet_pt = []

    for _ in range(nevents):
        if not py.next():
            continue

        pts, etas, phis, masses = [], [], [], []
        ht = 0.0
        for i in range(py.event.size()):
            p = py.event[i]
            if not p.isFinal():
                continue
            if abs(p.eta()) > 5.0:
                continue
            pts.append(p.pT())
            etas.append(p.eta())
            phis.append(p.phi())
            masses.append(p.m())
            ht += p.pT()

        all_ht.append(ht)

        pseudojets = []
        for pt, eta, phi, m in zip(pts, etas, phis, masses):
            pj = fastjet.PseudoJet()
            pj.reset_PtYPhiM(pt, eta, phi, m)
            pseudojets.append(pj)

        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.5)
        cs = fastjet.ClusterSequence(pseudojets, jet_def)
        jets = fastjet.sorted_by_pt(cs.inclusive_jets(10.0))

        jet_pts_ev = [j.pt() for j in jets]
        all_jet_multiplicities.append(len(jets))
        all_jet_pts.extend(jet_pts_ev)

        if len(jets) > 0:
            all_leading_jet_pt.append(jets[0].pt())

        if len(jets) >= 2:
            j1, j2 = jets[0], jets[1]
            px = j1.px() + j2.px()
            py_val = j1.py() + j2.py()
            pz = j1.pz() + j2.pz()
            e = j1.e() + j2.e()
            mjj = np.sqrt(max(0, e**2 - px**2 - py_val**2 - pz**2))
            all_mjj.append(mjj)

    py.stat()
    return {
        "jet_pt": np.array(all_jet_pts),
        "jet_mult": np.array(all_jet_multiplicities),
        "mjj": np.array(all_mjj),
        "ht": np.array(all_ht),
        "leading_jet_pt": np.array(all_leading_jet_pt),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nevents", type=int, default=5000)
    parser.add_argument("-o", "--output", default="figures/process_comparison_sm_bsm.pdf")
    args = parser.parse_args()

    common = [
        "Beams:eCM = 13000.",
        "Beams:idA = 2212",
        "Beams:idB = 2212",
        "Next:numberShowEvent = 0",
        "Print:quiet = on",
    ]

    sm_settings = common + [
        "HiggsSM:gg2H = on",
        "25:onMode = off",
        "25:onIfMatch = 23 23",
        "23:onMode = off",
        "23:onIfAny = 11 13 15",
    ]

    bsm_settings = common + [
        "Higgs:useBSM = on",
        "HiggsBSM:gg2H2 = on",
        "35:m0 = 125.0",
        "HiggsH2:coup2u = 1.0",
        "HiggsH2:coup2d = 1.0",
        "HiggsH2:coup2A3A3 = 1.0",
        "35:onMode = off",
        "35:onIfAll = 36 36",
        "36:m0 = 20.0",
        "HiggsA3:coup2d = 1.0",
        "HiggsA3:coup2u = 1.0",
        "HiggsA3:coup2l = 1.0",
        "36:onMode = off",
        "36:onIfAny = 21 13",
    ]

    print(f"Generating {args.nevents} SM events (H→ZZ→4l)...")
    sm = generate_events(sm_settings, args.nevents)
    print(f"Generating {args.nevents} BSM events (H→aa→ggmumu)...")
    bsm = generate_events(bsm_settings, args.nevents)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    hist_kw = dict(density=True, histtype="step", linewidth=1.5)
    sm_color, bsm_color = "#2962FF", "#D32F2F"
    sm_label = r"$H{\to}4\ell$ (SM)"
    bsm_label = r"$H{\to}aa{\to}gg\mu\mu$ (BSM)"

    ax = axes[0]
    bins = np.linspace(0, 300, 31)
    ax.hist(sm["mjj"], bins=bins, color=sm_color, label=sm_label, **hist_kw)
    ax.hist(bsm["mjj"], bins=bins, color=bsm_color, label=bsm_label, **hist_kw)
    ax.set_xlabel(r"$m_{jj}$ [GeV]")
    ax.set_ylabel("Normalized")
    ax.set_title(r"(a) Dijet invariant mass $m_{jj}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = np.arange(-0.5, 15.5, 1)
    ax.hist(sm["jet_mult"], bins=bins, color=sm_color, label=sm_label, **hist_kw)
    ax.hist(bsm["jet_mult"], bins=bins, color=bsm_color, label=bsm_label, **hist_kw)
    ax.set_xlabel("Jet multiplicity")
    ax.set_ylabel("Normalized")
    ax.set_title(r"(b) Anti-$k_t$ jet multiplicity")
    ax.legend(fontsize=8)

    ax = axes[2]
    bins = np.linspace(0, 200, 31)
    mask_sm = sm["leading_jet_pt"] < 200
    mask_bsm = bsm["leading_jet_pt"] < 200
    ax.hist(sm["leading_jet_pt"][mask_sm], bins=bins, color=sm_color, label=sm_label, **hist_kw)
    ax.hist(bsm["leading_jet_pt"][mask_bsm], bins=bins, color=bsm_color, label=bsm_label, **hist_kw)
    ax.set_xlabel(r"Leading jet $p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Normalized")
    ax.set_title(r"(c) Leading jet $p_\mathrm{T}$")
    ax.legend(fontsize=8)

    ax = axes[3]
    bins = np.linspace(0, 500, 31)
    ax.hist(sm["ht"][sm["ht"] < 500], bins=bins, color=sm_color, label=sm_label, **hist_kw)
    ax.hist(bsm["ht"][bsm["ht"] < 500], bins=bins, color=bsm_color, label=bsm_label, **hist_kw)
    ax.set_xlabel(r"$H_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Normalized")
    ax.set_title(r"(d) $H_\mathrm{T}$")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

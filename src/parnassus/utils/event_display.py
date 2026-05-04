"""Event display utility: eta-phi plane with particle classes and jets.

Produces a two-panel figure showing truth-level particles (left) and
Parnassus reconstructed particles (right) in the eta-phi plane with
marker size proportional to pT and dashed circles for jets.

Usage (standalone):
    python -m parnassus.utils.event_display output.root --event 0 -o display.pdf

Usage (library):
    from parnassus.utils.event_display import plot_event
    fig = plot_event(truth_particles, reco_particles, truth_jets, reco_jets)
    fig.savefig("display.pdf")
"""

from __future__ import annotations

from typing import Any

import numpy as np

PARTICLE_STYLES: dict[str, dict[str, Any]] = {
    "charged_hadron": {"color": "#2962FF", "marker": "o", "label": "Charged hadrons"},
    "electron": {"color": "#D32F2F", "marker": "^", "label": "Electrons"},
    "muon": {"color": "#F57C00", "marker": "s", "label": "Muons"},
    "neutral_hadron": {"color": "#7B1FA2", "marker": "D", "label": "Neutral hadrons"},
    "photon": {"color": "#388E3C", "marker": "*", "label": "Photons"},
}

CLASS_ID_MAP: dict[int, str] = {
    0: "charged_hadron",
    1: "electron",
    2: "muon",
    3: "neutral_hadron",
    4: "photon",
}


def plot_event(
    truth_eta: np.ndarray,
    truth_phi: np.ndarray,
    truth_pt: np.ndarray,
    truth_class: np.ndarray,
    reco_eta: np.ndarray,
    reco_phi: np.ndarray,
    reco_pt: np.ndarray,
    reco_class: np.ndarray,
    truth_jets: list[dict[str, float]] | None = None,
    reco_jets: list[dict[str, float]] | None = None,
    jet_radius: float = 0.5,
    pt_scale: float = 5.0,
    figsize: tuple[float, float] = (14, 5),
):
    """Create a two-panel eta-phi event display.

    Parameters
    ----------
    truth_eta, truth_phi, truth_pt, truth_class :
        Arrays of truth-level particle kinematics and class IDs.
    reco_eta, reco_phi, reco_pt, reco_class :
        Arrays of reconstructed particle kinematics and class IDs.
    truth_jets, reco_jets :
        Lists of dicts with keys 'eta', 'phi', 'pt' for jet positions.
    jet_radius : float
        Radius for jet cone circles.
    pt_scale : float
        Marker size = pt_scale * sqrt(pT).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, (ax_truth, ax_reco) = plt.subplots(1, 2, figsize=figsize)

    def _scatter(ax, eta, phi, pt, cls, jets, title):
        for class_id, style_name in CLASS_ID_MAP.items():
            style = PARTICLE_STYLES[style_name]
            mask = cls == class_id
            if not np.any(mask):
                continue
            sizes = pt_scale * np.sqrt(pt[mask])
            ax.scatter(
                eta[mask],
                phi[mask],
                s=sizes,
                c=style["color"],
                marker=style["marker"],
                alpha=0.7,
                edgecolors="none",
                label=style["label"],
            )

        if jets:
            for jet in jets:
                circle = mpatches.Circle(
                    (jet["eta"], jet["phi"]),
                    jet_radius,
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    edgecolor="gray",
                )
                ax.add_patch(circle)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\phi$")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    _scatter(ax_truth, truth_eta, truth_phi, truth_pt, truth_class, truth_jets, "Truth particles")
    _scatter(ax_reco, reco_eta, reco_phi, reco_pt, reco_class, reco_jets, "Parnassus reconstructed")

    handles = []
    for style in PARTICLE_STYLES.values():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=style["color"],
                markersize=8,
                label=style["label"],
            )
        )
    handles.append(
        mpatches.Circle((0, 0), 0.1, fill=False, linestyle="--", edgecolor="gray", label=f"anti-$k_t$ jet ($R={jet_radius}$)")
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_event_from_root(
    root_path: str,
    event_index: int = 0,
    jet_collection: str = "PflowJetsAntiKt05",
    truth_jet_collection: str = "TruthJetsAntiKt05",
    jet_radius: float = 0.5,
    output_path: str | None = None,
):
    """Load a Parnassus ROOT output file and produce an event display.

    Parameters
    ----------
    root_path : str
        Path to the Parnassus output ROOT file.
    event_index : int
        Which event to display.
    output_path : str, optional
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import uproot

    f = uproot.open(root_path)
    tree = f["Parnassus"]

    truth_eta = tree["Truth.eta"].array()[event_index]
    truth_phi = tree["Truth.phi"].array()[event_index]
    truth_pt = tree["Truth.pt"].array()[event_index]
    truth_class = tree["Truth.particle_class"].array()[event_index]

    reco_eta = tree["Pflow.eta"].array()[event_index]
    reco_phi = tree["Pflow.phi"].array()[event_index]
    reco_pt = tree["Pflow.pt"].array()[event_index]
    reco_class = tree["Pflow.particle_class"].array()[event_index]

    def _load_jets(collection):
        try:
            jet_eta = tree[f"{collection}.eta"].array()[event_index]
            jet_phi = tree[f"{collection}.phi"].array()[event_index]
            jet_pt = tree[f"{collection}.pt"].array()[event_index]
            return [
                {"eta": float(e), "phi": float(p), "pt": float(pt)}
                for e, p, pt in zip(jet_eta, jet_phi, jet_pt)
            ]
        except Exception:
            return None

    truth_jets = _load_jets(truth_jet_collection)
    reco_jets = _load_jets(jet_collection)

    fig = plot_event(
        np.asarray(truth_eta),
        np.asarray(truth_phi),
        np.asarray(truth_pt),
        np.asarray(truth_class),
        np.asarray(reco_eta),
        np.asarray(reco_phi),
        np.asarray(reco_pt),
        np.asarray(reco_class),
        truth_jets=truth_jets,
        reco_jets=reco_jets,
        jet_radius=jet_radius,
    )

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parnassus event display")
    parser.add_argument("root_file", help="Path to Parnassus output ROOT file")
    parser.add_argument("--event", type=int, default=0, help="Event index")
    parser.add_argument("-o", "--output", default="event_display.pdf", help="Output path")
    parser.add_argument("--jet-radius", type=float, default=0.5, help="Jet cone radius")
    args = parser.parse_args()

    fig = plot_event_from_root(args.root_file, args.event, jet_radius=args.jet_radius, output_path=args.output)
    print(f"Saved event display to {args.output}")

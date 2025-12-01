from collections.abc import Sequence
from functools import partial
from typing import final, override

import numpy as np

from parnassus.configs.accessors import Accessor, ParticleAccessor
from parnassus.configs.pipeline import IsolationConfig
from parnassus.configs.scheme import (
    GenEvent,
)
from parnassus.utils import calculate_dr
from parnassus.utils.executor import process_batches
from parnassus.utils.typing import FloatArray, IntArray

from .base import GenPipeline

type IsolationData = FloatArray

ISOLATION_ACESSORS = [
    partial(ParticleAccessor, name=name, dtype="float32")
    for name in ["iso_var", "sum_pt", "sum_pt_ch", "sum_pt_neut"]
]


def calculate_photon_isolation(
    pt: FloatArray,
    dR_matrix: FloatArray,  # noqa: N803
    class_id: IntArray,
    dr_cut: float = 0.4,
) -> FloatArray:
    """Calculate photon isolation scores.

    Parameters
    ----------
    pt : FloatArray
        Particle transverse momenta.
    dR_matrix : FloatArray
        Delta-R distance matrix between particles.
    class_id : IntArray
        Particle class IDs.
    dr_cut : float, optional
        Delta-R cut for isolation calculation, by default 0.4.

    Returns
    -------
    FloatArray
        Isolation scores for photons; non-photons get a score of 1000.
    """
    ch_hadron_mask = class_id == 0
    neut_mask = class_id == 3
    phot_mask = class_id == 4

    dR_mask_phot = dR_matrix < dr_cut  # noqa: N806
    ch_pt_sum_phot = ((pt * dR_mask_phot) * ch_hadron_mask).sum(-1)
    neut_pt_sum_phot = ((pt * dR_mask_phot) * neut_mask).sum(-1)
    iso_score_phot = (ch_pt_sum_phot + np.maximum(0, neut_pt_sum_phot)) / pt
    iso_score_phot[~phot_mask] = 1000

    return iso_score_phot


def calculate_lepton_isolation(
    lepton_id: int,
    pt: FloatArray,
    dR_matrix: FloatArray,  # noqa: N803
    class_id: IntArray,
    dr_cut: float = 0.4,
) -> IsolationData:
    """Calculate lepton isolation variables.

    Parameters
    ----------
    lepton_id : int
        Particle class ID for the lepton type (1=electron, 2=muon).
    pt : FloatArray
        Particle transverse momenta.
    dR_matrix : FloatArray
        Delta-R distance matrix between particles.
    class_id : IntArray
        Particle class IDs.
    dr_cut : float, optional
        Delta-R cut for isolation calculation, by default 0.4.

    Returns
    -------
    IsolationData
        Array of shape (n_leptons, 4) with columns: pt_sum, pt_sum_ch, pt_sum_neut, iso_score.
    """
    ch_hadron_mask = class_id == 0
    neut_mask = class_id == 3
    phot_mask = class_id == 4

    iso_score_phot = calculate_photon_isolation(pt, dR_matrix, class_id, 0.3)
    # Exclude final-state radiation (FSR) photons
    # Following https://cds.cern.ch/record/1460664, page 3
    iso_phot_mask = iso_score_phot < 1
    iso_phot_mask_close = ~(iso_phot_mask & (pt > 2) & (dR_matrix < 0.07))
    iso_phot_mask_far = ~(iso_phot_mask & (pt > 4) & (dR_matrix >= 0.07) & (dR_matrix < 0.5))
    dR_mask = dR_matrix < dr_cut  # noqa: N806

    masked_pt = pt * dR_mask
    pt_sum = masked_pt.sum(-1)

    pt_sum_ch = (masked_pt * ch_hadron_mask).sum(-1)
    phot_mask = (iso_phot_mask_close & phot_mask) | (iso_phot_mask_far & phot_mask)
    pt_sum_neut = (masked_pt * neut_mask).sum(-1)
    pt_sum_phot = (masked_pt * phot_mask).sum(-1)
    iso_score = (pt_sum_ch + np.maximum(0, pt_sum_neut + pt_sum_phot)) / pt

    return np.stack([pt_sum, pt_sum_ch, pt_sum_neut, iso_score], -1)[class_id == lepton_id]


def calculate_isolation(
    lepton_id: int, particle_data: dict[str, FloatArray | IntArray], config: IsolationConfig
) -> IsolationData:
    """Calculate isolation variables for leptons.

    Parameters
    ----------
    lepton_id : int
        Particle class ID for the lepton type (1=electron, 2=muon).
    particle_data : dict[str, FloatArray | IntArray]
        Dictionary with keys: pt, eta, phi, class_id.
    config : IsolationConfig
        Isolation configuration.

    Returns
    -------
    IsolationData
        Array of shape (n_leptons, 4) with columns: pt_sum, pt_sum_ch, pt_sum_neut, iso_score.
    """
    pt: FloatArray = particle_data["pt"]  # pyright: ignore[reportAssignmentType]
    eta: FloatArray = particle_data["eta"]  # pyright: ignore[reportAssignmentType]
    phi: FloatArray = particle_data["phi"]  # pyright: ignore[reportAssignmentType]
    class_id: IntArray = particle_data["class_id"]  # pyright: ignore[reportAssignmentType]

    dR_matrix = calculate_dr(  # noqa: N806
        eta[:, None],
        phi[:, None],
        eta[None, :],
        phi[None, :],
    )
    return calculate_lepton_isolation(lepton_id, pt, dR_matrix, class_id, dr_cut=config.dr)


def calculate_isolation_batch(
    particle_data_batch: list[dict[str, FloatArray | IntArray]], config: IsolationConfig
) -> tuple[list[IsolationData], list[IsolationData]]:
    """Worker function to calculate isolation for a batch of events.

    Parameters
    ----------
    particle_data_batch : list[dict[str, FloatArray | IntArray]]
        List of particle data dictionaries (one per event), each with keys: pt, eta, phi, class_id.
    config : IsolationConfig
        Isolation configuration.

    Returns
    -------
    tuple[list[IsolationData], list[IsolationData]]
        Tuple of (electrons_data, muons_data) lists, one entry per event in batch.
    """
    assert config.collection in {"electrons", "muons", "all"}, (
        f"Can't calculate isolation for {config.collection}"
    )
    electrons_data: list[IsolationData] = []
    muons_data: list[IsolationData] = []
    for particle_data in particle_data_batch:
        if config.collection in {"electrons", "all"}:
            electrons_data.append(calculate_isolation(1, particle_data, config))
        if config.collection in {"muons", "all"}:
            muons_data.append(calculate_isolation(2, particle_data, config))
    return electrons_data, muons_data


def extract_isolation_data(
    events: Sequence[GenEvent], batch_indices: range, _: IsolationConfig
) -> list[dict[str, FloatArray | IntArray]]:
    """Extract particle data for isolation calculation from events.

    Parameters
    ----------
    events : Sequence[GenEvent]
        All events being processed.
    batch_indices : range
        Indices of events in this batch.
    _ : IsolationConfig
        Isolation configuration (not used but required by executor interface).

    Returns
    -------
    list[dict[str, FloatArray | IntArray]]
        List of particle data dicts (one per event), each with keys: pt, eta, phi, class_id.
    """
    particle_data_batch: list[dict[str, FloatArray | IntArray]] = []
    for i in batch_indices:
        particles = events[i].pflow_particles
        assert particles.class_id is not None, "Can't calculate isolation without particle classes."
        particle_data_batch.append({
            "pt": particles.pt,
            "eta": particles.eta,
            "phi": particles.phi,
            "class_id": particles.class_id,
        })
    return particle_data_batch


@final
class IsolationPipeline(GenPipeline):
    """Pipeline to calculate lepton isolation variables from particle collections."""

    @override
    def __init__(self, config: IsolationConfig):
        self.config = config

    @override
    def get_accessors(self) -> dict[str, list[Accessor]]:
        if self.config.collection == "all":
            # If collection is "all", we return accessors for both electrons and muons
            return {
                "Electrons": [accessor(collection="electrons") for accessor in ISOLATION_ACESSORS],
                "Muons": [accessor(collection="muons") for accessor in ISOLATION_ACESSORS],
            }
        # Otherwise, we return accessors for the specified collection
        return {
            self.config.collection.capitalize(): [
                accessor(collection=self.config.collection) for accessor in ISOLATION_ACESSORS
            ],
        }

    @override
    def process(self, events: Sequence[GenEvent]):
        # Use the shared executor utility for batch processing
        batch_results = process_batches(
            events=events,
            config=self.config,
            worker_fn=calculate_isolation_batch,
            extract_fn=extract_isolation_data,
            description=f"Calculating isolation for {self.config.collection}",
        )

        # Flatten batch results into per-event lists
        electrons_data: list[IsolationData] = []
        muons_data: list[IsolationData] = []
        for batch_electrons, batch_muons in batch_results:
            electrons_data.extend(batch_electrons)
            muons_data.extend(batch_muons)

        # Assign results back to events
        for i in range(len(events)):
            if self.config.collection in {"electrons", "all"}:
                events[i].electrons.sum_pt = electrons_data[i][:, 0]
                events[i].electrons.sum_pt_ch = electrons_data[i][:, 1]
                events[i].electrons.sum_pt_neut = electrons_data[i][:, 2]
                events[i].electrons.iso_var = electrons_data[i][:, 3]
            if self.config.collection in {"muons", "all"}:
                events[i].muons.sum_pt = muons_data[i][:, 0]
                events[i].muons.sum_pt_ch = muons_data[i][:, 1]
                events[i].muons.sum_pt_neut = muons_data[i][:, 2]
                events[i].muons.iso_var = muons_data[i][:, 3]

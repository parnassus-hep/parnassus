import multiprocessing as mp
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any, final, override

import numpy as np

from parnassus.configs.accessors import Accessor, ParticleAccessor
from parnassus.configs.pipeline import IsolationConfig
from parnassus.configs.scheme import (
    GenEvent,
    GenParticleCollection,
)
from parnassus.utils import calculate_dr
from parnassus.utils.logger import ProgressBar
from parnassus.utils.typing import FloatArray, IntArray

from .base import GenPipeline

type IsolationData = FloatArray

ISOLATION_ACESSORS = [
    partial(ParticleAccessor, name=name, dtype="float32")
    for name in ["iso_var", "sum_pt", "sum_pt_ch", "sum_pt_neut"]
]


def calculate_photon_isolation(
    pt: FloatArray,
    dR_matrix: FloatArray,
    class_id: IntArray,
    dr_cut: float = 0.4,
) -> FloatArray:
    ch_hadron_mask = class_id == 0
    neut_mask = class_id == 3
    phot_mask = class_id == 4

    dR_mask_phot = dR_matrix < dr_cut
    ch_pt_sum_phot = ((pt * dR_mask_phot) * ch_hadron_mask).sum(-1)
    neut_pt_sum_phot = ((pt * dR_mask_phot) * neut_mask).sum(-1)
    iso_score_phot = (ch_pt_sum_phot + np.maximum(0, neut_pt_sum_phot)) / pt
    iso_score_phot[~phot_mask] = 1000

    return iso_score_phot


def calculate_lepton_isolation(
    lepton_id: int,
    pt: FloatArray,
    dR_matrix: FloatArray,
    class_id: IntArray,
    dr_cut: float = 0.4,
) -> IsolationData:
    ch_hadron_mask = class_id == 0
    neut_mask = class_id == 3
    phot_mask = class_id == 4

    iso_score_phot = calculate_photon_isolation(pt, dR_matrix, class_id, 0.3)
    # Exclude final-state radiation (FSR) photons
    # Following https://cds.cern.ch/record/1460664, page 3
    iso_phot_mask = iso_score_phot < 1
    iso_phot_mask_close = ~(iso_phot_mask & (pt > 2) & (dR_matrix < 0.07))
    iso_phot_mask_far = ~(iso_phot_mask & (pt > 4) & (dR_matrix >= 0.07) & (dR_matrix < 0.5))
    dR_mask = dR_matrix < dr_cut

    masked_pt = pt * dR_mask
    pt_sum = masked_pt.sum(-1)

    pt_sum_ch = (masked_pt * ch_hadron_mask).sum(-1)
    phot_mask = (iso_phot_mask_close & phot_mask) | (iso_phot_mask_far & phot_mask)
    pt_sum_neut = (masked_pt * neut_mask).sum(-1)
    pt_sum_phot = (masked_pt * phot_mask).sum(-1)
    iso_score = (pt_sum_ch + np.maximum(0, pt_sum_neut + pt_sum_phot)) / pt

    return np.stack([pt_sum, pt_sum_ch, pt_sum_neut, iso_score], -1)[class_id == lepton_id]


def calculate_isolation(
    lepton_id: int, particles: GenParticleCollection, config: IsolationConfig
) -> IsolationData:
    assert particles.class_id is not None, "Can't calculate isolation without particle classes."
    dR_matrix = calculate_dr(
        particles.eta[:, None],
        particles.phi[:, None],
        particles.eta[None, :],
        particles.phi[None, :],
    )
    return calculate_lepton_isolation(
        lepton_id, particles.pt, dR_matrix, particles.class_id, dr_cut=config.dr
    )


def process_events(
    event_list: list[GenEvent], config: IsolationConfig
) -> tuple[list[IsolationData], list[IsolationData]]:
    assert config.collection in {"electrons", "muons", "all"}, (
        f"Can't calculate isolation for {config.collection}"
    )
    electrons_data: list[IsolationData] = []
    muons_data: list[IsolationData] = []
    for event in event_list:
        if config.collection in {"electrons", "all"}:
            electrons_data.append(calculate_isolation(1, event.pflow_particles, config))
        if config.collection in {"muons", "all"}:
            muons_data.append(calculate_isolation(2, event.pflow_particles, config))
    return electrons_data, muons_data


def process_events_wrapper(args: Iterable[Any]):
    return process_events(*args)


@final
class IsolationPipeline(GenPipeline):
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
        n_events = len(events)
        batch_size = 2000
        n_batches = n_events // batch_size
        n_batches += 1 if n_events % batch_size != 0 else 0

        input_batched_data = [
            (events[i * batch_size : (i + 1) * batch_size], self.config) for i in range(n_batches)
        ]
        n_events_in_batch = (len(data[0]) for data in input_batched_data)
        electrons_data: list[IsolationData] = []
        muons_data: list[IsolationData] = []
        with mp.Pool(processes=self.config.num_processes) as pool, ProgressBar() as progress:
            task = progress.add_task(
                f"[green]Calculating isolation for {self.config.collection}", total=n_events
            )
            for data_ in pool.imap(process_events_wrapper, input_batched_data):
                electrons_data.extend(data_[0])
                muons_data.extend(data_[1])
                progress.update(task, advance=next(n_events_in_batch))
        for i in range(n_events):
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

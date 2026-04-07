from collections.abc import Sequence
from contextlib import nullcontext
from typing import final, override

import energyflow as ef
import fastjet as fj
import numpy as np

from parnassus.configs.accessors import (
    Accessor,
    AccessorListBuilder,
    AccessorTemplates,
    ParticleAccessor,
)
from parnassus.configs.pipeline import JetClusteringConfig
from parnassus.configs.scheme import (
    GenEvent,
    GenJetCollection,
)
from parnassus.utils.executor import process_batches
from parnassus.utils.logger import stdout_redirected
from parnassus.utils.typing import FloatArray, IntArray

from .base import GenPipeline


@final
class Jet:
    """Jet object wrapping fastjet PseudoJet with additional utilities."""

    def __init__(self, fj_jet: fj.PseudoJet, dr: float, calc_substructure: bool = False):
        self.fj_jet = fj_jet
        self.dR = dr
        self.nconstituents = len(self.constituents())
        self.constituents_pt = np.array([c.pt() for c in self.constituents()])
        self.constituents_eta = np.array([c.eta() for c in self.constituents()])
        self.constituents_phi = np.array([
            c.phi() if c.phi() <= np.pi else c.phi() - 2 * np.pi for c in self.constituents()
        ])
        self.constituents_m = np.array([c.m() for c in self.constituents()])
        self.constituents_idx = np.array([c.user_index() for c in self.constituents()])
        self.pt_order_constituents()

        self.dR_matrix = None
        self.ecf = {0: 1, 1: -1, 2: -1, 3: -1}
        self.substructure = {"c2": np.nan, "d2": np.nan}

        if calc_substructure:
            self.calc_substructure()

    def __getattr__(self, name: str):
        if name in {
            "pt",
            "eta",
            "phi",
            "phi_std",
            "e",
            "m",
            "constituents",
            "px",
            "py",
            "pz",
            "E",
        }:
            return getattr(self.fj_jet, name)
        if name in self.__dict__:
            return getattr(self, name)
        raise AttributeError(f"'Jet' object has no attribute '{name}'")

    def pt_order_constituents(self):
        idx = np.argsort(self.constituents_pt)[::-1]
        self.constituents_pt = self.constituents_pt[idx]
        self.constituents_eta = self.constituents_eta[idx]
        self.constituents_phi = self.constituents_phi[idx]
        self.constituents_m = self.constituents_m[idx]
        self.constituents_idx = self.constituents_idx[idx]

    def calc_substructure(self):
        d2_calc = ef.D2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
        c2_calc = ef.C2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)

        pt_eta_phi_m = np.stack(
            [
                self.constituents_pt,
                self.constituents_eta,
                self.constituents_phi,
                self.constituents_m,
            ],
            axis=1,
        )

        self.substructure["d2"] = d2_calc.compute(pt_eta_phi_m)
        self.substructure["c2"] = c2_calc.compute(pt_eta_phi_m)


def get_cluster_sequence(
    jet_definition: fj.JetDefinition,
    px: FloatArray,
    py: FloatArray,
    pz: FloatArray,
    e: FloatArray,
    user_indices: list[int] | None = None,
) -> fj.ClusterSequence:
    """Create fastjet ClusterSequence from particle four-vectors.

    Parameters
    ----------
    jet_definition : fj.JetDefinition
        Jet clustering algorithm definition.
    px, py, pz, e : FloatArray
        Particle four-momentum components.
    user_indices : list[int] | None
        Optional user indices to assign to particles.

    Returns
    -------
    fj.ClusterSequence
        Fastjet cluster sequence object.
    """
    pj_array: list[fj.PseudoJet] = []

    for i in range(len(px)):
        pj = fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(e[i]))
        if user_indices is not None:
            pj.set_user_index(user_indices[i])
        else:
            pj.set_user_index(i)
        pj_array.append(pj)

    return fj.ClusterSequence(pj_array, jet_definition)


def cluster_jets(
    particle_data: dict[str, FloatArray], config: JetClusteringConfig
) -> tuple[list[Jet], IntArray]:
    """Cluster jets from particle four-vectors.

    Parameters
    ----------
    particle_data : dict[str, FloatArray]
        Dictionary with keys: px, py, pz, e (particle four-momentum components).
    config : JetClusteringConfig
        Jet clustering configuration.

    Returns
    -------
    tuple[list[Jet], IntArray]
        List of clustered jets and array of jet indices for each particle.
    """
    n_particles = len(particle_data["px"])
    cs = get_cluster_sequence(
        config.jet_definition,
        particle_data["px"],
        particle_data["py"],
        particle_data["pz"],
        particle_data["e"],
        user_indices=list(range(n_particles)),
    )
    jets = cs.inclusive_jets(config.pt_min)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, config.dr, calc_substructure=True) for j in jets]

    used_indices: set[int] = set()
    jet_idxs = np.zeros(n_particles, dtype=int)
    for jet_idx, jet in enumerate(jets):
        particle_idx = jet.constituents_idx
        jet_idxs[particle_idx] = jet_idx
        used_indices.update(particle_idx)
    particle_idx = np.arange(n_particles)
    particle_idx = particle_idx[~np.isin(particle_idx, list(used_indices))]
    jet_idxs[particle_idx] = -1
    return [j for j in jets if j.nconstituents >= config.nconst_min], jet_idxs


def convert_to_jet_collection(name: str, jets: list[Jet]) -> GenJetCollection:
    """Convert list of Jet objects to GenJetCollection.

    Parameters
    ----------
    name : str
        Name of the jet collection.
    jets : list[Jet]
        List of Jet objects to convert.

    Returns
    -------
    GenJetCollection
        The converted GenJetCollection object.
    """
    return GenJetCollection(
        name=name,
        pt=np.array([jet.pt() for jet in jets]),
        eta=np.array([jet.eta() for jet in jets]),
        phi=np.array([jet.phi() for jet in jets]),
        d2=np.array([jet.substructure["d2"] for jet in jets]),
        c2=np.array([jet.substructure["c2"] for jet in jets]),
    )


def cluster_jets_batch(
    particle_data_batch: list[dict[str, FloatArray]],
    config: JetClusteringConfig,
) -> tuple[list[GenJetCollection], list[IntArray]]:
    """Worker function to cluster jets for a batch of events.

    Parameters
    ----------
    particle_data_batch : list[dict[str, FloatArray]]
        List of particle data dictionaries (one per event), each with keys: px, py, pz, e.
    config : JetClusteringConfig
        Jet clustering configuration.

    Returns
    -------
    tuple[list[GenJetCollection], list[IntArray]]
        Lists of jet collections and particle indices (one per event in batch).
    """
    jets: list[GenJetCollection] = []
    idxs: list[IntArray] = []

    # Redirect stdout if configured (only safe in multiprocessing workers)
    context_manager = (
        stdout_redirected()
        if (config.redirect_stdout and config.num_processes > 1)
        else nullcontext()
    )
    with context_manager:
        for particle_data in particle_data_batch:
            evt_jets, jet_idxs = cluster_jets(particle_data, config)
            jets.append(convert_to_jet_collection(config.name, evt_jets))
            idxs.append(jet_idxs)
    return jets, idxs


def extract_clustering_data(
    events: Sequence[GenEvent], batch_indices: range, config: JetClusteringConfig
) -> list[dict[str, FloatArray]]:
    """Extract particle data for jet clustering from events.

    Parameters
    ----------
    events : Sequence[GenEvent]
        All events being processed.
    batch_indices : range
        Indices of events in this batch.
    config : JetClusteringConfig
        Configuration specifying which particle collection to use.

    Returns
    -------
    list[dict[str, FloatArray]]
        List of particle data dictionaries (one per event), each with keys: px, py, pz, E.
    """
    assert config.collection in {"truth", "pflow"}, f"Can't cluster {config.collection}"
    particle_data_batch: list[dict[str, FloatArray]] = []
    for i in batch_indices:
        particles = (
            events[i].truth_particles if config.collection == "truth" else events[i].pflow_particles
        )
        # Get 4-vectors via awkward and convert to numpy
        np_4vecs = particles.get4vecs_numpy()
        particle_data_batch.append({
            "px": np_4vecs[..., 0],
            "py": np_4vecs[..., 1],
            "pz": np_4vecs[..., 2],
            "e": np_4vecs[..., 3],
        })
    return particle_data_batch


@final
class JetClusteringPipeline(GenPipeline):
    """Pipeline to cluster jets from particle collections."""

    @override
    def __init__(self, config: JetClusteringConfig):
        self.config = config

    @override
    def get_accessors(self) -> dict[str, list[Accessor]]:
        return {
            self.config.name: (
                AccessorListBuilder
                .for_jets(self.config.name)
                .add_from_specs(AccessorTemplates.KINEMATICS)
                .add_from_specs(AccessorTemplates.JET_SUBSTRUCTURE)
                .build()
            ),
            self.config.collection.capitalize(): [
                ParticleAccessor(
                    f"jet_idx/{self.config.name}",
                    f"{self.config.collection}_particles",
                    f"{self.config.name}_idx",
                    dtype="int32",
                )
            ],
        }

    @override
    def process(self, events: Sequence[GenEvent]):
        # Use the shared executor utility for batch processing
        # Note: stdout redirection only happens in multiprocessing workers,
        # not in synchronous execution to avoid conflicts with progress bar
        batch_results = process_batches(
            events=events,
            config=self.config,
            worker_fn=cluster_jets_batch,
            extract_fn=extract_clustering_data,
            description=f"Cluster {self.config.name} jets",
        )

        # Flatten batch results into per-event lists
        jets: list[GenJetCollection] = []
        jet_idxs: list[IntArray] = []
        for batch_jets, batch_idxs in batch_results:
            jets.extend(batch_jets)
            jet_idxs.extend(batch_idxs)

        # Assign results back to events
        for i in range(len(events)):
            events[i].jets[self.config.name] = jets[i]
            if self.config.collection == "truth":
                events[i].truth_particles.jet_idx[self.config.name] = jet_idxs[i]
            if self.config.collection == "pflow":
                events[i].pflow_particles.jet_idx[self.config.name] = jet_idxs[i]

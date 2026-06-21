import numpy as np
import pytest

from parnassus.configs.pipeline import ParticleFilteringConfig
from parnassus.configs.scheme import GenEvent, GenParticleCollection, GenTowerCollection
from parnassus.pipelines.filter import ParticleFilteringPipeline


def make_event() -> GenEvent:
    truth = GenParticleCollection(
        name="truth",
        pt=np.array([0.2, 5.0, 30.0, 50.0]),
        eta=np.array([0.1, 3.5, -0.5, 0.5]),
        phi=np.array([0.0, 1.0, 2.0, -1.0]),
        pdg_id=np.array([211, 211, 12, 13]),  # one neutrino (12)
    )
    pflow = GenParticleCollection(
        name="pflow",
        pt=np.array([10.0, 20.0]),
        eta=np.array([0.2, 0.3]),
        phi=np.array([0.1, 0.2]),
        pdg_id=np.array([11, 13]),  # electron + muon
    )
    return GenEvent(event_number=0, truth_particles=truth, pflow_particles=pflow)


def test_pt_threshold_filter():
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pt", "op": ">", "value": 1.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])
    # 0.2 GeV particle dropped -> 3 survive
    assert event.truth_particles.num_particles == 3
    assert np.all(event.truth_particles.pt > 1.0)
    # derived fields stay length-consistent
    assert len(event.truth_particles.pdg_id) == 3


def test_abs_eta_filter():
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "eta", "op": "<=", "value": 3.0, "abs": True}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])
    # |eta|=3.5 dropped -> 3 survive
    assert event.truth_particles.num_particles == 3
    assert np.all(np.abs(event.truth_particles.eta) <= 3.0)


def test_drop_neutrinos_not_in():
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pdg_id", "op": "not in", "value": [12, 14, 16]}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])
    assert 12 not in event.truth_particles.pdg_id
    assert event.truth_particles.num_particles == 3


def test_combine_all_vs_any():
    event_all = make_event()
    cfg_all = ParticleFilteringConfig(
        name="f",
        collection="truth",
        combine="all",
        conditions=[  # type: ignore[list-item]
            {"field": "pt", "op": ">", "value": 1.0},
            {"field": "eta", "op": "<=", "value": 3.0, "abs": True},
        ],
    )
    ParticleFilteringPipeline(cfg_all).process([event_all])
    # AND: must pass both -> drops pt=0.2 (also eta ok) and eta=3.5 (pt ok) -> 2 survive
    assert event_all.truth_particles.num_particles == 2

    event_any = make_event()
    cfg_any = ParticleFilteringConfig(
        name="f",
        collection="truth",
        combine="any",
        conditions=[  # type: ignore[list-item]
            {"field": "pt", "op": ">", "value": 1.0},
            {"field": "eta", "op": "<=", "value": 3.0, "abs": True},
        ],
    )
    ParticleFilteringPipeline(cfg_any).process([event_any])
    # OR: pass either -> only a particle failing BOTH would drop; none does -> 4 survive
    assert event_any.truth_particles.num_particles == 4


def test_filter_named_tower_collection():
    event = make_event()
    event.collections["Tower"] = GenTowerCollection(
        name="Tower",
        e=np.array([1.0, 50.0, 100.0]),
        et=np.array([0.5, 20.0, 80.0]),
        eta=np.array([0.0, 0.1, 0.2]),
        phi=np.array([0.0, 0.1, 0.2]),
        t=np.array([0.0, 0.0, 0.0]),
    )
    cfg = ParticleFilteringConfig(
        name="f",
        collection="Tower",
        conditions=[{"field": "et", "op": ">=", "value": 10.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])
    tower = event.collections["Tower"]
    assert len(tower) == 2
    assert np.all(tower.et >= 10.0)  # type: ignore[attr-defined]


def test_missing_field_raises():
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "nonexistent", "op": ">", "value": 1.0}],  # type: ignore[list-item]
    )
    with pytest.raises((ValueError, AttributeError), match="nonexistent"):
        ParticleFilteringPipeline(cfg).process([event])


def test_no_conditions_is_noop():
    event = make_event()
    cfg = ParticleFilteringConfig(name="f", collection="truth", conditions=[])
    ParticleFilteringPipeline(cfg).process([event])
    assert event.truth_particles.num_particles == 4


def test_unknown_collection_raises():
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="Nope",
        conditions=[{"field": "pt", "op": ">", "value": 1.0}],  # type: ignore[list-item]
    )
    with pytest.raises((KeyError, ValueError), match="Nope"):
        ParticleFilteringPipeline(cfg).process([event])


def test_get_accessors_empty():
    cfg = ParticleFilteringConfig(name="f", collection="truth")
    assert ParticleFilteringPipeline(cfg).get_accessors() == {}


def test_pipeline_exported_from_package():
    from parnassus.pipelines import ParticleFilteringPipeline as Exported

    assert Exported is ParticleFilteringPipeline


def test_runs_before_clustering_sees_survivors():
    # Filtering pflow then accessing the collection reflects the cut.
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="pflow",
        conditions=[{"field": "pt", "op": ">=", "value": 15.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])
    assert event.pflow_particles.num_particles == 1
    assert event.pflow_particles.pt[0] == 20.0

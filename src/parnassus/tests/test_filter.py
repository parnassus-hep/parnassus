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
    assert event.pflow_particles.pt[0] == pytest.approx(20.0)


def test_jet_idx_and_particle_jet_idx_masked():
    # 4 particles; filter keeps only those with pt > 10 (particles 1 and 2, 0-indexed).
    # particle_jet_idx and jet_idx must be masked to match survivors.
    truth = GenParticleCollection(
        name="truth",
        pt=np.array([5.0, 20.0, 40.0, 3.0]),
        eta=np.array([0.0, 0.1, -0.2, 0.3]),
        phi=np.array([0.0, 0.5, 1.0, 1.5]),
        pdg_id=np.array([211, 211, 211, 211]),
        particle_jet_idx=np.array([0, 0, 1, 1], dtype=np.int32),
        jet_idx={"AK4": np.array([3, 3, 7, 7], dtype=np.int32)},
    )
    pflow = GenParticleCollection(
        name="pflow",
        pt=np.array([10.0, 20.0]),
        eta=np.array([0.2, 0.3]),
        phi=np.array([0.1, 0.2]),
        pdg_id=np.array([11, 13]),
    )
    event = GenEvent(event_number=0, truth_particles=truth, pflow_particles=pflow)

    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pt", "op": ">", "value": 10.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])

    # Survivors: particles with pt=20 and pt=40 (original indices 1 and 2)
    assert event.truth_particles.num_particles == 2
    assert np.all(event.truth_particles.pt > 10.0)

    # particle_jet_idx must be masked to the survivors (values 0 and 1)
    assert event.truth_particles.particle_jet_idx is not None
    assert len(event.truth_particles.particle_jet_idx) == 2
    assert np.array_equal(event.truth_particles.particle_jet_idx, np.array([0, 1]))

    # jet_idx dict must also be masked; AK4 values for survivors are 3 and 7
    assert "AK4" in event.truth_particles.jet_idx
    assert len(event.truth_particles.jet_idx["AK4"]) == 2
    assert np.array_equal(event.truth_particles.jet_idx["AK4"], np.array([3, 7]))


def test_filter_electrons_lepton_collection():
    # Build a pflow with three electrons and one muon so event.electrons has 3 entries.
    pflow = GenParticleCollection(
        name="pflow",
        pt=np.array([5.0, 15.0, 30.0, 25.0]),
        eta=np.array([0.1, 0.2, 0.3, 0.4]),
        phi=np.array([0.0, 0.1, 0.2, 0.3]),
        pdg_id=np.array([11, 11, 11, 13]),  # 3 electrons + 1 muon
    )
    truth = GenParticleCollection(
        name="truth",
        pt=np.array([10.0]),
        eta=np.array([0.0]),
        phi=np.array([0.0]),
        pdg_id=np.array([211]),
    )
    event = GenEvent(event_number=0, truth_particles=truth, pflow_particles=pflow)

    # Sanity: derived electrons collection has 3 entries
    assert event.electrons.num_particles == 3

    # Filter electrons: keep only pt >= 10.0 (drops the 5.0 GeV electron)
    cfg = ParticleFilteringConfig(
        name="f",
        collection="electrons",
        conditions=[{"field": "pt", "op": ">=", "value": 10.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])

    assert event.electrons.num_particles == 2
    assert np.all(event.electrons.pt >= 10.0)
    assert np.array_equal(np.sort(event.electrons.pt), np.array([15.0, 30.0]))


def test_zero_survivors():
    # Filter with an impossible condition: no particle has pt > 1e9.
    event = make_event()
    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pt", "op": ">", "value": 1e9}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])

    assert event.truth_particles.num_particles == 0
    assert len(event.truth_particles.pt) == 0
    assert event.truth_particles.pdg_id is not None
    assert len(event.truth_particles.pdg_id) == 0
    assert event.truth_particles.mass is not None
    assert len(event.truth_particles.mass) == 0


def test_equality_and_in_ops():
    # Exercise the "==" and "in" operators (complements the existing ">", ">=", "<=", "not in").
    # truth pdg_ids are [211, 211, 12, 13].

    # "in" keeps only particles whose pdg_id is in the list [13] → just the muon
    event_in = make_event()
    cfg_in = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pdg_id", "op": "in", "value": [13]}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg_in).process([event_in])
    assert event_in.truth_particles.num_particles == 1
    assert event_in.truth_particles.pdg_id is not None
    assert np.all(event_in.truth_particles.pdg_id == 13)

    # "==" keeps only particles whose pdg_id == 211 → the two pions
    event_eq = make_event()
    cfg_eq = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pdg_id", "op": "==", "value": 211}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg_eq).process([event_eq])
    assert event_eq.truth_particles.num_particles == 2
    assert event_eq.truth_particles.pdg_id is not None
    assert np.all(event_eq.truth_particles.pdg_id == 211)


def test_event_features_updated_after_filtering():
    event = make_event()
    # Capture stale (pre-filter) HT, then drop the 0.2 GeV truth particle.
    ht_before = event.truth_ht
    cfg = ParticleFilteringConfig(
        name="f",
        collection="truth",
        conditions=[{"field": "pt", "op": ">", "value": 1.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])

    # HT must reflect the survivors, not the original collection.
    expected_ht = np.sum(event.truth_particles.pt)
    assert event.truth_ht == pytest.approx(expected_ht)
    assert event.truth_ht == pytest.approx(ht_before - 0.2)

    # MET is recomputed consistently from the surviving particles.
    expected_met_x = np.sum(event.truth_particles.pt * np.cos(event.truth_particles.phi))
    expected_met_y = np.sum(event.truth_particles.pt * np.sin(event.truth_particles.phi))
    assert event.truth_met_x == pytest.approx(expected_met_x)
    assert event.truth_met_y == pytest.approx(expected_met_y)
    assert event.truth_met == pytest.approx(np.sqrt(expected_met_x**2 + expected_met_y**2))

    # pflow was untouched, so its features are unchanged.
    assert event.pflow_ht == pytest.approx(np.sum(event.pflow_particles.pt))


def test_pflow_features_updated_but_leptons_not_rederived():
    event = make_event()
    n_electrons_before = event.electrons.num_particles
    cfg = ParticleFilteringConfig(
        name="f",
        collection="pflow",
        conditions=[{"field": "pt", "op": ">=", "value": 15.0}],  # type: ignore[list-item]
    )
    ParticleFilteringPipeline(cfg).process([event])
    # pflow HT reflects the single surviving particle (pt=20).
    assert event.pflow_ht == pytest.approx(20.0)
    # Leptons are intentionally NOT re-derived (documented limitation).
    assert event.electrons.num_particles == n_electrons_before

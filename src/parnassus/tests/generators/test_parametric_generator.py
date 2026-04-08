"""Tests for ParametricEventGenerator and its conversion helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from parnassus.configs.generators.parametric import ParametricGeneratorConfig
from parnassus.configs.scheme import GenParticleCollection, GenTowerCollection
from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.pipelines.generators.parametric import (
    ParametricEventGenerator,
    _make_particle_collection,
    _make_tower_collection,
    _tensors_to_gen_events,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STUB_LOG = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None, debug=lambda *_: None)


def _make_config(card: str = "cms", seed: int | None = None) -> ParametricGeneratorConfig:
    return ParametricGeneratorConfig(name="test", card=card, seed=seed)


def _particle_tensor(
    n: int,
    event_number: int,
    pt: float = 10.0,
    eta: float = 0.5,
    phi: float = 1.0,
    e: float = 12.0,
    pid: int = 211,
    status: int = 1,
) -> np.ndarray:
    """Build a minimal (n, N_FEATURES) particle array for testing."""
    arr = np.zeros((n, N_FEATURES), dtype=np.float64)
    arr[:, ColumnMap.PID] = pid
    arr[:, ColumnMap.STATUS] = status
    arr[:, ColumnMap.CHARGE] = np.sign(pid)
    arr[:, ColumnMap.E] = e
    arr[:, ColumnMap.PT] = pt
    arr[:, ColumnMap.ETA] = eta
    arr[:, ColumnMap.PHI] = phi
    arr[:, ColumnMap.MASS] = 0.140
    arr[:, ColumnMap.T] = 0.001
    arr[:, ColumnMap.X] = 0.1
    arr[:, ColumnMap.Y] = 0.2
    arr[:, ColumnMap.Z] = 0.3
    arr[:, ColumnMap.EVENT_NUMBER] = event_number
    return arr


def _particle_torch(n: int, event_number: int, **kwargs) -> torch.Tensor:
    return torch.from_numpy(_particle_tensor(n, event_number, **kwargs))


def _make_stub_card(results: dict[str, torch.Tensor]):
    """Return a callable that ignores its input and returns *results*."""
    card = MagicMock()
    card.return_value = results
    card.eval = MagicMock()
    card.to = MagicMock(return_value=card)
    return card


# ---------------------------------------------------------------------------
# Conversion helper tests
# ---------------------------------------------------------------------------


def test_make_particle_collection_extracts_columns():
    arr = _particle_tensor(3, event_number=7, pt=5.0, eta=0.3, phi=1.2, pid=13)
    col = _make_particle_collection(arr, name="truth")

    assert isinstance(col, GenParticleCollection)
    assert len(col) == 3
    assert np.allclose(col.pt, 5.0)
    assert np.allclose(col.eta, 0.3)
    assert np.allclose(col.phi, 1.2)
    assert np.all(col.status == 1)


@pytest.mark.parametrize("fix_neutral_hadrons", [True, False])
def test_make_particle_collection_fixes_neutral_hadron_pid(fix_neutral_hadrons):
    arr = _particle_tensor(2, event_number=1, pid=0)
    col = _make_particle_collection(arr, name="pflow", fix_neutral_hadrons=fix_neutral_hadrons)
    if fix_neutral_hadrons:
        assert np.all(col.pdg_id == 130)
    else:
        assert np.all(col.pdg_id == 0)


def test_make_particle_collection_as_track_extracts_kinematics():
    arr = _particle_tensor(4, event_number=5, pt=8.0)
    tracks = _make_particle_collection(arr, name="Track")
    assert tracks.name == "Track"
    assert np.allclose(tracks.pt, 8.0)


def test_make_tower_collection_empty_input_yields_zero_length():
    empty = np.empty((0, N_FEATURES), dtype=np.float64)
    towers = _make_tower_collection(empty)
    assert isinstance(towers, GenTowerCollection)
    assert len(towers) == 0


def test_make_tower_collection_et_equals_e_over_cosh_eta():
    arr = _particle_tensor(3, event_number=2, e=20.0, eta=1.0)
    towers = _make_tower_collection(arr)
    assert isinstance(towers, GenTowerCollection)
    assert np.allclose(towers.et, 20.0 / np.cosh(1.0), rtol=1e-5)


def test_tensors_to_gen_events_produces_one_event_per_unique_event_number():
    ev1 = _particle_torch(3, event_number=10)
    ev2 = _particle_torch(2, event_number=20)
    combined = torch.cat([ev1, ev2])

    events = _tensors_to_gen_events(
        truth=combined,
        event_numbers=torch.tensor([10, 20]),
        results={"EFlowObject": combined, "Track": combined, "Tower": combined},
    )

    assert len(events) == 2
    assert {e.event_number for e in events} == {10, 20}
    assert len(events[0].truth_particles) == 3
    assert len(events[1].truth_particles) == 2


def test_tensors_to_gen_events_empty_branches_have_zero_length():
    truth = _particle_torch(2, event_number=1)
    empty = torch.zeros((0, N_FEATURES), dtype=torch.float64)

    events = _tensors_to_gen_events(
        truth=truth,
        event_numbers=torch.tensor([1]),
        results={"EFlowObject": truth, "Track": empty, "Tower": empty},
    )

    assert len(events) == 1
    assert len(events[0].collections["Track"]) == 0
    assert len(events[0].collections["Tower"]) == 0


# ---------------------------------------------------------------------------
# ParametricEventGenerator class tests
# ---------------------------------------------------------------------------


def test_generator_init_dispatches_card():
    from parnassus.torch_delphes.defaults import (
        ATLASEnergyFlowDefault,
        CMSEnergyFlowDefault,
    )

    assert isinstance(
        ParametricEventGenerator(_make_config("cms"), _STUB_LOG).card,
        CMSEnergyFlowDefault,
    )
    assert isinstance(
        ParametricEventGenerator(_make_config("atlas"), _STUB_LOG).card,
        ATLASEnergyFlowDefault,
    )


def test_generator_unknown_card_raises():
    with pytest.raises(KeyError):
        ParametricEventGenerator(_make_config("lhcb"), _STUB_LOG)


def test_generator_to_moves_card_and_returns_self():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    stub_card = _make_stub_card({})
    gen.card = stub_card

    cpu = torch.device("cpu")
    assert gen.to(cpu) is gen
    stub_card.to.assert_called_once_with(cpu)
    assert gen.device == cpu


@pytest.mark.parametrize("same_seed", [True, False])
def test_generator_random_seed_reproducibility(same_seed: bool):
    N_PARTICLES = 100
    batch = {
        "stable_particles": torch.rand((N_PARTICLES, N_FEATURES), dtype=torch.float64),
        "all_particles": torch.rand((N_PARTICLES, N_FEATURES), dtype=torch.float64),
        "event_numbers": torch.tensor([1]),
        "n_particles": torch.tensor([N_PARTICLES]),
    }
    batch["stable_particles"][:, ColumnMap.EVENT_NUMBER] = 1
    batch["all_particles"][:, ColumnMap.EVENT_NUMBER] = 1

    events_list = []
    for _ in range(2):
        gen_batch = {k: v.clone() for k, v in batch.items()}
        with ParametricEventGenerator(
            _make_config("cms", seed=123 if same_seed else None), _STUB_LOG
        ) as gen:
            gen.initialize(n_events=1, n_batches=1)
            gen.process_batch(gen_batch)
            events_list.append(gen.get_events())

    events1, events2 = events_list
    assert len(events1) == len(events2) == 1
    for var in ["pt", "eta", "phi", "status"]:
        if same_seed:
            np.testing.assert_allclose(
                getattr(events1[0].pflow_particles, var),
                getattr(events2[0].pflow_particles, var),
            )
        else:
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(
                    getattr(events1[0].pflow_particles, var),
                    getattr(events2[0].pflow_particles, var),
                )


def test_generator_process_batch_accumulates_events():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    t1 = _particle_torch(3, event_number=1)
    t2 = _particle_torch(2, event_number=2)
    gen.card = _make_stub_card({"EFlowObject": t1, "Track": t1, "Tower": t1})
    gen.initialize(n_events=2, n_batches=2)

    gen.process_batch({
        "stable_particles": t1,
        "all_particles": t1,
        "event_numbers": torch.tensor([1]),
        "n_particles": torch.tensor([len(t1)]),
    })
    gen.card.return_value = {"EFlowObject": t2, "Track": t2, "Tower": t2}
    gen.process_batch({
        "stable_particles": t2,
        "all_particles": t2,
        "event_numbers": torch.tensor([2]),
        "n_particles": torch.tensor([len(t2)]),
    })

    assert len(gen.get_events()) == 2


def test_generator_process_batch_preserves_truth_when_card_mutates_input():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    truth = _particle_torch(2, event_number=1)
    truth[:, ColumnMap.T] = 0.0

    def _mutating_card(particles: torch.Tensor) -> dict[str, torch.Tensor]:
        particles[:, ColumnMap.T] = 123.0
        return {"EFlowObject": particles, "Track": particles[:0], "Tower": particles[:0]}

    gen.card = _make_stub_card({})
    gen.card.side_effect = _mutating_card
    gen.initialize(n_events=1, n_batches=1)

    gen.process_batch({
        "stable_particles": truth,
        "all_particles": truth.clone(),
        "event_numbers": torch.tensor([1]),
        "n_particles": torch.tensor([2]),
    })

    events = gen.get_events()
    assert len(events) == 1
    assert events[0].truth_particles.t is not None
    np.testing.assert_allclose(
        events[0].truth_particles.t, np.zeros_like(events[0].truth_particles.t)
    )


def test_generator_process_batch_preserves_events_without_stable_particles():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    stable = _particle_torch(1, event_number=2)
    all_particles = torch.cat([_particle_torch(1, event_number=1, status=2), stable])
    gen.card = _make_stub_card({"EFlowObject": stable, "Track": stable[:0], "Tower": stable[:0]})
    gen.initialize(n_events=2, n_batches=1)

    gen.process_batch({
        "stable_particles": stable,
        "all_particles": all_particles,
        "event_numbers": torch.tensor([1, 2]),
        "n_particles": torch.tensor([1, 1]),
    })

    events = gen.get_events()
    assert [event.event_number for event in events] == [1, 2]
    assert len(events[0].truth_particles) == 0
    assert len(events[0].pflow_particles) == 0
    assert len(events[1].truth_particles) == 1


def test_generator_process_batch_skips_empty_without_calling_card():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    stub_card = _make_stub_card({})
    gen.card = stub_card
    gen.initialize(n_events=0, n_batches=1)

    gen.process_batch({
        "stable_particles": torch.zeros((0, N_FEATURES), dtype=torch.float64),
        "all_particles": torch.zeros((0, N_FEATURES), dtype=torch.float64),
        "event_numbers": torch.tensor([]),
        "n_particles": torch.tensor([]),
    })

    stub_card.assert_not_called()
    gen.get_events()


def test_generator_guards_before_initialize():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    with pytest.raises(AssertionError):
        gen.get_events()
    with pytest.raises(AssertionError):
        gen.process_batch({
            "stable_particles": torch.zeros((1, N_FEATURES)),
            "all_particles": torch.zeros((1, N_FEATURES)),
        })


def test_generator_context_manager_resets_dtype_to_float32():
    gen = ParametricEventGenerator(_make_config(), _STUB_LOG)
    gen.card = _make_stub_card({})
    with gen:
        gen.initialize(n_events=0, n_batches=1)
    assert torch.get_default_dtype() == torch.float32


def test_get_accessors_exposes_four_branches():
    accessors = ParametricEventGenerator(_make_config(), _STUB_LOG).get_accessors()
    assert set(accessors.keys()) == {"Truth", "Pflow", "Track", "Tower"}


def test_get_accessors_particle_has_expected_output_names():
    accessors = ParametricEventGenerator(_make_config(), _STUB_LOG).get_accessors()
    truth_names = {a.output_name for a in accessors["Truth"]}
    assert {"PT", "Eta", "Phi", "Mass", "PID", "Status"} <= truth_names
    assert "D0" not in truth_names  # no impact parameters for parametric


def test_get_accessors_tower_output_names_and_collection():
    accessors = ParametricEventGenerator(_make_config(), _STUB_LOG).get_accessors()
    tower_accessors = accessors["Tower"]
    assert {a.output_name for a in tower_accessors} == {"E", "ET", "Eta", "Phi", "T"}
    assert all(a.collection == "Tower" for a in tower_accessors)

"""Tests for NeuralEventGenerator internals and the cms_2011 pretrained model."""

import numpy as np
import pytest
import torch

from parnassus.configs.generators.neural import NEURAL_GENERATORS_REGISTRY, NeuralGeneratorConfig
from parnassus.pipelines.generators.neural import (
    NeuralEventGenerator,
    _GenerationBuffers,
)
from parnassus.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# _GenerationBuffers
# ---------------------------------------------------------------------------


def _make_buffers(n_events: int, max_particles: int = 2) -> _GenerationBuffers:
    keys = ["pt", "eta", "phi", "vx", "vy", "vz", "class", "ind"]
    return _GenerationBuffers(
        truth_data={k: np.zeros((n_events, max_particles), dtype=np.float32) for k in keys},
        pflow_data={k: np.zeros((n_events, max_particles), dtype=np.float32) for k in keys},
        event_numbers=np.arange(1, n_events + 1, dtype=np.int32),
        count=n_events,
    )


def test_buffers_trim_slices_to_count():
    """trim() must discard rows beyond count."""
    buffers = _make_buffers(n_events=5)
    buffers.count = 2
    trimmed = buffers.trim()

    assert trimmed.truth_data["pt"].shape == (2, 2)
    assert trimmed.pflow_data["pt"].shape == (2, 2)
    assert np.array_equal(trimmed.event_numbers, [1, 2])


# ---------------------------------------------------------------------------
# Registry and config
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cms_config() -> NeuralGeneratorConfig:
    return NEURAL_GENERATORS_REGISTRY["cms_2011_flow_v00"]  # type: ignore[return-value]


def test_registry_contains_cms_2011(cms_config):
    assert cms_config.max_particles == 400
    assert cms_config.impact_model_config is not None  # includes impact parameter model


def test_registry_config_has_correct_fs_vars(cms_config):
    assert "npflow" in cms_config.event_model_config.fs_vars
    assert "pflow_ptrel" in cms_config.particle_model_config.fs_vars
    assert "pflow_d0" in cms_config.impact_model_config.fs_vars  # type: ignore[union-attr]


def test_set_num_steps_propagates_to_all_models(cms_config):
    import copy

    cfg = copy.deepcopy(cms_config)
    cfg.set_num_steps(3)

    assert cfg.event_model_config.sampler_config.num_steps == 3
    assert cfg.particle_model_config.sampler_config.num_steps == 3
    assert cfg.impact_model_config is not None
    assert cfg.impact_model_config.sampler_config.num_steps == 3


# ---------------------------------------------------------------------------
# Generator instantiation and smoke generation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cms_generator(cms_config) -> NeuralEventGenerator:
    import copy

    cfg = copy.deepcopy(cms_config)
    cfg.set_num_steps(1)  # single step — fast but exercises the full sampling path
    return NeuralEventGenerator(cfg, setup_logger())


def test_generator_loads_all_models(cms_generator):
    assert cms_generator.event_model is not None
    assert cms_generator.particle_model is not None
    assert cms_generator.impact_model is not None
    assert cms_generator.has_impact_model


def test_generator_smoke_generation(cms_generator):
    """End-to-end: process one fake batch and verify GenEvent objects are returned."""
    cfg = cms_generator.config
    max_p = cfg.max_particles

    # ctxt_vars = [ptrel, eta, phi, vx, vy, vz, class]
    # After encoding: phi → (sin, cos) = 2 values; class → one_hot(5) = 5 values; others = 1 each
    # Total is: 1 + 1 + 2 + 1 + 1 + 1 + 5 = 12
    N_CTXT = 12

    # ctxt_global_vars_stripped filters pflow_* → ["means", "ht", "met_x", "met_y", "ntruth"]
    # "means" expands to 6 (non-class: ptrel, eta, phi, vx, vy, vz); 4 scalars → total 10
    N_GLOBAL = 10

    # Build a minimal fake batch: 2 events, each with a few non-masked particles
    batch_size = 2
    n_real = 5  # particles per event

    ctxt = torch.zeros(batch_size, max_p, N_CTXT)
    ctxt[:, :n_real, :] = 0.1  # small non-zero context

    mask = torch.zeros(batch_size, max_p, dtype=torch.bool)
    mask[:, :n_real] = True

    global_data = torch.zeros(batch_size, N_GLOBAL)
    global_data[:, 0] = 1.0  # means slot

    event_number = torch.arange(batch_size, dtype=torch.long).unsqueeze(-1)

    batch = {
        "ctxt_data": ctxt,
        "ctxt_global_data": global_data,
        "mask": mask,
        "event_number": event_number,
    }

    with cms_generator:
        cms_generator.initialize(n_events=batch_size, n_batches=1)
        cms_generator.process_batch(batch)
        events = cms_generator.get_events()

    # Some events may be filtered by good_evt_mask; we only require no crash
    # and that any returned events have the right structure.
    assert isinstance(events, list)
    for ev in events:
        assert ev.truth_particles is not None
        assert ev.pflow_particles is not None
        assert ev.pflow_particles.d0 is not None  # impact params present

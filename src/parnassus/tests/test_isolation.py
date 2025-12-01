import re

import numpy as np
import pytest

from parnassus.configs.pipeline import IsolationConfig
from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.pipelines.isolation import (
    IsolationPipeline,
    calculate_isolation,
    calculate_lepton_isolation,
    calculate_photon_isolation,
)
from parnassus.utils import calculate_dr
from parnassus.utils.typing import FloatArray, IntArray

type IsolationInputData = tuple[FloatArray, FloatArray, IntArray]


@pytest.fixture
def sample_data() -> IsolationInputData:
    # Create sample test data
    pt = np.array([10.0, 20.0, 5.0, 15.0, 8.0])
    eta = np.array([0.1, 1.1, 0.2, -0.5, 0.3])
    phi = np.array([0.1, 2.1, 0.3, -1.0, 0.5])
    dR_matrix = calculate_dr(eta[:, None], phi[:, None], eta[None, :], phi[None, :])
    # class_id: 0=charged hadron, 1=electron, 2=muon, 3=neutral hadron, 4=photon
    class_id = np.array([0, 1, 3, 4, 2])
    return pt, dR_matrix, class_id


@pytest.mark.parametrize("dr_cut", [0.3, 0.4, 0.5])
def test_calculate_photon_isolation(sample_data: IsolationInputData, dr_cut: float):
    pt, dR_matrix, class_id = sample_data

    iso_scores = calculate_photon_isolation(pt, dR_matrix, class_id, dr_cut)

    # Test photon isolation calculation
    assert isinstance(iso_scores, np.ndarray)
    assert iso_scores.shape == (5,)  # One score per row in dR_matrix

    # Non-photon particles should have isolation score of 1000
    mask = class_id != 4
    assert np.all(iso_scores[mask] == 1000)


@pytest.mark.parametrize("lepton_id", [1, 2])
@pytest.mark.parametrize("dr_cut", [0.3, 0.4, 0.5])
def test_calculate_lepton_isolation(sample_data: IsolationInputData, lepton_id: int, dr_cut: float):
    pt, dR_matrix, class_id = sample_data
    iso_data = calculate_lepton_isolation(lepton_id, pt, dR_matrix, class_id, dr_cut)

    # Check shape and content
    assert isinstance(iso_data, np.ndarray)
    # (n_leptons, [pt_sum, pt_sum_ch, pt_sum_neut, iso_score])
    assert iso_data.shape == (1, 4)

    # Test that isolation scores are non-negative
    assert np.all(iso_data >= 0)


@pytest.mark.parametrize("lepton_id", [1, 2])
@pytest.mark.parametrize("dr_cut", [0.3, 0.4, 0.5])
def test_calculate_isolation(lepton_id: int, dr_cut: float):
    # Create minimal test event
    particles = GenParticleCollection(
        name="test_particles",
        pt=np.array([10.0, 20.0, 5.0, 15.0, 8.0, 12.0]),
        eta=np.array([0.1, 1.1, 0.2, -0.5, 0.3, 0.4]),
        phi=np.array([0.1, 2.1, 0.3, -1.0, 0.5, 1.5]),
        class_id=np.array([
            0,  # charged hadron
            1,  # electron
            3,  # neutral hadron
            2,  # muon
            4,  # photon
            2,  # muon
        ]),
    )

    config = IsolationConfig(name="test_iso", dr=dr_cut, collection="all", num_processes=1)

    # Convert GenParticleCollection to particle data dictionary
    particle_data = {
        "pt": particles.pt,
        "eta": particles.eta,
        "phi": particles.phi,
        "class_id": particles.class_id,
    }
    iso_data = calculate_isolation(lepton_id, particle_data, config)

    assert isinstance(iso_data, np.ndarray)
    # (n_leptons, [pt_sum, pt_sum_ch, pt_sum_neut, iso_score])
    if lepton_id == 1:
        assert iso_data.shape == (1, 4)
    elif lepton_id == 2:
        assert iso_data.shape == (2, 4)


def test_calculate_isolation_scores():
    pt = np.array([
        1.0821273,
        34.719383,
        1.2258172,
        30.221874,
        1.2549235,
        1.0460212,
        8.374317,
        6.3921337,
        2.4247317,
        1.2531377,
        1.3789331,
        2.4851658,
        1.9721998,
        1.1720936,
        1.5717405,
    ])
    eta = np.array([
        -0.04862569,
        -0.24119054,
        -1.0451822,
        1.6813488,
        1.1240231,
        0.9329206,
        1.9190673,
        2.1989734,
        -2.107885,
        -2.5792665,
        2.444524,
        2.8127582,
        2.4132555,
        2.9101477,
        -0.9159662,
    ])
    phi = np.array([
        -2.2690468,
        -2.1760383,
        -1.4556972,
        -0.0147252,
        0.3480272,
        1.3035926,
        1.5603805,
        2.3269625,
        -2.0593581,
        -3.0620444,
        2.962961,
        -0.5954394,
        1.8368979,
        0.95319086,
        2.8386505,
    ])
    class_id = np.array([0, 2, 0, 2, 0, 0, 2, 0, 3, 3, 4, 3, 3, 3, 3], dtype=np.int32)

    particles = GenParticleCollection(
        name="test_particles", pt=pt, eta=eta, phi=phi, class_id=class_id
    )

    config = IsolationConfig(name="test_iso", dr=0.4, collection="all", num_processes=1)

    # Convert GenParticleCollection to particle data dictionary
    particle_data = {
        "pt": particles.pt,
        "eta": particles.eta,
        "phi": particles.phi,
        "class_id": particles.class_id,
    }
    iso_data = calculate_isolation(2, particle_data, config)
    np.testing.assert_allclose(iso_data[:, 0], np.array([35.80151, 30.221874, 8.374317]))
    np.testing.assert_allclose(iso_data[:, 1], np.array([1.0821273, 0.0, 0.0]))
    np.testing.assert_allclose(iso_data[:, 2], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(iso_data[:, 3], np.array([0.0311678148, 0.0, 0.0]))


def test_calculate_isolation_batch():
    # Create test particle data for a batch
    from parnassus.pipelines.isolation import calculate_isolation_batch

    particle_data_batch = [
        {
            "pt": np.array([10.0, 20.0, 5.0, 15.0, 8.0, 12.0]),
            "eta": np.array([0.1, 1.1, 0.2, -0.5, 0.3, 0.4]),
            "phi": np.array([0.1, 2.1, 0.3, -1.0, 0.5, 1.5]),
            "class_id": np.array([
                0,  # charged hadron
                1,  # electron
                3,  # neutral hadron
                2,  # muon
                4,  # photon
                2,  # muon
            ]),
        }
    ]

    config = IsolationConfig(name="test_iso", dr=0.4, collection="all", num_processes=1)

    electrons_data, muons_data = calculate_isolation_batch(particle_data_batch, config)

    assert len(electrons_data) == len(particle_data_batch)
    assert len(muons_data) == len(particle_data_batch)
    # (n_electrons/muons, [pt_sum, pt_sum_ch, pt_sum_neut, iso_score])
    assert electrons_data[0].shape == (1, 4)
    assert muons_data[0].shape == (2, 4)


@pytest.mark.parametrize("num_processes", [1, 2])
def test_isolation_pipeline(num_processes: int):
    # Test the full pipeline
    config = IsolationConfig(
        name="test_iso", dr=0.4, collection="all", num_processes=num_processes, batch_size=50
    )
    pipeline = IsolationPipeline(config)

    # Check accessors
    accessors = pipeline.get_accessors()
    assert "Electrons" in accessors
    assert "Muons" in accessors
    assert len(accessors["Electrons"]) == 4  # iso_var, sum_pt, sum_pt_ch, sum_pt_neut
    assert len(accessors["Muons"]) == 4  # iso_var, sum_pt, sum_pt_ch, sum_pt_neut

    # Test processing
    pflow_particles = GenParticleCollection(
        name="pflow",
        pt=np.array([10.0, 5.0, 50, 40]),
        eta=np.array([0.1, 1.1, 0.0, 1.0]),
        phi=np.array([0.1, 1.1, 0.0, 1.0]),
        class_id=np.array([0, 3, 1, 2]),  # charged hadron, neutral hadron, electron, muon
    )

    truth_particles = GenParticleCollection(
        name="truth", pt=np.array([]), eta=np.array([]), phi=np.array([]), class_id=np.array([])
    )

    event = GenEvent(
        event_number=1, truth_particles=truth_particles, pflow_particles=pflow_particles
    )

    event_list = [event for _ in range(200)]
    pipeline.process(event_list)

    # Check that isolation variables were added to the event
    for event in event_list:
        assert hasattr(event.electrons, "iso_var")
        assert hasattr(event.electrons, "sum_pt")
        assert hasattr(event.electrons, "sum_pt_ch")
        assert hasattr(event.electrons, "sum_pt_neut")
        assert hasattr(event.muons, "iso_var")
        assert hasattr(event.muons, "sum_pt")
        assert hasattr(event.muons, "sum_pt_ch")
        assert hasattr(event.muons, "sum_pt_neut")


def test_invalid_collection():
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Requested isolation for invalid, only "electrons" and "muons", '
            'and "all" (both of them) are supported.'
        ),
    ):
        _ = IsolationConfig(name="test_iso", dr=0.4, collection="invalid", num_processes=1)

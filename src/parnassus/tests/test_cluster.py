import fastjet as fj
import numpy as np
import pytest

from parnassus.configs.pipeline import JetClusteringConfig
from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.pipelines.cluster import Jet


@pytest.fixture
def mock_particle_collection() -> GenParticleCollection:
    return GenParticleCollection(
        name="test_particles",
        pt=np.array([50.0, 30.0, 20.0]),
        eta=np.array([0.4, 0.6, 0.5]),
        phi=np.array([0.9, 1.1, 1.0]),
        pdg_id=np.array([211, 211, 211]),
    )


def test_get_cluster_sequence(mock_particle_collection: GenParticleCollection):
    """Test get_cluster_sequence function."""
    from parnassus.pipelines.cluster import get_cluster_sequence

    jetdef = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    four_vectors = mock_particle_collection.get4vecs_awkward()
    user_indices = list(range(len(mock_particle_collection)))
    # Extract numpy arrays from awkward array
    px = np.array([v.px.item() for v in four_vectors])
    py = np.array([v.py.item() for v in four_vectors])
    pz = np.array([v.pz.item() for v in four_vectors])
    E = np.array([v.E.item() for v in four_vectors])
    cs = get_cluster_sequence(jetdef, px, py, pz, E, user_indices)
    assert len(cs.inclusive_jets(0.0)) == 1, "Expected to have one jet."
    assert len(cs.inclusive_jets(0.0)[0].constituents()) == 3, (
        "Expected to have 3 particles in jet."
    )


def test_cluster_jets(mock_particle_collection: GenParticleCollection):
    """Test cluster_jets function."""
    from parnassus.pipelines.cluster import cluster_jets

    config = JetClusteringConfig(
        name="test_cluster", algorithm="antikt", dr=0.4, nconst_min=2, min_pt=0
    )
    # Convert GenParticleCollection to particle data dictionary
    four_vectors = mock_particle_collection.get4vecs_numpy()
    particle_data = {
        "px": four_vectors[..., 0],
        "py": four_vectors[..., 1],
        "pz": four_vectors[..., 2],
        "e": four_vectors[..., 3],
    }
    jets, idxs = cluster_jets(particle_data, config)
    assert len(jets) == 1
    assert idxs.shape == (3,)
    assert jets[0].nconstituents == 3


def get_mock_constituent(pt: float, eta: float, phi: float, user_index: int) -> fj.PseudoJet:
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = pt * np.cosh(eta)
    pseudojet = fj.PseudoJet(px, py, pz, E)
    pseudojet.set_user_index(user_index)
    return pseudojet


@pytest.fixture
def mock_fj_jet_cs():
    # Add constituents
    constituents = [
        get_mock_constituent(pt=30.0, eta=0.6, phi=1.1, user_index=0),
        get_mock_constituent(pt=50.0, eta=0.4, phi=0.9, user_index=1),
        get_mock_constituent(pt=20.0, eta=0.5, phi=1.0, user_index=2),
    ]
    jetdef = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    cs = fj.ClusterSequence(constituents, jetdef)
    return fj.sorted_by_pt(cs.inclusive_jets(0.0)), cs


@pytest.fixture
def mock_jet(mock_fj_jet_cs) -> Jet:
    mock_fj_jet, _ = mock_fj_jet_cs
    return Jet(mock_fj_jet[0], dr=0.4, calc_substructure=True)


def test_jet_init_basic(mock_fj_jet_cs):
    """Test basic Jet initialization without substructure calculation."""
    mock_fj_jet, _ = mock_fj_jet_cs
    jet = Jet(mock_fj_jet[0], dr=0.4, calc_substructure=False)
    assert jet.dR == 0.4
    assert jet.nconstituents == 3
    assert len(jet.constituents_pt) == 3
    assert len(jet.constituents_eta) == 3
    assert len(jet.constituents_phi) == 3
    assert len(jet.constituents_m) == 3
    assert len(jet.constituents_idx) == 3

    # Check if constituents are ordered by pt
    assert np.all(np.diff(jet.constituents_pt) <= 0)  # Should be descending

    # Check initial substructure values
    assert np.isnan(jet.substructure["c2"])
    assert np.isnan(jet.substructure["d2"])


def test_jet_init_with_substructure(mock_jet: Jet):
    """Test Jet initialization with substructure calculation."""
    # Check if substructure values were calculated
    assert not np.isnan(mock_jet.substructure["c2"])
    assert not np.isnan(mock_jet.substructure["d2"])


def test_jet_getattr(mock_jet: Jet):
    """Test __getattr__ method for accessing PseudoJet properties."""
    np.testing.assert_approx_equal(mock_jet.pt(), 99.62034462103314)
    np.testing.assert_approx_equal(mock_jet.eta(), 0.48344593734981606)
    np.testing.assert_approx_equal(mock_jet.phi(), 0.9799558810407399)
    np.testing.assert_approx_equal(mock_jet.m(), 12.328849995939832)


def test_pt_ordering(mock_jet: Jet):
    """Test if constituents are properly ordered by pt."""
    # Check if constituents are ordered by decreasing pt
    np.testing.assert_allclose(mock_jet.constituents_pt, np.array([50.0, 30.0, 20.0]))
    # Check if other arrays maintain the same ordering
    np.testing.assert_allclose(mock_jet.constituents_eta, np.array([0.4, 0.6, 0.5]))
    np.testing.assert_allclose(mock_jet.constituents_idx, np.array([1, 0, 2]))


def test_invalid_attribute(mock_jet: Jet):
    """Test accessing invalid attribute."""
    with pytest.raises(AttributeError):
        _ = mock_jet.invalid_attribute


def test_convert_to_jet_collection(mock_jet: Jet):
    from parnassus.pipelines.cluster import convert_to_jet_collection

    jet_collection = convert_to_jet_collection("Test", [mock_jet, mock_jet])
    assert jet_collection.name == "Test"
    assert len(jet_collection) == 2
    np.testing.assert_allclose(jet_collection.pt, np.array([99.62034462103314, 99.62034462103314]))
    np.testing.assert_allclose(
        jet_collection.eta, np.array([0.48344593734981606, 0.48344593734981606])
    )


def test_jet_clustering_batch(mock_particle_collection: GenParticleCollection):
    """Test jet clustering in batch mode."""
    from parnassus.pipelines.cluster import cluster_jets_batch

    config = JetClusteringConfig(
        name="test_cluster", algorithm="antikt", dr=0.4, nconst_min=2, min_pt=0
    )
    # Convert GenParticleCollection to particle data dictionary
    four_vectors = mock_particle_collection.get4vecs_numpy()
    particle_data = {
        "px": four_vectors[..., 0],
        "py": four_vectors[..., 1],
        "pz": four_vectors[..., 2],
        "e": four_vectors[..., 3],
    }
    jets_collection_batch, idxs_batch = cluster_jets_batch(
        [particle_data, particle_data, particle_data], config
    )
    assert len(jets_collection_batch) == 3
    for jets, idxs in zip(jets_collection_batch, idxs_batch, strict=True):
        assert idxs.shape == (3,)
        assert jets.num_jets == 1


@pytest.mark.parametrize("num_processes", [1, 2])
def test_jet_clustering_pipeline(
    mock_particle_collection: GenParticleCollection, num_processes: int
):
    """Test the full jet clustering pipeline."""
    from parnassus.pipelines.cluster import JetClusteringPipeline

    config = JetClusteringConfig(
        name="test_cluster",
        algorithm="antikt",
        dr=0.4,
        nconst_min=2,
        min_pt=0,
        redirect_stdout=True,
        batch_size=50,
        num_processes=num_processes,
    )
    pipeline = JetClusteringPipeline(config)
    event_list = [
        GenEvent(
            event_number=i,
            pflow_particles=mock_particle_collection,
            truth_particles=mock_particle_collection,
        )
        for i in range(200)
    ]

    pipeline.process(event_list)

    idx = np.random.randint(0, len(event_list))
    assert "test_cluster" in event_list[idx].jets
    jets = event_list[idx].jets["test_cluster"]
    assert jets.num_jets == 1
    np.testing.assert_allclose(jets.pt, np.array([99.62034462103314]))
    np.testing.assert_allclose(jets.eta, np.array([0.48344593734981606]))
    np.testing.assert_allclose(jets.phi, np.array([0.9799558810407399]))

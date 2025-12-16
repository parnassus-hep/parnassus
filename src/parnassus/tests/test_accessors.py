import numpy as np
import pytest

from parnassus.configs.accessors import (
    AccessorError,
    AccessorListBuilder,
    AccessorSpec,
    AccessorStore,
    AccessorTemplates,
    JetAccessor,
    ParticleAccessor,
)
from parnassus.configs.scheme import GenParticleCollection


# Fixtures
@pytest.fixture
def mock_particle_collection():
    """Create a mock particle collection for testing."""
    return GenParticleCollection(
        name="test_particles",
        pt=np.array([10.0, 20.0, 30.0]),
        eta=np.array([0.5, 1.0, 1.5]),
        phi=np.array([0.0, 1.57, 3.0]),
        mass=np.array([0.0, 0.0, 0.0]),
        pdg_id=np.array([11, 13, 22]),
        d0=np.array([0.1, 0.2, 0.3]),
        z0=np.array([1.0, 2.0, 3.0]),
        d0_error=np.array([0.01, 0.02, 0.03]),
        z0_error=np.array([0.1, 0.2, 0.3]),
        jet_idx={"ak04": np.array([0, 1, 0])},
    )


@pytest.fixture
def mock_event(mock_particle_collection):
    """Create a mock GenEvent for testing."""

    class MockJetCollection:
        def __init__(self):
            self.pt = np.array([50.0, 60.0])
            self.eta = np.array([0.0, 1.0])
            self.phi = np.array([0.0, 1.57])

    class MockJets:
        def __init__(self):
            self.ak04 = MockJetCollection()

        def __getitem__(self, key):
            return getattr(self, key)

    class MockEvent:
        def __init__(self):
            self.particles = mock_particle_collection
            self.jets = MockJets()

    return MockEvent()


# Test Accessor base class
class TestAccessor:
    """Test suite for Accessor base class."""

    def test_invalid_dtype_raises_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            ParticleAccessor(name="pt", collection="particles", dtype="invalid_type")

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
    def test_valid_dtypes(self, dtype):
        """Test that valid dtypes are accepted."""
        accessor = ParticleAccessor(name="pt", collection="particles", dtype=dtype)
        assert accessor.dtype == dtype

    def test_output_name_defaults_to_name(self):
        """Test that output_name defaults to name if not provided."""
        accessor = ParticleAccessor(name="pt", collection="particles")
        assert accessor.output_name == "pt"

    def test_custom_output_name(self):
        """Test custom output_name."""
        accessor = ParticleAccessor(
            name="pt", collection="particles", output_name="transverse_momentum"
        )
        assert accessor.output_name == "transverse_momentum"


# Test ParticleAccessor
class TestParticleAccessor:
    """Test suite for ParticleAccessor class."""

    def test_get_simple_attribute(self, mock_event):
        """Test accessing simple particle attributes."""
        accessor = ParticleAccessor(name="pt", collection="particles")
        result = accessor.get(mock_event)
        np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))

    def test_get_nested_dict_attribute(self, mock_event):
        """Test accessing nested dictionary attributes."""
        accessor = ParticleAccessor(name="jet_idx/ak04", collection="particles")
        result = accessor.get(mock_event)
        np.testing.assert_array_equal(result, np.array([0, 1, 0]))

    def test_missing_collection_raises_error(self, mock_event):
        """Test that missing collection raises AccessorError."""
        accessor = ParticleAccessor(name="pt", collection="missing_collection")
        with pytest.raises(AccessorError, match="has no collection"):
            accessor.get(mock_event)

    def test_missing_attribute_raises_error(self, mock_event):
        """Test that missing attribute raises AccessorError."""
        accessor = ParticleAccessor(name="nonexistent", collection="particles")
        with pytest.raises(AccessorError, match="has no attribute"):
            accessor.get(mock_event)

    def test_missing_dict_field_raises_error(self, mock_event):
        """Test that missing dictionary field raises AccessorError."""
        accessor = ParticleAccessor(name="jet_idx/missing_field", collection="particles")
        with pytest.raises(AccessorError, match="Cannot access"):
            accessor.get(mock_event)


# Test JetAccessor
class TestJetAccessor:
    """Test suite for JetAccessor class."""

    def test_get_jet_attribute(self, mock_event):
        """Test accessing jet attributes."""
        accessor = JetAccessor(name="pt", collection="ak04")
        result = accessor.get(mock_event)
        np.testing.assert_array_equal(result, np.array([50.0, 60.0]))


# Test AccessorSpec
class TestAccessorSpec:
    """Test suite for AccessorSpec class."""

    def test_default_values(self):
        """Test AccessorSpec default values."""
        spec = AccessorSpec(name="pt")
        assert spec.name == "pt"
        assert not spec.output_name
        assert spec.dtype == "float32"

    def test_custom_values(self):
        """Test AccessorSpec with custom values."""
        spec = AccessorSpec(name="pt", output_name="momentum", dtype="float64")
        assert spec.name == "pt"
        assert spec.output_name == "momentum"
        assert spec.dtype == "float64"


# Test AccessorListBuilder
class TestAccessorListBuilder:
    """Test suite for AccessorListBuilder class."""

    def test_for_particles(self):
        """Test creating builder for particles."""
        accessors = AccessorListBuilder.for_particles("electrons").add(["pt"]).build()
        assert accessors[0].collection == "electrons"
        assert isinstance(accessors[0], ParticleAccessor)

    def test_for_jets(self):
        """Test creating builder for jets."""
        accessors = AccessorListBuilder.for_jets("ak04").add(["pt"]).build()
        assert accessors[0].collection == "ak04"
        assert isinstance(accessors[0], JetAccessor)

    def test_add_single_name(self):
        """Test adding accessors with single dtype."""
        accessors = AccessorListBuilder.for_particles("electrons").add(["pt", "eta", "phi"]).build()
        assert len(accessors) == 3
        assert all(a.collection == "electrons" for a in accessors)
        assert [a.name for a in accessors] == ["pt", "eta", "phi"]
        assert all(a.dtype == "float32" for a in accessors)

    def test_add_with_dtype_list(self):
        """Test adding accessors with different dtypes."""
        accessors = (
            AccessorListBuilder.for_particles("electrons")
            .add(["pt", "class_id"], dtype=["float32", "int32"])
            .build()
        )
        assert len(accessors) == 2
        assert accessors[0].dtype == "float32"
        assert accessors[1].dtype == "int32"

    def test_add_dtype_mismatch_raises_error(self):
        """Test that mismatched dtype list raises error."""
        builder = AccessorListBuilder.for_particles("electrons")
        with pytest.raises(ValueError, match="Length of dtype list"):
            builder.add(["pt", "eta"], dtype=["float32"])

    def test_add_with_output(self):
        """Test adding accessor with custom output name."""
        accessors = (
            AccessorListBuilder.for_particles("electrons")
            .add_with_output("pt", "electron_pt", dtype="float64")
            .build()
        )
        assert len(accessors) == 1
        assert accessors[0].name == "pt"
        assert accessors[0].output_name == "electron_pt"
        assert accessors[0].dtype == "float64"

    def test_add_from_specs(self):
        """Test adding accessors from specs."""
        specs = [
            AccessorSpec("pt", dtype="float32"),
            AccessorSpec("eta", dtype="float32"),
        ]
        accessors = AccessorListBuilder.for_particles("electrons").add_from_specs(specs).build()
        assert len(accessors) == 2
        assert [a.name for a in accessors] == ["pt", "eta"]

    def test_chaining_multiple_operations(self):
        """Test chaining multiple builder operations."""
        accessors = (
            AccessorListBuilder.for_particles("electrons")
            .add(["pt", "eta"])
            .add_with_output("phi", "azimuth")
            .add(["class_id"], dtype=["int32"])
            .build()
        )
        assert len(accessors) == 4
        assert accessors[2].output_name == "azimuth"
        assert accessors[3].dtype == "int32"


# Test AccessorTemplates
class TestAccessorTemplates:
    """Test suite for AccessorTemplates class."""

    def test_kinematics_template(self):
        """Test KINEMATICS template."""
        assert len(AccessorTemplates.KINEMATICS) == 3
        names = [spec.name for spec in AccessorTemplates.KINEMATICS]
        assert names == ["pt", "eta", "phi"]

    def test_impact_parameters_template(self):
        """Test IMPACT_PARAMETERS template."""
        assert len(AccessorTemplates.IMPACT_PARAMETERS) == 4
        names = [spec.name for spec in AccessorTemplates.IMPACT_PARAMETERS]
        assert names == ["d0", "z0", "d0_error", "z0_error"]

    def test_templates_can_be_used_with_builder(self):
        """Test that templates work with builder."""
        accessors = (
            AccessorListBuilder.for_particles("electrons")
            .add_from_specs(AccessorTemplates.KINEMATICS)
            .build()
        )
        assert len(accessors) == 3
        assert all(a.collection == "electrons" for a in accessors)


# Test AccessorStore
class TestAccessorStore:
    """Test suite for AccessorStore class."""

    def test_empty_store(self):
        """Test creating empty AccessorStore."""
        store = AccessorStore()
        assert store.accessors_dict == {}

    def test_from_dict(self):
        """Test creating AccessorStore from dict."""
        accessors = [
            ParticleAccessor(name="pt", collection="electrons"),
            ParticleAccessor(name="eta", collection="electrons"),
        ]
        store = AccessorStore.from_dict({"electrons": accessors})
        assert "electrons" in store.accessors_dict
        assert len(store.accessors_dict["electrons"]) == 2

    def test_update_from_dict_new_collection(self):
        """Test updating store with new collection."""
        store = AccessorStore()
        accessors = [ParticleAccessor(name="pt", collection="electrons")]
        store.update_from_dict({"electrons": accessors})
        assert "electrons" in store.accessors_dict
        assert len(store.accessors_dict["electrons"]) == 1

    def test_update_from_dict_existing_collection(self):
        """Test updating store with existing collection."""
        accessor1 = ParticleAccessor(name="pt", collection="electrons")
        store = AccessorStore.from_dict({"electrons": [accessor1]})
        accessor2 = ParticleAccessor(name="eta", collection="electrons")
        store.update_from_dict({"electrons": [accessor2]})
        assert len(store.accessors_dict["electrons"]) == 2

    def test_update_from_dict_no_duplicates(self):
        """Test that update doesn't add duplicates."""
        accessor = ParticleAccessor(name="pt", collection="electrons")
        store = AccessorStore.from_dict({"electrons": [accessor]})
        store.update_from_dict({"electrons": [accessor]})
        assert len(store.accessors_dict["electrons"]) == 1

    def test_get_branch_types(self):
        """Test getting branch types for ROOT output."""
        accessors = [
            ParticleAccessor(name="pt", collection="electrons", dtype="float32"),
            ParticleAccessor(name="class_id", collection="electrons", dtype="int32"),
        ]
        store = AccessorStore.from_dict({"electrons": accessors})
        branch_types = store.get_branch_types()
        assert "electrons" in branch_types
        assert '"pt" : float32' in branch_types["electrons"]
        assert '"class_id" : int32' in branch_types["electrons"]

    def test_init_data_dict(self):
        """Test initializing data dictionary."""
        accessors = [
            ParticleAccessor(name="pt", collection="electrons", output_name="electron_pt"),
            ParticleAccessor(name="eta", collection="electrons", output_name="electron_eta"),
        ]
        store = AccessorStore.from_dict({"electrons": accessors})
        data_dict = store.init_data_dict()
        assert "electrons" in data_dict
        assert "electron_pt" in data_dict["electrons"]
        assert "electron_eta" in data_dict["electrons"]
        assert data_dict["electrons"]["electron_pt"] == []

    def test_update_data_dict(self, mock_event):
        """Test updating data dictionary with event data."""
        accessors = [
            ParticleAccessor(name="pt", collection="particles"),
            ParticleAccessor(name="eta", collection="particles"),
        ]
        store = AccessorStore.from_dict({"particles": accessors})
        data_dict = store.init_data_dict()
        store.update_data_dict(mock_event, data_dict)
        assert len(data_dict["particles"]["pt"]) == 1
        np.testing.assert_array_equal(data_dict["particles"]["pt"][0], np.array([10.0, 20.0, 30.0]))

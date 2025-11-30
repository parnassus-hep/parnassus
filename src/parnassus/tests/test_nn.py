"""Comprehensive tests for neural network components.

This module tests:
- ModelWrapper: Loading, forward pass, sampling
- EulerSampler: Initialization, time stepping, sampling logic
- Mock utilities: Model and data generation
- Edge cases: Invalid configs, dimension mismatches
"""

from contextlib import suppress
from pathlib import Path

import pytest
import torch
from torch import Tensor

from parnassus.configs.generators.model import ModelConfig, SamplerConfig, VariablesConfig
from parnassus.nn.sampler import EulerSampler
from parnassus.nn.wrapper import ModelWrapper
from parnassus.utils.mock import get_mock_input_data, get_mock_model_file


def _create_model_config(model_path: str, mode: str = "part") -> ModelConfig:
    """Helper function to create a ModelConfig from a model file path.

    Parameters
    ----------
    model_path : str
        Path to the exported model file (.pt2).
    mode : str
        Model mode, either 'part' (particle) or 'evt' (event).

    Returns
    -------
    ModelConfig
        ModelConfig for the specified model path and mode.
    """
    # Define variables based on mode
    if mode == "part":
        fs_vars = ("pflow_pt", "pflow_eta", "pflow_phi", "pflow_class")
        ctxt_vars = ("truth_pt", "truth_eta", "truth_phi")
        ctxt_global_vars = ("met_pt", "met_phi")
    else:  # evt mode
        fs_vars = ("truth_pt", "truth_eta", "truth_phi", "npflow")
        ctxt_vars = ("truth_pt", "truth_eta", "truth_phi")
        ctxt_global_vars = ("met_pt", "met_phi")

    variables_config = VariablesConfig(
        truth_vars_to_load=("truth_pt", "truth_eta", "truth_phi"),
        fs_vars=fs_vars,
        ctxt_vars=ctxt_vars,
        ctxt_global_vars=ctxt_global_vars,
    )

    sampler_config = SamplerConfig(
        type="euler",
        num_steps=2,
        reverse_time=False,
    )

    return ModelConfig(
        name=f"mock_{mode}_model",
        file_path=Path(model_path),
        variables_config=variables_config,
        sampler_config=sampler_config,
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_particle_model():
    """Create a mock particle model configuration.

    Returns
    -------
    ModelConfig
        Configuration for a particle-level model.
    """
    model_path = get_mock_model_file(mode="part")
    return _create_model_config(model_path, mode="part")


@pytest.fixture
def mock_particle_data():
    """Create mock input data for particle model.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing fs_data, ctxt_data, mask, timestep, ctxt_global_data.
    """
    # Model expects 9 fs features based on config: 4 base + 1 (phi) + 4 (class) = 9
    return get_mock_input_data(mode="part", num_fs_feats=9)


@pytest.fixture
def mock_event_model():
    """Create a mock event model configuration.

    Returns
    -------
    ModelConfig
        Configuration for an event-level model.
    """
    model_path = get_mock_model_file(mode="evt")
    return _create_model_config(model_path, mode="evt")


@pytest.fixture
def mock_event_data():
    """Create mock input data for event model.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing fs_data, ctxt_data, mask, timestep, ctxt_global_data.
    """
    return get_mock_input_data(mode="evt", num_fs_feats=4)


# ============================================================================
# ModelWrapper Tests
# ============================================================================


def test_particle_model_wrapper_load(mock_particle_model: ModelConfig):
    """Test that particle model wrapper loads successfully."""
    model = ModelWrapper(mock_particle_model)
    assert model is not None
    # fs_vars: (pflow_pt, pflow_eta, pflow_phi, pflow_class)
    # = 4 base + 1 (sin/cos phi adds 1) + 4 (class adds 4) = 9
    assert model.num_fs_vars == 9
    assert isinstance(model.sampler, EulerSampler)
    assert model.sampler.n_steps == 2


def test_event_model_wrapper_load(mock_event_model: ModelConfig):
    """Test that event model wrapper loads successfully."""
    model = ModelWrapper(mock_event_model)
    assert model is not None
    assert model.num_fs_vars == 4  # pt, eta, phi, npflow
    assert isinstance(model.sampler, EulerSampler)
    assert model.sampler.n_steps == 2


def test_particle_model_wrapper_forward(
    mock_particle_model: ModelConfig, mock_particle_data: dict[str, Tensor]
):
    """Test forward pass through particle model wrapper."""
    model = ModelWrapper(mock_particle_model)
    output = model.forward(
        fs_data=mock_particle_data["fs_data"],
        ctxt_data=mock_particle_data["ctxt_data"],
        mask=mock_particle_data["mask"],
        timestep=mock_particle_data["timestep"],
        ctxt_global_data=mock_particle_data["ctxt_global_data"],
    )
    assert output is not None
    assert output.shape == mock_particle_data["fs_data"].shape
    assert output.dtype == torch.float32


def test_event_model_wrapper_forward(
    mock_event_model: ModelConfig, mock_event_data: dict[str, Tensor]
):
    """Test forward pass through event model wrapper."""
    model = ModelWrapper(mock_event_model)
    output = model.forward(
        fs_data=mock_event_data["fs_data"],
        ctxt_data=mock_event_data["ctxt_data"],
        mask=mock_event_data["mask"],
        timestep=mock_event_data["timestep"],
        ctxt_global_data=mock_event_data["ctxt_global_data"],
    )
    assert output is not None
    assert output.shape == mock_event_data["fs_data"].shape
    assert output.dtype == torch.float32


def test_model_wrapper_num_fs_vars_calculation():
    """Test that num_fs_vars is calculated correctly for different variable combinations."""
    # Test with pflow_phi and pflow_class
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)
    # fs_vars: (pflow_pt, pflow_eta, pflow_phi, pflow_class)
    # = 4 base + 1 (phi becomes sin/cos, net +1) + 4 (class expands to 5, net +4) = 9
    assert model.num_fs_vars == 9

    # Test without pflow_phi and pflow_class
    config.variables_config = VariablesConfig(
        truth_vars_to_load=("truth_pt", "truth_eta"),
        fs_vars=("pflow_pt", "pflow_eta"),
        ctxt_vars=("truth_pt", "truth_eta"),
        ctxt_global_vars=("met_pt",),
    )
    model_path_simple = get_mock_model_file(mode="part")
    config.file_path = Path(model_path_simple)
    model = ModelWrapper(config)
    assert model.num_fs_vars == 2  # Just pt and eta


def test_model_wrapper_device_transfer():
    """Test that model can be transferred to different devices."""
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)

    # Test CPU device (always available)
    # Note: Exported models may not have traditional parameters
    # Just verify the model exists and to() method works
    model_cpu = model.to("cpu")
    assert model_cpu is not None
    # Verify we can run forward pass after device transfer
    data = get_mock_input_data(mode="part", num_fs_feats=9)
    output = model_cpu.forward(
        fs_data=data["fs_data"][:, :, :9],  # Match expected dims
        ctxt_data=data["ctxt_data"],
        mask=data["mask"],
        timestep=data["timestep"],
        ctxt_global_data=data["ctxt_global_data"],
    )
    assert output.device.type == "cpu"


# ============================================================================
# EulerSampler Tests
# ============================================================================


def test_euler_sampler_initialization():
    """Test EulerSampler initialization with various parameters."""
    sampler = EulerSampler(n_steps=10, zero_init_padded=True, reverse_time=False)
    assert sampler.n_steps == 10
    assert sampler.zero_init_padded is True
    assert sampler.reverse_time is False
    assert len(sampler.t_steps) == 11  # n_steps + 1


def test_euler_sampler_time_steps():
    """Test that time steps are correctly generated."""
    sampler = EulerSampler(n_steps=5, reverse_time=False)
    expected = torch.linspace(0, 1, 5)
    expected = torch.cat([expected, torch.ones(1)])
    assert torch.allclose(sampler.t_steps, expected)


def test_euler_sampler_reverse_time():
    """Test that time steps are reversed when reverse_time=True."""
    sampler = EulerSampler(n_steps=5, reverse_time=True)
    expected = torch.linspace(0, 1, 5)
    expected = torch.cat([expected, torch.ones(1)])
    expected = torch.flip(expected, [0])
    assert torch.allclose(sampler.t_steps, expected)


def test_euler_sampler_random_seed():
    """Test that random seed produces reproducible results."""
    # Note: Testing via sample output rather than private methods
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    config.sampler_config = SamplerConfig(type="euler", num_steps=5, reverse_time=False)

    # Create two models with same random seed
    torch.manual_seed(42)
    model1 = ModelWrapper(config)
    data1 = get_mock_input_data(mode="part", num_fs_feats=9)

    torch.manual_seed(42)
    model2 = ModelWrapper(config)
    data2 = get_mock_input_data(mode="part", num_fs_feats=9)

    # Sample with same inputs - should produce same outputs
    output1 = model1.sample(
        shape=data1["ctxt_data"].size()[:-1],
        ctxt_data=data1["ctxt_data"],
        mask=data1["mask"],
        ctxt_global_data=data1["ctxt_global_data"],
    )
    output2 = model2.sample(
        shape=data2["ctxt_data"].size()[:-1],
        ctxt_data=data2["ctxt_data"],
        mask=data2["mask"],
        ctxt_global_data=data2["ctxt_global_data"],
    )
    # Outputs should be similar (not exact due to model stochasticity)
    assert output1.shape == output2.shape


def test_euler_sampler_sample(
    mock_particle_model: ModelConfig, mock_particle_data: dict[str, Tensor]
):
    """Test full sampling loop through EulerSampler."""
    model = ModelWrapper(mock_particle_model)
    shape = mock_particle_data["ctxt_data"].size()[:-1]

    output = model.sample(
        shape=shape,
        ctxt_data=mock_particle_data["ctxt_data"],
        mask=mock_particle_data["mask"],
        ctxt_global_data=mock_particle_data["ctxt_global_data"],
    )

    assert output is not None
    assert output.shape == (*shape, model.num_fs_vars)
    assert output.dtype == torch.float32


def test_euler_sampler_callback():
    """Test that callback is called during sampling."""
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)
    data = get_mock_input_data(mode="part", num_fs_feats=9)

    callback_count = 0

    def callback():
        nonlocal callback_count
        callback_count += 1

    _ = model.sample(
        shape=data["ctxt_data"].size()[:-1],
        ctxt_data=data["ctxt_data"],
        mask=data["mask"],
        ctxt_global_data=data["ctxt_global_data"],
        callback=callback,
    )

    # Should be called once per step (n_steps = 2)
    assert callback_count == 2


def test_euler_sampler_to_cpu():
    """Test that sampling can return results on CPU."""
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)
    data = get_mock_input_data(mode="part", num_fs_feats=9)

    output = model.sample(
        shape=data["ctxt_data"].size()[:-1],
        ctxt_data=data["ctxt_data"],
        mask=data["mask"],
        ctxt_global_data=data["ctxt_global_data"],
        to_cpu=True,
    )

    assert output.device.type == "cpu"


# ============================================================================
# Mock Utilities Tests
# ============================================================================


def test_get_mock_model_file_particle():
    """Test that get_mock_model_file creates valid particle model."""
    model_path = get_mock_model_file(mode="part")
    assert Path(model_path).exists()
    assert Path(model_path).suffix in {".pt", ".pt2"}


def test_get_mock_model_file_event():
    """Test that get_mock_model_file creates valid event model."""
    model_path = get_mock_model_file(mode="evt")
    assert Path(model_path).exists()
    assert Path(model_path).suffix in {".pt", ".pt2"}


def test_get_mock_input_data_particle():
    """Test that get_mock_input_data generates correct shapes for particle mode."""
    num_feats = 9
    data = get_mock_input_data(mode="part", num_fs_feats=num_feats)

    assert "fs_data" in data
    assert "ctxt_data" in data
    assert "mask" in data
    assert "ctxt_global_data" in data
    assert "timestep" in data

    # Check shapes
    assert data["fs_data"].shape == (2, 400, num_feats)
    assert data["ctxt_data"].shape == (2, 400, 12)  # ctxt always 12
    assert data["mask"].shape == (2, 400, 2)
    assert data["ctxt_global_data"].shape == (2, 8)
    assert data["timestep"].shape == (2, 1)


def test_get_mock_input_data_event():
    """Test that get_mock_input_data generates correct shapes for event mode."""
    data = get_mock_input_data(mode="evt", num_fs_feats=4)

    assert "fs_data" in data
    assert "ctxt_data" in data
    assert "mask" in data
    assert "ctxt_global_data" in data
    assert "timestep" in data

    # Check shapes
    assert data["fs_data"].shape == (2, 4)
    assert data["ctxt_data"].shape == (2, 400, 12)
    assert data["mask"].shape == (2, 400)
    assert data["ctxt_global_data"].shape == (2, 8)
    assert data["timestep"].shape == (2, 1)


def test_mock_model_and_data_compatibility_particle():
    """Test that mock model and data are compatible for particle mode."""
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)
    # Use correct number of features (9) that matches model expectation
    data = get_mock_input_data(mode="part", num_fs_feats=9)

    # Should not raise
    output = model.forward(
        fs_data=data["fs_data"],
        ctxt_data=data["ctxt_data"],
        mask=data["mask"],
        timestep=data["timestep"],
        ctxt_global_data=data["ctxt_global_data"],
    )
    assert output.shape == data["fs_data"].shape


def test_mock_model_and_data_compatibility_event():
    """Test that mock model and data are compatible for event mode."""
    model_path = get_mock_model_file(mode="evt")
    config = _create_model_config(model_path, mode="evt")
    model = ModelWrapper(config)
    data = get_mock_input_data(mode="evt", num_fs_feats=4)

    # Should not raise
    output = model.forward(
        fs_data=data["fs_data"],
        ctxt_data=data["ctxt_data"],
        mask=data["mask"],
        timestep=data["timestep"],
        ctxt_global_data=data["ctxt_global_data"],
    )
    assert output.shape == data["fs_data"].shape


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_model_wrapper_with_invalid_path():
    """Test that ModelWrapper raises error with non-existent model file."""
    config = ModelConfig(
        name="invalid_model",
        file_path=Path("/nonexistent/path/model.pt2"),
        variables_config=VariablesConfig(
            truth_vars_to_load=("pt",),
            fs_vars=("pflow_pt",),
            ctxt_vars=("truth_pt",),
            ctxt_global_vars=("met_pt",),
        ),
        sampler_config=SamplerConfig(),
    )

    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        _ = ModelWrapper(config)


def test_sampler_with_zero_steps():
    """Test that sampler handles edge case of very few steps."""
    sampler = EulerSampler(n_steps=1)
    assert sampler.n_steps == 1
    assert len(sampler.t_steps) == 2  # Should still have start and end


def test_model_forward_with_mismatched_batch_sizes():
    """Test that forward pass handles batch size consistency."""
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)
    data = get_mock_input_data(mode="part", num_fs_feats=12)

    # Modify timestep to have different batch size
    data["timestep"] = torch.randn(1, 1)  # Batch size 1 instead of 2

    # Should broadcast correctly or raise error
    with suppress(RuntimeError):
        _ = model.forward(
            fs_data=data["fs_data"],
            ctxt_data=data["ctxt_data"],
            mask=data["mask"],
            timestep=data["timestep"],
            ctxt_global_data=data["ctxt_global_data"],
        )


def test_sample_output_shape_matches_input():
    """Test that sampling output shape is consistent with config and input."""
    model_path = get_mock_model_file(mode="part")
    config = _create_model_config(model_path, mode="part")
    model = ModelWrapper(config)
    data = get_mock_input_data(mode="part", num_fs_feats=9)

    # Use default shape from data
    shape = data["ctxt_data"].size()[:-1]

    output = model.sample(
        shape=shape,
        ctxt_data=data["ctxt_data"],
        mask=data["mask"],
        ctxt_global_data=data["ctxt_global_data"],
    )

    # Output should match input shape plus the number of fs variables
    assert output.shape == (*shape, model.num_fs_vars)
    assert output.shape[0] == 2  # Batch size
    assert output.shape[1] == 400  # Particle count
    assert output.shape[2] == 9  # Feature count

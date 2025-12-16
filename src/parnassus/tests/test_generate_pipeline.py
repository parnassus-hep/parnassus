from types import SimpleNamespace

import numpy as np
import pytest

from parnassus.configs.scheme import GenEvent
from parnassus.data import build_dataset
from parnassus.pipelines.generate import GenerationBuffers, GenerationPipeline


class StubSampler:
    n_steps = 1


class StubModel:
    def __init__(self):
        self.sampler = StubSampler()


class StubDataLoader:
    def __init__(self, dataset_len: int, batches: list[dict]):
        self.dataset = [None] * dataset_len
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def make_stub_event_generator(with_impact: bool = False, max_particles: int = 2):
    class _StubEventGenerator:
        def __init__(self):
            base_pflow_vars = ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
            impact_vars = ["d0", "z0", "d0Error", "z0Error"] if with_impact else []
            self.config = SimpleNamespace(
                truth_output_vars=["pt", "eta", "phi", "vx", "vy", "vz", "class"],
                pflow_output_vars=[*base_pflow_vars, *impact_vars],
                max_particles=max_particles,
            )
            self.event_model = StubModel()
            self.particle_model = StubModel()
            self.impact_model = StubModel() if with_impact else None
            self.calls = 0

        @property
        def truth_output_vars(self):
            return self.config.truth_output_vars

        @property
        def pflow_output_vars(self):
            return self.config.pflow_output_vars

        @property
        def has_impact_model(self):
            return self.impact_model is not None

        @property
        def event_sampler_steps(self):
            return self.event_model.sampler.n_steps

        @property
        def particle_sampler_steps(self):
            return self.particle_model.sampler.n_steps

        @property
        def impact_sampler_steps(self):
            return self.impact_model.sampler.n_steps if self.impact_model else None

        def get_accessors(self):
            """Return stub accessor partials."""
            from parnassus.configs.accessors import AccessorListBuilder

            accessors_builder = (
                AccessorListBuilder.for_particles("Pflow")
                .add(["pt", "eta", "phi", "vx", "vy", "vz"])
                .add(["class_id"], dtype="int32")
            )

            if self.has_impact_model:
                accessors_builder.add(["d0", "z0", "d0Error", "z0Error"])

            return {"Pflow": accessors_builder.build(), "Truth": accessors_builder.build()}

        def generate_event(
            self,
            _batch,
            event_callback=None,
            particle_callback=None,
            impact_callback=None,
        ):
            if event_callback:
                event_callback()
            if particle_callback:
                particle_callback()
            if impact_callback:
                impact_callback()
            self.calls += 1
            gen_size = 1
            val = float(self.calls)
            mask = np.array([[1, 0]], dtype=bool)
            data_block = np.full((gen_size, max_particles), val, dtype=np.float32)
            tr_data: dict[str, np.ndarray] = {
                "pt": data_block,
                "eta": data_block,
                "phi": data_block,
                "vx": data_block,
                "vy": data_block,
                "vz": data_block,
                "class": data_block,
            }
            pf_data = {**tr_data}
            pf_data["pt"] = np.array([[0.5, 2.5]], dtype=np.float32)
            if with_impact:
                pf_data.update({
                    "d0": data_block,
                    "z0": data_block,
                    "d0Error": data_block,
                    "z0Error": data_block,
                })
            common = {
                "bad_idxs": np.array([], dtype=np.int64),
                "event_number": np.array([[10 + self.calls]], dtype=np.int64),
                "fs_mask": np.array([[1, 1]], dtype=bool),
                "tr_mask": mask,
            }
            return tr_data, pf_data, common

        def generate_batch(
            self,
            _batch,
            event_callback=None,
            particle_callback=None,
            impact_callback=None,
        ):
            return self.generate_event(_batch, event_callback, particle_callback, impact_callback)

    return _StubEventGenerator()


@pytest.fixture
def default_event_generator():
    """Fixture for a default stub event generator without impact model."""
    return make_stub_event_generator(with_impact=False)


def test_build_dataset_uses_custom_builder(tmp_path):
    """Test that build_dataset uses the provided custom dataset builder based on file extension."""
    called = {}

    class DummyDataset:
        def __init__(self, cfg, var_transform_dict):
            called["cfg"] = cfg
            called["vars"] = var_transform_dict

    dataset_config = SimpleNamespace(file_path=tmp_path / "data.root")
    dataset_config.file_path.touch()
    transform_registry = SimpleNamespace(to_var_transform_dict=lambda: {"a": 1})

    dataset = build_dataset(
        dataset_config,
        transform_registry,
        dataset_builders={".root": DummyDataset},
    )

    assert isinstance(dataset, DummyDataset)
    assert called["cfg"] is dataset_config
    assert called["vars"] == {"a": 1}


def test_build_dataset_validates_path_and_extension(tmp_path):
    """Test that build_dataset raises errors for missing files and unsupported extensions."""
    transform_registry = SimpleNamespace(to_var_transform_dict=dict)
    missing_config = SimpleNamespace(file_path=tmp_path / "missing.root")

    with pytest.raises(FileNotFoundError):
        build_dataset(
            missing_config, transform_registry, dataset_builders={".root": lambda *_: None}
        )

    bad_ext_path = tmp_path / "data.txt"
    bad_ext_path.touch()
    bad_ext_config = SimpleNamespace(file_path=bad_ext_path)

    with pytest.raises(ValueError):
        build_dataset(
            bad_ext_config, transform_registry, dataset_builders={".root": lambda *_: None}
        )


def test_run_sampling_trims_to_generated_events(default_event_generator):
    """Test that _run_sampling correctly trims buffers to the number of generated events."""
    dataset_config = SimpleNamespace(max_particles=2)
    config = SimpleNamespace(dataset_config=dataset_config, model=None, batch_size=1, device="cpu")
    pipeline = GenerationPipeline(config)
    dataloader = StubDataLoader(dataset_len=3, batches=[{}, {}])
    pipeline.generator = default_event_generator

    buffers = pipeline._run_sampling(dataloader)

    assert buffers.count == 2
    assert buffers.truth_data["pt"].shape == (2, 2)
    assert buffers.pflow_data["pt"].shape == (2, 2)
    assert buffers.event_numbers.tolist() == [11, 12]
    assert buffers.truth_data["ind"].shape == (2, 2)


def test_build_events_filters_and_adds_impact():
    """Test that _build_events correctly builds GenEvent objects with impact parameters."""
    config = SimpleNamespace(dataset_config=None, model=None, batch_size=1, device="cpu")
    pipeline = GenerationPipeline(config)
    pipeline.generator = make_stub_event_generator(with_impact=True)

    buffers = GenerationBuffers(
        truth_data={
            "pt": np.array([[1.0, 0.0]], dtype=np.float32),
            "eta": np.array([[0.1, 0.0]], dtype=np.float32),
            "phi": np.array([[0.0, 0.0]], dtype=np.float32),
            "vx": np.array([[0.0, 0.0]], dtype=np.float32),
            "vy": np.array([[0.0, 0.0]], dtype=np.float32),
            "vz": np.array([[0.0, 0.0]], dtype=np.float32),
            "class": np.array([[1.0, 0.0]], dtype=np.float32),
            "ind": np.array([[1.0, 0.0]], dtype=np.float32),
        },
        pflow_data={
            "pt": np.array([[0.5, 2.0]], dtype=np.float32),
            "eta": np.array([[0.2, 0.3]], dtype=np.float32),
            "phi": np.array([[0.0, 0.5]], dtype=np.float32),
            "vx": np.array([[0.0, 0.0]], dtype=np.float32),
            "vy": np.array([[0.0, 0.0]], dtype=np.float32),
            "vz": np.array([[0.0, 0.0]], dtype=np.float32),
            "class": np.array([[2.0, 3.0]], dtype=np.float32),
            "ind": np.array([[1.0, 1.0]], dtype=np.float32),
            "d0": np.array([[0.1, 0.2]], dtype=np.float32),
            "z0": np.array([[0.3, 0.4]], dtype=np.float32),
            "d0Error": np.array([[0.01, 0.02]], dtype=np.float32),
            "z0Error": np.array([[0.03, 0.04]], dtype=np.float32),
        },
        event_numbers=np.array([42], dtype=np.int32),
        count=1,
    )

    events = pipeline._build_events(buffers)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, GenEvent)
    assert event.event_number == 42
    assert list(event.truth_particles.pt) == [1.0]
    assert list(event.pflow_particles.pt) == [2.0]
    assert event.pflow_particles.d0 is not None
    assert list(event.pflow_particles.d0) == [0.2]
    assert event.pflow_particles.z0_error is not None
    assert list(event.pflow_particles.z0_error) == [0.04]


def test_build_accessors_respects_impact_presence(default_event_generator):
    """Test that _build_accessors includes impact parameters when present in the model."""
    config = SimpleNamespace(dataset_config=None, model=None, batch_size=1, device="cpu")
    base_pipeline = GenerationPipeline(config)
    base_pipeline.generator = default_event_generator

    impact_pipeline = GenerationPipeline(config)
    impact_pipeline.generator = make_stub_event_generator(with_impact=True)

    no_impact_accessors = base_pipeline._build_accessors()
    with_impact_accessors = impact_pipeline._build_accessors()

    assert len(with_impact_accessors["Pflow"]) > len(no_impact_accessors["Pflow"])


def test_generate_wiring_with_stubs(monkeypatch, default_event_generator):
    """Test the generate pipeline wiring using stub components."""
    dataset_config = SimpleNamespace(max_particles=2, file_path="ignored")
    model_config = SimpleNamespace(transform_registry=SimpleNamespace())
    config = SimpleNamespace(
        model=model_config,
        dataset_config=dataset_config,
        batch_size=1,
        device="cpu",
    )

    dataloader = StubDataLoader(dataset_len=1, batches=[{}])
    generative_model = default_event_generator

    pipeline = GenerationPipeline(config)
    monkeypatch.setattr(pipeline, "_build_dataset", lambda: dataloader.dataset)
    monkeypatch.setattr(pipeline, "_build_dataloader", lambda ds: dataloader)
    monkeypatch.setattr(pipeline, "_init_generator", lambda: generative_model)

    events, accessors = pipeline.run()

    assert len(events) == 1
    assert isinstance(events[0], GenEvent)
    assert len(events) == 1
    assert pipeline.get_accessors() == {key: list(val) for key, val in accessors.items()}

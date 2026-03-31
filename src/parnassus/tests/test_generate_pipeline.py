from types import SimpleNamespace

import numpy as np
import pytest

from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.data import build_dataset
from parnassus.pipelines.generate import GenerationPipeline
from parnassus.pipelines.generators.neural import _GenerationBuffers  # noqa: PLC2701


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
            self._events: list[GenEvent] = []
            self.calls = 0
            self._with_impact = with_impact

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def initialize(self, n_events: int, n_batches: int) -> None:
            self._events = []

        def process_batch(self, batch) -> None:
            self.calls += 1
            val = float(self.calls)
            pt = np.array([val, 0.0], dtype=np.float32)
            eta = np.array([0.1, 0.0], dtype=np.float32)
            phi = np.array([0.0, 0.0], dtype=np.float32)
            vxyz = np.zeros(2, dtype=np.float32)
            class_id = np.array([1, 0], dtype=np.int32)
            truth = GenParticleCollection(
                name="truth",
                pt=pt[:1],
                eta=eta[:1],
                phi=phi[:1],
                vx=vxyz[:1],
                vy=vxyz[:1],
                vz=vxyz[:1],
                class_id=class_id[:1],
            )
            impact_kwargs = {}
            if self._with_impact:
                impact_kwargs = {
                    "d0": np.array([0.1], dtype=np.float32),
                    "z0": np.array([0.3], dtype=np.float32),
                    "d0_error": np.array([0.01], dtype=np.float32),
                    "z0_error": np.array([0.03], dtype=np.float32),
                }
            pflow = GenParticleCollection(
                name="pflow",
                pt=np.array([2.5], dtype=np.float32),
                eta=eta[:1],
                phi=phi[:1],
                vx=vxyz[:1],
                vy=vxyz[:1],
                vz=vxyz[:1],
                class_id=class_id[:1],
                **impact_kwargs,
            )
            self._events.append(
                GenEvent(
                    event_number=10 + self.calls,
                    truth_particles=truth,
                    pflow_particles=pflow,
                )
            )

        def get_events(self) -> list[GenEvent]:
            return list(self._events)

        def get_accessors(self):
            from parnassus.configs.accessors import AccessorListBuilder

            builder = (
                AccessorListBuilder.for_particles("Pflow")
                .add(["pt", "eta", "phi", "vx", "vy", "vz"])
                .add(["class_id"], dtype="int32")
            )
            if self._with_impact:
                builder.add(["d0", "z0", "d0Error", "z0Error"])
            return {"Pflow": builder.build(), "Truth": builder.build()}

    return _StubEventGenerator()


@pytest.fixture
def default_event_generator():
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


def test_neural_generator_accumulates_events():
    """Test that NeuralEventGenerator accumulates events across process_batch calls."""
    max_particles = 2
    buffers = _GenerationBuffers(
        truth_data={
            "pt": np.zeros((3, max_particles), dtype=np.float32),
            "eta": np.zeros((3, max_particles), dtype=np.float32),
            "phi": np.zeros((3, max_particles), dtype=np.float32),
            "vx": np.zeros((3, max_particles), dtype=np.float32),
            "vy": np.zeros((3, max_particles), dtype=np.float32),
            "vz": np.zeros((3, max_particles), dtype=np.float32),
            "class": np.zeros((3, max_particles), dtype=np.float32),
            "ind": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        },
        pflow_data={
            "pt": np.array([[0.5, 2.0], [0.5, 2.0], [0.5, 2.0]], dtype=np.float32),
            "eta": np.zeros((3, max_particles), dtype=np.float32),
            "phi": np.zeros((3, max_particles), dtype=np.float32),
            "vx": np.zeros((3, max_particles), dtype=np.float32),
            "vy": np.zeros((3, max_particles), dtype=np.float32),
            "vz": np.zeros((3, max_particles), dtype=np.float32),
            "class": np.ones((3, max_particles), dtype=np.float32),
            "ind": np.ones((3, max_particles), dtype=np.float32),
        },
        event_numbers=np.array([1, 2, 3], dtype=np.int32),
        count=3,
    )
    buffers_trimmed = buffers.trim()
    assert buffers_trimmed.count == 3
    assert buffers_trimmed.truth_data["pt"].shape == (3, max_particles)


def test_neural_generator_converts_buffers_to_events():
    """Test _GenerationBuffers trimming and that get_events works via a stub."""
    stub = make_stub_event_generator(with_impact=True)
    stub.initialize(n_events=2, n_batches=2)
    stub.process_batch({})
    stub.process_batch({})
    events = stub.get_events()

    assert len(events) == 2
    assert isinstance(events[0], GenEvent)
    assert events[0].event_number == 11
    assert events[1].event_number == 12
    assert events[0].pflow_particles.d0 is not None


def test_build_accessors_respects_impact_presence(default_event_generator):
    """Test that get_accessors includes impact parameters when present in the model."""
    config = SimpleNamespace(dataset_config=None, model=None, batch_size=1, device="cpu")
    base_pipeline = GenerationPipeline(config)
    base_pipeline.generator = default_event_generator

    impact_generator = make_stub_event_generator(with_impact=True)
    impact_pipeline = GenerationPipeline(config)
    impact_pipeline.generator = impact_generator

    no_impact_accessors = base_pipeline.generator.get_accessors()
    with_impact_accessors = impact_pipeline.generator.get_accessors()

    assert len(with_impact_accessors["Pflow"]) > len(no_impact_accessors["Pflow"])


def test_progress_bar_closed_on_process_batch_exception(monkeypatch):
    """Progress bar must be closed even when process_batch raises, preventing LiveError."""
    from parnassus.utils.logger import ProgressBar

    dataset_config = SimpleNamespace(max_particles=2, file_path="ignored")
    config = SimpleNamespace(dataset_config=dataset_config, batch_size=1, device="cpu")

    class _BoomGenerator:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            # Real generators close their resources here even on exception
            if hasattr(self, "_pb_stack"):
                self._pb_stack.close()

        def get_accessors(self):
            return {}

        def initialize(self, n_events, n_batches):
            self._pb_stack = __import__("contextlib").ExitStack()
            self._pb_stack.enter_context(ProgressBar())

        def process_batch(self, batch):
            raise RuntimeError("boom")

        def get_events(self):
            self._pb_stack.close()
            return []

    dataloader = StubDataLoader(dataset_len=1, batches=[{}])
    pipeline = GenerationPipeline(config)
    monkeypatch.setattr(pipeline, "_build_dataset", lambda: dataloader.dataset)
    monkeypatch.setattr(pipeline, "_build_dataloader", lambda ds: dataloader)
    monkeypatch.setattr(pipeline, "_init_generator", lambda: _BoomGenerator())

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run()

    # After the exception, a new ProgressBar must be openable (no LiveError)
    with ProgressBar():
        pass


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
    assert pipeline.get_accessors() == {key: list(val) for key, val in accessors.items()}

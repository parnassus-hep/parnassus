"""Integration tests for GenerationPipeline wiring and dataset building."""

from types import SimpleNamespace

import pytest

from parnassus.configs.scheme import GenEvent
from parnassus.data import build_dataset
from parnassus.pipelines.generate import GenerationPipeline
from parnassus.utils.logger import ProgressBar

from .conftest import StubDataLoader, StubEventGenerator

# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def test_build_dataset_uses_custom_builder(tmp_path):
    """build_dataset raises FileNotFoundError for missing ROOT files."""
    transform_registry = SimpleNamespace(to_var_transform_dict=lambda: {"a": 1})

    missing_config = SimpleNamespace(file_path=tmp_path / "missing.root")
    with pytest.raises(FileNotFoundError):
        build_dataset(
            missing_config,  # pyright: ignore[reportArgumentType]
            transform_registry,  # pyright: ignore[reportArgumentType]
        )


def test_build_dataset_validates_path_and_extension(tmp_path):
    """build_dataset raises for missing files and unsupported extensions."""
    transform_registry = SimpleNamespace(to_var_transform_dict=dict)

    missing_config = SimpleNamespace(file_path=tmp_path / "missing.root")
    with pytest.raises(FileNotFoundError):
        build_dataset(
            missing_config,  # pyright: ignore[reportArgumentType]
            transform_registry,  # pyright: ignore[reportArgumentType]
        )

    bad_ext_path = tmp_path / "data.txt"
    bad_ext_path.touch()
    bad_ext_config = SimpleNamespace(file_path=bad_ext_path)
    with pytest.raises(ValueError):
        build_dataset(
            bad_ext_config,  # pyright: ignore[reportArgumentType]
            transform_registry,  # pyright: ignore[reportArgumentType]
        )


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_pipeline_run_wires_dataset_dataloader_generator(monkeypatch, stub_event_generator):
    """GenerationPipeline.run() calls initialize, process_batch, get_events in order."""
    config = SimpleNamespace(
        dataset_config=SimpleNamespace(max_particles=2, file_path="ignored"),
        batch_size=1,
        device="cpu",
    )
    dataloader = StubDataLoader(dataset_len=1, batches=[{}])
    pipeline = GenerationPipeline(config)  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(pipeline, "_build_dataset", lambda: dataloader.dataset)
    monkeypatch.setattr(pipeline, "_build_dataloader", lambda _: dataloader)
    monkeypatch.setattr(pipeline, "_init_generator", lambda: stub_event_generator)

    events, accessors = pipeline.run()

    assert len(events) == 1
    assert isinstance(events[0], GenEvent)
    assert pipeline.get_accessors() == {key: list(val) for key, val in accessors.items()}


def test_pipeline_get_accessors_is_empty_before_run():
    config = SimpleNamespace(dataset_config=None, batch_size=1, device="cpu")
    pipeline = GenerationPipeline(config)  # pyright: ignore[reportArgumentType]
    assert pipeline.get_accessors() == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_progress_bar_closed_on_process_batch_exception(monkeypatch):
    """Progress bar is closed even when process_batch raises, preventing LiveError."""
    config = SimpleNamespace(
        dataset_config=SimpleNamespace(max_particles=2, file_path="ignored"),
        batch_size=1,
        device="cpu",
    )

    class _BoomGenerator:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            if hasattr(self, "_pb_stack"):
                self._pb_stack.close()

        def get_accessors(self):
            return {}

        def initialize(self, n_events, n_batches):  # noqa: ARG002
            self._pb_stack = __import__("contextlib").ExitStack()
            self._pb_stack.enter_context(ProgressBar())

        def process_batch(self, batch):  # noqa: ARG002
            raise RuntimeError("boom")

        def get_events(self):
            self._pb_stack.close()
            return []

    dataloader = StubDataLoader(dataset_len=1, batches=[{}])
    pipeline = GenerationPipeline(config)  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(pipeline, "_build_dataset", lambda: dataloader.dataset)
    monkeypatch.setattr(pipeline, "_build_dataloader", lambda _: dataloader)
    monkeypatch.setattr(pipeline, "_init_generator", _BoomGenerator)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.run()

    # After the exception a new ProgressBar must be openable (no LiveError)
    with ProgressBar():
        pass


def test_pipeline_run_multiple_batches(monkeypatch):
    """All batches are processed and all events are returned."""
    config = SimpleNamespace(
        dataset_config=SimpleNamespace(max_particles=2, file_path="ignored"),
        batch_size=1,
        device="cpu",
    )
    n_batches = 4
    dataloader = StubDataLoader(dataset_len=n_batches, batches=[{}] * n_batches)
    generator = StubEventGenerator(with_impact=False)

    pipeline = GenerationPipeline(config)  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(pipeline, "_build_dataset", lambda: dataloader.dataset)
    monkeypatch.setattr(pipeline, "_build_dataloader", lambda _: dataloader)
    monkeypatch.setattr(pipeline, "_init_generator", lambda: generator)

    events, _ = pipeline.run()
    assert len(events) == n_batches

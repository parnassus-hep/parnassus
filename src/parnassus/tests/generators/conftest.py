"""Shared test fixtures and stubs for the parnassus test suite."""

import numpy as np
import pytest

from parnassus.configs.scheme import GenEvent, GenParticleCollection


class StubDataLoader:
    """Minimal DataLoader substitute that replays a fixed list of batches."""

    def __init__(self, dataset_len: int, batches: list[dict]):
        self.dataset = [None] * dataset_len
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class StubEventGenerator:
    """Minimal EventGenerator stub that produces synthetic GenEvents.

    Each :meth:`process_batch` call appends one event with a monotonically
    increasing event number so tests can verify ordering and counts.

    Parameters
    ----------
    with_impact : bool
        When *True* the generated pflow particles include d0/z0 impact params
        and ``get_accessors`` exposes them.
    max_particles : int
        Particle-array width (unused for logic, kept for API parity).
    """

    def __init__(self, *, with_impact: bool = False, max_particles: int = 2):  # noqa: ARG002
        self._events: list[GenEvent] = []
        self.calls = 0
        self._with_impact = with_impact

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def initialize(self, n_events: int, n_batches: int) -> None:  # noqa: ARG002
        self._events = []

    def process_batch(self, batch) -> None:  # noqa: ARG002
        self.calls += 1
        val = float(self.calls)
        pt = np.array([val], dtype=np.float32)
        eta = np.array([0.1], dtype=np.float32)
        phi = np.array([0.0], dtype=np.float32)
        vxyz = np.zeros(1, dtype=np.float32)
        class_id = np.array([1], dtype=np.int32)

        truth = GenParticleCollection(
            name="truth",
            pt=pt,
            eta=eta,
            phi=phi,
            vx=vxyz,
            vy=vxyz,
            vz=vxyz,
            class_id=class_id,
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
            eta=eta,
            phi=phi,
            vx=vxyz,
            vy=vxyz,
            vz=vxyz,
            class_id=class_id,
            **impact_kwargs,  # pyright: ignore[reportArgumentType]
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
            AccessorListBuilder
            .for_particles("pflow_particles")
            .add(["pt", "eta", "phi", "vx", "vy", "vz"])
            .add(["class_id"], dtype="int32")
        )
        if self._with_impact:
            builder.add(["d0", "z0", "d0_error", "z0_error"])
        accessors = builder.build()
        return {"Pflow": accessors, "Truth": accessors}


@pytest.fixture
def stub_event_generator() -> StubEventGenerator:
    """A basic StubEventGenerator without impact parameters."""
    return StubEventGenerator(with_impact=False)

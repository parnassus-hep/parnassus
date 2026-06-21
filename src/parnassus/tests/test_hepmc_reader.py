"""Regression tests for the line-based :class:`HepMCDataset` reader.

Asserts the fast pyarrow/line parser produces tensors bit-identical to the
reference :mod:`pyhepmc` path it replaces, and that gzip + ``num_events``
handling behave correctly.
"""

import gzip
import shutil
from pathlib import Path

import numpy as np
import pyhepmc
import pytest
import torch

from parnassus.data.hepmc import HepMCDataset
from parnassus.data.particle_io import ColumnMap, hepmc_particles_to_tensor

TEST_FILE = Path(__file__).parent / "h4lep_test_100.hepmc"
N_EVENTS = 50


def _pyhepmc_reference(path: Path, num_events: int) -> list[torch.Tensor]:
    """Reference tensors built the old way: pyhepmc objects → tensor."""
    out: list[torch.Tensor] = []
    with pyhepmc.open(str(path)) as f:
        for i, event in enumerate(f):
            if i >= num_events:
                break
            out.append(hepmc_particles_to_tensor(event.particles, event.event_number))
    return out


@pytest.mark.skipif(not TEST_FILE.exists(), reason="benchmark HepMC file not available")
def test_hepmc_matches_pyhepmc():
    ds = HepMCDataset(TEST_FILE, num_events=N_EVENTS)
    ref = _pyhepmc_reference(TEST_FILE, N_EVENTS)

    assert len(ds) == len(ref)
    for i, expected in enumerate(ref):
        got = ds[i]["particles"]
        assert got.shape == expected.shape, f"event {i}: {got.shape} != {expected.shape}"
        np.testing.assert_array_equal(got.numpy(), expected.numpy(), err_msg=f"event {i}")


@pytest.mark.skipif(not TEST_FILE.exists(), reason="benchmark HepMC file not available")
def test_hepmc_event_numbers_match():
    ds = HepMCDataset(TEST_FILE, num_events=N_EVENTS)
    with pyhepmc.open(str(TEST_FILE)) as f:
        ref_numbers = [int(ev.event_number) for i, ev in enumerate(f) if i < N_EVENTS]
    assert [ds[i]["event_number"] for i in range(len(ds))] == ref_numbers
    # event number is also stored per particle in the tensor
    for i, num in enumerate(ref_numbers):
        particles = ds[i]["particles"]
        if particles.shape[0]:
            assert torch.all(particles[:, ColumnMap.EVENT_NUMBER] == num)


@pytest.mark.skipif(not TEST_FILE.exists(), reason="benchmark HepMC file not available")
def test_hepmc_gzip_matches_plain(tmp_path: Path):
    gz_path = tmp_path / "events.hepmc.gz"
    with TEST_FILE.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    plain = HepMCDataset(TEST_FILE, num_events=10)
    gzipped = HepMCDataset(gz_path, num_events=10)

    assert len(plain) == len(gzipped)
    for i in range(len(plain)):
        np.testing.assert_array_equal(
            gzipped[i]["particles"].numpy(), plain[i]["particles"].numpy()
        )


@pytest.mark.skipif(not TEST_FILE.exists(), reason="benchmark HepMC file not available")
def test_hepmc_num_events_none_loads_all():
    ds = HepMCDataset(TEST_FILE, num_events=None)
    with pyhepmc.open(str(TEST_FILE)) as f:
        total = sum(1 for _ in f)
    assert len(ds) == total

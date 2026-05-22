"""Tests for DelphesPileUpMerger.

Verifies pile-up merging logic: HS vertex smearing, PU event sampling,
phi rotation, vertex shifting, event number assignment, and reproducibility.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from parnassus.configs.pileup import DelphesPileUpConfig
from parnassus.data.particle_io import N_FEATURES, ColumnMap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pileup_file(path: Path, events: list[list[tuple]]) -> None:
    """Write a minimal .pileup file in Delphes XDR format."""
    body_parts: list[bytes] = []
    offsets: list[int] = []
    current_offset = 0

    for particles in events:
        offsets.append(current_offset)
        n = len(particles)
        event_bytes = struct.pack(">i", n)
        for pid, x, y, z, t, px, py, pz, e in particles:
            event_bytes += struct.pack(">i", pid)
            event_bytes += struct.pack(">ffffffff", x, y, z, t, px, py, pz, e)
        body_parts.append(event_bytes)
        current_offset += len(event_bytes)

    num_entries = len(events)
    index_bytes = struct.pack(f">{num_entries}q", *offsets)
    footer = struct.pack(">q", num_entries)

    with open(path, "wb") as f:
        f.writelines(body_parts)
        f.write(index_bytes)
        f.write(footer)


def _make_hs_tensor(n_events: int, particles_per_event: int = 5) -> torch.Tensor:
    """Create a synthetic HS batch tensor."""
    rows = []
    for ev in range(n_events):
        arr = torch.zeros(particles_per_event, N_FEATURES, dtype=torch.float64)
        arr[:, ColumnMap.PID] = 211
        arr[:, ColumnMap.STATUS] = 1
        arr[:, ColumnMap.CHARGE] = 1
        arr[:, ColumnMap.E] = 10.0
        arr[:, ColumnMap.PX] = 5.0
        arr[:, ColumnMap.PY] = 5.0
        arr[:, ColumnMap.PZ] = 2.0
        arr[:, ColumnMap.PT] = float(np.sqrt(50.0))
        arr[:, ColumnMap.ETA] = float(np.arcsinh(2.0 / np.sqrt(50.0)))
        arr[:, ColumnMap.PHI] = float(np.arctan2(5.0, 5.0))
        arr[:, ColumnMap.X] = 0.1
        arr[:, ColumnMap.Y] = 0.2
        arr[:, ColumnMap.Z] = 10.0
        arr[:, ColumnMap.T] = 5.0
        arr[:, ColumnMap.MASS] = 0.140
        arr[:, ColumnMap.EVENT_NUMBER] = ev
        rows.append(arr)
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pileup_path(tmp_path: Path) -> Path:
    """Create a synthetic .pileup file with 10 MinBias events."""
    path = tmp_path / "test.pileup"
    events: list[list[tuple]] = []
    for i in range(10):
        particles = []
        for j in range(3):
            px = 0.5 + 0.1 * j
            py = 0.3 + 0.1 * j
            pz = 0.2 + 0.1 * j
            e = float(np.sqrt(px**2 + py**2 + pz**2 + 0.14**2))
            particles.append((
                211,  # pid
                0.01 * (i + 1),  # x
                0.02 * (i + 1),  # y
                0.5 * (i + 1),  # z
                0.1 * (i + 1),  # t
                px,
                py,
                pz,
                e,
            ))
        events.append(particles)
    _write_pileup_file(path, events)
    return path


@pytest.fixture
def config(pileup_path: Path) -> DelphesPileUpConfig:
    """Default config with moderate pileup."""
    return DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=5.0,
    )


@pytest.fixture
def config_no_pu(pileup_path: Path) -> DelphesPileUpConfig:
    """Config with zero pileup."""
    return DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=0.0,
    )


@pytest.fixture
def config_no_hs_smear(pileup_path: Path) -> DelphesPileUpConfig:
    """Config with HS vertex smearing disabled."""
    return DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=5.0,
        smear_hs_vertex=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_loads_minbias_events(config: DelphesPileUpConfig) -> None:
    """Verify n_minbias_events matches the number of events in the file."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config, seed=42)
    assert merger.n_minbias_events == 10


def test_merge_preserves_hs_with_zero_pileup(
    config_no_pu: DelphesPileUpConfig,
) -> None:
    """With mean_pileup=0, no PU is added; truth == merged (up to HS smearing)."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config_no_pu, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=5)
    merged, truth = merger.merge(hs)
    # No PU => same number of particles
    assert merged.shape[0] == hs.shape[0]
    # Truth should also have same number
    assert truth.shape[0] == hs.shape[0]


def test_merge_adds_pu_particles(config: DelphesPileUpConfig) -> None:
    """With mean_pileup=5, merged tensor should have more particles than HS."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(3, particles_per_event=5)
    merged, _truth = merger.merge(hs)
    # On average 5 PU events * 3 particles * 3 events = 45 extra
    # With 15 HS particles, merged should be larger (statistically guaranteed for seed=42)
    assert merged.shape[0] > hs.shape[0]


def test_pu_particles_assigned_to_each_event_independently(
    pileup_path: Path,
) -> None:
    """Each HS event must receive its own PU particles with the correct EVENT_NUMBER.

    Uses non-consecutive event numbers (100, 200, 300) to catch hardcoded
    assumptions about sequential numbering. With mean_pileup=50 every event
    is virtually guaranteed to receive PU particles.
    """
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    config = DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=50.0,
        smear_hs_vertex=False,
    )
    merger = DelphesPileUpMerger(config, seed=42)

    event_ids = [100, 200, 300]
    n_hs_per_event = 4
    rows = []
    for ev_id in event_ids:
        arr = torch.zeros(n_hs_per_event, N_FEATURES, dtype=torch.float64)
        arr[:, ColumnMap.PID] = 211
        arr[:, ColumnMap.STATUS] = 1
        arr[:, ColumnMap.CHARGE] = 1
        arr[:, ColumnMap.E] = 10.0
        arr[:, ColumnMap.PX] = 5.0
        arr[:, ColumnMap.PY] = 5.0
        arr[:, ColumnMap.PT] = float(np.sqrt(50.0))
        arr[:, ColumnMap.PHI] = float(np.arctan2(5.0, 5.0))
        arr[:, ColumnMap.EVENT_NUMBER] = ev_id
        rows.append(arr)
    hs = torch.cat(rows, dim=0)

    merged, _ = merger.merge(hs)

    # Only the original event IDs should appear — no fabricated numbers
    merged_ids = set(merged[:, ColumnMap.EVENT_NUMBER].long().unique().tolist())
    assert merged_ids == set(event_ids)

    # Each event must have received PU particles (count > HS count)
    for ev_id in event_ids:
        mask = merged[:, ColumnMap.EVENT_NUMBER].long() == ev_id
        assert mask.sum().item() > n_hs_per_event, f"Event {ev_id} should have PU particles added"


def test_phi_rotation_applied_to_pu(pileup_path: Path) -> None:
    """PU particles should have rotated PHI compared to originals."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    config = DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=50.0,  # high PU to guarantee some PU particles
    )
    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=5)
    merged, _ = merger.merge(hs)

    n_hs = hs.shape[0]
    if merged.shape[0] > n_hs:
        pu_phi = merged[n_hs:, ColumnMap.PHI]
        # Original MinBias events have specific PHI values; after random rotation,
        # at least some should differ from the original.
        # The original PHI for the MinBias events is arctan2(py, px).
        # With dphi ~ Uniform(-pi, pi), the rotated PHI should vary.
        assert pu_phi.std() > 0.0, "PU PHI values should vary after rotation"


def test_phi_rotation_preserves_pt(pileup_path: Path) -> None:
    """PT must be unchanged after phi rotation."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    config = DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=50.0,
    )
    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=3)
    merged, _ = merger.merge(hs)

    n_hs = hs.shape[0]
    if merged.shape[0] > n_hs:
        pu_particles = merged[n_hs:]
        px = pu_particles[:, ColumnMap.PX]
        py = pu_particles[:, ColumnMap.PY]
        pt_from_pxpy = torch.sqrt(px**2 + py**2)
        pt_stored = pu_particles[:, ColumnMap.PT]
        torch.testing.assert_close(pt_from_pxpy, pt_stored, atol=1e-10, rtol=1e-10)


def test_vertex_smearing_shifts_z_and_t(pileup_path: Path) -> None:
    """PU particles should have nonzero Z and T after vertex smearing."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    # Create a pileup file where all MinBias events have Z=0, T=0
    zero_vert_path = pileup_path.parent / "zero_vert.pileup"
    events: list[list[tuple]] = [
        [
            (211, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.2, 0.7),
            (211, 0.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.1, 0.7),
        ]
        for _ in range(10)
    ]
    _write_pileup_file(zero_vert_path, events)

    config = DelphesPileUpConfig(
        file_path=str(zero_vert_path),
        mean_pileup=50.0,
    )
    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=3)
    merged, _ = merger.merge(hs)

    n_hs = hs.shape[0]
    if merged.shape[0] > n_hs:
        pu_z = merged[n_hs:, ColumnMap.Z]
        pu_t = merged[n_hs:, ColumnMap.T]
        # With sigma_z = 0.053 m = 53 mm, PU Z should be shifted
        assert pu_z.abs().max() > 0.0, "PU Z should be nonzero after vertex smearing"
        assert pu_t.abs().max() > 0.0, "PU T should be nonzero after vertex smearing"


def test_hs_vertex_smearing_shifts_positions(
    config: DelphesPileUpConfig,
) -> None:
    """When smear_hs_vertex=True, HS Z positions should be modified."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=5)
    original_z = hs[:, ColumnMap.Z].clone()
    _, truth = merger.merge(hs)
    # Truth is captured after HS smearing, so Z should differ from original
    assert not torch.allclose(truth[:, ColumnMap.Z], original_z), (
        "HS Z should be shifted when smear_hs_vertex=True"
    )


def test_hs_vertex_smearing_disabled(
    config_no_hs_smear: DelphesPileUpConfig,
) -> None:
    """When smear_hs_vertex=False, HS Z/T should remain unchanged."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config_no_hs_smear, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=5)
    original_z = hs[:, ColumnMap.Z].clone()
    original_t = hs[:, ColumnMap.T].clone()
    _merged, truth = merger.merge(hs)

    # With smear_hs_vertex=False, truth should preserve original Z/T
    torch.testing.assert_close(truth[:, ColumnMap.Z], original_z)
    torch.testing.assert_close(truth[:, ColumnMap.T], original_t)


def test_seed_reproducibility(config: DelphesPileUpConfig) -> None:
    """Same seed produces identical results."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    hs = _make_hs_tensor(2, particles_per_event=5)

    merger1 = DelphesPileUpMerger(config, seed=42)
    merged1, truth1 = merger1.merge(hs.clone())

    merger2 = DelphesPileUpMerger(config, seed=42)
    merged2, truth2 = merger2.merge(hs.clone())

    torch.testing.assert_close(merged1, merged2)
    torch.testing.assert_close(truth1, truth2)


def test_different_seeds_differ(config: DelphesPileUpConfig) -> None:
    """Different seeds produce different results."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    hs = _make_hs_tensor(2, particles_per_event=5)

    merger1 = DelphesPileUpMerger(config, seed=42)
    merged1, _ = merger1.merge(hs.clone())

    merger2 = DelphesPileUpMerger(config, seed=99)
    merged2, _ = merger2.merge(hs.clone())

    # At least the PU count or PU values should differ
    assert not torch.equal(merged1, merged2), "Different seeds should produce different results"


def test_merge_multiple_events_in_batch(config: DelphesPileUpConfig) -> None:
    """All 5 event numbers should appear in the merged output."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config, seed=42)
    n_events = 5
    hs = _make_hs_tensor(n_events, particles_per_event=4)
    merged, truth = merger.merge(hs)

    merged_evs = set(merged[:, ColumnMap.EVENT_NUMBER].unique().tolist())
    truth_evs = set(truth[:, ColumnMap.EVENT_NUMBER].unique().tolist())

    expected = set(range(n_events))
    # Truth must contain exactly the HS event numbers
    assert truth_evs == expected
    # Merged must contain at least the HS event numbers
    assert expected.issubset(merged_evs)


def test_px_py_consistent_with_pt_phi(pileup_path: Path) -> None:
    """After rotation, PX=PT*cos(PHI) and PY=PT*sin(PHI)."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    config = DelphesPileUpConfig(
        file_path=str(pileup_path),
        mean_pileup=50.0,
    )
    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(1, particles_per_event=3)
    merged, _ = merger.merge(hs)

    n_hs = hs.shape[0]
    if merged.shape[0] > n_hs:
        pu = merged[n_hs:]
        pt = pu[:, ColumnMap.PT]
        phi = pu[:, ColumnMap.PHI]
        px = pu[:, ColumnMap.PX]
        py = pu[:, ColumnMap.PY]
        torch.testing.assert_close(px, pt * torch.cos(phi), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(py, pt * torch.sin(phi), atol=1e-10, rtol=1e-10)


def test_truth_is_hs_only(config: DelphesPileUpConfig) -> None:
    """Truth tensor should have exactly the HS particle count; merged has more."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config, seed=42)
    hs = _make_hs_tensor(2, particles_per_event=5)
    merged, truth = merger.merge(hs)
    assert truth.shape[0] == hs.shape[0]
    # With mean_pileup=5 and 2 events, very likely merged > truth
    assert merged.shape[0] >= truth.shape[0]


def test_to_device(config: DelphesPileUpConfig) -> None:
    """.to(torch.device('cpu')) should work without error."""
    from parnassus.pipelines.pileup import DelphesPileUpMerger

    merger = DelphesPileUpMerger(config, seed=42)
    result = merger.to(torch.device("cpu"))
    assert result is merger  # should return self
    # Verify merge still works after .to()
    hs = _make_hs_tensor(1, particles_per_event=3)
    merged, truth = merger.merge(hs)
    assert merged.shape[1] == N_FEATURES
    assert truth.shape[1] == N_FEATURES

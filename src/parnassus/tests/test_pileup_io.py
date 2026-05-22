"""Tests for pileup_io.read_pileup_file.

The .pileup binary format (XDR big-endian):
  [event_0_data] ... [event_N-1_data] [index_table] [num_entries: int64]

Each event:
  int32 entry_size
  entry_size * (int32 pid, float32 x, y, z, t, px, py, pz, e)

Index table:
  num_entries * int64  (byte offsets to start of each event)

Final 8 bytes:
  int64 num_entries
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from parnassus.data.particle_io import N_FEATURES, ColumnMap

# ---------------------------------------------------------------------------
# Helper: synthetic .pileup writer
# ---------------------------------------------------------------------------


def _write_pileup_file(path: Path, events: list[list[tuple]]) -> None:
    """Write a synthetic .pileup file (big-endian XDR format).

    Parameters
    ----------
    path : Path
        Output file path.
    events : list[list[tuple]]
        Each element is a list of particle tuples:
        ``(pid, x, y, z, t, px, py, pz, e)``
        where pid is int and the rest are float.
    """
    body_parts: list[bytes] = []
    offsets: list[int] = []
    current_offset = 0

    for particles in events:
        offsets.append(current_offset)
        n = len(particles)
        # int32 entry_size
        event_bytes = struct.pack(">i", n)
        for pid, x, y, z, t, px, py, pz, e in particles:
            event_bytes += struct.pack(">i", pid)
            event_bytes += struct.pack(">ffffffff", x, y, z, t, px, py, pz, e)
        body_parts.append(event_bytes)
        current_offset += len(event_bytes)

    num_entries = len(events)
    # index table: num_entries * int64
    index_bytes = struct.pack(f">{num_entries}q", *offsets)
    # final int64: num_entries
    footer = struct.pack(">q", num_entries)

    with open(path, "wb") as f:
        f.writelines(body_parts)
        f.write(index_bytes)
        f.write(footer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_pileup(tmp_path: Path) -> Path:
    """Return path for a temporary pileup file."""
    return tmp_path / "test.pileup"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_correct_number_of_events(tmp_pileup: Path) -> None:
    """read_pileup_file returns exactly as many arrays as events."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [
        [(211, 0.1, 0.2, 0.3, 0.0, 0.5, 0.6, 0.7, 1.0)],
        [(2212, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)],
        [
            (11, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.0, 0.5),
            (-11, 0.0, 0.0, 0.0, 0.0, -0.3, -0.4, 0.0, 0.5),
        ],
    ]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)
    assert len(result) == 3


def test_particle_columns_populated(tmp_pileup: Path) -> None:
    """Verify PID, PX, PY, PZ, E, X, Y, Z, T, PT, PHI, ETA columns."""
    from parnassus.data.pileup_io import read_pileup_file

    pid_val = 211
    x, y, z, t = 1.0, 2.0, 3.0, 4.0
    px, py, pz, e = 0.3, 0.4, 0.5, 0.707

    events = [[(pid_val, x, y, z, t, px, py, pz, e)]]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    arr = result[0]
    assert arr.shape == (1, N_FEATURES)
    assert arr[0, ColumnMap.PID] == pytest.approx(pid_val)
    assert arr[0, ColumnMap.PX] == pytest.approx(px, rel=1e-5)
    assert arr[0, ColumnMap.PY] == pytest.approx(py, rel=1e-5)
    assert arr[0, ColumnMap.PZ] == pytest.approx(pz, rel=1e-5)
    assert arr[0, ColumnMap.E] == pytest.approx(e, rel=1e-5)
    assert arr[0, ColumnMap.X] == pytest.approx(x, rel=1e-5)
    assert arr[0, ColumnMap.Y] == pytest.approx(y, rel=1e-5)
    assert arr[0, ColumnMap.Z] == pytest.approx(z, rel=1e-5)
    assert arr[0, ColumnMap.T] == pytest.approx(t, rel=1e-5)

    expected_pt = np.sqrt(px**2 + py**2)
    expected_phi = np.arctan2(py, px)
    expected_eta = np.arcsinh(pz / expected_pt)

    assert arr[0, ColumnMap.PT] == pytest.approx(expected_pt, rel=1e-5)
    assert arr[0, ColumnMap.PHI] == pytest.approx(expected_phi, rel=1e-5)
    assert arr[0, ColumnMap.ETA] == pytest.approx(expected_eta, rel=1e-5)


def test_charge_from_pdg_id(tmp_pileup: Path) -> None:
    """pi+ (211) should have charge +1; positron (-11) should have charge +1."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [
        [
            (211, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1),  # pi+  -> +1
            (-11, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1),  # e+   -> +1
        ]
    ]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)
    arr = result[0]

    assert arr[0, ColumnMap.CHARGE] == pytest.approx(1.0)
    assert arr[1, ColumnMap.CHARGE] == pytest.approx(1.0)


def test_empty_event(tmp_pileup: Path) -> None:
    """An event with 0 particles must produce an array of shape (0, N_FEATURES)."""
    from parnassus.data.pileup_io import read_pileup_file

    events: list[list[tuple]] = [
        [(211, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1)],
        [],  # empty event
        [(2212, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)],
    ]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    assert len(result) == 3
    assert result[1].shape == (0, N_FEATURES)


def test_mass_from_pdg_lookup(tmp_pileup: Path) -> None:
    """Proton (2212) should have mass ~0.93827 GeV."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [[(2212, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.93827)]]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    assert result[0][0, ColumnMap.MASS] == pytest.approx(0.93827, rel=1e-4)


def test_file_not_found() -> None:
    """read_pileup_file raises FileNotFoundError for a non-existent path."""
    from parnassus.data.pileup_io import read_pileup_file

    with pytest.raises(FileNotFoundError):
        read_pileup_file("/tmp/definitely_does_not_exist_xyz.pileup")


def test_status_is_one(tmp_pileup: Path) -> None:
    """All MinBias particles should have STATUS = 1."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [
        [
            (211, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 1.0),
            (2212, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0),
        ]
    ]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    assert np.all(result[0][:, ColumnMap.STATUS] == pytest.approx(1.0))


def test_propagator_columns_zero(tmp_pileup: Path) -> None:
    """ETA_OUTER, PHI_OUTER, PASS_PROP, TRACK_RESOLUTION, EVENT_NUMBER should be 0."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [[(211, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 1.0)]]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)
    arr = result[0]

    assert arr[0, ColumnMap.ETA_OUTER] == pytest.approx(0.0)
    assert arr[0, ColumnMap.PHI_OUTER] == pytest.approx(0.0)
    assert arr[0, ColumnMap.PASS_PROP] == pytest.approx(0.0)
    assert arr[0, ColumnMap.TRACK_RESOLUTION] == pytest.approx(0.0)
    assert arr[0, ColumnMap.EVENT_NUMBER] == pytest.approx(0.0)


def test_low_pt_eta_sentinel(tmp_pileup: Path) -> None:
    """Particles with pt < PT_MIN should get eta = +999.9 (Delphes convention)."""
    from parnassus.data.pileup_io import read_pileup_file

    # px=py=0 => pt=0; pz > 0 => eta should be +999.9
    events = [[(211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)]]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    assert result[0][0, ColumnMap.ETA] == pytest.approx(999.9)


def test_low_pt_eta_sentinel_negative(tmp_pileup: Path) -> None:
    """Particles with pt < PT_MIN and pz < 0 should get eta = -999.9."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [[(211, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0)]]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    assert result[0][0, ColumnMap.ETA] == pytest.approx(-999.9)


def test_output_dtype_float64(tmp_pileup: Path) -> None:
    """Output arrays must use float64."""
    from parnassus.data.pileup_io import read_pileup_file

    events = [[(211, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 1.0)]]
    _write_pileup_file(tmp_pileup, events)
    result = read_pileup_file(tmp_pileup)

    assert result[0].dtype == np.float64


def test_smoke_real_minbias_file() -> None:
    """Smoke test: read the real MinBias.pileup and check basic sanity."""
    from parnassus.data.pileup_io import read_pileup_file

    real_path = Path(__file__).parents[3] / "delphes_cmp" / "MinBias.pileup"
    if not real_path.exists():
        pytest.skip(f"MinBias.pileup not found at {real_path}")

    events = read_pileup_file(real_path)

    # Sanity: should have at least one event
    assert len(events) > 0

    for arr in events:
        # Each event array must have N_FEATURES columns
        assert arr.ndim == 2
        assert arr.shape[1] == N_FEATURES
        assert arr.dtype == np.float64

        if arr.shape[0] > 0:
            # PT must be non-negative
            assert np.all(arr[:, ColumnMap.PT] >= 0.0)
            # PHI must be in [-pi, pi]
            assert np.all(arr[:, ColumnMap.PHI] >= -np.pi - 1e-6)
            assert np.all(arr[:, ColumnMap.PHI] <= np.pi + 1e-6)
            # STATUS == 1 for all particles
            assert np.all(arr[:, ColumnMap.STATUS] == pytest.approx(1.0))

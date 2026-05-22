"""Reader for Delphes binary .pileup files (XDR big-endian format).

File layout::

    [event_0_data] [event_1_data] ... [event_N-1_data] [index_table] [num_entries: int64]

Each event's data::

    int32  entry_size          # number of particles
    entry_size x record        # one record per particle

Particle record (9 fields, all big-endian)::

    int32   pid
    float32 x, y, z, t        # production vertex (mm, mm/c)
    float32 px, py, pz, e     # 4-momentum (GeV)

Index table::

    num_entries x int64        # byte offsets pointing to start of each event

Footer::

    int64 num_entries          # total number of events in the file

Reference C++ implementation: ``delphes_cmp/DelphesPileUpReader.cc``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from parnassus.data.particle_io import (
    N_FEATURES,
    PT_MIN,
    ColumnMap,
    get_charge_from_pdg_id,
    get_mass_from_pdg_id,
)

__all__ = ["read_pileup_file"]

# Number of float32 fields per particle record (after the int32 pid)
_N_FLOAT_FIELDS = 8  # x, y, z, t, px, py, pz, e
# Total bytes per particle: 4 (pid) + 8*4 (floats)
_RECORD_BYTES = 4 + _N_FLOAT_FIELDS * 4  # = 36

# Structured numpy dtype for one particle record: int32 pid + 8 float32s
_RECORD_DTYPE = np.dtype([("pid", ">i4"), ("floats", ">f4", 8)])


def _raw_to_columnmap(raw: np.ndarray) -> np.ndarray:
    """Convert raw particle records to ColumnMap format.

    Parameters
    ----------
    raw : np.ndarray
        Shape ``(N, 9)`` with columns ``[pid, x, y, z, t, px, py, pz, e]``.
        The pid column is stored as a float (cast from int32) so that the array
        is uniform float64.

    Returns
    -------
    np.ndarray
        Shape ``(N, N_FEATURES)`` with dtype ``float64``.
    """
    n = raw.shape[0]
    out = np.zeros((n, N_FEATURES), dtype=np.float64)

    if n == 0:
        return out

    pids = raw[:, 0].astype(np.int64)

    # Raw columns: pid=0, x=1, y=2, z=3, t=4, px=5, py=6, pz=7, e=8
    x = raw[:, 1]
    y = raw[:, 2]
    z = raw[:, 3]
    t = raw[:, 4]
    px = raw[:, 5]
    py = raw[:, 6]
    pz = raw[:, 7]
    e = raw[:, 8]

    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)

    # Match C++ Delphes: ±999.9 for zero-pt particles
    low_pt = pt < PT_MIN
    eta = np.where(
        low_pt,
        np.sign(pz) * 999.9,
        np.arcsinh(pz / np.where(low_pt, 1.0, pt)),
    )
    # If pt==0 AND pz==0, eta = 0.0 (np.sign(0.0) == 0.0, already handled)

    charges = get_charge_from_pdg_id(pids)
    masses = get_mass_from_pdg_id(pids)

    out[:, ColumnMap.PID] = pids
    out[:, ColumnMap.STATUS] = 1.0  # all MinBias particles are final-state
    out[:, ColumnMap.CHARGE] = charges
    out[:, ColumnMap.E] = e
    out[:, ColumnMap.PX] = px
    out[:, ColumnMap.PY] = py
    out[:, ColumnMap.PZ] = pz
    out[:, ColumnMap.PT] = pt
    out[:, ColumnMap.ETA] = eta
    out[:, ColumnMap.PHI] = phi
    out[:, ColumnMap.T] = t
    out[:, ColumnMap.X] = x
    out[:, ColumnMap.Y] = y
    out[:, ColumnMap.Z] = z
    out[:, ColumnMap.MASS] = masses
    # ETA_OUTER, PHI_OUTER, PASS_PROP, TRACK_RESOLUTION, EVENT_NUMBER remain 0 (default)

    return out


def read_pileup_file(path: Path | str) -> list[np.ndarray]:
    """Read a Delphes binary .pileup file.

    Parameters
    ----------
    path : Path | str
        Path to the ``.pileup`` file.

    Returns
    -------
    list[np.ndarray]
        One ``(n_particles, N_FEATURES)`` float64 array per MinBias event,
        in ``ColumnMap`` column order.  Empty events produce shape ``(0, N_FEATURES)``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pile-up file not found: {path}")

    data = path.read_bytes()

    # Read footer: num_entries (int64 big-endian, last 8 bytes)
    (num_entries,) = struct.unpack_from(">q", data, len(data) - 8)

    # Read index table: num_entries int64s just before the footer
    index_offset = len(data) - 8 - num_entries * 8
    offsets: list[int] = list(struct.unpack_from(f">{num_entries}q", data, index_offset))

    # Parse each event
    results: list[np.ndarray] = []
    for offset in offsets:
        # int32 entry_size at offset
        (entry_size,) = struct.unpack_from(">i", data, offset)
        pos = offset + 4  # skip the entry_size int32

        if entry_size == 0:
            results.append(np.zeros((0, N_FEATURES), dtype=np.float64))
            continue

        records = np.frombuffer(data, dtype=_RECORD_DTYPE, count=entry_size, offset=pos)
        raw = np.zeros((entry_size, 9), dtype=np.float64)
        raw[:, 0] = records["pid"]
        raw[:, 1:] = records["floats"]
        results.append(_raw_to_columnmap(raw))

    return results

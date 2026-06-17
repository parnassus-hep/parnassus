"""HepMC dataset in ColumnMap format.

Provides :class:`HepMCDataset`, a ``torch.utils.data.Dataset`` that loads all
particles from a HepMC3 file and stores them as per-event tensors using the
:class:`~parnassus.data.particle_io.ColumnMap` feature layout.

No selection cuts, no transforms — raw physics variables only.  Lightweight
adapters in :mod:`parnassus.data.adapters` wrap this dataset for neural-network
generation or parametric (torch_delphes) simulation.

Implementation
--------------
Rather than building full :mod:`pyhepmc` event objects, the file is parsed
line-by-line (mirroring C++ ``DelphesHepMC3Reader``): only ``P`` (particle),
``V`` (vertex) and ``E`` (event) records are read.  All particle numbers are
parsed in one shot by pyarrow's multithreaded CSV reader and the physics
(``pt``/``eta``/``phi``/charge) is computed vectorized over the whole file.
This is ~15-20x faster than the per-particle pyhepmc attribute access while
producing bit-identical tensors.  Only the ``GEV``/``MM`` and ``MEV``/``CM``
unit conventions are supported (constant per file, matching real generator
output); per-event unit changes are not handled.
"""

import gzip
import io
from typing import override

import numpy as np
import pyarrow.csv as pacsv
import torch

from parnassus.utils.logger import ProgressBar

from .base import RawDataset
from .particle_io import N_FEATURES, ColumnMap, build_particle_array

# HepMC3 ASCII record types, as leading bytes.
_P = ord("P")  # particle
_V = ord("V")  # vertex
_E = ord("E")  # event header
_U = ord("U")  # units
_AT = ord("@")  # vertex-position marker

# Number of whitespace-separated fields on a ``P`` line after the leading "P ".
# Order: id, out_vertex, pid, px, py, pz, e, mass, status
_N_P_FIELDS = 9


def _unit_coefficients(line: bytes) -> tuple[float, float]:
    """Return ``(momentum, position)`` scale factors from a ``U`` line.

    ``MEV`` momenta are scaled to GeV (by 1e-3) and ``CM`` positions to mm
    (by 10), matching C++ Delphes.  ``GEV``/``MM`` give unity.

    Returns
    -------
    tuple[float, float]
        The momentum and position scale coefficients.
    """
    parts = line.split()
    mom = 1.0e-3 if len(parts) > 1 and parts[1] == b"MEV" else 1.0
    pos = 10.0 if len(parts) > 2 and parts[2] == b"CM" else 1.0
    return mom, pos


class HepMCDataset(RawDataset):
    """Dataset of HepMC particles in ColumnMap tensor format.

    Each item is a dictionary with the per-event particle tensor and metadata.
    Events with no particles are retained (empty tensor) so that event indices
    remain predictable.

    Parameters
    ----------
    file_path : Path | str
        Path to HepMC file (``.hepmc`` / ``.hepmc3`` / ``.hepmc.gz``).
    num_events : int | None, optional
        Maximum number of events to load.  ``None`` loads all events.
    """

    @override
    def _load(self) -> None:
        p_lines: list[bytes] = []  # raw "P" lines (without leading "P ")
        counts: list[int] = []  # particles per event
        vtx_per_event: list[dict[int, tuple[float, float, float, float]]] = []
        cur_count = 0
        cur_vtx: dict[int, tuple[float, float, float, float]] = {}
        in_event = False
        mom_coef, pos_coef = 1.0, 1.0

        opener = gzip.open if self.file_path.suffix == ".gz" else open
        with opener(self.file_path, "rb") as fh:
            for raw in fh:
                kind = raw[0]
                if kind == _P:
                    p_lines.append(raw[2:])
                    cur_count += 1
                elif kind == _V:
                    if _AT in raw:
                        parts = raw.split()
                        at = parts.index(b"@")
                        cur_vtx[int(parts[1])] = (
                            float(parts[at + 1]),
                            float(parts[at + 2]),
                            float(parts[at + 3]),
                            float(parts[at + 4]),
                        )
                elif kind == _E:
                    if in_event:
                        counts.append(cur_count)
                        vtx_per_event.append(cur_vtx)
                        if self.num_events is not None and len(counts) >= self.num_events:
                            break
                        cur_count = 0
                        cur_vtx = {}
                    self._event_numbers.append(int(raw.split()[1]))
                    in_event = True
                elif kind == _U:
                    mom_coef, pos_coef = _unit_coefficients(raw)
            else:
                # Reached EOF without an extra E line: finalize the last event.
                if in_event:
                    counts.append(cur_count)
                    vtx_per_event.append(cur_vtx)

        # ``_event_numbers`` may have one trailing entry if we broke at the E
        # line of event ``num_events`` before recording its particles.
        del self._event_numbers[len(counts) :]
        self._build_event_tensors(p_lines, counts, vtx_per_event, mom_coef, pos_coef)

    def _build_event_tensors(
        self,
        p_lines: list[bytes],
        counts: list[int],
        vtx_per_event: list[dict[int, tuple[float, float, float, float]]],
        mom_coef: float,
        pos_coef: float,
    ) -> None:
        """Parse collected ``P`` lines and split into per-event tensors."""
        n_total = sum(counts)
        if n_total == 0:
            self._event_tensors = [torch.zeros((c, N_FEATURES)) for c in counts]
            return

        table = pacsv.read_csv(
            io.BytesIO(b"".join(p_lines[:n_total])),
            parse_options=pacsv.ParseOptions(delimiter=" "),
            read_options=pacsv.ReadOptions(autogenerate_column_names=True),
        )
        cols = [table.column(i).to_numpy().astype(np.float64) for i in range(_N_P_FIELDS)]
        _id, out_vtx, pid, px, py, pz, e, mass, status = cols

        full = build_particle_array(
            pid=pid,
            status=status,
            e=e * mom_coef,
            px=px * mom_coef,
            py=py * mom_coef,
            pz=pz * mom_coef,
            mass=mass * mom_coef,
            x=np.zeros(n_total),
            y=np.zeros(n_total),
            z=np.zeros(n_total),
            t=np.zeros(n_total),
            event_number=0.0,
        )
        self._assign_vertices_and_events(
            full, out_vtx, counts, vtx_per_event, self._event_numbers, pos_coef
        )

        bounds = np.cumsum(counts)[:-1]
        with ProgressBar() as progress:
            task = progress.add_task("[green]Reading data from HepMC file", total=len(counts))
            self._event_tensors = []
            for chunk in np.split(full, bounds):
                self._event_tensors.append(torch.from_numpy(np.ascontiguousarray(chunk)))
                progress.update(task, advance=1)

    @staticmethod
    def _assign_vertices_and_events(
        full: np.ndarray,
        out_vtx: np.ndarray,
        counts: list[int],
        vtx_per_event: list[dict[int, tuple[float, float, float, float]]],
        event_numbers: list[int],
        pos_coef: float,
    ) -> None:
        """Write per-event ``EVENT_NUMBER`` and production-vertex positions.

        Each particle's production vertex is the vertex whose code equals its
        out-vertex field; positions are resolved per event with a vectorized
        ``searchsorted`` lookup.
        """
        out_vtx = out_vtx.astype(np.int64)
        offset = 0
        for k, n in enumerate(counts):
            sl = slice(offset, offset + n)
            full[sl, ColumnMap.EVENT_NUMBER] = event_numbers[k]
            vtx = vtx_per_event[k]
            if vtx and n:
                codes = np.fromiter(vtx.keys(), dtype=np.int64, count=len(vtx))
                positions = np.array(list(vtx.values()), dtype=np.float64)
                order = np.argsort(codes)
                codes, positions = codes[order], positions[order]
                idx = np.clip(np.searchsorted(codes, out_vtx[sl]), 0, len(codes) - 1)
                valid = codes[idx] == out_vtx[sl]
                xyzt = np.where(valid[:, None], positions[idx], 0.0) * pos_coef
                full[sl, ColumnMap.X] = xyzt[:, 0]
                full[sl, ColumnMap.Y] = xyzt[:, 1]
                full[sl, ColumnMap.Z] = xyzt[:, 2]
                full[sl, ColumnMap.T] = xyzt[:, 3]
            offset += n

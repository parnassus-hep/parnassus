"""Lightweight adapters from :class:`HepMCRawDataset` to generator-specific formats.

Two adapters are provided:

* :class:`NeuralAdapter` — standalone ``Dataset`` that applies particle selection
  cuts and variable transforms, producing the same
  ``{ctxt_data, ctxt_global_data, mask, event_number}`` dict that
  :class:`~parnassus.data.base.BaseDataset` subclasses produce.

* :class:`ParametricAdapter` — thin wrapper that returns per-event particle
  tensors in ColumnMap format.  Use with :func:`parametric_collate_fn` to
  concatenate events into the flat tensor expected by the torch_delphes modules.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.utils import pid_to_class
from parnassus.utils.transform import VarTransform
from parnassus.utils.typing import FloatArray

from .hepmc_raw import HepMCRawDataset
from .transforms import (
    SCALAR_KEYS,
    get_event_data,
    prepare_ctxt_global_data,
    preprocess_flat_arrays,
)

if TYPE_CHECKING:
    from parnassus.utils.typing import IntArray, LongArray

_pid_to_class_vec = np.vectorize(pid_to_class)


# ==================== NEURAL ADAPTER ====================


class NeuralAdapter(Dataset[dict[str, Tensor]]):
    """Standalone dataset that wraps :class:`HepMCRawDataset` for neural generation.

    Applies particle selection cuts, computes derived variables (``ptrel``,
    ``ht``, ``met_x``, ``met_y``), scales variables via ``VarTransform``, and
    returns padded, masked batches identical to those produced by
    :class:`~parnassus.data.base.BaseDataset` subclasses.

    Parameters
    ----------
    raw : HepMCRawDataset
        Pre-loaded raw dataset to read particles from.
    cfg : DatasetConfig
        Dataset configuration (max_particles, truth_vars_to_load, etc.).
    var_transform_dict : dict[str, VarTransform] | None
        Variable transformation registry for scaling.  ``None`` → no scaling.
    """

    def __init__(
        self,
        raw: HepMCRawDataset,
        cfg: DatasetConfig,
        var_transform_dict: dict[str, VarTransform] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.var_transform_dict: dict[str, VarTransform] = var_transform_dict or {}
        self.ctxt_vars: list[str] = cfg.variable_requirements.ctxt_vars_stripped
        self.ctxt_global_vars: list[str] = cfg.variable_requirements.ctxt_global_vars_stripped

        self.full_data_array: dict[str, FloatArray] = {}
        self.n_truth_particles: IntArray
        self.truth_cumsum: LongArray
        self.eventNumber: LongArray

        if not Path(cfg.file_path).exists():
            raise FileNotFoundError(f"Trying to load file {cfg.file_path}, no file exists!")

        self._load_data(raw)
        preprocess_flat_arrays(self.full_data_array, SCALAR_KEYS)

        self.n_events = len(self.n_truth_particles)
        self.scaled_ctxt_global_data: Tensor = prepare_ctxt_global_data(
            self.full_data_array,
            self.n_truth_particles,
            self.truth_cumsum,
            self.ctxt_vars,
            self.ctxt_global_vars,
            self.var_transform_dict,
        )

    def _load_data(self, raw: HepMCRawDataset) -> None:
        """Load particles from the raw dataset, applying neural selection cuts."""
        n_events_raw = len(raw)
        n_alloc = n_events_raw * self.cfg.max_particles

        self.n_truth_particles = np.zeros(n_events_raw, dtype=np.int32)
        for var in (*self.cfg.truth_vars_to_load, "ptrel"):
            self.full_data_array[var] = np.zeros(n_alloc, dtype=np.float32)

        self.eventNumber = np.zeros(n_events_raw, dtype=np.int64)
        self.full_data_array["ht"] = np.zeros(n_events_raw, dtype=np.float32)
        self.full_data_array["met_x"] = np.zeros(n_events_raw, dtype=np.float32)
        self.full_data_array["met_y"] = np.zeros(n_events_raw, dtype=np.float32)

        curr_event_idx = 0
        curr_particle_idx = 0

        for raw_idx in range(n_events_raw):
            item = raw[raw_idx]
            particles: Tensor = item["particles"]  # (N_i, N_FEATURES), float32
            event_number: int = item["event_number"]

            if particles.shape[0] == 0:
                continue

            pids_np = particles[:, ColumnMap.PID].numpy()
            status_np = particles[:, ColumnMap.STATUS].numpy()
            pt_np = particles[:, ColumnMap.PT].numpy()
            eta_np = particles[:, ColumnMap.ETA].numpy()
            phi_np = particles[:, ColumnMap.PHI].numpy()

            # Neural selection cuts
            mask = (
                (status_np == 1)
                & (np.abs(eta_np) < 2.7)
                & (pt_np > 0.25)
                & ~np.isin(np.abs(pids_np), [12, 14, 16])
            )
            num_particles = int(mask.sum())

            if num_particles >= self.cfg.max_particles:
                # Drop this event — too many particles
                continue

            # Write particle data to flat arrays
            event_start = curr_particle_idx
            end = curr_particle_idx + num_particles
            self.full_data_array["pt"][event_start:end] = pt_np[mask]
            self.full_data_array["eta"][event_start:end] = eta_np[mask]
            self.full_data_array["phi"][event_start:end] = phi_np[mask]
            self.full_data_array["class"][event_start:end] = _pid_to_class_vec(
                pids_np[mask].astype(int)
            )
            if "vx" in self.cfg.truth_vars_to_load:
                self.full_data_array["vx"][event_start:end] = particles[:, ColumnMap.X].numpy()[
                    mask
                ]
            if "vy" in self.cfg.truth_vars_to_load:
                self.full_data_array["vy"][event_start:end] = particles[:, ColumnMap.Y].numpy()[
                    mask
                ]
            if "vz" in self.cfg.truth_vars_to_load:
                self.full_data_array["vz"][event_start:end] = particles[:, ColumnMap.Z].numpy()[
                    mask
                ]
            curr_particle_idx = end

            pt_slice = self.full_data_array["pt"][event_start:curr_particle_idx]
            ht = float(pt_slice.sum())
            self.full_data_array["ht"][curr_event_idx] = ht
            self.full_data_array["ptrel"][event_start:curr_particle_idx] = (
                pt_slice / ht if ht > 0 else pt_slice
            )
            phi_slice = self.full_data_array["phi"][event_start:curr_particle_idx]
            self.full_data_array["met_x"][curr_event_idx] = float(
                (pt_slice * np.cos(phi_slice)).sum()
            )
            self.full_data_array["met_y"][curr_event_idx] = float(
                (pt_slice * np.sin(phi_slice)).sum()
            )
            self.n_truth_particles[curr_event_idx] = num_particles
            self.eventNumber[curr_event_idx] = event_number
            curr_event_idx += 1

        # Trim to actual size and drop raw pt (use ptrel instead)
        _ = self.full_data_array.pop("pt")
        for key in self.ctxt_vars:
            self.full_data_array[key] = self.full_data_array[key][:curr_particle_idx]
        for key in ("ht", "met_x", "met_y"):
            self.full_data_array[key] = self.full_data_array[key][:curr_event_idx]
        self.eventNumber = self.eventNumber[:curr_event_idx]
        self.n_truth_particles = self.n_truth_particles[:curr_event_idx]
        self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])

    def __len__(self) -> int:
        return len(self.n_truth_particles)

    def __getitem__(self, idx: Any) -> dict[str, Tensor]:  # pyright: ignore[reportImplicitOverride]
        ctxt_data, ctxt_global_data, mask = get_event_data(
            idx,
            self.full_data_array,
            self.n_truth_particles,
            self.truth_cumsum,
            self.ctxt_vars,
            self.scaled_ctxt_global_data,
            self.cfg.max_particles,
            self.var_transform_dict,
        )
        event_number = torch.tensor(self.eventNumber[idx], dtype=torch.long).unsqueeze(-1)
        return {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
            "mask": mask,
            "event_number": event_number,
        }


# ==================== PARAMETRIC ADAPTER ====================


class ParametricAdapter(Dataset):
    """Thin wrapper around :class:`HepMCRawDataset` for parametric simulation.

    Returns per-event particle tensors in ColumnMap format without any cuts
    or transforms.  Use :func:`parametric_collate_fn` as the ``collate_fn``
    when constructing a ``DataLoader``.

    Parameters
    ----------
    raw : HepMCRawDataset
        Pre-loaded raw dataset.
    """

    def __init__(self, raw: HepMCRawDataset) -> None:
        self._raw = raw

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        return self._raw[idx]


def parametric_collate_fn(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate function for :class:`ParametricAdapter` batches.

    Concatenates per-event particle tensors into a single flat tensor.  The
    ``EVENT_NUMBER`` column in each particle row already identifies event
    membership, so no additional bookkeeping is needed for the torch_delphes
    detector modules.

    Parameters
    ----------
    batch : list[dict]
        List of items from :class:`ParametricAdapter.__getitem__`.

    Returns
    -------
    dict with keys:

    ``"particles"``
        ``Tensor(N_total, N_FEATURES)`` — all particles in the batch concatenated.
    ``"event_numbers"``
        ``list[int]`` — event number per original event (length = batch size).
    ``"n_particles"``
        ``list[int]`` — particle count per event (length = batch size).
    """
    parts = [item["particles"] for item in batch]
    event_numbers = [item["event_number"] for item in batch]
    n_particles = [item["n_particles"] for item in batch]

    non_empty = [p for p in parts if p.shape[0] > 0]
    if non_empty:
        all_particles = torch.cat(non_empty, dim=0)
    else:
        all_particles = torch.zeros((0, N_FEATURES), dtype=torch.float32)

    stable_particles = all_particles[all_particles[:, ColumnMap.STATUS] == 1]

    return {
        "all_particles": all_particles,
        "stable_particles": stable_particles,
        "event_numbers": event_numbers,
        "n_particles": n_particles,
    }

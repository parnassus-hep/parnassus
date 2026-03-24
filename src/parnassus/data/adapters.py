"""Lightweight adapters from :class:`HepMCRawDataset` to generator-specific formats.

Two adapters are provided:

* :class:`NeuralAdapter` — wraps :class:`HepMCRawDataset` to produce the same
  ``{ctxt_data, ctxt_global_data, mask, event_number}`` dict that
  :class:`~parnassus.data.base.BaseDataset` subclasses produce, making it a
  drop-in replacement for :class:`~parnassus.data.hepmc.HepMCDataset` in the
  neural generation pipeline.

* :class:`ParametricAdapter` — thin wrapper that returns per-event particle
  tensors in ColumnMap format.  Use with :func:`parametric_collate_fn` to
  concatenate events into the flat tensor expected by the torch_delphes modules.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.utils import pid_to_class
from parnassus.utils.transform import VarTransform

from .base import BaseDataset
from .hepmc_raw import HepMCRawDataset

if TYPE_CHECKING:
    pass


# ==================== NEURAL ADAPTER ====================


class NeuralAdapter(BaseDataset):
    """Wraps :class:`HepMCRawDataset` to produce the neural-generator data format.

    Applies the same particle selection cuts, derived-variable computation, and
    flat-array bookkeeping as :class:`~parnassus.data.hepmc.HepMCDataset` so
    that :class:`~parnassus.pipelines.generate.GenerationPipeline` receives
    identical batches from both dataset implementations.

    Parameters
    ----------
    raw : HepMCRawDataset
        Pre-loaded raw dataset to read particles from.
    cfg : DatasetConfig
        Dataset configuration (max_particles, truth_vars_to_load, etc.).
    var_transform_dict : dict[str, VarTransform]
        Variable transformation registry for scaling.
    """

    def __init__(
        self,
        raw: HepMCRawDataset,
        cfg: DatasetConfig,
        var_transform_dict: dict[str, VarTransform] | None = None,
    ) -> None:
        # Assign _raw BEFORE calling super().__init__(), which triggers load_data()
        self._raw = raw
        super().__init__(cfg=cfg, var_transform_dict=var_transform_dict or {})
        # Free raw tensors — all data is now in flat numpy arrays
        self._raw = None  # type: ignore[assignment]

    def _load_data(self) -> None:
        """Load particles from the raw dataset, applying neural selection cuts."""
        assert self._raw is not None, "Raw dataset must be assigned before load_data()"

        n_events_raw = len(self._raw)
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
            item = self._raw[raw_idx]
            particles: Tensor = item["particles"]  # (N_i, N_FEATURES), float32
            event_number: int = item["event_number"]

            if particles.shape[0] == 0:
                continue

            event_start = curr_particle_idx
            num_particles = 0

            # Access relevant columns as numpy for fast iteration
            pids_np = particles[:, ColumnMap.PID].numpy()
            pt_np = particles[:, ColumnMap.PT].numpy()
            eta_np = particles[:, ColumnMap.ETA].numpy()
            phi_np = particles[:, ColumnMap.PHI].numpy()

            # Optional columns
            x_np = (
                particles[:, ColumnMap.X].numpy() if "vx" in self.cfg.truth_vars_to_load else None
            )
            y_np = (
                particles[:, ColumnMap.Y].numpy() if "vy" in self.cfg.truth_vars_to_load else None
            )
            z_np = (
                particles[:, ColumnMap.Z].numpy() if "vz" in self.cfg.truth_vars_to_load else None
            )

            for i in range(particles.shape[0]):
                pid = int(pids_np[i])
                pt = float(pt_np[i])
                eta = float(eta_np[i])

                # Neural selection cuts (status==1 already guaranteed by HepMCRawDataset)
                if abs(eta) >= 2.7 or pt <= 0.25 or abs(pid) in {12, 14, 16}:
                    continue

                self.full_data_array["pt"][curr_particle_idx] = pt
                self.full_data_array["eta"][curr_particle_idx] = eta
                self.full_data_array["phi"][curr_particle_idx] = float(phi_np[i])
                self.full_data_array["class"][curr_particle_idx] = float(pid_to_class(pid))
                if x_np is not None:
                    self.full_data_array["vx"][curr_particle_idx] = float(x_np[i])
                if y_np is not None:
                    self.full_data_array["vy"][curr_particle_idx] = float(y_np[i])
                if z_np is not None:
                    self.full_data_array["vz"][curr_particle_idx] = float(z_np[i])

                num_particles += 1
                curr_particle_idx += 1

            if num_particles >= self.cfg.max_particles:
                # Drop this event — too many particles
                curr_particle_idx -= num_particles
                continue

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

        # Trim to actual size
        _ = self.full_data_array.pop("pt")
        for key in self.ctxt_vars:
            self.full_data_array[key] = self.full_data_array[key][:curr_particle_idx]
        for key in ("ht", "met_x", "met_y"):
            self.full_data_array[key] = self.full_data_array[key][:curr_event_idx]
        self.eventNumber = self.eventNumber[:curr_event_idx]
        self.n_truth_particles = self.n_truth_particles[:curr_event_idx]
        self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])


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

    return {
        "particles": all_particles,
        "event_numbers": event_numbers,
        "n_particles": n_particles,
    }

"""Lightweight adapters from :class:`RawDataset` to generator-specific formats.

Two adapters are provided:

* :class:`NeuralAdapter` — standalone ``Dataset`` that applies particle selection
  cuts and variable transforms at ``__getitem__`` time, producing
  ``{ctxt_data, ctxt_global_data, mask, event_number}`` batches.

* :class:`ParametricAdapter` — thin wrapper that returns per-event particle
  tensors in ColumnMap format.  Use with :func:`parametric_collate_fn` to
  concatenate events into the flat tensor expected by the torch_delphes modules.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.utils import pid_to_class
from parnassus.utils.transform import VarTransform

from .base import RawDataset
from .transforms import do_padding

_pid_to_class_vec = np.vectorize(pid_to_class)


# ==================== NEURAL ADAPTER ====================


class NeuralAdapter(Dataset[dict[str, Tensor]]):
    """Dataset adapter that applies neural selection cuts and variable transforms.

    Wraps any :class:`~parnassus.data.base.RawDataset` and converts per-event
    ColumnMap tensors to padded, masked batches suitable for neural generation.

    Selection cuts applied at both init (for filtering) and ``__getitem__``
    (for actual data extraction):

    * ``status == 1`` (final-state particles only)
    * ``|eta| < 2.7``
    * ``pt > 0.25 GeV``
    * Neutrino PIDs (12, 14, 16) excluded

    Parameters
    ----------
    raw : RawDataset
        Pre-loaded raw dataset to read particles from.
    cfg : DatasetConfig
        Dataset configuration (max_particles, truth_vars_to_load, etc.).
    var_transform_dict : dict[str, VarTransform] | None
        Variable transformation registry for scaling.  ``None`` → no scaling.
    """

    def __init__(
        self,
        raw: RawDataset,
        cfg: DatasetConfig,
        var_transform_dict: dict[str, VarTransform] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.var_transform_dict: dict[str, VarTransform] = var_transform_dict or {}
        self.ctxt_vars: list[str] = cfg.variable_requirements.ctxt_vars_stripped
        self.ctxt_global_vars: list[str] = cfg.variable_requirements.ctxt_global_vars_stripped

        self._raw = raw
        # Lightweight pre-pass: record which raw event indices survive the cuts.
        # No particle data is copied — only integer indices are stored.
        self._valid_idx: list[int] = []
        self._event_numbers: list[int] = []
        for i in range(len(raw)):
            item = raw[i]
            p: Tensor = item["particles"].float()
            if p.shape[0] == 0:
                continue
            mask = self._selection_mask(p)
            n_filtered = int(mask.sum())
            if 0 < n_filtered < cfg.max_particles:
                self._valid_idx.append(i)
                self._event_numbers.append(int(item["event_number"]))

    @staticmethod
    def _selection_mask(p: Tensor) -> Tensor:
        """Return boolean mask selecting neural-cut particles from a ColumnMap tensor.

        Returns
        -------
        Tensor
            Boolean mask of shape ``(N,)``.
        """
        abs_pid = p[:, ColumnMap.PID].abs()
        return (
            (p[:, ColumnMap.STATUS] == 1)
            & (p[:, ColumnMap.ETA].abs() < 2.7)
            & (p[:, ColumnMap.PT] > 0.25)
            & (abs_pid != 12)
            & (abs_pid != 14)
            & (abs_pid != 16)
        )

    def __len__(self) -> int:
        return len(self._valid_idx)

    def __getitem__(self, idx: Any) -> dict[str, Tensor]:  # pyright: ignore[reportImplicitOverride]
        raw_idx = self._valid_idx[idx]
        particles: Tensor = self._raw[raw_idx]["particles"].float()

        # Apply selection and extract as float32 numpy
        mask = self._selection_mask(particles)
        filtered = particles[mask]
        n_part = filtered.shape[0]

        pt = filtered[:, ColumnMap.PT].float()
        eta = torch.clip(filtered[:, ColumnMap.ETA].float(), -3.0, 3.0)
        phi_raw = filtered[:, ColumnMap.PHI].float()
        phi = torch.arctan2(torch.sin(phi_raw), torch.cos(phi_raw))
        pid = filtered[:, ColumnMap.PID].long()

        # Derived event quantities
        ht = pt.sum()
        ptrel = pt / ht if ht > 0.0 else pt
        met_x = (pt * torch.cos(phi)).sum()
        met_y = (pt * torch.sin(phi)).sum()

        # Sort particles by descending ptrel
        sort_idx = torch.argsort(ptrel, descending=True)

        # Build per-variable sorted arrays (used for ctxt_data and means)
        var_arrays: dict[str, torch.Tensor] = {
            "ptrel": ptrel[sort_idx],
            "eta": eta[sort_idx],
            "phi": phi[sort_idx],
            "class": torch.tensor(_pid_to_class_vec(pid[sort_idx].numpy())),
        }
        for vname, col in (("vx", ColumnMap.X), ("vy", ColumnMap.Y), ("vz", ColumnMap.Z)):
            if vname in self.ctxt_vars:
                var_arrays[vname] = filtered[:, col].float()[sort_idx]

        # Build per-particle context tensor
        apply_transforms = bool(self.var_transform_dict)
        ctxt_data_list: list[Tensor] = []
        for var in self.ctxt_vars:
            x = var_arrays[var].view(-1, 1)
            if var == "phi" and apply_transforms:
                ctxt_data_list.extend([torch.sin(x), torch.cos(x)])
            elif var == "class" and apply_transforms:
                ctxt_data_list.append(F.one_hot(x.long().squeeze(-1), num_classes=5).float())
            elif apply_transforms and var in self.var_transform_dict:
                ctxt_data_list.append(self.var_transform_dict[var].transform(x).float())
            else:
                ctxt_data_list.append(x)

        ctxt_data = do_padding(torch.cat(ctxt_data_list, dim=-1), self.cfg.max_particles)

        # Per-event global context
        ctxt_global_data = self._compute_global_context(
            n_part, ht, met_x, met_y, var_arrays, apply_transforms
        )

        mask_tensor = torch.zeros(self.cfg.max_particles, dtype=torch.bool)
        mask_tensor[:n_part] = True

        event_number = torch.tensor(self._event_numbers[idx], dtype=torch.long).unsqueeze(-1)

        return {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
            "mask": mask_tensor,
            "event_number": event_number,
        }

    def _compute_global_context(
        self,
        n_part: int,
        ht: Tensor,
        met_x: Tensor,
        met_y: Tensor,
        var_arrays: dict[str, Tensor],
        apply_transforms: bool,
    ) -> Tensor:
        """Build the global context vector for a single event.

        Returns
        -------
        Tensor
            Shape ``(n_global_features,)``.
        """
        scalar_map = {"ht": ht, "met_x": met_x, "met_y": met_y}
        parts: list[Tensor] = []

        for var in self.ctxt_global_vars:
            if var == "means":
                # Mean of each non-class per-particle ctxt_var
                mean_vals: list[Tensor] = []
                for cv in self.ctxt_vars:
                    if "class" not in cv and cv in var_arrays:
                        arr = var_arrays[cv]
                        mean_vals.append(
                            arr.mean() if len(arr) > 0 else torch.tensor(0.0, dtype=torch.float32)
                        )
                parts.append(torch.stack(mean_vals))
            elif var.startswith("ntruth"):
                data = torch.tensor(float(n_part), dtype=torch.float32)
                if apply_transforms:
                    data = self.var_transform_dict["npart"].transform(data)
                parts.append(data.view(1))  # shape (1,)
            else:
                data = scalar_map[var]
                if apply_transforms and var in self.var_transform_dict:
                    data = self.var_transform_dict[var].transform(data)
                parts.append(data.view(1))  # shape (1,)

        return torch.cat(parts, dim=0)


# ==================== PARAMETRIC ADAPTER ====================


class ParametricAdapter(Dataset):
    """Thin wrapper around :class:`RawDataset` for parametric simulation.

    Returns per-event particle tensors in ColumnMap format without any cuts
    or transforms.  Use :func:`parametric_collate_fn` as the ``collate_fn``
    when constructing a ``DataLoader``.

    Parameters
    ----------
    raw : RawDataset
        Pre-loaded raw dataset.
    """

    def __init__(self, raw: RawDataset) -> None:
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
        "event_numbers": torch.tensor(event_numbers),
        "n_particles": torch.tensor(n_particles),
    }

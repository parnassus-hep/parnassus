"""Shared ML transform helpers for parnassus data modules.

Free functions that implement the common preprocessing and per-event
transform logic shared by :class:`~parnassus.data.base.BaseDataset`
(used by ``RootDataset`` / ``PythiaDataset``) and
:class:`~parnassus.data.adapters.NeuralAdapter`.

Provides
--------
- :func:`do_padding` — pad a tensor to a fixed length
- :func:`preprocess_flat_arrays` — clip eta, wrap phi, concatenate jagged arrays
- :func:`prepare_ctxt_global_data` — compute pre-scaled global context tensor
- :func:`get_event_data` — apply per-event transforms, sort, and pad
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from parnassus.utils.logger import ProgressBar
from parnassus.utils.transform import VarTransform
from parnassus.utils.typing import FloatArray

if TYPE_CHECKING:
    from parnassus.utils.typing import BoolArray, IntArray, LongArray


# Keys whose arrays are per-event scalars (not per-particle, never concatenated)
SCALAR_KEYS: frozenset[str] = frozenset({"ht", "eventNumber", "met_x", "met_y"})


# ==================== PADDING ====================


def do_padding(tensor: Tensor, max_len: int) -> Tensor:
    """Pad a tensor to the specified maximum length along the first dimension.

    Parameters
    ----------
    tensor : Tensor
        Input tensor to pad.
    max_len : int
        Desired maximum length along the first dimension.

    Returns
    -------
    Tensor
        Padded tensor of shape ``(max_len, ...)``.
    """
    shape = tensor.shape
    new_shape = (max_len, *shape[1:])
    x = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    x[: shape[0]] = tensor
    return x


# ==================== PREPROCESSING ====================


def preprocess_flat_arrays(
    full_data_array: dict[str, FloatArray],
    scalar_keys: frozenset[str] = SCALAR_KEYS,
    n_particle_mask: BoolArray | None = None,
) -> None:
    """Apply physics preprocessing to flat particle arrays in place.

    Performs three operations on every entry in ``full_data_array``:

    1. **Event masking** — if ``n_particle_mask`` is provided, filters rows
       (used by :class:`~parnassus.data.root.RootDataset` to drop events with
       too many particles).
    2. **Concatenation** — object-dtype arrays (jagged, from ROOT) are
       concatenated into a single flat array.
    3. **Physics normalisation** — eta clipped to ``[-3, 3]``;
       phi wrapped to ``(-π, π)``.

    Parameters
    ----------
    full_data_array : dict[str, FloatArray]
        Mapping of variable name → numpy array. **Mutated in place.**
    scalar_keys : frozenset[str]
        Keys that hold per-event scalars and must not be concatenated.
        Defaults to ``{"ht", "eventNumber", "met_x", "met_y"}``.
    n_particle_mask : BoolArray | None
        Boolean mask of shape ``(n_events,)`` applied to all arrays when
        provided.  ``None`` skips masking.
    """
    with ProgressBar() as progress:
        task = progress.add_task("[green]Preprocessing data", total=len(full_data_array))
        for var in list(full_data_array):
            arr = full_data_array[var]
            if n_particle_mask is not None:
                arr = arr[n_particle_mask]
            if var not in scalar_keys and arr.dtype == object:
                arr = np.concatenate(arr)
            if "eta" in var:
                arr = np.clip(arr, -3, 3)
            elif "phi" in var:
                arr = np.atan2(np.sin(arr), np.cos(arr))
            full_data_array[var] = arr
            progress.update(task, advance=1)


# ==================== GLOBAL CONTEXT ====================


def _calculate_means(
    full_data_array: dict[str, FloatArray],
    n_truth_particles: IntArray,
    truth_cumsum: LongArray,
    ctxt_vars: list[str],
    n_events: int,
) -> FloatArray:
    """Compute per-event particle means for non-class context variables.

    Returns
    -------
    FloatArray
        Shape ``(n_events, n_non_class_vars)``.
    """
    n_vars = len([v for v in ctxt_vars if "class" not in v])
    means = np.zeros((n_events, n_vars), dtype=np.float32)
    with ProgressBar() as progress:
        task = progress.add_task("[green]Calculating means", total=len(ctxt_vars))
        for i, var in enumerate(ctxt_vars):
            if "class" not in var:
                means[:, i] = (
                    np.add.reduceat(
                        full_data_array[var],
                        truth_cumsum[:-1],
                    )
                    / n_truth_particles
                )
            progress.update(task, advance=1)
    return means


def prepare_ctxt_global_data(
    full_data_array: dict[str, FloatArray],
    n_truth_particles: IntArray,
    truth_cumsum: LongArray,
    ctxt_vars: list[str],
    ctxt_global_vars: list[str],
    var_transform_dict: dict[str, VarTransform],
) -> Tensor:
    """Compute the pre-scaled global context tensor for all events.

    Handles three kinds of global variables:

    * ``"means"`` — per-event particle-feature means
    * ``"ntruth*"`` — particle count, optionally scaled by ``"npart"`` transform
    * anything else — scalar from ``full_data_array``, optionally scaled

    Parameters
    ----------
    full_data_array : dict[str, FloatArray]
        Flat particle arrays and per-event scalar arrays.
    n_truth_particles : IntArray
        Particle count per event, shape ``(n_events,)``.
    truth_cumsum : LongArray
        Cumulative particle counts, shape ``(n_events + 1,)``.
    ctxt_vars : list[str]
        Per-particle context variable names (for mean computation).
    ctxt_global_vars : list[str]
        Global context variable names to concatenate.
    var_transform_dict : dict[str, VarTransform]
        Scaling registry; empty dict → no scaling.

    Returns
    -------
    Tensor
        Shape ``(n_events, n_global_features)``.
    """
    apply_transforms = bool(var_transform_dict)
    n_events = len(n_truth_particles)
    means = _calculate_means(full_data_array, n_truth_particles, truth_cumsum, ctxt_vars, n_events)

    parts: list[Tensor] = []
    for var in ctxt_global_vars:
        if var == "means":
            parts.append(torch.tensor(means, dtype=torch.float32))
        elif var.startswith("ntruth"):
            data = torch.tensor(n_truth_particles, dtype=torch.float32).view(-1, 1)
            if apply_transforms:
                data = var_transform_dict["npart"].transform(data)
            parts.append(data)
        else:
            data = torch.tensor(full_data_array[var], dtype=torch.float32).view(-1, 1)
            if apply_transforms:
                data = var_transform_dict[var].transform(data)
            parts.append(data)
    return torch.cat(parts, dim=-1)


# ==================== PER-EVENT TRANSFORM ====================


def get_event_data(
    idx: int,
    full_data_array: dict[str, FloatArray],
    n_truth_particles: IntArray,
    truth_cumsum: LongArray,
    ctxt_vars: list[str],
    scaled_ctxt_global_data: Tensor,
    max_particles: int,
    var_transform_dict: dict[str, VarTransform],
) -> tuple[Tensor, Tensor, Tensor]:
    """Get transformed, padded data for a single event.

    Sorts particles by descending ``ptrel``, applies per-variable transforms
    (sin/cos for phi, one-hot for class, ``VarTransform`` for others), pads
    to ``max_particles``, and constructs the boolean mask.

    Parameters
    ----------
    idx : int
        Event index.
    full_data_array : dict[str, FloatArray]
        Flat particle and event arrays.
    n_truth_particles : IntArray
        Particle count per event.
    truth_cumsum : LongArray
        Cumulative particle counts.
    ctxt_vars : list[str]
        Per-particle context variable names (in output column order).
    scaled_ctxt_global_data : Tensor
        Pre-computed global context, shape ``(n_events, n_global_features)``.
    max_particles : int
        Padding target length.
    var_transform_dict : dict[str, VarTransform]
        Scaling registry; empty dict → raw values, no expansion.

    Returns
    -------
    ctxt_data : Tensor
        Shape ``(max_particles, n_features)``.
    ctxt_global_data : Tensor
        Shape ``(n_global_features,)``.
    mask : Tensor
        Boolean mask of shape ``(max_particles,)``.
    """
    apply_transforms = bool(var_transform_dict)
    n_part = int(n_truth_particles[idx])
    truth_start = int(truth_cumsum[idx])
    truth_end = int(truth_cumsum[idx + 1])

    truth_idx = np.argsort(full_data_array["ptrel"][truth_start:truth_end], axis=0)[::-1]

    ctxt_data_list: list[Tensor] = []
    for var in ctxt_vars:
        x = torch.tensor(full_data_array[var][truth_start:truth_end][truth_idx]).view(-1, 1)
        if var == "phi" and apply_transforms:
            ctxt_data_list.extend([torch.sin(x).float(), torch.cos(x).float()])
        elif var == "class" and apply_transforms:
            ctxt_data_list.append(F.one_hot(x.long().squeeze(-1), num_classes=5).float())
        elif apply_transforms:
            ctxt_data_list.append(var_transform_dict[var].transform(x).float())
        else:
            ctxt_data_list.append(x.float())

    ctxt_data = do_padding(torch.cat(ctxt_data_list, dim=-1), max_particles)
    ctxt_global_data = scaled_ctxt_global_data[idx]
    mask = torch.zeros((max_particles,), dtype=torch.bool)
    mask[:n_part] = 1
    return ctxt_data, ctxt_global_data, mask

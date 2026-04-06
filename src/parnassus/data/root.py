from pathlib import Path
from typing import TYPE_CHECKING, Any, final

import numpy as np
import torch
import uproot
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.utils.transform import VarTransform
from parnassus.utils.typing import FloatArray

from .transforms import (
    SCALAR_KEYS,
    get_event_data,
    prepare_ctxt_global_data,
    preprocess_flat_arrays,
)

if TYPE_CHECKING:
    from parnassus.utils.typing import BoolArray, IntArray, LongArray


@final
class RootDataset(Dataset[dict[str, Tensor]]):
    """Legacy dataset class for loading events from a ROOT file.

    Reads truth-level particle arrays from a ROOT ``evt_tree``, pre-computes
    derived quantities (ht, met_x, met_y, ptrel), and stores everything in
    flat numpy arrays with a cumulative-sum index for fast per-event access.
    """

    def __init__(self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform]):
        super().__init__()
        self.cfg = cfg
        self.var_transform_dict: dict[str, VarTransform] = var_transform_dict or {}
        self.truth_vars_to_load = cfg.truth_vars_to_load
        self.ctxt_vars: list[str] = cfg.variable_requirements.ctxt_vars_stripped
        self.ctxt_global_vars: list[str] = cfg.variable_requirements.ctxt_global_vars_stripped

        self.full_data_array: dict[str, FloatArray] = {}
        self.n_particle_mask: BoolArray  # type: ignore[annotation-unchecked]
        self.n_truth_particles: IntArray  # type: ignore[annotation-unchecked]
        self.truth_cumsum: LongArray  # type: ignore[annotation-unchecked]
        self.eventNumber: Any

        if not Path(cfg.file_path).exists():
            raise FileNotFoundError(f"Trying to load file {cfg.file_path}, no file exist!")

        self._load_data()

        for attr in ("n_truth_particles", "truth_cumsum", "eventNumber"):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise AttributeError(f"'{attr}' not set by _load_data().")

        preprocess_flat_arrays(
            self.full_data_array,
            SCALAR_KEYS,
            getattr(self, "n_particle_mask", None),
        )

        self.n_events = len(self.n_truth_particles)
        self.scaled_ctxt_global_data: Tensor = prepare_ctxt_global_data(
            self.full_data_array,
            self.n_truth_particles,
            self.truth_cumsum,
            self.ctxt_vars,
            self.ctxt_global_vars,
            self.var_transform_dict,
        )

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
        event_number = np.asarray(self.eventNumber[idx])
        return {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
            "mask": mask,
            "event_number": torch.tensor(event_number, dtype=torch.long).unsqueeze(-1),
        }

    def _load_data(self) -> None:
        with uproot.open(self.cfg.file_path) as f:  # pyright: ignore  # noqa: PGH003
            tree = f["evt_tree"]
            self._update_num_events(tree)
            self._load_truth(tree)
            self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])

    def _update_num_events(self, tree: Any) -> None:
        tree_num_entries = tree.num_entries
        if self.cfg.entry_start > tree_num_entries:
            raise ValueError(
                f"Requested entry_start exceeds number of events in the file {self.cfg.file_path}"
            )
        if self.cfg.num_events > (tree_num_entries - self.cfg.entry_start):
            raise ValueError(
                f"Requested num_events ({self.cfg.num_events}) exceeds number of events"
                f" in the file {self.cfg.file_path}"
            )
        self.entry_start = self.cfg.entry_start
        self.entry_stop = self.cfg.entry_start + self.cfg.num_events

    def _load_truth(self, tree: Any) -> None:
        self.n_truth_particles = tree["ntruth"].array(
            library="np",
            entry_stop=self.entry_stop,
            entry_start=self.entry_start,
        )
        self.n_particle_mask = self.n_truth_particles < self.cfg.max_particles
        arrays = tree.arrays(
            [f"truth_{var}" for var in self.truth_vars_to_load] + ["eventNumber"],
            library="np",
            entry_stop=self.entry_stop,
            entry_start=self.entry_start,
        )
        for var in self.truth_vars_to_load:
            self.full_data_array[var] = arrays[f"truth_{var}"]
        self.eventNumber = arrays["eventNumber"]

        self.full_data_array["ht"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_x"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_y"] = np.zeros(self.cfg.num_events, dtype=np.float32)

        for j in range(self.cfg.num_events):
            self.full_data_array["ht"][j] = self.full_data_array["pt"][j].sum()
            self.full_data_array["met_x"][j] = (
                self.full_data_array["pt"][j] * np.cos(self.full_data_array["phi"][j])
            ).sum()
            self.full_data_array["met_y"][j] = (
                self.full_data_array["pt"][j] * np.sin(self.full_data_array["phi"][j])
            ).sum()

        self.full_data_array["ptrel"] = np.array(
            [x / x.sum() for x in self.full_data_array["pt"]], dtype=object
        )
        _ = self.full_data_array.pop("pt")

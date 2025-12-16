from typing import Any, final, override

import numpy as np
import uproot

from parnassus.configs.data import DatasetConfig
from parnassus.utils.transform import VarTransform

from .base import BaseDataset


@final
class RootDataset(BaseDataset):
    """Dataset class for loading events from a ROOT file."""

    def __init__(self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform]):
        super().__init__(cfg=cfg, var_transform_dict=var_transform_dict)

    def _update_num_events(self, tree: Any):
        tree_num_entries = tree.num_entries
        if self.cfg.entry_start > tree_num_entries:
            raise ValueError(
                f"Requested entry_start exceeds number of events in the file {self.cfg.file_path}"
            )
        if self.cfg.num_events > (tree_num_entries - self.cfg.entry_start):
            raise ValueError(
                f"""Requested num_events ({self.cfg.num_events}) exceeds number of events
                    in the file {self.cfg.file_path}"""
            )
        self.entry_start = self.cfg.entry_start
        self.entry_stop = self.cfg.entry_start + self.cfg.num_events

    def _load_truth(self, tree: Any):
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

    @override
    def load_data(self):
        with uproot.open(self.cfg.file_path) as f:  # pyright: ignore  # noqa: PGH003
            tree = f["evt_tree"]
            self._update_num_events(tree)
            self._load_truth(tree)
            self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])

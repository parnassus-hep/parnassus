from typing import final, override

import numpy as np
import pyhepmc

from parnassus.configs.data import DatasetConfig
from parnassus.utils import pid_to_class
from parnassus.utils.logger import ProgressBar
from parnassus.utils.transform import VarTransform

from .base import BaseDataset


@final
class HepMCDataset(BaseDataset):
    """Dataset wrapper for HepMC events.

    This class loads events from a HepMC file and prepares arrays for model
    training and evaluation.
    """

    def __init__(self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform]):
        super().__init__(cfg=cfg, var_transform_dict=var_transform_dict)

    @override
    def load_data(self):
        self.n_truth_particles = np.zeros(self.cfg.num_events, dtype=np.int32)
        for var in (*self.cfg.truth_vars_to_load, "ptrel"):
            self.full_data_array[var] = np.zeros(
                self.cfg.num_events * self.cfg.max_particles, dtype=np.float32
            )

        self.eventNumber = np.zeros(self.cfg.num_events, dtype=np.int64)
        self.full_data_array["ht"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_x"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_y"] = np.zeros(self.cfg.num_events, dtype=np.float32)

        curr_event_idx = 0
        curr_particle_idx = 0
        with pyhepmc.open(self.cfg.file_path, "r") as f:
            evt: pyhepmc.GenEvent
            with ProgressBar() as progress:
                task = progress.add_task(
                    "[green]Loading data from HepMC file", total=self.cfg.num_events
                )
                for evt in f:
                    if curr_event_idx == self.cfg.num_events:
                        break
                    event_start_particle_idx = curr_particle_idx
                    num_particles = 0
                    for vtx in evt.vertices:
                        for part in vtx.particles_out:
                            if (
                                part.status != 1
                                or np.abs(part.momentum.eta()) >= 2.7
                                or part.momentum.pt() <= 0.25
                                or abs(part.pid) in {12, 14, 16}
                            ):
                                continue
                            pid = pid_to_class(part.pid)
                            self.full_data_array["pt"][curr_particle_idx] = part.momentum.pt()
                            self.full_data_array["eta"][curr_particle_idx] = part.momentum.eta()
                            self.full_data_array["phi"][curr_particle_idx] = part.momentum.phi()
                            self.full_data_array["class"][curr_particle_idx] = float(pid)
                            self.full_data_array["vx"][curr_particle_idx] = vtx.position.x
                            self.full_data_array["vy"][curr_particle_idx] = vtx.position.y
                            self.full_data_array["vz"][curr_particle_idx] = vtx.position.z

                            num_particles += 1
                            curr_particle_idx += 1
                    if num_particles >= self.cfg.max_particles:
                        curr_particle_idx -= num_particles
                        continue

                    self.full_data_array["ht"][curr_event_idx] = self.full_data_array["pt"][
                        event_start_particle_idx:curr_particle_idx
                    ].sum()
                    self.full_data_array["ptrel"][event_start_particle_idx:curr_particle_idx] = (
                        self.full_data_array["pt"][event_start_particle_idx:curr_particle_idx]
                        / self.full_data_array["ht"][curr_event_idx]
                    )
                    self.full_data_array["met_x"][curr_event_idx] = (
                        self.full_data_array["pt"][event_start_particle_idx:curr_particle_idx]
                        * np.cos(
                            self.full_data_array["phi"][event_start_particle_idx:curr_particle_idx]
                        )
                    ).sum()
                    self.full_data_array["met_y"][curr_event_idx] = (
                        self.full_data_array["pt"][event_start_particle_idx:curr_particle_idx]
                        * np.sin(
                            self.full_data_array["phi"][event_start_particle_idx:curr_particle_idx]
                        )
                    ).sum()
                    self.n_truth_particles[curr_event_idx] = num_particles
                    self.eventNumber[curr_event_idx] = int(evt.event_number)
                    curr_event_idx += 1
                    progress.update(task, advance=1)
                if self.cfg.num_events > curr_event_idx:
                    print("Requested more events than in file")
        _ = self.full_data_array.pop("pt")
        for key in self.ctxt_vars:
            self.full_data_array[key] = self.full_data_array[key][:curr_particle_idx]
        for key in ["ht", "met_x", "met_y"]:
            self.full_data_array[key] = self.full_data_array[key][:curr_event_idx]
        self.eventNumber = self.eventNumber[:curr_event_idx]
        self.n_truth_particles = self.n_truth_particles[:curr_event_idx]
        self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])

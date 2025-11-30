from pathlib import Path
from typing import final, override

import numpy as np

from parnassus.configs.data import DatasetConfig
from parnassus.utils import pid_to_class
from parnassus.utils.logger import ProgressBar
from parnassus.utils.transform import VarTransform

from .base import BaseDataset


def _make_pythia(pythia_card: Path | str, seed: int | None = None):
    """
    Helper that returns an initialized Pythia instance.
    Replace the body with your own bindings if needed.

    Parameters
    ----------
    pythia_card : str
        Path to Pythia8 steering card.
    seed : int | None, optional
        Optional random seed to set. If None, uses default Pythia behavior.


    Returns
    -------
    pythia : pythia8mc.Pythia
        Initialized Pythia instance.
    """
    import pythia8mc  # noqa: PLC0415

    pythia = pythia8mc.Pythia()

    # Read steering card
    if pythia_card:
        if isinstance(pythia_card, Path):
            pythia_card = pythia_card.as_posix()
        pythia.readFile(pythia_card)

    # Optional manual seed
    if seed is not None:
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")

    pythia.init()
    return pythia


@final
class PythiaDataset(BaseDataset):
    """Dataset class for reading events generated with Pythia8.

    This class uses the pythia8mc Python bindings to generate events on-the-fly
    according to the provided steering card (.cmnd file). It applies selection cuts
    similar to those in HepMCDataset and fills per-particle and event-level variables.
    """

    def __init__(
        self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform] | None = None
    ):
        super().__init__(cfg=cfg, var_transform_dict=var_transform_dict)

    @override
    def load_data(self):
        # Same allocation pattern as HepMCDataset
        self.n_truth_particles = np.zeros(self.cfg.num_events, dtype=np.int32)

        # Per-particle variables (plus derived "ptrel")
        for var in (*self.cfg.truth_vars_to_load, "pt", "ptrel"):
            # Allocate a flat array of size (num_events * max_particles)
            # and we will later crop to curr_particle_idx
            self.full_data_array[var] = np.zeros(
                self.cfg.num_events * self.cfg.max_particles,
                dtype=np.float32,
            )

        # Event-level quantities
        self.eventNumber = np.zeros(self.cfg.num_events, dtype=np.int64)
        self.full_data_array["ht"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_x"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_y"] = np.zeros(self.cfg.num_events, dtype=np.float32)

        # Initialise Pythia
        pythia = _make_pythia(self.cfg.file_path)

        curr_event_idx = 0
        curr_particle_idx = 0

        with ProgressBar() as progress:
            task = progress.add_task(
                "[green]Generating events with Pythia8", total=self.cfg.num_events
            )

            # Loop until we have cfg.num_events *accepted* events
            while curr_event_idx < self.cfg.num_events:
                if not pythia.next():
                    # Failed event; skip and try again
                    continue

                event_start_particle_idx = curr_particle_idx
                num_particles = 0

                for part in pythia.event:
                    # Only final-state, visible particles -- mirror HepMCDataset cuts
                    # status == 1      : final state
                    # |eta| < 2.7      : central-ish detector coverage
                    # pt > 0.25 GeV    : minimal pt cut
                    # |pid| not in {12,14,16} : remove neutrinos
                    if not part.isFinal():
                        continue

                    pt = part.pT()
                    eta = part.eta()
                    phi = part.phi()
                    pid = part.id()

                    if abs(eta) >= 2.7 or pt <= 0.25 or abs(pid) in {12, 14, 16}:
                        continue

                    # Map PDG ID -> class index
                    cls = pid_to_class(pid)

                    # Fill per-particle arrays
                    self.full_data_array["pt"][curr_particle_idx] = pt
                    if "eta" in self.full_data_array:
                        self.full_data_array["eta"][curr_particle_idx] = eta
                    if "phi" in self.full_data_array:
                        self.full_data_array["phi"][curr_particle_idx] = phi
                    if "class" in self.full_data_array:
                        self.full_data_array["class"][curr_particle_idx] = float(cls)

                    # Vertex coordinates if requested in truth_vars_to_load
                    if "vx" in self.full_data_array:
                        self.full_data_array["vx"][curr_particle_idx] = part.xProd()
                    if "vy" in self.full_data_array:
                        self.full_data_array["vy"][curr_particle_idx] = part.yProd()
                    if "vz" in self.full_data_array:
                        self.full_data_array["vz"][curr_particle_idx] = part.zProd()

                    num_particles += 1
                    curr_particle_idx += 1

                # If this event has too many truth particles, drop it
                if num_particles >= self.cfg.max_particles:
                    curr_particle_idx -= num_particles
                    continue

                # Event-level quantities from the particles we just filled
                pt_slice = slice(event_start_particle_idx, curr_particle_idx)
                pt_arr = self.full_data_array["pt"][pt_slice]
                phi_arr = (
                    self.full_data_array["phi"][pt_slice] if "phi" in self.full_data_array else None
                )

                ht = pt_arr.sum()
                self.full_data_array["ht"][curr_event_idx] = ht

                if ht > 0.0:
                    self.full_data_array["ptrel"][pt_slice] = pt_arr / ht

                if phi_arr is not None:
                    self.full_data_array["met_x"][curr_event_idx] = (pt_arr * np.cos(phi_arr)).sum()
                    self.full_data_array["met_y"][curr_event_idx] = (pt_arr * np.sin(phi_arr)).sum()

                # Store number of truth particles for this event
                self.n_truth_particles[curr_event_idx] = num_particles

                # Use current event index as event number
                self.eventNumber[curr_event_idx] = int(curr_event_idx)

                curr_event_idx += 1
                progress.update(task, advance=1)

        if self.cfg.num_events > curr_event_idx:
            print("Requested more events than could be generated")

        # Drop temporary "pt" array
        _ = self.full_data_array.pop("pt")

        # Crop to actually used particles / events
        for key in self.ctxt_vars:
            # ctxt_vars should be a subset of the truth-level keys
            if key in self.full_data_array:
                self.full_data_array[key] = self.full_data_array[key][:curr_particle_idx]

        for key in ["ht", "met_x", "met_y"]:
            self.full_data_array[key] = self.full_data_array[key][:curr_event_idx]

        self.eventNumber = self.eventNumber[:curr_event_idx]
        self.n_truth_particles = self.n_truth_particles[:curr_event_idx]

        # Same convention as HepMCDataset
        self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])

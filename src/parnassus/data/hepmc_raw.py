"""Unified HepMC dataset in ColumnMap format.

Provides :class:`HepMCRawDataset`, a ``torch.utils.data.Dataset`` that loads
stable (status == 1) particles from a HepMC file and stores them as per-event
tensors using the :class:`~parnassus.data.particle_io.ColumnMap` feature layout.

No selection cuts, no transforms — raw physics variables only.  Lightweight
adapters in :mod:`parnassus.data.adapters` wrap this dataset for neural-network
training or parametric (torch_delphes) simulation.
"""

from pathlib import Path
from typing import Any

import pyhepmc
import torch
from torch.utils.data import Dataset

from parnassus.utils.logger import ProgressBar

from .particle_io import N_FEATURES, particles_to_tensor


class HepMCRawDataset(Dataset):
    """Dataset of stable HepMC particles in ColumnMap tensor format.

    Each item is a dictionary with the per-event particle tensor and metadata.
    Events with no stable particles are retained (empty tensor) so that event
    indices remain predictable.

    Parameters
    ----------
    file_path : Path | str
        Path to HepMC file (.hepmc / .hepmc3 / .hepmc.gz).
    num_events : int | None, optional
        Maximum number of events to load.  ``None`` loads all events.
    """

    def __init__(self, file_path: Path | str, num_events: int | None = None) -> None:
        self.file_path = Path(file_path)
        self.num_events = num_events
        if not self.file_path.exists():
            raise FileNotFoundError(f"HepMC file not found: {self.file_path}")
        self._event_tensors: list[torch.Tensor] = []
        self._event_numbers: list[int] = []
        self._load()

    def _load(self) -> None:
        with ProgressBar() as progress:
            task = progress.add_task("[green]Reading data from HepMC file", total=self.num_events)
            with pyhepmc.open(self.file_path) as f:
                for event_idx, event in enumerate(f):
                    if self.num_events is not None and event_idx >= self.num_events:
                        break
                    stable = [p for p in event.particles if p.status == 1]
                    tensor = particles_to_tensor(stable, event.event_number)
                    self._event_tensors.append(tensor)
                    self._event_numbers.append(int(event.event_number))
                    progress.update(task, advance=1)

    def __len__(self) -> int:
        return len(self._event_tensors)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        particles = self._event_tensors[idx]
        return {
            "particles": particles,  # shape (N_i, N_FEATURES), float32
            "event_number": self._event_numbers[idx],  # int
            "n_particles": particles.shape[0],  # int
        }

    @property
    def n_features(self) -> int:
        """Number of features per particle (= ``N_FEATURES``)."""
        return N_FEATURES

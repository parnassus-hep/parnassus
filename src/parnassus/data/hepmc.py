"""HepMC dataset in ColumnMap format.

Provides :class:`HepMCDataset`, a ``torch.utils.data.Dataset`` that loads
all particles from a HepMC file and stores them as per-event tensors using
the :class:`~parnassus.data.particle_io.ColumnMap` feature layout.

No selection cuts, no transforms — raw physics variables only.  Lightweight
adapters in :mod:`parnassus.data.adapters` wrap this dataset for neural-network
generation or parametric (torch_delphes) simulation.
"""

from typing import override

import pyhepmc

from parnassus.utils.logger import ProgressBar

from .base import RawDataset
from .particle_io import hepmc_particles_to_tensor


class HepMCDataset(RawDataset):
    """Dataset of HepMC particles in ColumnMap tensor format.

    Each item is a dictionary with the per-event particle tensor and metadata.
    Events with no particles are retained (empty tensor) so that event
    indices remain predictable.

    Parameters
    ----------
    file_path : Path | str
        Path to HepMC file (.hepmc / .hepmc3 / .hepmc.gz).
    num_events : int | None, optional
        Maximum number of events to load.  ``None`` loads all events.
    """

    @override
    def _load(self) -> None:
        with ProgressBar() as progress:
            task = progress.add_task("[green]Reading data from HepMC file", total=self.num_events)
            with pyhepmc.open(self.file_path) as f:
                for event_idx, event in enumerate(f):
                    if self.num_events is not None and event_idx >= self.num_events:
                        break
                    tensor = hepmc_particles_to_tensor(event.particles, event.event_number)
                    self._event_tensors.append(tensor)
                    self._event_numbers.append(int(event.event_number))
                    progress.update(task, advance=1)

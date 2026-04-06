"""Pythia8 dataset in ColumnMap format.

Provides :class:`PythiaDataset`, a :class:`~parnassus.data.base.RawDataset`
subclass that generates events on-the-fly using Pythia8 and stores them as
per-event ColumnMap tensors — the same format as :class:`HepMCDataset`.

No selection cuts, no transforms.  All particles (including intermediates) are
stored with their status codes, mirroring a Pythia→HepMC write.  Adapters in
:mod:`parnassus.data.adapters` apply selection cuts for specific use-cases.
"""

from pathlib import Path
from typing import final, override

from parnassus.utils.logger import ProgressBar

from .base import RawDataset
from .particle_io import pythia_particles_to_tensor


def _make_pythia(pythia_card: Path | str, seed: int | None = None):
    """Return an initialized Pythia instance from a steering card.

    Parameters
    ----------
    pythia_card : Path | str
        Path to Pythia8 steering card (.cmnd file).
    seed : int | None, optional
        Random seed.  ``None`` uses default Pythia behaviour.

    Returns
    -------
    pythia8mc.Pythia
    """
    import pythia8mc  # noqa: PLC0415

    pythia = pythia8mc.Pythia()
    if pythia_card:
        if isinstance(pythia_card, Path):
            pythia_card = pythia_card.as_posix()
        pythia.readFile(pythia_card)
    if seed is not None:
        pythia.readString("Random:setSeed = on")
        pythia.readString(f"Random:seed = {seed}")
    pythia.init()
    return pythia


@final
class PythiaDataset(RawDataset):
    """Dataset of Pythia8-generated events in ColumnMap tensor format.

    Generates exactly ``num_events`` accepted events and stores all particles
    (including intermediates) per event, matching what a Pythia→HepMC write
    would produce.

    Parameters
    ----------
    file_path : Path | str
        Path to Pythia8 steering card (.cmnd file).
    num_events : int | None
        Number of events to generate.  ``None`` is not supported and will raise.
    seed : int | None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        file_path: Path | str,
        num_events: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._seed = seed
        super().__init__(file_path, num_events)

    @override
    def _load(self) -> None:
        if self.num_events is None:
            raise ValueError("PythiaDataset requires num_events to be set.")

        pythia = _make_pythia(self.file_path, self._seed)
        event_idx = 0

        with ProgressBar() as progress:
            task = progress.add_task("[green]Generating events with Pythia8", total=self.num_events)
            while event_idx < self.num_events:
                if not pythia.next():
                    continue
                tensor = pythia_particles_to_tensor(pythia.event, event_idx)
                self._event_tensors.append(tensor)
                self._event_numbers.append(event_idx)
                event_idx += 1
                progress.update(task, advance=1)

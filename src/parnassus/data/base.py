"""Abstract base dataset for raw per-event ColumnMap tensor datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset

from .particle_io import N_FEATURES


class RawDataset(Dataset[dict[str, Any]], ABC):
    """Abstract base for raw per-event ColumnMap datasets.

    Subclasses implement :meth:`_load` to populate ``_event_tensors`` and
    ``_event_numbers``.  No selection cuts or variable transforms are applied —
    the raw physics variables are stored as-is.

    Parameters
    ----------
    file_path : Path | str
        Path to the input file.
    num_events : int | None
        Maximum number of events to load.  ``None`` loads all events.
    """

    def __init__(self, file_path: Path | str, num_events: int | None = None) -> None:
        super().__init__()
        self.file_path = Path(file_path)
        self.num_events = num_events
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self._event_tensors: list[Tensor] = []
        self._event_numbers: list[int] = []
        self._load()

    @abstractmethod
    def _load(self) -> None:
        """Load events into ``_event_tensors`` and ``_event_numbers``."""

    def __len__(self) -> int:
        return len(self._event_tensors)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        particles = self._event_tensors[idx]
        return {
            "particles": particles,  # Tensor(N_i, N_FEATURES)
            "event_number": self._event_numbers[idx],
            "n_particles": particles.shape[0],
        }

    @property
    def n_features(self) -> int:
        """Number of features per particle (= ``N_FEATURES``)."""
        return N_FEATURES


__all__ = ["RawDataset"]

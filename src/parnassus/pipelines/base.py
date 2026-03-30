from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

from parnassus.configs.accessors import Accessor
from parnassus.configs.scheme import GenEvent

if TYPE_CHECKING:
    from parnassus.utils.typing import TensorDict


class EventGenerator(Protocol):
    """Protocol for event generation backends (NN, parametric, Pythia, etc.)."""

    def get_accessors(self) -> dict[str, list[Accessor]]:
        """Return dictionary of accessors for generated output."""
        ...

    def initialize(self, n_events: int, n_batches: int) -> None:
        """Set up internal storage and progress tracking before the batch loop.

        Parameters
        ----------
        n_events : int
            Total number of events to generate (for buffer pre-allocation).
        n_batches : int
            Total number of batches (for progress tracking).
        """
        ...

    def process_batch(self, batch: "TensorDict") -> None:
        """Process one batch, accumulate results internally, and update progress.

        Parameters
        ----------
        batch : TensorDict
            Input batch from the dataloader.
        """
        ...

    def get_events(self) -> list[GenEvent]:
        """Finalise generation and return the list of generated events.

        Closes any open progress contexts and converts accumulated data to
        GenEvent objects.

        Returns
        -------
        list[GenEvent]
            Generated events.
        """
        ...


class GenPipeline(ABC):
    """Abstract base class for general pipelines."""

    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass

    @abstractmethod
    def get_accessors(self) -> dict[str, list[Accessor]]:
        pass

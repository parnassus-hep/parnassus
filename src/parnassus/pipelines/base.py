from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

import numpy as np

from parnassus.configs.accessors import Accessor
from parnassus.configs.scheme import GenEvent

if TYPE_CHECKING:
    from parnassus.utils.typing import TensorDict


class EventGenerator(Protocol):
    """Protocol for event generation backends (NN, parametric, Pythia, etc.)."""

    @property
    def has_impact_model(self) -> bool:
        """Whether this generator produces impact parameters."""
        ...

    @property
    def max_particles(self) -> int:
        """Maximum number of particles per event."""
        ...

    @property
    def truth_output_vars(self) -> list[str]:
        """List of truth-level output variable names."""
        ...

    @property
    def pflow_output_vars(self) -> list[str]:
        """List of particle-flow output variable names."""
        ...

    @property
    def event_sampler_steps(self) -> int | None:
        """Number of sampling steps for event generation (None for parametric)."""
        ...

    @property
    def particle_sampler_steps(self) -> int | None:
        """Number of sampling steps for particle generation (None for parametric)."""
        ...

    @property
    def impact_sampler_steps(self) -> int | None:
        """Number of sampling steps for impact generation (None if no impact model)."""
        ...

    def get_accessors(self) -> list:
        """Return list of partial accessor constructors for this generator's output."""
        ...

    def generate_batch(
        self,
        data_dict: "TensorDict",
        event_callback=None,
        particle_callback=None,
        impact_callback=None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Generate truth and pflow particles for a batch.

        Parameters
        ----------
        data_dict : TensorDict
            Input batch containing ctxt_data, ctxt_global_data, mask, event_number.
        event_callback : Callable | None
            Optional callback for event-level generation progress.
        particle_callback : Callable | None
            Optional callback for particle-level generation progress.
        impact_callback : Callable | None
            Optional callback for impact parameter generation progress.

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]
            (truth_data_dict, pflow_data_dict, metadata_dict) where metadata contains
            event_number, fs_mask, tr_mask, bad_idxs.
        """
        ...


class GenPipeline(ABC):
    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass

    @abstractmethod
    def get_accessors(self) -> dict[str, list[Accessor]]:
        pass


class SourcePipeline(ABC):
    @abstractmethod
    def run(self) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
        """Generate events from an external source."""

    @abstractmethod
    def get_accessors(self) -> dict[str, list[Accessor]]:
        """Accessors exposed by this pipeline (after run)."""

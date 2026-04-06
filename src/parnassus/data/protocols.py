"""Dataset protocols for parnassus data pipelines.

Two runtime-checkable protocols define the two dataset contracts used
throughout the parnassus pipeline:

* :class:`NeuralDataset` — padded/masked format for neural generation
* :class:`ParametricDataset` — raw ColumnMap format for parametric simulation
"""

from typing import Any, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class NeuralDataset(Protocol):
    """Protocol for neural-generation datasets.

    Yields padded, masked batches ready for the neural generation pipeline.

    ``__getitem__`` must return a dict with keys:
    ``ctxt_data``, ``ctxt_global_data``, ``mask``, ``event_number``.

    Satisfied by :class:`~parnassus.data.adapters.NeuralAdapter`
    and :class:`~parnassus.data.root.RootDataset`.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: Any) -> dict[str, Tensor]: ...


@runtime_checkable
class ParametricDataset(Protocol):
    """Protocol for parametric-simulation datasets.

    Yields raw ColumnMap-format particle dicts for torch_delphes modules.
    Use :func:`~parnassus.data.adapters.parametric_collate_fn` when building
    a ``DataLoader`` over this dataset type.

    ``__getitem__`` must return a dict with keys:
    ``particles`` (``Tensor[N, N_FEATURES]``), ``event_number``, ``n_particles``.

    Satisfied by :class:`~parnassus.data.adapters.ParametricAdapter`,
    :class:`~parnassus.data.hepmc.HepMCDataset`,
    and :class:`~parnassus.data.pythia.PythiaDataset`.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: Any) -> dict[str, Any]: ...

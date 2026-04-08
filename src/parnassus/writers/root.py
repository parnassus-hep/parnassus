from typing import TYPE_CHECKING, Any, final, override

import awkward as ak
import numpy as np
from awkward.contents import ListOffsetArray, NumpyArray, RecordArray
from awkward.index import Index64
from uproot import WritableTree, recreate

from parnassus.configs.scheme import GenEvent
from parnassus.utils.logger import ProgressBar

from .base import BaseWriter

if TYPE_CHECKING:
    from uproot.writing.writable import WritableDirectory


BATCH_SIZE = 1000


def _make_collection_array(var_data: dict[str, list[Any]]) -> ak.Array:
    """Build a variable-length record array, computing offsets once across all variables.

    Parameters
    ----------
    var_data : dict[str, list[Any]]
        Dictionary of variable data, where each value is a list of arrays (one per event).
        All lists must have the same length (number of events).

    Returns
    -------
    ak.Array
        An Awkward Array with a ListOffsetArray structure, where each record contains the variables.
    """
    if not var_data:
        return ak.Array([])
    first = next(iter(var_data.values()))
    counts = np.fromiter((len(arr) for arr in first), dtype=np.int64, count=len(first))
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    index = Index64(offsets)
    fields = list(var_data.keys())
    inner = RecordArray(
        [NumpyArray(np.concatenate(var_data[f])) for f in fields],  # type: ignore[arg-type]
        fields,
    )
    return ak.Array(ListOffsetArray(index, inner))


def clear_dicts(data: dict[Any, Any]):
    """Recursively clear lists in a nested dictionary."""
    for value in data.values():
        if isinstance(value, dict):
            clear_dicts(value)
        elif isinstance(value, list):
            value.clear()


def custom_field_name(outer: str, inner: str) -> str:
    """Custom field name for the ROOT file.

    Returns
    -------
    str
        The field name to use in the ROOT file.
    """
    return inner if not outer else outer + "." + inner


@final
class RootWriter(BaseWriter):
    """Writer class for outputting generated events to a ROOT file."""

    def write_to_tree(self, tree: WritableTree, data: dict[str, dict[str, Any]]):
        tree.extend({collection: _make_collection_array(data[collection]) for collection in data})
        clear_dicts(data)

    @override
    def write(self, events: list[GenEvent]):
        f: WritableDirectory
        accessor_store = self.config.accessor_store
        with recreate(self.config.file_path) as f:
            f.mktree(
                "Parnassus",
                branch_types=accessor_store.get_branch_types(),
                field_name=custom_field_name,
            )

            data = accessor_store.init_data_dict()
            events_in_queue = 0
            tree: WritableTree = f["Parnassus"]  # type: ignore[assignment]
            with ProgressBar() as progress:
                task = progress.add_task("[green]Writing data to file", total=len(events))
                for event in events:
                    accessor_store.update_data_dict(event, data)
                    events_in_queue += 1
                    if events_in_queue == BATCH_SIZE:
                        self.write_to_tree(tree, data)
                        progress.update(task, advance=BATCH_SIZE)
                        events_in_queue = 0
                if events_in_queue != 0:
                    self.write_to_tree(tree, data)
                    progress.update(task, advance=events_in_queue)

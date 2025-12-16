from typing import TYPE_CHECKING, Any, final, override

import awkward as ak
from uproot import WritableTree, recreate

from parnassus.configs.scheme import GenEvent
from parnassus.utils.logger import ProgressBar

from .base import BaseWriter

if TYPE_CHECKING:
    from uproot.writing.writable import WritableDirectory


BATCH_SIZE = 100


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
        extend_data = {
            collection: ak.zip({
                var_name: ak.Array(data[collection][var_name]) for var_name in data[collection]
            })
            for collection in data
        }
        tree.extend(extend_data)
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
            with ProgressBar() as progress:
                task = progress.add_task("[green]Writing data to file", total=len(events))
                for event in events:
                    accessor_store.update_data_dict(event, data)
                    events_in_queue += 1
                    if events_in_queue == BATCH_SIZE:
                        self.write_to_tree(f["Parnassus"], data)
                        progress.update(task, advance=BATCH_SIZE)
                        events_in_queue = 0
                if events_in_queue != 0:
                    self.write_to_tree(f["Parnassus"], data)
                    progress.update(task, advance=events_in_queue)

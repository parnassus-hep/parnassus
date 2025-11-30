from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .hepmc import HepMCDataset
from .root import RootDataset

if TYPE_CHECKING:
    from parnassus.configs.data import DatasetConfig
    from parnassus.data.base import BaseDataset

__all__ = ["HepMCDataset", "RootDataset", "build_dataset"]


def build_dataset(
    dataset_config: "DatasetConfig",
    transform_registry=None,
    dataset_builders: Mapping[str, Callable[..., "BaseDataset"]] | None = None,
) -> "BaseDataset":
    """Factory function to create dataset instances based on file type.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration containing file path and other settings.
    transform_registry : TransformRegistry | None
        Optional registry of variable transformations.
    dataset_builders : Mapping[str, Callable] | None
        Optional custom builders for file types.
        Defaults to {".root": RootDataset, ".hepmc": HepMCDataset}.

    Returns
    -------
    BaseDataset
        Dataset instance appropriate for the input file type.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file type is not supported.
    """
    input_file = dataset_config.file_path
    assert isinstance(input_file, Path)
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Trying to load file {input_file}, no file exists!")

    var_transform_dict = transform_registry.to_var_transform_dict() if transform_registry else {}
    builders = dataset_builders or {".root": RootDataset, ".hepmc": HepMCDataset}
    suffix = input_file.suffix.lower()
    if suffix not in builders:
        raise ValueError(
            f"Only ROOT or HepMC files are supported as input, got {dataset_config.file_path}"
        )
    dataset_factory = builders[suffix]
    return dataset_factory(dataset_config, var_transform_dict=var_transform_dict)

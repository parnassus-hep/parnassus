"""Data module providing dataset classes and factory function."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from parnassus.utils.transform import TransformRegistry

from .adapters import NeuralAdapter, ParametricAdapter, parametric_collate_fn
from .base import RawDataset
from .hepmc import HepMCDataset
from .protocols import NeuralDataset, ParametricDataset
from .pythia import PythiaDataset
from .root import RootDataset

if TYPE_CHECKING:
    from parnassus.configs.data import DatasetConfig

__all__ = [
    "HepMCDataset",
    "NeuralAdapter",
    "NeuralDataset",
    "ParametricAdapter",
    "ParametricDataset",
    "PythiaDataset",
    "RawDataset",
    "RootDataset",
    "build_dataset",
    "parametric_collate_fn",
]

_RAW_BUILDERS: dict[str, Callable[["DatasetConfig"], RawDataset]] = {
    ".hepmc": lambda cfg: HepMCDataset(cfg.file_path, num_events=cfg.num_events),
    ".cmnd": lambda cfg: PythiaDataset(cfg.file_path, num_events=cfg.num_events),
}


def build_dataset(
    dataset_config: "DatasetConfig",
    transform_registry: TransformRegistry | None = None,
    mode: Literal["neural", "parametric", "raw"] = "neural",
) -> "NeuralDataset | ParametricDataset":
    """Factory function to create dataset instances based on file type.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration containing file path and other settings.
    transform_registry : TransformRegistry | None
        Optional registry of variable transformations (used in ``"neural"`` mode).
    mode : {"neural", "parametric", "raw"}
        Dataset mode:

        * ``"neural"`` *(default)* — returns a :class:`NeuralAdapter`
          (HepMC / Pythia) or :class:`RootDataset` (ROOT, legacy).
        * ``"raw"`` — returns :class:`HepMCDataset` or :class:`PythiaDataset`.
        * ``"parametric"`` — returns :class:`ParametricAdapter`.

    Returns
    -------
    NeuralDataset | ParametricDataset
        Dataset instance appropriate for the requested mode and file type.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file type is not supported for the requested mode.
    """
    input_file = dataset_config.file_path
    assert isinstance(input_file, Path)
    if not input_file.exists():
        raise FileNotFoundError(f"Trying to load file {input_file}, no file exists!")

    suffix = input_file.suffix.lower()

    # Legacy ROOT path: RootDataset handles its own flat-array loading
    if suffix == ".root":
        if mode != "neural":
            raise ValueError(f"ROOT files only support mode='neural', got mode='{mode}'")
        transforms = transform_registry.to_var_transform_dict() if transform_registry else {}
        return RootDataset(dataset_config, var_transform_dict=transforms)

    if suffix not in _RAW_BUILDERS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {list(_RAW_BUILDERS)} and '.root'."
        )

    raw = _RAW_BUILDERS[suffix](dataset_config)

    if mode == "raw":
        return raw
    if mode == "parametric":
        return ParametricAdapter(raw)
    transforms = transform_registry.to_var_transform_dict() if transform_registry else {}
    return NeuralAdapter(raw, dataset_config, transforms)

"""Data module providing dataset classes and factory function."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from parnassus.utils.transform import TransformRegistry

from .adapters import NeuralAdapter, ParametricAdapter, parametric_collate_fn
from .hepmc import HepMCDataset
from .hepmc_raw import HepMCRawDataset
from .pythia import PythiaDataset
from .root import RootDataset


def _build_hepmc_neural(
    cfg: "DatasetConfig",
    var_transform_dict: dict | None = None,
) -> NeuralAdapter:
    raw = HepMCRawDataset(cfg.file_path, num_events=cfg.num_events)
    return NeuralAdapter(raw, cfg, var_transform_dict)


if TYPE_CHECKING:
    from parnassus.configs.data import DatasetConfig
    from parnassus.data.base import BaseDataset

__all__ = [
    "HepMCDataset",
    "HepMCRawDataset",
    "NeuralAdapter",
    "ParametricAdapter",
    "PythiaDataset",
    "RootDataset",
    "build_dataset",
    "parametric_collate_fn",
]


def build_dataset(
    dataset_config: "DatasetConfig",
    transform_registry: TransformRegistry | None = None,
    dataset_builders: Mapping[str, Callable[..., "BaseDataset"]] | None = None,
    mode: Literal["neural", "parametric", "raw"] = "neural",
) -> "BaseDataset | HepMCRawDataset | ParametricAdapter":
    """Factory function to create dataset instances based on file type.

    Parameters
    ----------
    dataset_config : DatasetConfig
        Dataset configuration containing file path and other settings.
    transform_registry : TransformRegistry | None
        Optional registry of variable transformations (used in ``"neural"`` mode).
    dataset_builders : Mapping[str, Callable] | None
        Optional custom builders for file types (``"neural"`` mode only).
        Defaults to ``{".root": RootDataset, ".hepmc": _build_hepmc_neural,
        ".cmnd": PythiaDataset}``.
    mode : {"neural", "parametric", "raw"}
        Dataset mode:

        * ``"neural"`` *(default)* — returns a :class:`NeuralAdapter`
          (HepMC), :class:`RootDataset`, or :class:`PythiaDataset`
          subclass of :class:`BaseDataset`.
        * ``"raw"`` — returns :class:`HepMCRawDataset` (HepMC files only).
        * ``"parametric"`` — returns :class:`ParametricAdapter` wrapping a
          :class:`HepMCRawDataset` (HepMC files only).

    Returns
    -------
    BaseDataset | HepMCRawDataset | ParametricAdapter
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
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Trying to load file {input_file}, no file exists!")

    if mode in {"raw", "parametric"}:
        suffix = input_file.suffix.lower()
        if suffix != ".hepmc":
            raise ValueError(
                f"mode='{mode}' is only supported for HepMC files (.hepmc), got '{suffix}'"
            )
        raw = HepMCRawDataset(input_file, num_events=dataset_config.num_events)
        if mode == "raw":
            return raw
        return ParametricAdapter(raw)

    # mode == "neural" — original behaviour
    var_transform_dict = transform_registry.to_var_transform_dict() if transform_registry else {}
    builders = dataset_builders or {
        ".root": RootDataset,
        ".hepmc": _build_hepmc_neural,
        ".cmnd": PythiaDataset,
    }
    suffix = input_file.suffix.lower()
    if suffix not in builders:
        raise ValueError(
            f"Only ROOT or HepMC files are supported as input, got {dataset_config.file_path}"
        )
    dataset_factory = builders[suffix]
    return dataset_factory(dataset_config, var_transform_dict=var_transform_dict)

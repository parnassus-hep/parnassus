from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.utils.transform import VarTransform
from parnassus.utils.typing import FloatArray, VarNameTuple

from .transforms import (
    SCALAR_KEYS,
    do_padding,  # re-exported for backward compatibility
    get_event_data,
    prepare_ctxt_global_data,
    preprocess_flat_arrays,
)

if TYPE_CHECKING:
    from parnassus.utils.typing import BoolArray, IntArray, LongArray

__all__ = ["BaseDataset", "do_padding"]


class BaseDataset(Dataset[dict[str, Tensor]]):
    """Base dataset class for loading event data."""

    def __init__(
        self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform] | None = None
    ):
        super().__init__()
        self.cfg: DatasetConfig = cfg

        self.var_transform_dict: dict[str, VarTransform] = var_transform_dict or {}

        self.truth_vars_to_load: VarNameTuple = cfg.truth_vars_to_load
        self.ctxt_vars: list[str] = cfg.variable_requirements.ctxt_vars_stripped
        self.ctxt_global_vars: list[str] = cfg.variable_requirements.ctxt_global_vars_stripped

        self.full_data_array: dict[str, FloatArray] = {}

        self.n_particle_mask: BoolArray
        self.n_truth_particles: IntArray
        self.truth_cumsum: LongArray
        self.eventNumber: LongArray

        if not Path(self.cfg.file_path).exists():
            raise FileNotFoundError(f"Trying to load file {self.cfg.file_path}, no file exist!")
        self._load_data()
        self._validate_required_attributes()

        preprocess_flat_arrays(
            self.full_data_array,
            SCALAR_KEYS,
            getattr(self, "n_particle_mask", None),
        )

        self.n_events = len(self.n_truth_particles)
        self.scaled_ctxt_global_data: Tensor = prepare_ctxt_global_data(
            self.full_data_array,
            self.n_truth_particles,
            self.truth_cumsum,
            self.ctxt_vars,
            self.ctxt_global_vars,
            self.var_transform_dict,
        )

    def _validate_required_attributes(self) -> None:
        """Validate that all required attributes are set by load_data().

        Raises
        ------
        AttributeError
            If any required attribute is not set or is None.
        """
        required_attrs = {
            "n_truth_particles": "IntArray",
            "truth_cumsum": "LongArray",
            "eventNumber": "IntArray",
        }

        for attr_name, expected_type in required_attrs.items():
            if not hasattr(self, attr_name):
                raise AttributeError(
                    f"'{attr_name}' not set in load_data(). Expected type: {expected_type}"
                )
            attr_value = getattr(self, attr_name)
            if attr_value is None:
                raise AttributeError(
                    f"'{attr_name}' is None after load_data(). Expected type: {expected_type}"
                )

    def __len__(self):
        return len(self.n_truth_particles)

    def __getitem__(self, idx: Any) -> dict[str, Tensor]:  # pyright: ignore[reportImplicitOverride]
        ctxt_data, ctxt_global_data, mask = get_event_data(
            idx,
            self.full_data_array,
            self.n_truth_particles,
            self.truth_cumsum,
            self.ctxt_vars,
            self.scaled_ctxt_global_data,
            self.cfg.max_particles,
            self.var_transform_dict,
        )
        event_number = np.asarray(self.eventNumber[idx])
        return {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
            "mask": mask,
            "event_number": torch.tensor(event_number, dtype=torch.long).unsqueeze(-1),
        }

    @abstractmethod
    def _load_data(self):
        pass

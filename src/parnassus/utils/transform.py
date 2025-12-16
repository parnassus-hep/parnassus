from dataclasses import dataclass, field
from pathlib import Path
from typing import final

import torch
import yaml
from torch import Tensor

from parnassus.utils.typing import FloatArray, VarNameTuple

TRANSFORM_TYPES = {
    "std",
    "min_max",
    "min_max_sym",
}

TRANSFORM_FUNCTIONS = {
    "log",
    "log1p",
    "sqrt",
    "tanh",
    "asinh",
    "atan",
    "pow",
    None,  # No function
}


@dataclass(kw_only=True)
class VarTransformConfig:
    """Configuration for variable transformation.

    Parameters
    ----------
    name : str
        Name of the variable.
    transform_type : str, optional
        Type of transformation ('std', 'min_max', 'min_max_sym'), by default 'std'.
    transform_fn : str | None, optional
        Function to apply ('log', 'log1p', 'sqrt', 'tanh', 'asinh', 'atan', 'pow'), by default None.
    mean : float | None, optional
        Mean value for 'std' transformation, by default None.
    std : float | None, optional
        Standard deviation for 'std' transformation, by default None.
    min : float | None, optional
        Minimum value for 'min_max' and 'min_max_sym' transformations, by default None.
    max : float | None, optional
        Maximum value for 'min_max' and 'min_max_sym' transformations, by default None
    """

    name: str
    transform_type: str = "std"
    transform_fn: str | None = None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None

    shift: float = field(init=False)
    scale: float = field(init=False)

    fn: str | None = None
    fn_scale: float = 1.0
    fn_shift: float = 0.0
    power: float | int = 1.0

    def __post_init__(self):
        if self.transform_fn not in TRANSFORM_FUNCTIONS:
            raise ValueError(
                f"Expected transform_fn for var {self.name} "
                f"be in {TRANSFORM_FUNCTIONS}, got {self.transform_fn}"
            )
        if self.transform_type not in TRANSFORM_TYPES:
            raise ValueError(
                f"Expected transform_type for var {self.name} "
                f"be in {TRANSFORM_TYPES}, got {self.transform_type}"
            )
        if self.transform_type == "std":
            if self.mean is None or self.std is None:
                raise ValueError(
                    f"For var {self.name} and 'std' transform_type mean and std values "
                    f"should be provided, got mean={self.mean}, std={self.std}"
                )
            self.shift = self.mean
            self.scale = self.std
        if self.transform_type in {"min_max", "min_max_sym"}:
            if self.min is None or self.max is None:
                raise ValueError(
                    f"For var {self.name} and '{self.transform_type}' transform_type "
                    f"min and max values should be provided, got min={self.min}, max={self.max}"
                )
            if self.transform_type == "min_max":
                self.shift = self.min
                self.scale = self.max - self.min
            else:
                self.shift = (self.max + self.min) / 2
                self.scale = (self.max - self.min) / 2


@final
class VarTransform:
    """Class to handle variable transformations for scaling and unscaling."""

    def __init__(self, cfg: VarTransformConfig):
        self.config = cfg
        self.name = cfg.name

        self.shift = cfg.shift
        self.scale = cfg.scale

        self.fn = cfg.fn
        self.fn_scale = cfg.fn_scale
        self.fn_shift = cfg.fn_shift
        self.power = cfg.power

    def transform(
        self, x: Tensor, shift: Tensor | float | None = None, scale: Tensor | float | None = None
    ) -> Tensor:
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = (x + self.fn_shift) / self.fn_scale
        match self.fn:
            case "log" | "log1p" | "sqrt" | "tanh" | "asinh" | "atan":
                x = getattr(torch, self.fn)(x)
            case "pow":
                x = torch.sign(x) * torch.pow(torch.abs(x), self.power)
            case _:
                pass
        return (x - shift) / scale

    def inverse_transform(
        self, x: Tensor, shift: Tensor | float | None = None, scale: Tensor | float | None = None
    ) -> Tensor:
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
        match self.fn:
            case "log":
                x = torch.exp(x)
            case "log1p":
                x = torch.expm1(x)
            case "sqrt":
                x = torch.pow(x, 2)
            case "tanh":
                x = torch.atanh(x)
            case "asinh":
                x = torch.sinh(x)
            case "atan":
                x = torch.tan(x)
            case "pow":
                x = torch.sign(x) * torch.pow(torch.abs(x), 1 / self.power)
            case _:
                # For other functions, we assume they are linear transformations
                # and do not need to be inverted.
                pass
        return x * self.fn_scale - self.fn_shift


@final
class Unscaler:
    """Class to unscale model outputs back to original variable space."""

    def __init__(
        self,
        transform_dict: dict[str, VarTransform],
        fs_vars: VarNameTuple,
        ctxt_vars: VarNameTuple,
        ctxt_global_vars: VarNameTuple,
    ):
        """Initialize the Unscaler with a FeatureScaler and lists of feature names.

        Parameters
        ----------
        transform_dict : dict[str, VarTransform]
            The dict of VarTransform instances used for scaling transformations.
        fs_vars : VarNameTuple
            Tuple of fast simulation feature names.
        ctxt_vars : VarNameTuple
            Tuple of context feature names.
        ctxt_global_vars : VarNameTuple
            Tuple of global context feature names.
        """
        self.transform_dict = transform_dict
        self.fs_vars = fs_vars
        self.ctxt_vars = ctxt_vars
        self.ctxt_global_vars = ctxt_global_vars

    def extract_var(
        self,
        data: Tensor,
        var_name: str,
        var_idx: int,
    ) -> tuple[Tensor, int]:
        index_shift = 0
        if var_name.endswith("phi"):
            var = torch.atan2(
                data[..., var_idx],
                data[..., var_idx + 1],
            )
            index_shift = 2
        elif var_name.endswith("class"):
            var = data[..., var_idx : var_idx + 5].argmax(-1)
            index_shift = 5
        else:
            var = self.transform_dict[var_name].inverse_transform(data[..., var_idx])
            index_shift = 1
        return var, index_shift

    def unscale_impact_variables(self, fs_data: Tensor) -> dict[str, FloatArray]:
        fs_data = fs_data.cpu()
        pf_data_dict: dict[str, FloatArray] = {}
        var_idx = 0
        for var_name in ["d0", "d0Error", "z0", "z0Error"]:
            fs_, shift = self.extract_var(fs_data, var_name, var_idx)
            var_idx += shift
            pf_data_dict[var_name] = fs_.numpy()
        return pf_data_dict

    def unscale_variables(
        self,
        fs_data: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
    ) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
        fs_data = fs_data.cpu()
        ctxt_data = ctxt_data.cpu()
        ctxt_global_data = ctxt_global_data.cpu()
        tr_data_dict: dict[str, FloatArray] = {}
        pf_data_dict: dict[str, FloatArray] = {}
        var_idx = 0
        tr_ht = self.transform_dict["ht"].inverse_transform(
            ctxt_global_data[..., self.ctxt_global_vars.index("truth_ht")]
        )
        pf_ht = self.transform_dict["ht"].inverse_transform(
            ctxt_global_data[..., self.ctxt_global_vars.index("pflow_ht")]
        )
        var_idx = 0
        for var_name in self.ctxt_vars:
            if var_name.startswith("pflow_"):
                continue
            var_name_ = var_name.replace("truth_", "")
            var, index_shift = self.extract_var(ctxt_data, var_name_, var_idx)
            var_idx += index_shift
            if var_name_ == "ptrel":
                var = var * tr_ht.reshape(-1, 1)
                var_name_ = "pt"
            tr_data_dict[var_name_] = var.numpy()
        var_idx = 0
        for var_name in self.fs_vars:
            var_name_ = var_name.replace("pflow_", "").replace("npflow", "npart")
            fs_, index_shift = self.extract_var(fs_data, var_name_, var_idx)
            var_idx += index_shift
            if var_name_ == "ptrel":
                fs_ = fs_ * pf_ht.reshape(-1, 1)
                var_name_ = "pt"
            pf_data_dict[var_name_] = fs_.numpy()
        return tr_data_dict, pf_data_dict


@final
@dataclass(frozen=True)
class TransformRegistry:
    """Manages variable transformations loaded from configuration.

    This class provides a centralized registry for variable transformations,
    decoupling transform loading from model configuration.
    """

    transforms: dict[str, VarTransformConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "TransformRegistry":
        """Load transforms from a YAML configuration file.

        Parameters
        ----------
        path : Path
            Path to the YAML file containing transform configurations.

        Returns
        -------
        TransformRegistry
            A new registry with loaded transformations.
        """
        with open(path) as f:
            config = yaml.safe_load(f)
            return cls(
                transforms={
                    key: VarTransformConfig(name=key, **value) for key, value in config.items()
                }
            )

    def get_transform(self, var_name: str) -> VarTransformConfig:
        """Get the transform configuration for a variable.

        Parameters
        ----------
        var_name : str
            Name of the variable to get transform for.

        Returns
        -------
        VarTransformConfig
            The transform configuration for the variable.
        """
        return self.transforms[var_name]

    def to_var_transform_dict(self) -> dict[str, VarTransform]:
        """Convert registry to dictionary of VarTransform instances.

        Returns
        -------
        dict[str, VarTransform]
            Dictionary mapping variable names to VarTransform instances.
        """
        return {key: VarTransform(value) for key, value in self.transforms.items()}

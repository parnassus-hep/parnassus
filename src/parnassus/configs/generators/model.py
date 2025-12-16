from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from parnassus.utils.typing import VarNameTuple


@dataclass(slots=True)
class VariablesConfig:
    """Configuration for variables used by a model.

    Parameters
    ----------
    truth_vars_to_load : VarNameTuple
        Tuple of truth-level variable names to load.
    fs_vars : VarNameTuple
        Tuple of fastsim variable names.
    ctxt_vars : VarNameTuple
        Tuple of context variable names.
    ctxt_global_vars : VarNameTuple
        Tuple of global context variable names.
    """

    truth_vars_to_load: VarNameTuple
    fs_vars: VarNameTuple
    ctxt_vars: VarNameTuple
    ctxt_global_vars: VarNameTuple


@dataclass(slots=True)
class SamplerConfig:
    """Configuration for the sampler used with a model.

    Parameters
    ----------
    type : Literal["euler"]
        Type of sampler to use. Currently only "euler" is supported.
    num_steps : int
        Number of sampling steps to perform.
    reverse_time : bool
        Whether to perform reverse-time sampling.
    """

    type: Literal["euler"] = "euler"
    num_steps: int = 50
    reverse_time: bool = False


@dataclass(slots=True)
class ModelConfig:
    """Configuration for a model.

    Parameters
    ----------
    name : str
        Name identifier for the model.
    file_path : Path
        Path to the model file.
    variables_config : VariablesConfig
        Configuration of variables used by the model.
    sampler_config : SamplerConfig, optional
        Configuration for the sampler used with the model. Defaults to a default SamplerConfig.
    """

    name: str
    file_path: Path
    variables_config: VariablesConfig
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from parnassus.utils.typing import VarNameTuple


@dataclass(slots=True)
class VariablesConfig:
    truth_vars_to_load: VarNameTuple
    fs_vars: VarNameTuple
    ctxt_vars: VarNameTuple
    ctxt_global_vars: VarNameTuple


@dataclass(slots=True)
class SamplerConfig:
    type: Literal["euler"] = "euler"
    num_steps: int = 50
    reverse_time: bool = False


@dataclass(slots=True)
class ModelConfig:
    name: str
    file_path: Path
    variables_config: VariablesConfig
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)

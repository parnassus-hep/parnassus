from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from parnassus.configs.accessors import AccessorStore


@dataclass(kw_only=True)
class WriterConfig:
    """Configuration for data writers.

    Parameters
    ----------
    file_path : Path | str
        Path to the output file.
    format : str, optional
        Output file format, by default "default".
    accessor_store : AccessorStore, optional
        Store of accessors for data writing, by default empty AccessorStore.
    """

    # Data writer configs
    file_path: Path | str

    format: str = "default"
    accessor_store: AccessorStore = field(default_factory=AccessorStore)

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls(**config)

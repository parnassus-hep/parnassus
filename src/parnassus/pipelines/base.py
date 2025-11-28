from abc import ABC, abstractmethod

from parnassus.configs.accessors import Accessor
from parnassus.configs.scheme import GenEvent


class GenPipeline(ABC):
    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass

    @abstractmethod
    def get_accessors(self) -> dict[str, list[Accessor]]:
        pass


class SourcePipeline(ABC):
    @abstractmethod
    def run(self) -> tuple[list[GenEvent], dict[str, list[Accessor]]]:
        """Generate events from an external source."""

    @abstractmethod
    def get_accessors(self) -> dict[str, list[Accessor]]:
        """Accessors exposed by this pipeline (after run)."""

"""Parametric event generator — sketch/placeholder."""

from typing import TYPE_CHECKING, final

from parnassus.configs.accessors import Accessor
from parnassus.configs.generators.parametric import ParametricGeneratorConfig
from parnassus.configs.scheme import GenEvent

if TYPE_CHECKING:
    from parnassus.utils.typing import TensorDict


@final
class ParametricEventGenerator:
    """Parametric event generator implementing the EventGenerator protocol.

    This is a placeholder skeleton. Generation logic is not yet implemented.
    """

    def __init__(self, config: ParametricGeneratorConfig) -> None:
        self.config = config

    def get_accessors(self) -> dict[str, list[Accessor]]:
        raise NotImplementedError

    def initialize(self, n_events: int, n_batches: int) -> None:
        raise NotImplementedError

    def process_batch(self, batch: "TensorDict") -> None:
        raise NotImplementedError

    def get_events(self) -> list[GenEvent]:
        raise NotImplementedError

import dataclasses
import operator
from collections.abc import Callable, Sequence
from typing import final, override

import numpy as np

from parnassus.configs.accessors import Accessor
from parnassus.configs.pipeline import FilterCondition, ParticleFilteringConfig
from parnassus.configs.scheme import GenCollection, GenEvent
from parnassus.utils.logger import setup_logger

from .base import GenPipeline

# Map config collection names to GenEvent attribute names.
_ATTR_MAP: dict[str, str] = {
    "truth": "truth_particles",
    "pflow": "pflow_particles",
    "electrons": "electrons",
    "muons": "muons",
}


def _resolve_collection(
    event: GenEvent, name: str
) -> tuple[GenCollection, Callable[[GenCollection], None]]:
    """Return the target collection and a setter that writes a replacement back.

    Parameters
    ----------
    event : GenEvent
        Event to resolve against.
    name : str
        "truth"/"pflow"/"electrons"/"muons" or a key in ``event.collections``.

    Returns
    -------
    tuple[GenCollection, Callable[[GenCollection], None]]
        The current collection and a setter to store the filtered replacement.
    """
    if name in _ATTR_MAP:
        attr = _ATTR_MAP[name]

        def _attr_setter(value: GenCollection, _attr: str = attr) -> None:
            setattr(event, _attr, value)

        return getattr(event, attr), _attr_setter

    if name in event.collections:

        def _coll_setter(value: GenCollection, _key: str = name) -> None:
            event.collections[_key] = value

        return event.collections[name], _coll_setter

    raise ValueError(
        f"Unknown collection '{name}'. Expected one of {sorted(_ATTR_MAP)} "
        f"or a key in event.collections ({sorted(event.collections)})."
    )


_OP_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda v, x: np.isin(v, list(x)),
    "not in": lambda v, x: ~np.isin(v, list(x)),
}


def _condition_mask(collection: GenCollection, cond: FilterCondition) -> np.ndarray:
    """Build a boolean keep-mask for one condition.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)`` — True where the particle passes.
    """
    values = getattr(collection, cond.field, None)
    if values is None:
        raise ValueError(
            f"Field '{cond.field}' is missing or None on collection '{collection.name}'."
        )
    values = np.asarray(values)
    if cond.abs:
        values = np.abs(values)

    fn = _OP_FUNCS.get(cond.op)
    if fn is None:  # pragma: no cover - validated in config
        raise ValueError(f"Unsupported op '{cond.op}'.")
    return fn(values, cond.value)


def _build_mask(collection: GenCollection, config: ParticleFilteringConfig) -> np.ndarray:
    """Combine all condition masks into a single keep-mask.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N,)`` — True where the particle passes all/any conditions.
    """
    n = len(collection)
    if not config.conditions:
        return np.ones(n, dtype=bool)
    masks = [_condition_mask(collection, c) for c in config.conditions]
    if config.combine == "all":
        return np.logical_and.reduce(masks)
    return np.logical_or.reduce(masks)


def _apply_mask(collection: GenCollection, mask: np.ndarray) -> GenCollection:
    """Rebuild ``collection`` keeping only entries where ``mask`` is True.

    Generic over collection type: masks every array-valued init field (and each
    array in the ``jet_idx`` dict), then rebuilds via ``dataclasses.replace`` so
    ``__post_init__`` re-derives counts and re-validates equal lengths.

    Returns
    -------
    GenCollection
        A new collection of the same type with only the kept entries.
    """
    changes: dict[str, object] = {}
    for f in dataclasses.fields(collection):  # type: ignore[arg-type]
        if not f.init or f.name == "name":
            continue
        val = getattr(collection, f.name)
        if val is None:
            continue
        if isinstance(val, dict):  # jet_idx: dict[str, IntArray]
            changes[f.name] = {k: np.asarray(v)[mask] for k, v in val.items()}
        elif isinstance(val, np.ndarray):
            changes[f.name] = val[mask]
    return dataclasses.replace(collection, **changes)  # type: ignore[type-var]


@final
class ParticleFilteringPipeline(GenPipeline):
    """Pipeline that filters particles from a collection by declarative conditions.

    Mutates the target collection in place. Declare a "filter" pipeline before
    "cluster"/"isolation" so they operate on survivors only.
    """

    @override
    def __init__(self, config: ParticleFilteringConfig):
        self.config = config

    @override
    def get_accessors(self) -> dict[str, list[Accessor]]:
        # Filtering introduces no new variables; the collection keeps its identity.
        return {}

    @override
    def process(self, events: Sequence[GenEvent]):
        log = setup_logger()
        log.info(
            f"[green]Filtering '{self.config.collection}' with "
            f"{len(self.config.conditions)} condition(s) (combine={self.config.combine})."
        )
        for event in events:
            collection, setter = _resolve_collection(event, self.config.collection)
            mask = _build_mask(collection, self.config)
            setter(_apply_mask(collection, mask))
            # Keep cached scalar HT/MET event features consistent with the
            # filtered particle collections (leptons are not re-derived).
            event.update_event_features()

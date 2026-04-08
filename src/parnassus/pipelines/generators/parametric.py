"""Parametric event generator using torch_delphes detector simulation."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Self, final

import numpy as np
import torch
from rich.progress import Progress, TaskID

from parnassus.configs.accessors import Accessor, AccessorListBuilder, AccessorSpec
from parnassus.configs.generators.parametric import ParametricGeneratorConfig
from parnassus.configs.scheme import (
    GenCollection,
    GenEvent,
    GenParticleCollection,
    GenTowerCollection,
)
from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes.defaults import (
    ATLASEnergyFlowDefault,
    CMSEnergyFlowDefault,
)
from parnassus.torch_delphes.defaults.base import DelphesBaseCard
from parnassus.utils.logger import ProgressBar

if TYPE_CHECKING:
    from parnassus.utils.typing import TensorDict

T_SCALE_CONVERSION = 1e-3 / 299792458.0  # Convert mm/c to seconds for Delphes convention

_CARD_REGISTRY: dict[str, type[DelphesBaseCard]] = {
    "cms": CMSEnergyFlowDefault,
    "atlas": ATLASEnergyFlowDefault,
}

# All fields produced by the detector card for particles and tracks.
# No impact parameters (those require the neural impact model).
_PARTICLE_SPECS: list[AccessorSpec] = [
    AccessorSpec("pt", output_name="PT"),
    AccessorSpec("eta", output_name="Eta"),
    AccessorSpec("phi", output_name="Phi"),
    AccessorSpec("mass", output_name="Mass"),
    AccessorSpec("vx", output_name="X"),
    AccessorSpec("vy", output_name="Y"),
    AccessorSpec("vz", output_name="Z"),
    AccessorSpec("t", output_name="T"),
    AccessorSpec("pdg_id", output_name="PID", dtype="int32"),
    AccessorSpec("class_id", output_name="ClassID", dtype="int32"),
    AccessorSpec("charge", output_name="Charge", dtype="int32"),
    AccessorSpec("status", output_name="Status", dtype="int32"),
]

_TOWER_SPECS: list[AccessorSpec] = [
    AccessorSpec("e", output_name="E"),
    AccessorSpec("et", output_name="ET"),
    AccessorSpec("eta", output_name="Eta"),
    AccessorSpec("phi", output_name="Phi"),
    AccessorSpec("t", output_name="T"),
]

# Card output keys that represent calorimeter towers (vs charged-particle tracks).
# These are stored as GenTowerCollection; all other keys become GenParticleCollection.
_TOWER_OUTPUT_KEYS: frozenset[str] = frozenset({
    "Tower",
    "ECalTower",
    "HCalTower",
    "EFlowPhoton",
    "EFlowNeutralHadron",
})

# Keys that are always handled specially: truth/pflow live on GenEvent directly,
# Track and Tower go to collections in both normal and debug mode.
_ALWAYS_COLLECTIONS: frozenset[str] = frozenset({"AllParticles", "Track", "Tower"})

# Keys present in normal (non-debug) card output that we skip in normal mode —
# they would be redundant with EFlowObject (pflow) and are only stored in debug.
_NORMAL_MODE_SKIP = frozenset({"EFlowTrack", "EFlowPhoton", "EFlowNeutralHadron"})

_DEBUG_CARD_KEYS: frozenset[str] = frozenset({
    "ParticleBeforeProp",
    "ParticleAfterProp",
    "ChargedHadron",
    "Electron",
    "Muon",
    "NeutralParticle",
    "ChargedHadronEfficiency",
    "ElectronEfficiency",
    "MuonEfficiency",
    "ChargedHadronSmeared",
    "ElectronSmeared",
    "MuonSmeared",
    "ECal_EFlowTrack",
    "EFlowTrack",
    "ECalTower",
    "HCalTower",
    "EFlowPhoton",
    "EFlowNeutralHadron",
})


@final
class ParametricEventGenerator:
    """Parametric event generator implementing the EventGenerator protocol.

    Routes truth-level HepMC particles through a torch_delphes detector
    simulation card (CMS or ATLAS) and accumulates GenEvent objects.

    In normal mode (``config.debug=False``) each event stores:
    truth particles, pflow particles (EFlowObject), tracks, and towers.

    In debug mode (``config.debug=True``) the card's full intermediate output
    is also stored — every branch the card emits is placed in
    ``GenEvent.collections`` under its card output key.
    """

    def __init__(self, config: ParametricGeneratorConfig, log) -> None:
        self.config = config
        self.log = log

        self.device = torch.device("cpu")
        self.card: DelphesBaseCard = _CARD_REGISTRY[config.card](debug=config.debug)
        self._events: list[GenEvent] | None = None
        self._exit_stack: ExitStack | None = None
        self._progress_bar: Progress | None = None
        self._task: TaskID | None = None

        self.log.info(
            f"Initialized ParametricEventGenerator with card='{config.card}', debug={config.debug}"
        )

    def to(self, device: torch.device) -> Self:
        self.card = self.card.to(device)
        self.device = device
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
            self._progress_bar = None

        self.log.debug("[green]Resetting precision to float32.")
        torch.set_default_dtype(torch.float32)
        # Move card back to CPU to free GPU memory, if applicable
        self.to(torch.device("cpu"))

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def initialize(self, n_events: int, n_batches: int) -> None:  # noqa: ARG002
        if self.config.seed is not None:
            self.set_seed(self.config.seed)
            self.log.info(f"Set random seed to {self.config.seed}")
        self.log.debug("[green]Setting precision to float64.")
        torch.set_default_dtype(torch.float64)
        self.card.eval()
        self._events = []
        self._exit_stack = ExitStack()
        self._progress_bar = self._exit_stack.enter_context(ProgressBar())
        self._task = self._progress_bar.add_task("[green]Parametric generation", total=n_batches)

    @torch.inference_mode()
    def process_batch(self, batch: TensorDict) -> None:
        assert self._events is not None, "Call initialize() before process_batch()"
        assert self._progress_bar is not None
        assert self._task is not None

        all_particles: torch.Tensor = batch["all_particles"]
        stable_particles: torch.Tensor = batch["stable_particles"]
        # Clone before passing to card: card may mutate input in-place, and
        # .to(device) returns the same tensor when already on that device.
        truth_particles = stable_particles.clone()
        if stable_particles.shape[0] > 0:
            results = self.card(stable_particles.to(self.device))
        else:
            results = _empty_card_results(all_particles, debug=self.config.debug)

        self._events.extend(
            _tensors_to_gen_events(
                truth=truth_particles,
                event_numbers=batch["event_numbers"],
                results=results | {"AllParticles": all_particles},
                debug=self.config.debug,
            )
        )

        self._progress_bar.update(self._task, advance=1)

    def get_events(self) -> list[GenEvent]:
        assert self._events is not None, "Call initialize() before get_events()"
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None
            self._progress_bar = None
        return self._events

    def get_accessors(self) -> dict[str, list[Accessor]]:
        accessors: dict[str, list[Accessor]] = {
            "Truth": AccessorListBuilder
            .for_particles("truth_particles")
            .add_from_specs(_PARTICLE_SPECS)
            .build(),
            "Pflow": AccessorListBuilder
            .for_particles("pflow_particles")
            .add_from_specs(_PARTICLE_SPECS)
            .build(),
            "Track": AccessorListBuilder
            .for_collection("Track")
            .add_from_specs(_PARTICLE_SPECS)
            .build(),
            "Tower": AccessorListBuilder
            .for_collection("Tower")
            .add_from_specs(_TOWER_SPECS)
            .build(),
        }
        if self.config.debug:
            # Add accessors for all intermediate card outputs stored in collections.
            # Card output key == collection key == ROOT branch name.
            for key in _DEBUG_CARD_KEYS:
                specs = _TOWER_SPECS if key in _TOWER_OUTPUT_KEYS else _PARTICLE_SPECS
                accessors[key] = (
                    AccessorListBuilder.for_collection(key).add_from_specs(specs).build()
                )
        return accessors


# ---------------------------------------------------------------------------
# Module-level tensor → GenEvent conversion helpers
# ---------------------------------------------------------------------------


def _split_by_event(arr: np.ndarray) -> dict[int, np.ndarray]:
    """Split a concatenated multi-event array into per-event sub-arrays.

    Sorts by event number then uses ``np.split`` for a single O(N log N) pass.
    Handles non-monotonic event numbers (e.g. card output reordering rows).

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (N_rows, N_features) in ColumnMap format, containing rows for
        multiple events concatenated together. Must include ColumnMap.EVENT_NUMBER.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from event number to sub-array of rows for that event.
    """
    if arr.shape[0] == 0:
        return {}
    event_col = arr[:, ColumnMap.EVENT_NUMBER].astype(np.int32)
    sort_idx = np.argsort(event_col, kind="stable")
    sorted_col = event_col[sort_idx]
    sorted_arr = arr[sort_idx]
    boundaries = np.flatnonzero(np.diff(sorted_col)) + 1
    unique_evs = sorted_col[np.concatenate(([0], boundaries))]
    return {
        int(ev): sub for ev, sub in zip(unique_evs, np.split(sorted_arr, boundaries), strict=True)
    }


def _empty_card_results(
    source: torch.Tensor,
    debug: bool = False,
) -> dict[str, torch.Tensor]:
    """Create empty card outputs matching the non-debug/debug branch contract.

    Parameters
    ----------
    source : torch.Tensor
        The input tensor to the card, used to determine the number of features
        and dtype for the empty outputs.
    debug : bool, optional
        Whether to include debug branches in the output. Defaults to False.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary with keys for each expected card output,
        where each value is an empty tensor of the appropriate shape and dtype.
    """
    empty = source.new_zeros((0, source.shape[1]))
    results: dict[str, torch.Tensor] = {
        "EFlowObject": empty,
        "Track": empty,
        "Tower": empty,
    }
    if debug:
        for key in _DEBUG_CARD_KEYS:
            results[key] = empty
    return results


def _tensors_to_gen_events(
    truth: torch.Tensor,
    event_numbers: torch.Tensor,
    results: dict[str, torch.Tensor],
    debug: bool = False,
) -> list[GenEvent]:
    """Convert a batch of card outputs into a list of GenEvent objects.

    Parameters
    ----------
    truth:
        Stable truth tensor in ColumnMap format, all events concatenated.
    event_numbers:
        Event numbers for the original batch order. Events with no stable particles
        are still emitted with empty truth/pflow collections.
    results:
        Output dict from the detector card forward pass.
    debug:
        When True, every card output key is stored in ``GenEvent.collections``.
        When False, only ``Track`` and ``Tower`` are stored in collections.

    Returns
    -------
    list[GenEvent]
        One GenEvent per input event number.
    """
    truth_np = truth.cpu().numpy()
    pflow_np = results["EFlowObject"].cpu().numpy()

    # Convert tensors to numpy once, then pre-split every array by event number
    # (O(N log N) sort + single split pass), rather than O(N_events x N_arrays)
    # repeated boolean scans.
    extra_np: dict[str, np.ndarray] = {
        key: tensor.cpu().numpy()
        for key, tensor in results.items()
        if key != "EFlowObject"
        and (debug or key in _ALWAYS_COLLECTIONS)
        and (debug or key not in _NORMAL_MODE_SKIP)
    }

    truth_splits = _split_by_event(truth_np)
    pflow_splits = _split_by_event(pflow_np)
    extra_splits = {key: _split_by_event(arr) for key, arr in extra_np.items()}

    empty: np.ndarray = np.empty((0, truth_np.shape[1]), dtype=truth_np.dtype)
    events = []
    for ev in event_numbers.cpu().numpy():
        truth_a = truth_splits.get(ev, empty)
        collections: dict[str, GenCollection] = {}
        for key, splits in extra_splits.items():
            a = splits.get(ev, empty)
            collections[key] = (
                _make_tower_collection(a, name=key)
                if key in _TOWER_OUTPUT_KEYS
                else _make_particle_collection(a, name=key)
            )

        pflow_a = pflow_splits.get(ev, empty)
        events.append(
            GenEvent(
                event_number=ev,
                truth_particles=_make_particle_collection(
                    truth_a, "truth", fix_neutral_hadrons=True
                ),
                pflow_particles=_make_particle_collection(
                    pflow_a, "pflow", fix_neutral_hadrons=True
                ),
                collections=collections,
            )
        )
    return events


def _make_particle_collection(
    a: np.ndarray, name: str, fix_neutral_hadrons: bool = False
) -> GenParticleCollection:
    """Build a ``GenParticleCollection`` from a pre-sliced single-event array.

    Parameters
    ----------
    a : np.ndarray
        Array of shape (N_particles, N_features) in ColumnMap format.
    name : str
        Name for the GenParticleCollection (e.g. "truth", "pflow", "Track").
    fix_neutral_hadrons : bool, optional
        If True, replace PDG ID 0 with 130 (K_L^0) to match Delphes convention for neutral hadrons.
        Defaults to False.

    Returns
    -------
    GenParticleCollection
        GenParticleCollection with fields populated from the input array.
    """
    pdg_id = a[:, ColumnMap.PID].astype(np.int32)
    if fix_neutral_hadrons:
        pdg_id = pdg_id.copy()
        pdg_id[pdg_id == 0] = 130  # K_L^0 for Delphes-convention neutral hadrons (PID=0)
    return GenParticleCollection(
        name=name,
        pt=a[:, ColumnMap.PT].astype(np.float32),
        eta=a[:, ColumnMap.ETA].astype(np.float32),
        phi=a[:, ColumnMap.PHI].astype(np.float32),
        mass=a[:, ColumnMap.MASS].astype(np.float32),
        pdg_id=pdg_id,
        charge=a[:, ColumnMap.CHARGE].astype(np.int32),
        vx=a[:, ColumnMap.X].astype(np.float32),
        vy=a[:, ColumnMap.Y].astype(np.float32),
        vz=a[:, ColumnMap.Z].astype(np.float32),
        t=(a[:, ColumnMap.T] * T_SCALE_CONVERSION).astype(np.float32),
        status=a[:, ColumnMap.STATUS].astype(np.int32),
    )


def _make_tower_collection(a: np.ndarray, name: str = "Tower") -> GenTowerCollection:
    """Build a ``GenTowerCollection`` from a pre-sliced single-event array.

    Parameters
    ----------
    a : np.ndarray
        Array of shape (N_towers, N_features) in ColumnMap format.
    name : str, optional
        Name for the GenTowerCollection (e.g. "Tower", "ECalTower").
        Defaults to "Tower".

    Returns
    -------
    GenTowerCollection
        GenTowerCollection with fields populated from the input array.
    """
    eta = a[:, ColumnMap.ETA].astype(np.float32)
    e = a[:, ColumnMap.E].astype(np.float32)
    return GenTowerCollection(
        name=name,
        e=e,
        et=(e / np.cosh(eta)).astype(np.float32),  # ET = E / cosh(η), matching Delphes convention
        eta=eta,
        phi=a[:, ColumnMap.PHI].astype(np.float32),
        t=(a[:, ColumnMap.T] * T_SCALE_CONVERSION).astype(np.float32),
    )

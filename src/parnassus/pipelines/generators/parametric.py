"""Parametric event generator using torch_delphes detector simulation."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Self, final

import numpy as np
import torch

from parnassus.configs.accessors import Accessor, AccessorListBuilder, AccessorSpec
from parnassus.configs.generators.parametric import ParametricGeneratorConfig
from parnassus.configs.scheme import (
    GenEvent,
    GenParticleCollection,
    GenTowerCollection,
    GenTrackCollection,
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
_ALWAYS_COLLECTIONS = ("Track", "Tower")

# Keys present in normal (non-debug) card output that we skip in normal mode —
# they would be redundant with EFlowObject (pflow) and are only stored in debug.
_NORMAL_MODE_SKIP = frozenset({"EFlowTrack", "EFlowPhoton", "EFlowNeutralHadron"})


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
        self._progress_bar: ProgressBar | None = None
        self._task = None

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

    def initialize(self, n_events: int, n_batches: int) -> None:  # noqa: ARG002
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
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

        all_particles: torch.Tensor = batch["all_particles"]
        stable_particles: torch.Tensor = batch["stable_particles"].to(self.device)
        if stable_particles.shape[0] > 0:
            results = self.card(stable_particles)
            self._events.extend(
                _tensors_to_gen_events(
                    truth=all_particles,
                    results=results,
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
            "Truth": AccessorListBuilder.for_particles("truth_particles")
            .add_from_specs(_PARTICLE_SPECS)
            .build(),
            "Pflow": AccessorListBuilder.for_particles("pflow_particles")
            .add_from_specs(_PARTICLE_SPECS)
            .build(),
            "Track": AccessorListBuilder.for_collection("Track")
            .add_from_specs(_PARTICLE_SPECS)
            .build(),
            "Tower": AccessorListBuilder.for_collection("Tower")
            .add_from_specs(_TOWER_SPECS)
            .build(),
        }
        if self.config.debug:
            # Add accessors for all intermediate card outputs stored in collections.
            # Card output key == collection key == ROOT branch name.
            debug_card_keys = {
                # Particle-type
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
                # Tower-type
                "ECalTower",
                "HCalTower",
                "EFlowPhoton",
                "EFlowNeutralHadron",
            }
            for key in debug_card_keys:
                specs = _TOWER_SPECS if key in _TOWER_OUTPUT_KEYS else _PARTICLE_SPECS
                accessors[key] = (
                    AccessorListBuilder.for_collection(key).add_from_specs(specs).build()
                )
        return accessors


# ---------------------------------------------------------------------------
# Module-level tensor → GenEvent conversion helpers
# ---------------------------------------------------------------------------


def _tensors_to_gen_events(
    truth: torch.Tensor,
    results: dict[str, torch.Tensor],
    debug: bool = False,
) -> list[GenEvent]:
    """Convert a batch of card outputs into a list of GenEvent objects.

    Parameters
    ----------
    truth:
        All-particle truth tensor (ColumnMap format, all events concatenated).
    results:
        Output dict from the detector card forward pass.
    debug:
        When True, every card output key is stored in ``GenEvent.collections``.
        When False, only ``Track`` and ``Tower`` are stored in collections.

    Returns
    -------
    list[GenEvent]
        One GenEvent per unique event number in the truth tensor.
    """
    truth_np = truth.cpu().numpy()
    pflow_np = results["EFlowObject"].cpu().numpy()

    # Pre-convert all extra tensors to numpy once, outside the per-event loop.
    extra_np: dict[str, np.ndarray] = {
        key: tensor.cpu().numpy()
        for key, tensor in results.items()
        if key != "EFlowObject"
        and (debug or key in _ALWAYS_COLLECTIONS)
        and (debug or key not in _NORMAL_MODE_SKIP)
    }

    event_nums = np.unique(truth_np[:, ColumnMap.EVENT_NUMBER].astype(np.int32))
    events = []
    for ev in event_nums:
        collections: dict = {}

        for key, arr in extra_np.items():
            if key in _TOWER_OUTPUT_KEYS:
                col = _make_tower_collection(arr, ev, name=key)
            elif key == "Track":
                col = _make_track_collection(arr, ev)
            else:
                col = _make_particle_collection(arr, ev, name=key) if arr.shape[0] > 0 else None
            if col is not None:
                collections[key] = col
        events.append(
            GenEvent(
                event_number=int(ev),
                truth_particles=_make_particle_collection(
                    truth_np, ev, "truth", fix_neutral_hadrons=True
                ),
                pflow_particles=_make_particle_collection(
                    pflow_np, ev, "pflow", fix_neutral_hadrons=True
                ),
                collections=collections,
            )
        )
    return events


def _mask_for_event(arr: np.ndarray, event_num: int) -> np.ndarray:
    return arr[:, ColumnMap.EVENT_NUMBER].astype(np.int32) == event_num


def _make_particle_collection(
    arr: np.ndarray, event_num: int, name: str, fix_neutral_hadrons: bool = False
) -> GenParticleCollection:
    a = arr[_mask_for_event(arr, event_num)]
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


def _make_track_collection(arr: np.ndarray, event_num: int) -> GenTrackCollection | None:
    a = arr[_mask_for_event(arr, event_num)]
    if a.shape[0] == 0:
        return None
    return GenTrackCollection(
        name="Track",
        pt=a[:, ColumnMap.PT].astype(np.float32),
        eta=a[:, ColumnMap.ETA].astype(np.float32),
        phi=a[:, ColumnMap.PHI].astype(np.float32),
        mass=a[:, ColumnMap.MASS].astype(np.float32),
        pdg_id=a[:, ColumnMap.PID].astype(np.int32),
        charge=a[:, ColumnMap.CHARGE].astype(np.int32),
        vx=a[:, ColumnMap.X].astype(np.float32),
        vy=a[:, ColumnMap.Y].astype(np.float32),
        vz=a[:, ColumnMap.Z].astype(np.float32),
        t=(a[:, ColumnMap.T] * T_SCALE_CONVERSION).astype(np.float32),
        status=a[:, ColumnMap.STATUS].astype(np.int32),
    )


def _make_tower_collection(
    arr: np.ndarray, event_num: int, name: str = "Tower"
) -> GenTowerCollection | None:
    a = arr[_mask_for_event(arr, event_num)]
    if a.shape[0] == 0:
        return None
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

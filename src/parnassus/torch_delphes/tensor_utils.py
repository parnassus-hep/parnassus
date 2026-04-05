"""Utility functions for converting between ROOT files and PyTorch tensors.

This module provides:
- Column index constants for the tensor representation
- HepMC → Tensor conversion (per particle type)
- Tensor → ROOT conversion (for writing output files)

``ColumnMap``, ``N_FEATURES``, and the low-level particle conversion helpers
live in ``parnassus.data.particle_io`` (a dependency-free leaf module).
They are re-exported here so all existing callers continue to work unchanged.
"""

import struct
from pathlib import Path

import awkward as ak
import numpy as np
import pyhepmc
import torch
import uproot
from tqdm import tqdm

# Re-export shared definitions from particle_io so existing imports still work.
from parnassus.data.particle_io import (
    N_FEATURES,
    ColumnMap,
    particles_to_tensor,
)

# ==================== TENSOR COLUMN INDICES ====================
# These bare-integer names are kept for backward compatibility with callers
# that do ``from parnassus.torch_delphes.tensor_utils import ETA, PT, ...``.
PID = ColumnMap.PID
STATUS = ColumnMap.STATUS
CHARGE = ColumnMap.CHARGE
E = ColumnMap.E
PX = ColumnMap.PX
PY = ColumnMap.PY
PZ = ColumnMap.PZ
PT = ColumnMap.PT
ETA = ColumnMap.ETA
PHI = ColumnMap.PHI
T = ColumnMap.T
X = ColumnMap.X
Y = ColumnMap.Y
Z = ColumnMap.Z
MASS = ColumnMap.MASS
ETA_OUTER = ColumnMap.ETA_OUTER
PHI_OUTER = ColumnMap.PHI_OUTER
EVENT_NUMBER = ColumnMap.EVENT_NUMBER
PASS_PROP = ColumnMap.PASS_PROP
TRACK_RESOLUTION = ColumnMap.TRACK_RESOLUTION

COLUMN_MAP = {
    "PID": PID,
    "STATUS": STATUS,
    "CHARGE": CHARGE,
    "E": E,
    "PX": PX,
    "PY": PY,
    "PZ": PZ,
    "PT": PT,
    "ETA": ETA,
    "PHI": PHI,
    "T": T,
    "X": X,
    "Y": Y,
    "Z": Z,
    "MASS": MASS,
    "ETA_OUTER": ETA_OUTER,
    "PHI_OUTER": PHI_OUTER,
    "EVENT_NUMBER": EVENT_NUMBER,
    "PASS_PROP": PASS_PROP,
    "TRACK_RESOLUTION": TRACK_RESOLUTION,
}


# ==================== BATCHING UTILITIES ====================


def compute_max_particles(event_tensors: list[torch.Tensor], scale: float = 1.2) -> int:
    """Compute max_particles for padding as scale * max particle count in dataset.

    Parameters
    ----------
    event_tensors: list[torch.Tensor]
        List of (N_i, N_FEATURES) tensors
    scale: float
        Scaling factor (default 1.2 = 20% buffer)

    Returns
    -------
    max_particles: int
        Integer max particles for padding
    """
    if len(event_tensors) == 0:
        return 0
    max_count = max(t.shape[0] for t in event_tensors)
    return int(max_count * scale)


def tensor_to_root_dict(
    batch_tensors: list[torch.Tensor],
    branch_name: str,
    expected_event_numbers: list[float] | None = None,
) -> dict[str, ak.Array]:
    """Convert list of event tensors to ROOT-compatible dictionary of awkward arrays.

    This creates the structure needed for writing to ROOT files with uproot.

    Parameters
    ----------
        batch_tensors: List of tensors, one per batch, each of shape (n_particles, N_FEATURES)
        branch_name: Name for the branch (e.g., "ChargedHadronEfficiency")
        expected_event_numbers: List of all expected event numbers. If provided, ensures
                               all events are represented (with empty arrays for missing events).

    Returns
    -------
        Dictionary with keys like "BranchName/BranchName.Attribute" → awkward array
    """
    # Determine the branch type based on name
    # Tower objects: Tower, EFlowPhoton (ECal), EFlowNeutralHadron (HCal)
    # ParticleFlowCandidate: EFlowObject (combines Track and Tower fields)
    # GenParticle: Particle (all particles from HepMC including unstable)
    is_tower = (
        any(keyword in branch_name for keyword in ["Tower", "EFlowPhoton", "EFlowNeutralHadron"])
        and "EFlowObject" not in branch_name
    )
    is_eflow = "EFlowObject" in branch_name
    is_genparticle = branch_name == "Particle"

    if is_genparticle:
        # GenParticle attributes: matches Delphes GenParticle class
        # See DelphesClasses.h for full list. Core attributes:
        # PID, Status, Charge, E, Px, Py, Pz, P, PT, Eta, Phi, Rapidity, Mass, T, X, Y, Z
        # M1, M2, D1, D2 (mother/daughter indices - not available in tensor)
        attributes = [
            "PID",
            "Status",
            "Charge",
            "E",
            "Px",
            "Py",
            "Pz",
            "P",
            "PT",
            "Eta",
            "Phi",
            "Mass",
            "T",
            "X",
            "Y",
            "Z",
        ]

        column_map = {
            "PID": PID,
            "Status": STATUS,
            "Charge": CHARGE,
            "E": E,
            "Px": PX,
            "Py": PY,
            "Pz": PZ,
            "P": None,  # Will compute from Px, Py, Pz
            "PT": PT,
            "Eta": ETA,
            "Phi": PHI,
            "Mass": MASS,
            "T": T,
            "X": X,
            "Y": Y,
            "Z": Z,
        }
    elif is_eflow:
        # ParticleFlowCandidate attributes: combination of Track and Tower fields
        # This matches the ParticleFlowCandidate class in DelphesClasses.h (lines 532-613)
        # Track fields: PID, Charge, E, P, PT, Eta, Phi, CtgTheta, C, Mass, EtaOuter, PhiOuter,
        #               T, X, Y, Z, TOuter, XOuter, YOuter, ZOuter, Xd, Yd, Zd, L, D0, DZ,
        #               Nclusters, dNdx, ErrorP, ErrorPT, ErrorPhi, ErrorCtgTheta, ErrorT,
        #               ErrorD0, ErrorDZ, ErrorC, ErrorD0Phi, ErrorD0C, ErrorD0DZ, ErrorD0CtgTheta,
        #               ErrorPhiC, ErrorPhiDZ, ErrorPhiCtgTheta, ErrorCDZ, ErrorCCtgTheta,
        #               ErrorDZCtgTheta, VertexIndex
        # Tower fields: NTimeHits, Eem, Ehad, Edges[4]
        # Note: For Track objects, Tower fields are zero.
        # For Tower objects, Track-specific fields are zero.
        attributes = [
            "PID",
            "Charge",
            "E",
            "P",
            "PT",
            "Eta",
            "Phi",
            "T",
            "X",
            "Y",
            "Z",
            "Eem",
            "Ehad",
        ]

        # Column indices for ParticleFlowCandidate attributes
        column_map = {
            "PID": PID,
            "Charge": CHARGE,
            "E": E,
            "P": None,  # Will compute from Px, Py, Pz
            "PT": PT,
            "Eta": ETA,
            "Phi": PHI,
            "T": T,
            "X": X,
            "Y": Y,
            "Z": Z,
            "Eem": None,  # Will be zero for Track objects (towers don't have this in tensor)
            "Ehad": None,  # Will be zero for Track objects (towers don't have this in tensor)
        }
    elif is_tower:
        # Tower attributes: E, ET, Eta, Phi, T, Eem, Ehad
        attributes = ["E", "ET", "Eta", "Phi", "T"]

        # Column indices for tower attributes
        column_map = {
            "E": E,
            "ET": None,  # Will compute as E / cosh(Eta)
            "Eta": ETA,  # Momentum eta
            "Phi": PHI,  # Momentum phi
            "T": T,
        }
    else:
        # Track attributes (existing code)
        attributes = ["PID", "Charge", "P", "PT", "Eta", "EtaOuter", "Phi", "T", "X", "Y", "Z"]

        # Column indices for each attribute in the tensor
        column_map = {
            "PID": PID,
            "Charge": CHARGE,
            "P": None,  # Will compute from Px, Py, Pz
            "PT": PT,
            "Eta": ETA,  # Will compute from Px, Py, Pz (momentum eta)
            "EtaOuter": ETA_OUTER,  # Position eta stored in ETA_OUTER column
            "Phi": PHI,
            "PhiOuter": PHI_OUTER,  # Position phi stored in PHI_OUTER column
            "T": T,
            "X": X,
            "Y": Y,
            "Z": Z,
        }

    # Build dictionary
    root_dict = {}

    # First, collect all particles grouped by event number
    # Concatenate all batches into a single tensor
    if len(batch_tensors) == 0 or all(b.shape[0] == 0 for b in batch_tensors):
        # No particles at all - create empty arrays for all expected events
        all_particles_np = None
        all_event_numbers = []
    else:
        # Single CPU transfer - do this once for all attributes
        all_particles = torch.cat([b for b in batch_tensors if b.shape[0] > 0], dim=0)
        all_particles_np = all_particles.cpu().numpy()
        all_event_numbers = list(np.unique(all_particles_np[:, ColumnMap.EVENT_NUMBER]))

    # Determine which event numbers to iterate over
    if expected_event_numbers is not None:
        event_nums_to_process = sorted(expected_event_numbers)
    else:
        event_nums_to_process = sorted(all_event_numbers)

    # Pre-group particles by event number for efficient access
    # This avoids repeated filtering per attribute
    if all_particles_np is not None and len(all_particles_np) > 0:
        event_indices = all_particles_np[:, ColumnMap.EVENT_NUMBER]
        sort_indices = np.argsort(event_indices)
        sorted_particles = all_particles_np[sort_indices]
        sorted_event_nums = event_indices[sort_indices]

        # Find boundaries where event number changes
        event_boundaries = np.searchsorted(sorted_event_nums, event_nums_to_process)
        event_end_boundaries = np.searchsorted(
            sorted_event_nums, event_nums_to_process, side="right"
        )

        # Build a dict mapping event_num -> slice of sorted_particles
        event_slices = {}
        for i, event_num in enumerate(event_nums_to_process):
            start_idx = event_boundaries[i]
            end_idx = event_end_boundaries[i]
            if start_idx < end_idx:
                event_slices[event_num] = sorted_particles[start_idx:end_idx]
            else:
                event_slices[event_num] = None
    else:
        event_slices = dict.fromkeys(event_nums_to_process)

    # Helper function to compute attribute values for a single event
    def compute_attr_values(event_np, attr):  # noqa: PLR0911
        """Compute attribute values for particles in one event."""  # noqa: DOC201
        if event_np is None or len(event_np) == 0:
            return np.array([], dtype=np.float64)

        if is_genparticle:
            # GenParticle-specific computations to match C++ Delphes behavior
            if attr == "P":
                # C++ Delphes never sets P for GenParticles, leaving it uninitialized
                # The uninitialized value is 0x99999999 = -1.58818668e-23
                # We use the same sentinel value for exact validation match

                sentinel = struct.unpack("f", bytes.fromhex("99999999"))[0]
                return np.full(event_np.shape[0], sentinel, dtype=np.float32)
            if attr in {"PID", "Status"}:
                return event_np[:, column_map[attr]].astype(np.int32)
            if attr in column_map and column_map[attr] is not None:
                return event_np[:, column_map[attr]]
            return np.zeros(event_np.shape[0])
        if is_eflow:
            # ParticleFlowCandidate-specific computations
            if attr == "P":
                px = event_np[:, ColumnMap.PX]
                py = event_np[:, ColumnMap.PY]
                pz = event_np[:, ColumnMap.PZ]
                return np.sqrt(px**2 + py**2 + pz**2)
            if attr == "Eta":
                return event_np[:, ColumnMap.ETA]
            if attr == "Eem":
                pid_vals = event_np[:, ColumnMap.PID]
                e_vals = event_np[:, ColumnMap.E]
                return np.where(pid_vals == 22, e_vals, 0.0)
            if attr == "Ehad":
                pid_vals = event_np[:, ColumnMap.PID]
                e_vals = event_np[:, ColumnMap.E]
                return np.where(pid_vals == 0, e_vals, 0.0)
            if attr == "T":
                return event_np[:, ColumnMap.T] * 1e-3 / 299792458.0
            if attr == "PID":
                return event_np[:, column_map[attr]].astype(np.int32)
            return event_np[:, column_map[attr]]
        if is_tower:
            # Tower-specific computations
            if attr == "ET":
                e = event_np[:, ColumnMap.E]
                eta = event_np[:, ColumnMap.ETA]
                return e / np.cosh(eta)
            if attr == "T":
                return event_np[:, ColumnMap.T] * 1e-3 / 299792458.0
            if attr in column_map and column_map[attr] is not None:
                return event_np[:, column_map[attr]]
            return np.zeros(event_np.shape[0])
        # Track-specific computations
        if attr == "P":
            px = event_np[:, ColumnMap.PX]
            py = event_np[:, ColumnMap.PY]
            pz = event_np[:, ColumnMap.PZ]
            return np.sqrt(px**2 + py**2 + pz**2)
        if attr == "Eta":
            px = event_np[:, ColumnMap.PX]
            py = event_np[:, ColumnMap.PY]
            pz = event_np[:, ColumnMap.PZ]
            pt = np.sqrt(px**2 + py**2)
            return np.arcsinh(pz / (pt + 1e-10))
        if attr == "T":
            return event_np[:, ColumnMap.T] * 1e-3 / 299792458.0
        if attr == "PID":
            return event_np[:, column_map[attr]].astype(np.int32)
        return event_np[:, column_map[attr]]

    # Process all attributes using pre-grouped events
    for attr in attributes:
        attr_values = [
            compute_attr_values(event_slices[event_num], attr)
            for event_num in event_nums_to_process
        ]

        # Convert to awkward array
        ak_array = ak.Array(attr_values)

        # Add to dictionary with ROOT branch naming
        key = f"{branch_name}/{branch_name}.{attr}"
        root_dict[key] = ak_array

    return root_dict


def write_root_file(
    output_file: str | Path,
    branches_dict: dict[str, dict[str, ak.Array]],
    tree_name: str = "Delphes",
):
    """Write multiple branches to a ROOT file.

    Parameters
    ----------
    output_file: str | Path
        Path to output ROOT file
    branches_dict: dict[str, dict[str, ak.Array]]
        Dictionary mapping branch names to their data dictionaries
        e.g., {"ChargedHadronEfficiency": {...}, "ElectronEfficiency": {...}}
    tree_name: str
        Name of the tree in ROOT file (default: "Delphes")
    """
    # Combine all branch dictionaries
    combined_dict = {}
    for branch_data in branches_dict.values():
        combined_dict.update(branch_data)

    # Write to ROOT file
    with uproot.recreate(output_file) as f:
        f[tree_name] = combined_dict


def hepmc_to_tensor(
    hepmc_file: str | Path, max_events: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HepMC file to PyTorch tensors.

    Reads particles from HepMC events and converts to tensor format.
    Returns both stable particles (for detector simulation) and all particles
    (for truth-level studies).

    Parameters
    ----------
    hepmc_file: str | Path
        Path to HepMC file (.hepmc, .hepmc3, or .hepmc.gz)
    max_events: int | None, optional
        Maximum number of events to process, or None to process all events (default: None)

    Returns
    -------
    stable_particles: torch.Tensor
        Tensor of shape (n_stable_total, N_FEATURES) with status==1 particles
    all_particles: torch.Tensor
        Tensor of shape (n_all_total, N_FEATURES) with all particles
    """
    stable_event_tensors = []
    all_event_tensors = []

    with pyhepmc.open(hepmc_file) as f:
        for event_idx, event in tqdm(enumerate(f), total=max_events):
            if max_events is not None and event_idx >= max_events:
                break

            event_number = event.event_number

            # Get all particles and stable particles
            all_particles_list = list(event.particles)
            stable_particles_list = [p for p in all_particles_list if p.status == 1]

            # Process stable particles
            stable_tensor = particles_to_tensor(stable_particles_list, event_number)
            stable_event_tensors.append(stable_tensor)

            # Process all particles
            all_tensor = particles_to_tensor(all_particles_list, event_number)
            all_event_tensors.append(all_tensor)

    stable_particles = (
        torch.cat(stable_event_tensors, dim=0)
        if stable_event_tensors
        else torch.zeros((0, N_FEATURES), dtype=torch.float64)
    )
    all_particles = (
        torch.cat(all_event_tensors, dim=0)
        if all_event_tensors
        else torch.zeros((0, N_FEATURES), dtype=torch.float64)
    )

    return stable_particles, all_particles

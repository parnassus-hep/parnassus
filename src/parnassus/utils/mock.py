from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import final

import awkward as ak
import numpy as np
import pyhepmc
import uproot
from numpy.random import Generator
from torch import Tensor, nn
from torch.export import Dim, export, save

from parnassus.configs.variables import VariableRequirements

from . import pid_to_class
from .transform import VarTransform, VarTransformConfig
from .typing import FloatArray, IntArray

PARTICLE_VARS = [
    "pt",
    "eta",
    "phi",
    "pdgId",
    "vx",
    "vy",
    "vz",
    "class",
]


def mock_particles(
    num_events: int = 1000,
    num_particles: int = 40,
    rng: Generator | None = None,
) -> dict[str, list[FloatArray]]:
    if rng is None:
        rng = np.random.default_rng(42)
    particles: dict[str, FloatArray | IntArray] = {}
    for var in PARTICLE_VARS:
        if var == "pdgId":
            value = rng.choice(
                [-11, 11, -13, 13, 211, -211, 111, 130, 22],
                size=(num_events, num_particles),
                replace=True,
            ).astype(np.int32)
        elif var == "class":
            value = np.array(
                [
                    pid_to_class(particles["pdgId"][i, j])
                    for i in range(num_events)
                    for j in range(num_particles)
                ],
                dtype=np.float32,
            ).reshape(num_events, num_particles)
            # value = rng.integers(0, 5, size=(num_events, num_particles)).astype(np.int32)
        else:
            value = rng.random((num_events, num_particles)).astype(np.float32) + 1
        particles[var] = value
    ind = rng.choice([True, False], size=(num_events, num_particles)).astype(bool)
    data: dict[str, list[FloatArray]] = {var: [] for var in PARTICLE_VARS}
    for i in range(num_events):
        for var in PARTICLE_VARS:
            data[var].append(particles[var][i][ind[i]])

    return data


def get_4momentum(
    pt: FloatArray | float,
    y: FloatArray | float,
    phi: FloatArray | float,
    mass: FloatArray | float,
) -> FloatArray:
    mt = np.sqrt(pt**2 + mass**2)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = mt * np.sinh(y)
    e = mt * np.cosh(y)
    return np.array([px, py, pz, e], dtype=np.float32)


def getParticleHepMC(
    pt: FloatArray | float,
    y: FloatArray | float,
    phi: FloatArray | float,
    pid: FloatArray | float,
    status: int = 1,
) -> pyhepmc.GenParticle:
    p = pyhepmc.GenParticle()
    p.momentum = pyhepmc.FourVector(get_4momentum(pt, y, phi, 0))
    p.pid = int(pid)
    p.status = status
    return p


def getVertexHepMC(
    vx: FloatArray | float,
    vy: FloatArray | float,
    vz: FloatArray | float,
) -> pyhepmc.GenVertex:
    v = pyhepmc.GenVertex()
    v.position = pyhepmc.FourVector([vx, vy, vz, 0])
    return v


def getEventHepMC(event_data: list[FloatArray], event_number: int) -> pyhepmc.GenEvent:
    event = pyhepmc.GenEvent()
    vtx_dict: dict[int, pyhepmc.GenVertex] = {}
    for pt, y, phi, pid, vx, vy, vz in zip(*event_data, strict=True):
        particle = getParticleHepMC(pt, y, phi, pid)

        vtx = vtx_dict.get(hash((vx, vy, vz)))
        if vtx is None:
            vtx = getVertexHepMC(vx, vy, vz)
            particle_in = getParticleHepMC(0.0001, 0, 0, 0, 0)
            event.add_particle(particle_in)
            vtx.add_particle_in(particle_in)
            vtx_dict[hash((vx, vy, vz))] = vtx
        vtx.add_particle_out(particle)
        event.add_particle(particle)
    for vtx in vtx_dict.values():
        event.add_vertex(vtx)
    event.event_number = event_number
    return event


def get_mock_root_file(
    num_events: int = 1000,
    fname: str | None = None,
    ttree_name: str = "evt_tree",
    num_particles: int = 40,
) -> str:
    rng = np.random.default_rng(42)
    truth_particles = mock_particles(num_events=num_events, num_particles=num_particles)
    event_numbers = rng.integers(0, num_events * 4, num_events)
    # create a tempfile in a new folder
    if fname is None:
        fname = NamedTemporaryFile(suffix=".root", dir=mkdtemp()).name
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    with uproot.recreate(fname) as f:
        f[ttree_name] = {
            "truth": ak.zip({var: ak.Array(val) for var, val in truth_particles.items()}),
            "eventNumber": event_numbers,
        }

    return fname


def get_mock_hepmc_file(
    num_events: int = 1000,
    fname: str | None = None,
    num_particles: int = 40,
) -> str:
    rng = np.random.default_rng(42)
    truth_particles = mock_particles(num_events=num_events, num_particles=num_particles)
    event_numbers = rng.integers(0, num_events * 4, num_events)
    # create a tempfile in a new folder
    if fname is None:
        fname = NamedTemporaryFile(suffix=".hepmc", dir=mkdtemp()).name
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    events = [
        getEventHepMC(
            [truth_particles[key][i] for key in ["pt", "eta", "phi", "pdgId", "vx", "vy", "vz"]],
            int(event_numbers[i]),
        )
        for i in range(num_events)
    ]
    with pyhepmc.open(fname, "w") as f:
        for event in events:
            f.write(event)
    return fname


def get_mock_pythia_file(
    fname: str | None = None,
) -> str:
    if fname is None:
        fname = NamedTemporaryFile(suffix=".cmnd", dir=mkdtemp()).name
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    with open(fname, "w") as f:
        f.write("! Mock Pythia configuration file\n")
        f.write("Beams:idA = 2212\n")
        f.write("Beams:idB = 2212\n")
        f.write("Beams:eCM = 13000.\n")
        f.write("HardQCD:all = on\n")
        f.write("PhaseSpace:pTHatMin = 20.0\n")
        f.write("Print:quiet = on\n")
        f.write("Random:setSeed = on\n")
        f.write("Random:seed = 42\n")
    return fname


def get_mock_transforms() -> dict[str, VarTransform]:
    """Create mock variable transformations for testing.

    Returns
    -------
    dict[str, VarTransform]
        Dictionary mapping variable names to VarTransform instances.
    """
    var_transform_dict: dict[str, VarTransform] = {}
    for var in [*PARTICLE_VARS, "met_x", "met_y", "ht", "npart"]:
        cfg = VarTransformConfig(name=var if var != "pt" else "ptrel", mean=0, std=1)
        var_transform_dict[cfg.name] = VarTransform(cfg)
    return var_transform_dict


def get_mock_variable_requirements() -> VariableRequirements:
    """Create mock VariableRequirements for testing.

    Returns
    -------
    VariableRequirements
        Mock variable requirements with typical particle physics variables.
    """
    return VariableRequirements(
        truth_vars_to_load=(
            "pt",
            "eta",
            "phi",
            "vx",
            "vy",
            "vz",
            "class",
        ),
        ctxt_vars=(
            "truth_ptrel",
            "truth_eta",
            "truth_phi",
            "truth_vx",
            "truth_vy",
            "truth_vz",
            "truth_class",
        ),
        ctxt_global_vars=(
            "means",
            "truth_ht",
            "truth_met_x",
            "truth_met_y",
            "ntruth",
        ),
    )


@final
class MockModel(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.net = nn.Identity()
        self.mode = mode

    def forward(
        self,
        fs_data: Tensor,
        timestep: Tensor,
        mask: Tensor,
        ctxt_data: Tensor,
        ctxt_global_data: Tensor,
    ) -> Tensor:
        if self.mode == "evt":
            return (
                self.net(fs_data)
                + ctxt_data.sum(dim=(1, 2)).view(-1, 1)
                + timestep.view(-1, 1)
                + ctxt_global_data.sum(dim=-1).view(-1, 1)
                + mask.sum(dim=1).view(-1, 1)
            )
        return (
            self.net(fs_data)
            + ctxt_data.sum(dim=(1, 2)).view(-1, 1, 1)
            + timestep.view(-1, 1, 1)
            + ctxt_global_data.sum(dim=-1).view(-1, 1, 1)
            + mask.sum(dim=(1, 2)).view(-1, 1, 1)
        )


def get_mock_input_data(
    mode: str, num_fs_feats: int, num_ctxt_feats: int = 12, num_global_feats: int = 8
) -> dict[str, Tensor]:
    rng = np.random.default_rng(42)
    BS, L = 2, 400
    assert mode in {"evt", "part"}, "Mode should be either evt or part."
    mask = rng.random((BS, L)) > 0.5 if mode == "evt" else rng.random((BS, L, 2)) > 0.5
    fs_data = rng.random((BS, num_fs_feats)) if mode == "evt" else rng.random((BS, L, num_fs_feats))
    return {
        "fs_data": Tensor(fs_data),
        "ctxt_data": Tensor(rng.random((BS, L, num_ctxt_feats))),
        "mask": Tensor(mask).bool(),
        "ctxt_global_data": Tensor(rng.random((BS, num_global_feats))),
        "timestep": Tensor(rng.random((BS, 1))),
    }


def get_mock_model_file(fname: str | None = None, mode: str = "evt") -> str:
    if fname is None:
        fname = NamedTemporaryFile(suffix=".pt", dir=mkdtemp()).name
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    model = MockModel(mode=mode)
    # For particle mode: fs_vars calculation is:
    # base_vars (4: pt, eta, phi, class) + phi expansion (+1) + class expansion (+4) = 9
    fs_feats = 4 if mode == "evt" else 9
    mock_data = get_mock_input_data(mode=mode, num_fs_feats=fs_feats)
    batch = Dim("batch", min=1, max=2048)
    program = export(
        model,
        (
            mock_data["fs_data"],
            mock_data["timestep"],
            mock_data["mask"],
            mock_data["ctxt_data"],
            mock_data["ctxt_global_data"],
        ),
        dynamic_shapes={
            "fs_data": {0: batch},
            "timestep": {0: batch},
            "mask": {0: batch},
            "ctxt_data": {0: batch},
            "ctxt_global_data": {0: batch},
        },
    )
    save(program, fname)
    return fname

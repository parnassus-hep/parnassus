from dataclasses import dataclass, field
from typing import Protocol, Self, override

import awkward as ak
import numpy as np

from parnassus.utils import class_to_pid_vectorized, pid_to_class
from parnassus.utils.typing import FloatArray, IntArray


@dataclass(slots=True)
class GenParticleCollection:
    """Class storing information about a collection of generic particles.

    This class represents a collection of generic particles
    and provides methods to access and manipulate their properties.

    Attributes
    ----------
    name : str
        Name identifier for the particle collection.
    num_particles : int
        Total number of particles in the collection (automatically computed).
    pt : FloatArray
        Transverse momentum of particles.
    eta : FloatArray
        Pseudorapidity of particles.
    phi : FloatArray
        Azimuthal angle of particles.
    mass : FloatArray | None, optional
        Mass of particles. Defaults to zeros if not provided.
    vx : FloatArray | None, optional
        Vertex x-coordinate.
    vy : FloatArray | None, optional
        Vertex y-coordinate.
    vz : FloatArray | None, optional
        Vertex z-coordinate.
    t  : FloatArray | None, optional
        Particle arrival time of flight.
    d0 : FloatArray | None, optional
        Transverse impact parameter.
    z0 : FloatArray | None, optional
        Longitudinal impact parameter.
    d0_error : FloatArray | None, optional
        Uncertainty in transverse impact parameter.
    z0_error : FloatArray | None, optional
        Uncertainty in longitudinal impact parameter.
    pdg_id : IntArray | None, optional
        PDG (Particle Data Group) identification codes.
    class_id : IntArray | None, optional
        Particle class identifier (derived from pdg_id if available).
    charge : IntArray | None, optional
        Electric charge of particles (derived from pdg_id if available).
    status : IntArray | None, optional
        Status code of particles.
    particle_jet_idx : IntArray | None, optional
        Index mapping particles to jets.
    jet_idx : dict[str, IntArray]
        Dictionary mapping jet algorithm names to particle-jet associations.
    """

    # Properties
    name: str
    num_particles: int = field(init=False)

    # Kinematic properties
    pt: FloatArray
    eta: FloatArray
    phi: FloatArray
    mass: FloatArray | None = None

    # Vertex information
    vx: FloatArray | None = None
    vy: FloatArray | None = None
    vz: FloatArray | None = None
    t: FloatArray | None = None

    # Impact parameters
    d0: FloatArray | None = None
    z0: FloatArray | None = None
    d0_error: FloatArray | None = None
    z0_error: FloatArray | None = None

    # Additional properties
    pdg_id: IntArray | None = None
    class_id: IntArray | None = None
    charge: IntArray | None = None
    status: IntArray | None = None

    # Jet idxs
    particle_jet_idx: IntArray | None = None
    jet_idx: dict[str, IntArray] = field(default_factory=dict)

    def __post_init__(self):
        self.num_particles = len(self.pt)
        if self.mass is None:
            self.mass = np.zeros_like(self.pt)
        if self.pdg_id is not None:
            if self.class_id is None:
                self.class_id = np.array([pid_to_class(el) for el in self.pdg_id], dtype=np.int32)
            if self.charge is None:
                self.charge = np.array([np.sign(el) for el in self.pdg_id], dtype=np.int32)
        if self.pdg_id is None and self.class_id is not None:
            self.pdg_id = class_to_pid_vectorized(self.class_id)
        for key in self.__slots__:
            if key in {"name", "num_particles", "jet_idx"}:
                continue
            attr = self.__getattribute__(key)
            if attr is None:
                continue
            attr_len = len(attr)
            assert attr_len == self.num_particles, (
                f"Assumed length of each features be {self.num_particles}, got"
                f" {attr_len} for {key} attribute"
            )

    def __len__(self):
        return self.num_particles

    @override
    def __repr__(self) -> str:
        return f"{self.name}"

    def get4vecs_numpy(self) -> FloatArray:
        mass = np.zeros_like(self.pt) if self.mass is None else self.mass
        return np.stack(
            [
                self.pt * np.cos(self.phi),
                self.pt * np.sin(self.phi),
                self.pt * np.sinh(self.eta),
                np.sqrt(self.pt**2 * np.cosh(self.eta) ** 2 + mass**2),
            ],
            axis=1,
        )

    def get4vecs_awkward(self) -> ak.Array:
        mass = np.zeros_like(self.pt) if self.mass is None else self.mass
        return ak.Array(
            {
                "px": self.pt * np.cos(self.phi),
                "py": self.pt * np.sin(self.phi),
                "pz": self.pt * np.sinh(self.eta),
                "E": np.sqrt(self.pt**2 * np.cosh(self.eta) ** 2 + mass**2),
            },
            with_name="Momentum4D",
        )

    def __getitem__(self, idx: int):
        assert idx < self.num_particles, f"Index {idx} out of range"


@dataclass(slots=True)
class GenLeptonCollection:
    """Class storing information about a collection of generic leptons.

    This class represents a collection of generic leptons
    and provides methods to access and manipulate their properties.

    Attributes
    ----------
    name : str
        Name identifier for the lepton collection.
    num_particles : int
        Total number of leptons in the collection (automatically computed).
    pt : FloatArray
        Transverse momentum of leptons.
    eta : FloatArray
        Pseudorapidity of leptons.
    phi : FloatArray
        Azimuthal angle of leptons.
    iso_var : FloatArray | None, optional
        Isolation variable of leptons.
    sum_pt : FloatArray | None, optional
        Sum of transverse momentum around the lepton.
    sum_pt_ch : FloatArray | None, optional
        Sum of charged transverse momentum around the lepton.
    sum_pt_neut : FloatArray | None, optional
        Sum of neutral transverse momentum around the lepton.
    """

    # Properties
    name: str
    num_particles: int = field(init=False)

    # Kinematic properties
    pt: FloatArray
    eta: FloatArray
    phi: FloatArray

    # Vertex information
    vx: FloatArray | None = None
    vy: FloatArray | None = None
    vz: FloatArray | None = None
    t: FloatArray | None = None

    # Impact parameters
    d0: FloatArray | None = None
    z0: FloatArray | None = None
    d0_error: FloatArray | None = None
    z0_error: FloatArray | None = None

    # Isolation variables
    iso_var: FloatArray | None = None
    sum_pt: FloatArray | None = None
    sum_pt_ch: FloatArray | None = None
    sum_pt_neut: FloatArray | None = None

    @staticmethod
    def get_class_id(name: str) -> int:
        if name == "electrons":
            return 1
        if name == "muons":
            return 2
        return -1

    @classmethod
    def from_particles(cls, particles: GenParticleCollection, name: str) -> Self:
        assert name in {"electrons", "muons"}, (
            "Can create lepton collection only for electrons (class 1) or muons (class 2),"
            f" got {name}"
        )
        assert particles.class_id is not None, "Expect particles to have class"

        class_mask = cls.get_class_id(name) == particles.class_id

        vertex_attrs = {}
        for key in ["vx", "vy", "vz", "t"]:
            attr = getattr(particles, key)
            if attr is not None:
                vertex_attrs[key] = attr[class_mask]

        impact_attrs = {}
        for key in ["d0", "z0", "d0_error", "z0_error"]:
            attr = getattr(particles, key)
            if attr is not None:
                impact_attrs[key] = attr[class_mask]

        return cls(
            name=name,
            pt=particles.pt[class_mask],
            eta=particles.eta[class_mask],
            phi=particles.phi[class_mask],
            **vertex_attrs,
            **impact_attrs,
        )

    def __post_init__(self):
        self.num_particles = len(self.pt)
        for key in self.__slots__:
            if key in {"name", "num_particles"}:
                continue
            attr = self.__getattribute__(key)
            if attr is None:
                continue
            attr_len = len(attr)
            assert attr_len == self.num_particles, (
                f"Assumed length of each features be {self.num_particles}, got"
                f" {attr_len} for {key} attribute"
            )

    def __len__(self):
        return self.num_particles

    @override
    def __repr__(self) -> str:
        return f"{self.name} collection with {len(self)} elements"

    def __getitem__(self, idx: int):
        assert idx < self.num_particles, f"Index {idx} out of range"


@dataclass(slots=True)
class GenTowerCollection:
    """Class storing information about a collection of towers.

    This class represents a collection of towers
    and provides methods to access and manipulate their properties.

    Attributes
    ----------
    name : str
        Name identifier for the tower collection.
    num_particles : int
        Total number of towers in the collection (automatically computed).
    e : FloatArray
        Energy of towers.
    et: FloatArray
        Transverse energy of towers.
    eta : FloatArray
        Pseudorapidity of towers.
    phi : FloatArray
        Azimuthal angle of towers.
    t: FloatArray | None, optional
        Calo deposit time, averaged by sqrt(EM energy) over all particles.
    """

    # Properties
    name: str
    num: int = field(init=False)
    e: FloatArray
    et: FloatArray
    eta: FloatArray
    phi: FloatArray
    t: FloatArray

    def __post_init__(self):
        self.num = len(self.e)
        for key in self.__slots__:
            if key in {"name", "num"}:
                continue
            attr = self.__getattribute__(key)
            if attr is None:
                continue
            attr_len = len(attr)
            assert attr_len == self.num, (
                f"Assumed length of each features be {self.num}, got {attr_len} for {key} attribute"
            )

    def __len__(self):
        return self.num

    @override
    def __repr__(self) -> str:
        return f"{self.name} collection with {len(self)} elements"

    def __getitem__(self, idx: int):
        assert idx < self.num, f"Index {idx} out of range"


class GenCollection(Protocol):
    """Protocol satisfied by all named, sized particle/tower/jet collections."""

    @property
    def name(self) -> str: ...

    def __len__(self) -> int: ...


@dataclass(slots=True)
class GenJetCollection:
    """Class storing information about generic jet collection.

    We use it to store information about jets inside an event.

    Parameters
    ----------
    name : str
        Name identifier for the jet collection.
    num_jets : int
        Total number of jets in the collection (automatically computed).
    pt : FloatArray
        Transverse momentum of jets.
    eta : FloatArray
        Pseudorapidity of jets.
    phi : FloatArray
        Azimuthal angle of jets.
    mass : FloatArray | None, optional
        Mass of jets.
    jec : FloatArray | None, optional
        Jet energy correction factors.
    d2 : FloatArray | None, optional
        D2 substructure variable.
    c2 : FloatArray | None, optional
        C2 substructure variable.
    """

    # Jet properties
    name: str
    num_jets: int = field(init=False)
    pt: FloatArray
    eta: FloatArray
    phi: FloatArray
    mass: FloatArray | None = None

    jec: FloatArray | None = None
    d2: FloatArray | None = None
    c2: FloatArray | None = None

    def __post_init__(self):
        self.num_jets = len(self.pt)
        for key in self.__slots__:
            if key in {"name", "num_jets"}:
                continue
            attr = self.__getattribute__(key)
            if attr is None:
                continue
            attr_len = len(attr)
            assert attr_len == self.num_jets, (
                f"Assumed length of each features be {self.num_jets}, got"
                f" {attr_len} for {key} attribute"
            )

    def __len__(self):
        return len(self.pt)

    @override
    def __repr__(self) -> str:
        return f"{self.name} with {len(self)} jets"


@dataclass(slots=True)
class GenEvent:
    """Class storing properties of event."""

    # Event properties
    event_number: int

    truth_particles: GenParticleCollection
    pflow_particles: GenParticleCollection

    muons: GenLeptonCollection = field(init=False)
    electrons: GenLeptonCollection = field(init=False)

    jets: dict[str, GenJetCollection] = field(default_factory=dict)

    # Generator-specific collections (e.g. tracks, towers) keyed by collection name
    collections: dict[str, GenCollection] = field(default_factory=dict)

    # Event features
    truth_ht: np.float32 = field(init=False)
    truth_met_x: np.float32 = field(init=False)
    truth_met_y: np.float32 = field(init=False)

    pflow_ht: np.float32 = field(init=False)
    pflow_met_x: np.float32 = field(init=False)
    pflow_met_y: np.float32 = field(init=False)

    def __post_init__(self):
        self.truth_ht = np.sum(self.truth_particles.pt)
        self.truth_met_x = np.sum(self.truth_particles.pt * np.cos(self.truth_particles.phi))
        self.truth_met_y = np.sum(self.truth_particles.pt * np.sin(self.truth_particles.phi))

        self.pflow_ht = np.sum(self.pflow_particles.pt)
        self.pflow_met_x = np.sum(self.pflow_particles.pt * np.cos(self.pflow_particles.phi))
        self.pflow_met_y = np.sum(self.pflow_particles.pt * np.sin(self.pflow_particles.phi))

        self.muons = GenLeptonCollection.from_particles(self.pflow_particles, name="muons")
        self.electrons = GenLeptonCollection.from_particles(self.pflow_particles, name="electrons")

    @override
    def __repr__(self) -> str:
        return (
            f"Event {self.event_number} with {len(self.truth_particles)} truth_particles"
            f", {len(self.pflow_particles)} pflow particles"
            f" and {len(self.jets)} jet collections: {list(self.jets.keys())}"
        )

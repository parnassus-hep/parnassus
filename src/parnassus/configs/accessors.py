from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, final, override

from .scheme import GenEvent


class AccessorError(Exception):
    """Raised when accessor cannot retrieve data from event."""


@dataclass(frozen=True)
class Accessor(ABC):
    """Abstract base class for accessors."""

    name: str
    collection: str
    output_name: str = ""
    dtype: str = "float32"

    def __post_init__(self):
        # Validate dtype
        valid_dtypes = {"float32", "float64", "int32", "int64", "bool"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        # Set output_name if not provided by user
        if not self.output_name:
            object.__setattr__(self, "output_name", self.name)

    @abstractmethod
    def get(self, event: GenEvent) -> Any:
        pass

    @override
    def __repr__(self) -> str:
        return f"{self.collection} '{self.name}' accessor, output_name: {self.output_name}"


@final
@dataclass(frozen=True)
class ParticleAccessor(Accessor):
    """Accessor for named particle collections on GenEvent (e.g. truth_particles, electrons)."""

    @override
    def get(self, event: GenEvent):
        try:
            collection = getattr(event, self.collection)
        except AttributeError as e:
            raise AccessorError(f"Event has no collection '{self.collection}'") from e

        if "/" in self.name:
            dict_name, field = self.name.split("/", 1)
            try:
                data_dict = getattr(collection, dict_name)
                return data_dict[field]
            except (AttributeError, KeyError, TypeError) as e:
                raise AccessorError(f"Cannot access {self.name} from {self.collection}") from e

        try:
            return getattr(collection, self.name)
        except AttributeError as e:
            raise AccessorError(
                f"Collection {self.collection} has no attribute '{self.name}'"
            ) from e


@final
@dataclass(frozen=True)
class CollectionAccessor(Accessor):
    """Accessor for generator-specific collections stored in GenEvent.collections."""

    @override
    def get(self, event: GenEvent):
        try:
            collection = event.collections[self.collection]
        except KeyError as e:
            raise AccessorError(f"Event has no entry '{self.collection}' in collections") from e

        try:
            return getattr(collection, self.name)
        except AttributeError as e:
            raise AccessorError(
                f"Collection '{self.collection}' has no attribute '{self.name}'"
            ) from e


@final
@dataclass(frozen=True)
class JetAccessor(Accessor):
    """Accessor for jet collections stored in GenEvent.jets."""

    @override
    def get(self, event: GenEvent):
        return getattr(event.jets[self.collection], self.name)


# Accessor creation #
@dataclass
class AccessorSpec:
    """Specification for creating an accessor (without collection)."""

    name: str
    output_name: str = ""
    dtype: str = "float32"


class AccessorListBuilder:
    """Fluent builder for creating lists of accessors.

    Examples
    --------
    Simple case:

    >>> accessors = (
    ...     AccessorListBuilder.for_particles("electrons")
    ...     .add(["pt", "eta", "phi"])
    ...     .build()
    ... )

    With custom dtypes:

    >>> accessors = (
    ...     AccessorListBuilder.for_particles("pflow_particles")
    ...     .add(["pt", "eta", "phi"], dtype="float32")
    ...     .add(["class_id"], dtype="int32")
    ...     .build()
    ... )

    With output names:

    >>> accessors = (
    ...     AccessorListBuilder.for_particles("electrons")
    ...     .add_with_output("iso_var", "electron_iso")
    ...     .add(["sum_pt", "sum_pt_ch"])
    ...     .build()
    ... )

    From specs (for reusable templates):

    >>> specs = [
    ...     AccessorSpec("pt"),
    ...     AccessorSpec("eta"),
    ...     AccessorSpec("phi"),
    ... ]
    >>> accessors = (
    ...     AccessorListBuilder.for_particles("electrons")
    ...     .add_from_specs(specs)
    ...     .build()
    ... )

    """

    def __init__(self, collection: str, accessor_type: type[Accessor]):
        self._collection = collection
        self._accessor_type = accessor_type
        self._specs: list[AccessorSpec] = []

    @classmethod
    def for_particles(cls, collection: str) -> "AccessorListBuilder":
        """Create builder for particle accessors.

        Parameters
        ----------
            collection: Name of the particle collection

        Returns
        -------
            AccessorListBuilder for chaining
        """
        return cls(collection, ParticleAccessor)

    @classmethod
    def for_collection(cls, collection: str) -> "AccessorListBuilder":
        """Create builder for accessors targeting GenEvent.collections.

        Parameters
        ----------
            collection: Key in GenEvent.collections

        Returns
        -------
            AccessorListBuilder for chaining
        """
        return cls(collection, CollectionAccessor)

    @classmethod
    def for_jets(cls, collection: str) -> "AccessorListBuilder":
        """Create builder for jet accessors.

        Parameters
        ----------
            collection: Name of the jet collection

        Returns
        -------
            AccessorListBuilder for chaining
        """
        return cls(collection, JetAccessor)

    def add(
        self, names: Sequence[str], dtype: str | Sequence[str] = "float32"
    ) -> "AccessorListBuilder":
        """Add accessors by name with optional dtype.

        Parameters
        ----------
            names: Sequence[str]
                Variable names to create accessors for
            dtype: str | Sequence[str]
                Data type for all these accessors (default: "float32")
                If a sequence is provided, it must match the length of `names`.

        Returns
        -------
            AccessorListBuilder for chaining
        """
        if isinstance(dtype, str):
            dtype = [dtype] * len(names)
        if len(dtype) != len(names):
            raise ValueError("Length of dtype list must match length of names list.")
        for name, dtype_ in zip(names, dtype, strict=False):
            self._specs.append(AccessorSpec(name=name, dtype=dtype_))
        return self

    def add_with_output(
        self, name: str, output_name: str, dtype: str = "float32"
    ) -> "AccessorListBuilder":
        """Add accessor with custom output name.

        Parameters
        ----------
            name: Variable name in event
            output_name: Name to use in output file
            dtype: Data type (default: "float32")

        Returns
        -------
            AccessorListBuilder for chaining
        """
        self._specs.append(AccessorSpec(name=name, output_name=output_name, dtype=dtype))
        return self

    def add_from_specs(self, specs: Sequence[AccessorSpec]) -> "AccessorListBuilder":
        """Add accessors from pre-defined specs.

        Parameters
        ----------
            specs: Sequence of AccessorSpec objects

        Returns
        -------
            Self for chaining
        """
        self._specs.extend(specs)
        return self

    def build(self) -> list[Accessor]:
        """Build the final list of accessors.

        Returns
        -------
            List of configured accessor instances
        """
        return [
            self._accessor_type(
                name=spec.name,
                collection=self._collection,
                output_name=spec.output_name,
                dtype=spec.dtype,
            )
            for spec in self._specs
        ]


class AccessorTemplates:
    """Common accessor specifications for reuse across pipelines."""

    # Particle kinematics
    KINEMATICS: ClassVar = [
        AccessorSpec("pt", output_name="PT"),
        AccessorSpec("eta", output_name="Eta"),
        AccessorSpec("phi", output_name="Phi"),
    ]

    IMPACT_PARAMETERS: ClassVar = [
        AccessorSpec("d0", output_name="D0"),
        AccessorSpec("z0", output_name="Z0"),
        AccessorSpec("d0_error", output_name="ErrorD0"),
        AccessorSpec("z0_error", output_name="ErrorZ0"),
    ]

    # Full particle info
    FULL_PARTICLE: ClassVar = [
        *KINEMATICS,
        AccessorSpec("vx", output_name="X"),
        AccessorSpec("vy", output_name="Y"),
        AccessorSpec("vz", output_name="Z"),
        AccessorSpec("class_id", output_name="ClassID", dtype="int32"),
        AccessorSpec("pdg_id", output_name="PID", dtype="int32"),
    ]

    # Isolation variables
    ISOLATION: ClassVar = [
        AccessorSpec("iso_var", output_name="IsolationVar"),
        AccessorSpec("sum_pt", output_name="SumPt"),
        AccessorSpec("sum_pt_ch", output_name="SumPtCharged"),
        AccessorSpec("sum_pt_neut", output_name="SumPtNeutral"),
    ]

    # Jet substructure
    JET_SUBSTRUCTURE: ClassVar = [
        AccessorSpec("d2", output_name="D2"),
        AccessorSpec("c2", output_name="C2"),
    ]

    # Particle classification
    CLASSIFICATION: ClassVar = [
        AccessorSpec("class_id", output_name="ClassID", dtype="int32"),
        AccessorSpec("charge", output_name="Charge", dtype="int32"),
    ]


@dataclass(slots=True)
class AccessorStore:
    """Store for accessors grouped by collection name."""

    accessors_dict: dict[str, list[Accessor]] = field(default_factory=dict)

    @override
    def __repr__(self) -> str:
        repr_string = ""
        row = "{:<40} | {:^40} | {:>40}"
        headers = ["Input collection", "Accessor name", "Output name"]
        for collection, accessors in self.accessors_dict.items():
            repr_string += f"{collection:=^126}\n"
            repr_string += "-" * 126 + "\n"
            repr_string += row.format(*headers) + "\n"
            repr_string += "-" * 126 + "\n"
            for accessor in accessors:
                repr_string += (
                    row.format(accessor.collection, accessor.name, accessor.output_name) + "\n"
                )
            repr_string += "=" * 126 + "\n\n"
        return repr_string

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[Accessor]]) -> "AccessorStore":
        accessors_dict = {key: list(accessors) for key, accessors in data.items()}
        return cls(accessors_dict)

    def update_from_dict(self, data: Mapping[str, Sequence[Accessor]]):
        for key, accessors in data.items():
            if key not in self.accessors_dict:
                self.accessors_dict[key] = list(accessors)
            else:
                for accessor in accessors:
                    if accessor in self.accessors_dict[key]:
                        continue
                    self.accessors_dict[key].append(accessor)

    def get_branch_types(self) -> dict[str, str]:
        return {
            name: "var * {"
            + ", ".join([f'"{accessor.output_name}" : {accessor.dtype}' for accessor in accessors])
            + "}"
            for name, accessors in self.accessors_dict.items()
        }

    def init_data_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {accessor.output_name: [] for accessor in accessors}
            for name, accessors in self.accessors_dict.items()
        }

    def update_data_dict(self, event: GenEvent, data_dict: dict[str, dict[str, Any]]):
        for name, accessors in self.accessors_dict.items():
            for accessor in accessors:
                data_dict[name][accessor.output_name].append(accessor.get(event))

"""Configuration for Delphes-style pile-up merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DelphesPileUpConfig:
    """Configuration for Delphes-style pile-up merging.

    Parameters
    ----------
    file_path : str
        Path to preprocessed MinBias ``.pileup`` file (Delphes XDR format).
    mean_pileup : float
        Average number of pile-up interactions per bunch crossing.
    max_z_spread : float
        Truncation bound for vertex z-smearing in meters.
    max_t_spread : float
        Truncation bound for vertex t-smearing in seconds.
    sigma_z : float
        Gaussian sigma for vertex z-smearing in meters.
    sigma_t : float
        Gaussian sigma for vertex t-smearing in seconds.
    smear_hs_vertex : bool
        Whether to also smear the hard-scatter vertex (matching Delphes PileUpMerger).
    """

    file_path: str
    mean_pileup: float
    max_z_spread: float = 0.25
    max_t_spread: float = 800e-12
    sigma_z: float = 0.053
    sigma_t: float = 160e-12
    smear_hs_vertex: bool = True

    def __post_init__(self):
        self.mean_pileup = float(self.mean_pileup)
        self.max_z_spread = float(self.max_z_spread)
        self.max_t_spread = float(self.max_t_spread)
        self.sigma_z = float(self.sigma_z)
        self.sigma_t = float(self.sigma_t)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DelphesPileUpConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

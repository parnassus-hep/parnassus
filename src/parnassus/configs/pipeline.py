from dataclasses import dataclass, field
from typing import Any, Self

from fastjet import JetDefinition, antikt_algorithm, ee_genkt_algorithm


@dataclass(slots=True)
class GenPipelineConfig:
    """Base configuration for generator pipelines.

    Parameters
    ----------
    name : str
        Name of the pipeline.
    batch_size : int, optional
        Number of events to process in a single batch, by default 2000.
    """

    name: str
    batch_size: int = 2000

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> Self:
        return cls(name, **{field: config[field] for field in cls.__slots__ if field in config})


@dataclass(slots=True)
class JetClusteringConfig(GenPipelineConfig):
    """Configuration for jet clustering pipeline.

    Parameters
    ----------
    name : str
        Name of the clustering pipeline.
    algorithm : str, optional
        Jet clustering algorithm to use ("antikt" or "genkt"), by default "antikt".
    collection : str, optional
        Particle collection to cluster ("pflow" or "truth"), by default "pflow".
    dr : float, optional
        Jet radius parameter, by default 0.5.
    nconst_min : int, optional
        Minimum number of constituents for a jet to be kept, by default 2.
    pt_min : float, optional
        Minimum jet transverse momentum to be kept, by default 0.
    num_processes : int, optional
        Number of processes for parallel execution, by default 1.
    redirect_stdout : bool, optional
        Whether to redirect stdout during clustering, by default True.
        Used to suppress FastJet output.
    """

    algorithm: str = "antikt"
    collection: str = "pflow"
    dr: float = 0.5
    nconst_min: int = 2
    pt_min: float = 0
    num_processes: int = 1

    redirect_stdout: bool = True

    jet_definition: JetDefinition = field(init=False)

    def __post_init__(self):
        if self.algorithm == "genkt":
            self.jet_definition = JetDefinition(ee_genkt_algorithm, self.dr, -1.0)
        elif self.algorithm == "antikt":
            self.jet_definition = JetDefinition(antikt_algorithm, self.dr)
        else:
            raise NotImplementedError(f"Jet algorithm {self.algorithm} is not supported!")
        if self.collection not in {"pflow", "truth"}:
            raise ValueError(
                f'Requested clustering {self.collection}, only "pflow" and "truth" are supported.'
            )


@dataclass(slots=True)
class IsolationConfig(GenPipelineConfig):
    """Configuration for lepton isolation pipeline.

    Parameters
    ----------
    collection: str
        Particle collection to calculate isolation for ("electrons", "muons", or "all").
    dr: float
        Delta R cone size for isolation calculation, by default 0.4.
    num_processes: int
        Number of processes for parallel execution, by default 1.
    """

    collection: str = "electrons"
    dr: float = 0.4
    num_processes: int = 1

    def __post_init__(self):
        if self.collection not in {"electrons", "muons", "all"}:
            raise ValueError(
                f"Requested isolation for {self.collection}, "
                'only "electrons" and "muons", and "all" (both of them) are supported.'
            )
        if self.dr <= 0:
            raise ValueError(f'Only positive values of "dR" are supported, asked for {self.dr}')


def get_pipeline_config(name: str, config: dict[str, Any]) -> GenPipelineConfig:
    """Factory function to create pipeline configuration objects.

    Parameters
    ----------
    name : str
        Name of the pipeline.
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    GenPipelineConfig
        Pipeline configuration object.
    """
    match config["type"]:
        case "cluster":
            return JetClusteringConfig.from_dict(name=name, config=config)
        case "isolation":
            return IsolationConfig.from_dict(name=name, config=config)
        case _:
            return GenPipelineConfig(name=name)

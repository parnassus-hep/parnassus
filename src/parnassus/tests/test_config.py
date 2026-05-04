from pathlib import Path

from parnassus.configs import Config
from parnassus.configs.pipeline import IsolationConfig, JetClusteringConfig, get_pipeline_config

DEFAULT_CONFIG_PATH = Path(__file__).cwd().joinpath("src/parnassus/configs/neural_config.yaml")


def test_config_from_yaml():
    _ = Config.from_yaml(DEFAULT_CONFIG_PATH)


def test_pipeline_execution_controls_are_loaded():
    """Pipeline configs load shared batch_size and type-specific num_processes."""
    cluster = get_pipeline_config(
        "Jets",
        {"type": "cluster", "batch_size": 17, "num_processes": 3},
    )
    isolation = get_pipeline_config(
        "ElectronIsolation",
        {"type": "isolation", "batch_size": 19, "num_processes": 4},
    )

    assert isinstance(cluster, JetClusteringConfig)
    assert cluster.batch_size == 17
    assert cluster.num_processes == 3
    assert isinstance(isolation, IsolationConfig)
    assert isolation.batch_size == 19
    assert isolation.num_processes == 4

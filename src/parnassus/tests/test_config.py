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


def test_particle_filtering_config_from_dict():
    from parnassus.configs.pipeline import (
        FilterCondition,
        ParticleFilteringConfig,
        get_pipeline_config,
    )

    cfg = get_pipeline_config(
        "TruthFilter",
        {
            "type": "filter",
            "collection": "truth",
            "combine": "all",
            "conditions": [
                {"field": "pt", "op": ">", "value": 0.5},
                {"field": "eta", "op": "<=", "value": 2.5, "abs": True},
                {"field": "pdg_id", "op": "not in", "value": [12, 14, 16]},
            ],
        },
    )

    assert isinstance(cfg, ParticleFilteringConfig)
    assert cfg.name == "TruthFilter"
    assert cfg.collection == "truth"
    assert cfg.combine == "all"
    assert len(cfg.conditions) == 3
    assert all(isinstance(c, FilterCondition) for c in cfg.conditions)
    assert cfg.conditions[1].abs is True
    assert cfg.conditions[2].value == [12, 14, 16]


def test_particle_filtering_config_validates_op():
    import pytest

    from parnassus.configs.pipeline import ParticleFilteringConfig

    with pytest.raises(ValueError, match="Unsupported filter op"):
        ParticleFilteringConfig(
            name="bad",
            conditions=[{"field": "pt", "op": "=>", "value": 1}],  # type: ignore[list-item]
        )


def test_particle_filtering_config_in_op_requires_list():
    import pytest

    from parnassus.configs.pipeline import ParticleFilteringConfig

    with pytest.raises(ValueError, match="requires a list value"):
        ParticleFilteringConfig(
            name="bad",
            conditions=[{"field": "pdg_id", "op": "in", "value": 12}],  # type: ignore[list-item]
        )


def test_particle_filtering_config_validates_combine():
    import pytest

    from parnassus.configs.pipeline import ParticleFilteringConfig

    with pytest.raises(ValueError, match="combine"):
        ParticleFilteringConfig(name="bad", combine="xor")

from pathlib import Path

from parnassus.configs import Config

DEFAULT_CONFIG_PATH = Path(__file__).cwd().joinpath("src/parnassus/configs/neural_config.yaml")


def test_config_from_yaml():
    _ = Config.from_yaml(DEFAULT_CONFIG_PATH)

from pathlib import Path

import numpy as np
import pytest
import uproot

from parnassus.main import main


def test_main_help(capsys):
    """Test the help message of the main function."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out


def test_main_init(tmp_path):
    """Test the init command of the main function."""
    main(["init", tmp_path.as_posix()])
    assert (tmp_path / "neural_config.yaml").exists()


def test_main_run_with_invalid_config():
    """Test the run command of the main function with an invalid config path."""
    with pytest.raises(FileNotFoundError):
        main(["run", "-c", "non_existent_config.yaml"])


EXPECTED_FILE_MAP = {
    "neural_config.yaml": "expected_results/h4lep_test_neural.root",
    "parametric_config.yaml": "expected_results/h4lep_test_parametric.root",
}


@pytest.mark.parametrize("ds_file", ["h4lep_test_100.hepmc", "HZZ4l.cmnd"])
@pytest.mark.parametrize("config_file", ["neural_config.yaml", "parametric_config.yaml"])
def test_main_run(tmp_path, ds_file: str, config_file: str):
    output_path = Path(tmp_path) / "test.root"
    current_dir = Path(__file__).parent
    args = [
        "run",
        "-c",
        (current_dir.parent / "configs" / config_file).as_posix(),
        "-i",
        (current_dir / ds_file).as_posix(),
        "-ne",
        "4",
        "-bs",
        "2",
        "-o",
        output_path.as_posix(),
        "--random_seed",
        "42",
    ]
    main(args)
    assert output_path.exists(), "Output file was not created"
    assert output_path.stat().st_size > 0, "Output file is empty"

    # Verify that the output ROOT file contains expected trees
    with uproot.open(output_path) as file:
        generated = file["Parnassus"].arrays(library="np")
    with uproot.open(current_dir / EXPECTED_FILE_MAP[config_file]) as expected_file:
        expected = expected_file["Parnassus"].arrays(library="np")
    assert set(generated.keys()) == set(expected.keys()), "Output keys do not match expected keys"
    if ds_file == "h4lep_test_100.hepmc":
        for key in expected:
            if "Truth" in key:
                for i in range(4):
                    np.testing.assert_allclose(
                        generated[key][i],
                        expected[key][i],
                        err_msg=f"Mismatch in key '{key}' for event {i}",
                        rtol=1e-6,
                    )
            if config_file == "parametric_config.yaml":
                for i in range(4):
                    np.testing.assert_allclose(
                        generated[key][i],
                        expected[key][i],
                        err_msg=f"Mismatch in key '{key}' for event {i}",
                        rtol=1e-6,
                    )

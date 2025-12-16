import numpy as np
import pytest

from parnassus.utils import class_to_pid, class_to_pid_vectorized, pid_to_class, reshape_phi


@pytest.mark.parametrize(
    ("pid", "expected_class"),
    [
        # Electrons/positrons (class 1)
        (11, 1),
        (-11, 1),
        # Muons/antimuons (class 2)
        (13, 2),
        (-13, 2),
        # Charged hadrons (class 0)
        (211, 0),  # π+
        (-211, 0),  # π-
        (321, 0),  # K+
        (-321, 0),  # K-
        (2212, 0),  # proton
        (-2212, 0),  # antiproton
        # Neutral hadrons (class 3)
        (111, 3),  # π0
        (311, 3),  # K0
        (2112, 3),  # neutron
        # Other particles (class 4)
        (22, 4),  # photon
        (12, 4),  # electron neutrino
        (-12, 4),  # electron antineutrino
        (14, 4),  # muon neutrino
        (23, 4),  # Z boson
    ],
)
def test_pid_to_class(pid: int, expected_class: int):
    """Test particle ID to class conversion for various particles."""
    assert pid_to_class(pid) == expected_class


def test_pid_to_class_charged_hadrons():
    """Test specifically charged hadrons classification."""
    for pid in [211, -211, 321, -321, 2212, -2212]:
        assert pid_to_class(pid) == 0, f"Failed for charged hadron PID {pid}"


def test_pid_to_class_leptons():
    """Test electron and muon classification."""
    # Test electrons
    assert pid_to_class(11) == 1
    assert pid_to_class(-11) == 1

    # Test muons
    assert pid_to_class(13) == 2
    assert pid_to_class(-13) == 2


def test_pid_to_class_neutral_hadrons():
    """Test neutral hadrons classification."""
    neutral_hadrons = [111, 311, 2112]  # π0, K0, neutron
    for pid in neutral_hadrons:
        assert pid_to_class(pid) == 3, f"Failed for neutral hadron PID {pid}"


def test_pid_to_class_other_particles():
    """Test classification of non-hadron, non-lepton particles."""
    other_particles = [22, 12, -12, 23]  # photon, neutrinos, Z boson
    for pid in other_particles:
        assert pid_to_class(pid) == 4, f"Failed for other particle PID {pid}"


def test_class_to_pid():
    """Test class to PID conversion."""
    # Test charged hadrons (class 0)
    assert class_to_pid(0) == 211  # π+

    # Test electrons (class 1)
    assert class_to_pid(1) == 11

    # Test muons (class 2)
    assert class_to_pid(2) == 13

    # Test neutral hadrons (class 3)
    assert class_to_pid(3) == 111  # π0

    # Test other particles (class 4)
    assert class_to_pid(4) == 22  # photon


def test_class_to_pid_vectorized():
    """Test vectorized class to PID conversion."""
    particle_classes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    pids = class_to_pid_vectorized(particle_classes)
    assert pids.tolist() == [211, 11, 13, 111, 22]


def test_reshape_phi():
    rng = np.random.default_rng(42)
    phi = rng.uniform(-4 * np.pi, 4 * np.pi, size=(20, 300)).astype(np.float32)
    new_phi = reshape_phi(phi)
    assert np.all(new_phi >= -np.pi)
    assert np.all(new_phi < np.pi)
    np.testing.assert_allclose(np.sin(new_phi), np.sin(phi), rtol=1e-3)

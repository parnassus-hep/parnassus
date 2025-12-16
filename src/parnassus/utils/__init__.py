"""Utility functions and classes for Parnassus."""

import numpy as np
from particle import PDGID

from .transform import TransformRegistry, Unscaler, VarTransform, VarTransformConfig
from .typing import FloatArray, IntArray


def reshape_phi(phi: FloatArray | IntArray) -> FloatArray:
    """Reshape phi angles to be within the range [-pi, pi].

    Parameters
    ----------
    phi : FloatArray | IntArray
        Input array of phi angles.

    Returns
    -------
    FloatArray
        Reshaped array of phi angles within [-pi, pi].
    """
    return np.arctan2(np.sin(phi), np.cos(phi))


def pid_to_class(pid: int) -> int:
    if abs(pid) == 11:
        return 1
    if abs(pid) == 13:
        return 2
    p = PDGID(pid)
    if p.is_hadron:
        if p.charge != 0:
            return 0
        return 3
    return 4


def class_to_pid(particle_class: int) -> int:
    if particle_class == 0:
        return 211
    if particle_class == 1:
        return 11
    if particle_class == 2:
        return 13
    if particle_class == 3:
        return 111
    return 22


def class_to_pid_vectorized(particle_class: IntArray) -> IntArray:
    pid = np.ones_like(particle_class) * 22
    pid[particle_class == 0] = 211
    pid[particle_class == 1] = 11
    pid[particle_class == 2] = 13
    pid[particle_class == 3] = 111
    return pid


def calculate_dr(
    src_eta: FloatArray, src_phi: FloatArray, dst_eta: FloatArray, dst_phi: FloatArray
) -> FloatArray:
    delta_eta = src_eta - dst_eta
    delta_phi = (src_phi - dst_phi + np.pi) % (2 * np.pi) - np.pi

    return np.sqrt(delta_eta**2 + delta_phi**2)


__all__ = [
    "TransformRegistry",
    "Unscaler",
    "VarTransform",
    "VarTransformConfig",
    "reshape_phi",
]

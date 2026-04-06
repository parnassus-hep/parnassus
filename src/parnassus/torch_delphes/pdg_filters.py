"""PDG ID filtering utilities for TorchDelphes modules.

Provides functions to filter particles based on their PDG IDs and charge.
These functions are used by ParticlePropagator, Efficiency, and other modules
to separate particles into different categories.
"""

import torch

from parnassus.data.particle_io import ColumnMap


def charged_hadron_filter(particles: torch.Tensor) -> torch.Tensor:
    """Filter charged hadrons based on PDG IDs.

    Charged hadrons are defined as charged particles that are NOT electrons or muons.
    This includes: charged pions, kaons, protons, etc.

    Parameters
    ----------
    particles: torch.Tensor
        Tensor of shape (N, N_FEATURES) or (B, N, N_FEATURES)
        Must contain PID and CHARGE columns

    Returns
    -------
    mask: torch.Tensor
        Boolean tensor indicating charged hadrons
        Same shape as particles[..., 0]

    Examples
    --------
    >>> particles = torch.randn(100, 25)
    >>> mask = charged_hadron_filter(particles)
    >>> charged_hadrons = particles[mask]
    """
    pid = particles[..., ColumnMap.PID]
    q_final = particles[..., ColumnMap.CHARGE]
    abs_pid = torch.abs(pid)
    is_charged = torch.abs(q_final) > 1.0e-9

    # Exclude electrons (|PDG| = 11) and muons (|PDG| = 13)
    electron_mask = (abs_pid == 11) & is_charged
    muon_mask = (abs_pid == 13) & is_charged
    pid_mask = is_charged & ~electron_mask & ~muon_mask

    return pid_mask


def electron_filter(particles: torch.Tensor) -> torch.Tensor:
    """Filter electrons based on PDG IDs.

    Electrons and positrons have PDG ID = ±11.

    Parameters
    ----------
    particles: torch.Tensor
        Tensor of shape (N, N_FEATURES) or (B, N, N_FEATURES)
        Must contain PID and CHARGE columns

    Returns
    -------
    mask: torch.Tensor
            Boolean tensor indicating electrons/positrons
            Same shape as particles[..., 0]

    Examples
    --------
    >>> particles = torch.randn(100, 25)
    >>> mask = electron_filter(particles)
    >>> electrons = particles[mask]
    """
    pid = particles[..., ColumnMap.PID]
    q_final = particles[..., ColumnMap.CHARGE]
    abs_pid = torch.abs(pid)
    is_charged = torch.abs(q_final) > 1.0e-9

    pid_mask = (abs_pid == 11) & is_charged

    return pid_mask


def muon_filter(particles: torch.Tensor) -> torch.Tensor:
    """Filter muons based on PDG IDs.

    Muons and antimuons have PDG ID = ±13.

    Parameters
    ----------
    particles: torch.Tensor
        Tensor of shape (N, N_FEATURES) or (B, N, N_FEATURES)
        Must contain PID and CHARGE columns

    Returns
    -------
    mask: torch.Tensor
        Boolean tensor indicating muons/antimuons
        Same shape as particles[..., 0]

    Examples
    --------
    >>> particles = torch.randn(100, 25)
    >>> mask = muon_filter(particles)
    >>> muons = particles[mask]
    """
    pid = particles[..., ColumnMap.PID]
    q_final = particles[..., ColumnMap.CHARGE]
    abs_pid = torch.abs(pid)
    is_charged = torch.abs(q_final) > 1.0e-9

    pid_mask = (abs_pid == 13) & is_charged

    return pid_mask


def neutral_filter(particles: torch.Tensor) -> torch.Tensor:
    """Filter neutral particles based on charge.

    Neutral particles have |charge| < 1e-9 (accounts for floating point precision).
    This includes: photons, neutrons, neutrinos, neutral kaons, etc.

    Parameters
    ----------
    particles: torch.Tensor
        Tensor of shape (N, N_FEATURES) or (B, N, N_FEATURES)
        Must contain CHARGE column.

    Returns
    -------
    mask: torch.Tensor
        Boolean tensor indicating neutral particles
        Same shape as particles[..., 0]

    Examples
    --------
    >>> particles = torch.randn(100, 25)
    >>> mask = neutral_filter(particles)
    >>> neutrals = particles[mask]
    """
    q_final = particles[..., ColumnMap.CHARGE]
    is_charged = torch.abs(q_final) > 1.0e-9

    pid_mask = ~is_charged

    return pid_mask


# Convenience function for getting multiple filters at once
def get_particle_masks(particles: torch.Tensor) -> dict[str, torch.Tensor]:
    """Get all particle type masks at once.

    This is more efficient than calling individual filter functions
    when you need multiple masks, as it computes shared quantities only once.

    Parameters
    ----------
    particles: torch.Tensor
        Tensor of shape (N, N_FEATURES) or (B, N, N_FEATURES)
        Must contain PID and CHARGE columns

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing masks for each particle type:
        - 'charged_hadron': Charged hadrons (not e/μ)
        - 'electron': Electrons and positrons
        - 'muon': Muons and antimuons
        - 'neutral': Neutral particles
        - 'charged': All charged particles

    Examples
    --------
    >>> particles = torch.randn(100, 25)
    >>> masks = get_particle_masks(particles)
    >>> print(f"Found {masks['electron'].sum()} electrons")
    """
    pid = particles[..., ColumnMap.PID]
    q_final = particles[..., ColumnMap.CHARGE]
    abs_pid = torch.abs(pid)
    is_charged = torch.abs(q_final) > 1.0e-9

    electron_mask = (abs_pid == 11) & is_charged
    muon_mask = (abs_pid == 13) & is_charged
    charged_hadron_mask = is_charged & ~electron_mask & ~muon_mask
    neutral_mask = ~is_charged

    return {
        "charged_hadron": charged_hadron_mask,
        "electron": electron_mask,
        "muon": muon_mask,
        "neutral": neutral_mask,
        "charged": is_charged,
    }

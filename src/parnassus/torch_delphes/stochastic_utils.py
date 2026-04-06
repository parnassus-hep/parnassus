"""Stochastic sampling utilities for TorchDelphes modules.

Provides functions for sampling from probability distributions used in
detector simulation, including log-normal sampling for resolution smearing.
These functions maintain gradient flow for differentiable detector simulation.
"""

import torch


def log_normal_sample(mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Sample from a log-normal distribution ensuring positive values.

    This function is used for momentum and energy resolution smearing throughout
    TorchDelphes. The log-normal distribution ensures that smeared values remain
    positive (important for PT and energy), while maintaining gradient flow for
    differentiable detector simulation.


    Mathematical background:
    For a log-normal distribution with desired mean mu and std sigma:
    - Let Y ~ LogNormal(m, s^2) where m and s are the log-space parameters
    - Then: E[Y] = exp(m + s^2/2) = mu
    -       Var[Y] = (exp(s^2) - 1) x exp(2m + s^2) = sigma^2
    - Solving: s^2 = ln(1 + (sigma/mu)^2)
    -          m = ln(mu) - s^2/2
    - Sample: Y = exp(m + s x Z) where Z ~ N(0,1)

    Parameters
    ----------
    mean: torch.Tensor
        Mean value (original PT or energy).
        Shape: any tensor shape.
        Values should be positive for physical quantities.
    sigma: torch.Tensor
        Standard deviation (resolution uncertainty).
        Shape: same as mean.

    Returns
    -------
    sample: torch.Tensor
        Sampled values from log-normal distribution
        Shape: same as input
        For invalid inputs (mean ≤ 0), returns zero

    Notes
    -----
    - Uses small epsilon (1e-10) to avoid log(0) errors
    - For mean ≤ 0, returns 0 (physically invalid values)
    - Fully differentiable for gradient-based optimization
    - Device-agnostic (works on CPU and GPU)

    Used by:
    - MomentumSmearing: PT resolution smearing
    - SimpleCalorimeter.Tower: Energy resolution smearing
    - SimpleCalorimeter.EFlowTrack: Energy flow resolution
    - SimpleCalorimeter.EFlowPhoton: Photon energy resolution

    Examples
    --------
    >>> # Smear particle momenta
    >>> pt = torch.tensor([10.0, 50.0, 100.0])
    >>> resolution = torch.tensor([0.5, 1.0, 2.0])
    >>> smeared_pt = log_normal_sample(pt, resolution)

    >>> # Works with batched data
    >>> pt_batch = torch.rand(100, 500) * 100  # (batch, particles)
    >>> res_batch = torch.rand(100, 500) * 5
    >>> smeared = log_normal_sample(pt_batch, res_batch)

    >>> # Maintains gradients
    >>> pt = torch.tensor([10.0], requires_grad=True)
    >>> sigma = torch.tensor([1.0])
    >>> smeared = log_normal_sample(pt, sigma)
    >>> loss = (smeared - 12.0)**2
    >>> loss.backward()
    >>> print(pt.grad)  # Gradient flows through!

    """
    # Identify valid inputs (positive mean values)
    mask_positive = mean > 0.0

    # Compute log-normal parameters for all elements
    # Add epsilon to avoid log(0) - torch.where will handle selection
    s_squared = torch.log(1.0 + (sigma / (mean + 1e-10)) ** 2)
    s = torch.sqrt(s_squared)
    mu = torch.log(mean + 1e-10) - 0.5 * s_squared

    # Sample from standard normal and transform to log-normal
    z = torch.randn_like(mean)
    sample = torch.exp(mu + s * z)

    # Return sampled value for valid inputs, zero for invalid
    # Using torch.where maintains gradient flow through all operations
    result = torch.where(mask_positive, sample, torch.zeros_like(mean))

    return result

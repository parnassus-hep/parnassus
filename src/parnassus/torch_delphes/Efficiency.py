"""PyTorch implementation of Delphes Efficiency module.

Applies tracking efficiency based on particle kinematics (pT, η). The efficiency
determines the probability that a particle is successfully reconstructed as a track.

This module supports different efficiency formulas for:
- Charged hadrons (pions, kaons, protons)
- Electrons
- Muons

The efficiency is applied stochastically: particles pass with probability equal
to their computed efficiency value.

Reference:
    C++ Delphes: modules/Efficiency.cc
    CMS tracking efficiency: arXiv:1405.6569
"""

from collections.abc import Callable

import torch
from torch import nn

from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes import pdg_filters


class Efficiency(nn.Module):
    """PyTorch implementation of Delphes Efficiency module.

    Applies tracking efficiency based on particle kinematics. The efficiency
    formula takes (pT, η_outer) and returns a probability [0, 1]. Particles
    are then stochastically accepted or rejected based on this probability.

    The module uses **position-based η (EtaOuter)** for the efficiency formula,
    matching the C++ Delphes behavior where the track position at the detector
    surface determines reconstruction efficiency.

    Predefined efficiency formulas:

    - **charged_hadron_cms**: CMS charged hadron tracking efficiency
    - **electron_cms**: CMS electron tracking efficiency
    - **muon_cms**: CMS muon tracking efficiency

    Attributes
    ----------
    efficiency_formula: str
        Name of formula or callable
    efficiency_func: Callable
        The actual efficiency function
    pdg_filter_func: Optional[Callable[[torch.Tensor], torch.Tensor]]
        Optional PDG filter for particle selection

    Examples
    --------
    >>> eff_module = Efficiency(efficiency_formula='charged_hadron_cms')
    >>> passed_particles = eff_module(charged_hadrons)
    """

    def __init__(
        self,
        efficiency_formula: str
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "charged_hadron_cms",
    ) -> None:
        """Initialize the Efficiency module.

        Parameters
        ----------
        efficiency_formula: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
            Either a string naming a predefined formula
            ('charged_hadron_cms', 'electron_cms', 'muon_cms') or a
            callable that takes (pt, eta_outer) tensors and returns
            efficiency values in [0, 1].
        """
        super().__init__()
        self.efficiency_formula = efficiency_formula
        self.efficiency_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        self.pdg_filter_func: Callable[[torch.Tensor], torch.Tensor] | None

        # Load efficiency formula
        if self.efficiency_formula == "charged_hadron_cms":
            self.efficiency_func = self._charged_hadron_cms_efficiency
            self.pdg_filter_func = pdg_filters.charged_hadron_filter
        elif self.efficiency_formula == "electron_cms":
            self.efficiency_func = self._electron_cms_efficiency
            self.pdg_filter_func = pdg_filters.electron_filter
        elif self.efficiency_formula == "muon_cms":
            self.efficiency_func = self._muon_cms_efficiency
            self.pdg_filter_func = pdg_filters.muon_filter
        elif callable(self.efficiency_formula):
            self.efficiency_func = self.efficiency_formula
            self.pdg_filter_func = None
        else:
            raise ValueError(f"Unknown efficiency formula: {efficiency_formula}")

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """Apply efficiency filter to particles.

        Computes the efficiency probability for each particle based on its
        kinematics, then stochastically accepts or rejects particles.

        Parameters
        ----------
        particles: torch.Tensor
            Tensor of shape (N, N_FEATURES) containing particles.
            Required columns:
            - PT (col 7): Transverse momentum in GeV
            - ETA_OUTER (col 15): Position-based pseudorapidity at detector

        Returns
        -------
        particles_out: torch.Tensor
            Filtered tensor of shape (M, N_FEATURES) where M ≤ N containing
            only particles that passed the efficiency cut.
        """
        # We want to compute effiency vector based on particles that satisfy:
        # 1. particles that passed propagation (PASS_PROP == 1)
        # 2. particles of the desired type

        pt = particles[:, ColumnMap.PT]  # PT (transverse momentum)
        eta_outer = particles[:, ColumnMap.ETA_OUTER]  # EtaOuter (pseudorapidity at outer position)

        # Compute efficiency for each particle
        efficiency = self.efficiency_func(pt, eta_outer)

        # Apply efficiency stochastically
        passed = torch.rand_like(efficiency) < efficiency

        # Only real particles (mask==1) can pass efficiency
        particles_out = particles[passed]

        return particles_out

    @staticmethod
    def _charged_hadron_cms_efficiency(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """CMS charged hadron tracking efficiency formula.

        Parameters
        ----------
        pt: torch.Tensor
            Transverse momentum (GeV)
        eta_outer: torch.Tensor
            Pseudorapidity

        Returns
        -------
        efficiency: torch.Tensor
            Value between 0 and 1
        """
        abs_eta_outer = torch.abs(eta_outer)

        # Initialize with zeros
        eff = torch.zeros_like(pt)

        # Region 1: Central barrel, low pt (0.1 < pt <= 1.0, |eta_outer| <= 1.5)
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.70, device=pt.device), eff)

        # Region 2: Central barrel, high pt (pt > 1.0, |eta_outer| <= 1.5)
        mask2 = (pt > 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.95, device=pt.device), eff)

        # Region 3: Forward endcap, low pt (0.1 < pt <= 1.0, 1.5 < |eta_outer| <= 2.5)
        mask3 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask3, torch.tensor(0.60, device=pt.device), eff)

        # Region 4: Forward endcap, high pt (pt > 1.0, 1.5 < |eta_outer| <= 2.5)
        mask4 = (pt > 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.85, device=pt.device), eff)

        # Region 5: Very forward (|eta_outer| > 2.5) - efficiency is 0 (already initialized)

        return eff

    @staticmethod
    def _electron_cms_efficiency(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """CMS electron tracking efficiency formula.

        Parameters
        ----------
        pt: torch.Tensor
            Transverse momentum (GeV)
        eta_outer: torch.Tensor
            Pseudorapidity

        Returns
        -------
        efficiency: torch.Tensor
            Value between 0 and 1
        """
        abs_eta_outer = torch.abs(eta_outer)
        eff = torch.zeros_like(pt)

        # Central barrel
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.73, device=pt.device), eff)

        mask2 = (pt > 1.0) & (pt <= 1.0e2) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.95, device=pt.device), eff)

        mask3 = (pt > 1.0e2) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask3, torch.tensor(0.99, device=pt.device), eff)

        # Forward endcap
        mask4 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.50, device=pt.device), eff)

        mask5 = (pt > 1.0) & (pt <= 1.0e2) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask5, torch.tensor(0.83, device=pt.device), eff)

        mask6 = (pt > 1.0e2) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask6, torch.tensor(0.90, device=pt.device), eff)

        return eff

    @staticmethod
    def _muon_cms_efficiency(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """CMS muon tracking efficiency formula.

        Parameters
        ----------
        pt: torch.Tensor
            Transverse momentum (GeV)
        eta_outer: torch.Tensor
            Pseudorapidity

        Returns
        -------
        efficiency: torch.Tensor
            Value between 0 and 1
        """
        abs_eta_outer = torch.abs(eta_outer)
        eff = torch.zeros_like(pt)

        # Central barrel
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.75, device=pt.device), eff)

        mask2 = (pt > 1.0) & (pt <= 1.0e3) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.99, device=pt.device), eff)

        mask3 = (pt > 1.0e3) & (abs_eta_outer <= 1.5)
        eff3 = 0.99 * torch.exp(0.5 - pt * 5.0e-4)
        eff = torch.where(mask3, eff3, eff)

        # Forward endcap
        mask4 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.70, device=pt.device), eff)

        mask5 = (pt > 1.0) & (pt <= 1.0e3) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask5, torch.tensor(0.98, device=pt.device), eff)

        mask6 = (pt > 1.0e3) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff6 = 0.98 * torch.exp(0.5 - pt * 5.0e-4)
        eff = torch.where(mask6, eff6, eff)

        return eff

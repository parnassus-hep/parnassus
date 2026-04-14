"""Quick demo: Adam-tune a TorchDelphes resolution parameter against pseudo-data.

Because TorchDelphes is implemented as differentiable PyTorch modules (it uses
the reparameterization trick z ~ N(0,1); y = exp(mu + s * z) for log-normal
smearing), gradients flow end-to-end through the simulation. This script
demonstrates that by recovering the CMS central-barrel charged-hadron pT
resolution constant `a` in

    res(pT) = sqrt(a^2 + (b * pT)^2)

from "pseudo-data" generated with a known truth value, using torch.optim.Adam.

Run:
    python docs/prd/torch_delphes_tune_resolution.py
"""

import torch

from parnassus.torch_delphes.stochastic_utils import log_normal_sample


def smear(pt: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply the CMS central-barrel charged-hadron pT smearing."""
    res = torch.sqrt(a**2 + (b * pt) ** 2)        # relative resolution
    return log_normal_sample(pt, res * pt)        # absolute sigma = res * pt


def main() -> None:
    torch.manual_seed(0)

    # --- Truth pseudo-data ------------------------------------------------
    N = 100_000
    A_TRUE, B_TRUE = 0.06, 1.3e-3                 # CMS central-barrel constants
    pt = torch.empty(N).uniform_(1.0, 100.0)

    with torch.no_grad():
        pt_data = smear(pt, torch.tensor(A_TRUE), torch.tensor(B_TRUE))
    data_var = ((pt_data - pt) ** 2).mean()       # observed mean squared residual

    # --- Tunable parameter ------------------------------------------------
    log_a = torch.nn.Parameter(torch.log(torch.tensor(0.20)))   # start far from truth
    b = torch.tensor(B_TRUE)
    opt = torch.optim.Adam([log_a], lr=0.05)

    # --- Optimization loop ------------------------------------------------
    for step in range(401):
        opt.zero_grad()
        a = log_a.exp()
        pt_pred = smear(pt, a, b)
        pred_var = ((pt_pred - pt) ** 2).mean()
        loss = (pred_var - data_var) ** 2
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}  a = {a.item():.4f}  loss = {loss.item():.3e}")

    print(f"\nrecovered a = {log_a.exp().item():.4f}  (truth = {A_TRUE})")


if __name__ == "__main__":
    main()

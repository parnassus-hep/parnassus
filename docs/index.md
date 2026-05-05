# Parnassus

Parnassus is a fast detector simulation framework for particle physics. It reads truth-level particle data (HepMC3, ROOT, or Pythia8 input) and produces detector-level outputs using generative models.

Two simulation backends are available:

- **Neural**: Flow-based generative models that learn detector response from training data
- **Parametric**: PyTorch-based smearing and efficiency simulation reproducing Delphes-like behavior

## Available Generators

| Generator | Type | Description |
|-----------|------|-------------|
| `cms_2011_flow_v00` | Neural | CMS 2011 flow-based generative model |
| `cms` | Parametric | CMS detector card (tracker radius 1.29 m, B = 3.8 T) |
| `atlas` | Parametric | ATLAS detector card (tracker radius 1.15 m, B = 2.0 T) |

## Getting Started

1. [Install Parnassus](installation.md)
2. [Run your first simulation](quickstart.md)
3. Learn about [Neural](neural-mode.md) or [Parametric](parametric-mode.md) generator modes

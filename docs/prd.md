# Parnassus: Neural Fast Simulation for High Energy Physics

**Version 0.1.0** | **Author:** Dmitrii Kobylianskii | **License:** MIT

---

## 1. Introduction

Parnassus is a Python framework for fast detector simulation in High Energy Physics (HEP).
It replaces the computationally expensive full GEANT4 simulation chain with neural network-based
generative models that learn the mapping from truth-level (generator-level) particles to
detector-level particle-flow (pflow) objects. The framework is **process-agnostic**: the same
pre-trained models can be applied to arbitrary physics processes by simply providing different
input truth events.

### Design Goals

- **Speed**: Orders of magnitude faster than full simulation by replacing the detector response
  with neural diffusion models.
- **Fidelity**: Trained on real CMS 2011 data to reproduce realistic detector-level distributions
  including particle kinematics, vertex positions, impact parameters, and particle identification.
- **Flexibility**: Accept truth-level input from any source -- Pythia8 on-the-fly generation,
  HepMC files from external generators (MadGraph, Sherpa, Herwig, Powheg), or custom formats.
- **Completeness**: Full post-processing pipeline including jet clustering (FastJet) and lepton
  isolation, with output to ROOT format.

### What Parnassus Produces

Given a set of truth-level particles for each event, Parnassus generates:

| Output | Description |
|---|---|
| **Particle-flow particles** | Full kinematics (pT, eta, phi), vertex position (vx, vy, vz), particle class and PDG ID |
| **Impact parameters** | d0, z0 and their errors for charged particles |
| **Jets** | Clustered with configurable algorithms (anti-kT, generalized kT) including substructure variables (D2, C2) |
| **Lepton isolation** | Isolation scores and pT sums in configurable delta-R cones for electrons and muons |
| **Event-level quantities** | HT, MET components, particle multiplicity |

---

## 2. Architecture Overview

Parnassus uses a three-stage neural generation pipeline followed by physics-based
post-processing:

```
                         ┌─────────────────────┐
                         │   Input Truth Events │
                         │  (HepMC or Pythia8)  │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │     Event Model      │
                         │  Generates: HT, MET, │
                         │  n_particles          │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Particle Model     │
                         │  Generates per-particle: │
                         │  pT, eta, phi, vx,   │
                         │  vy, vz, class       │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Impact Model       │
                         │  (optional)          │
                         │  Generates: d0, z0,  │
                         │  d0_error, z0_error  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
           ┌────────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
           │ Jet Clustering │ │  Lepton    │ │   ROOT      │
           │   (FastJet)    │ │ Isolation  │ │   Writer    │
           └───────────────┘ └────────────┘ └─────────────┘
```

All three neural models are **diffusion models** sampled via Euler integration. Each model is
conditioned on truth-level particle information, making the generation process inherently
dependent on the input physics process without being tied to any specific one.

### Neural Model Details

| Model | Input Context | Output Variables | Sampler |
|---|---|---|---|
| **Event** | Truth particle kinematics (ptrel, eta, phi, vx, vy, vz, class), global sums (HT, MET, n_truth) | pflow HT, MET_x, MET_y, n_pflow | Euler, 50 steps |
| **Particle** | Same truth context + event model outputs | pflow ptrel, eta, phi, vx, vy, vz, class | Euler (reverse-time), 50 steps |
| **Impact** | Same truth context + event model outputs | d0, z0, d0_error, z0_error | Euler, 50 steps |

The number of diffusion steps is configurable at runtime via the `--num_steps` / `-n` flag,
allowing users to trade speed for quality.

---

## 3. Code Structure

```
src/parnassus/
├── main.py                          # CLI entry point (init, run)
├── configs/
│   ├── config.py                    # Top-level Config dataclass
│   ├── data.py                      # DatasetConfig
│   ├── writer.py                    # WriterConfig
│   ├── pipeline.py                  # JetClusteringConfig, IsolationConfig
│   ├── scheme.py                    # GenEvent, GenParticleCollection, GenJetCollection
│   ├── accessors.py                 # Accessor system for ROOT output
│   ├── variables.py                 # VariableRequirements (shared variable specs)
│   ├── default_config.yaml          # Default configuration template
│   └── generators/
│       ├── base.py                  # GeneratorConfig (ABC)
│       ├── neural.py                # NeuralGeneratorConfig + registry
│       ├── parametric.py            # ParametricGeneratorConfig (placeholder)
│       └── model.py                 # ModelConfig, SamplerConfig, VariablesConfig
├── data/
│   ├── base.py                      # BaseDataset (PyTorch Dataset)
│   ├── hepmc.py                     # HepMCDataset -- loads .hepmc files
│   └── pythia.py                    # PythiaDataset -- generates via Pythia8
├── nn/
│   ├── wrapper.py                   # ModelWrapper (loads exported .pt2 models)
│   └── sampler.py                   # EulerSampler (diffusion sampling)
├── pipelines/
│   ├── generate.py                  # GenerationPipeline (orchestrates everything)
│   ├── cluster.py                   # JetClusteringPipeline (FastJet)
│   ├── isolation.py                 # IsolationPipeline (lepton isolation)
│   └── generators/
│       └── neural.py                # NeuralEventGenerator
├── pythia/
│   ├── hepmc3_generator.py          # Parallel Pythia8 → HepMC3 generation
│   └── pythia8_to_hepmc3.py         # Event format conversion
├── writers/
│   └── root_writer.py               # ROOT output via uproot
├── utils/
│   ├── transform.py                 # VarTransform, TransformRegistry, Unscaler
│   ├── pid.py                       # PDG ID → particle class mapping
│   └── logger.py                    # Rich-based logging and progress bars
└── pretrained_models/
    └── cms_2011/
        ├── metadata.yaml            # Model specifications
        ├── var_transform.yaml       # Variable transformation configs
        ├── event.pt2                # Event-level diffusion model
        ├── particle.pt2             # Particle-level diffusion model
        └── impact.pt2               # Impact parameter diffusion model
```

### Key Abstractions

- **`Config`** -- Top-level configuration container parsed from YAML. Holds dataset, generator,
  pipeline, and writer configs.
- **`GeneratorConfig`** -- Abstract base for generation backends. Currently implemented:
  `NeuralGeneratorConfig` (neural diffusion) and `ParametricGeneratorConfig` (placeholder).
- **`BaseDataset`** -- PyTorch Dataset subclass that loads truth particles, applies variable
  transformations, pads to fixed length, and computes event-level quantities (HT, MET).
  Two concrete implementations: `HepMCDataset` (from files) and `PythiaDataset` (on-the-fly).
- **`GenerationPipeline`** -- Orchestrates the full workflow: build dataset, create dataloader,
  initialize generator, run batch sampling, convert to `GenEvent` objects.
- **`NeuralEventGenerator`** -- Loads pre-trained `.pt2` models, runs diffusion sampling,
  unscales outputs from normalized space back to physics space.
- **`GenEvent`** -- Complete event representation containing truth and pflow particle collections,
  extracted lepton collections, jet collections, and event-level properties.

---

## 4. Pre-trained Models

### CMS 2011 Flow v1 (`cms_2011_flow_v00`)

The bundled pre-trained model was trained on CMS 2011 open data. It supports:

- Up to **400 particles per event**
- **5 particle classes**: charged hadrons (0), electrons (1), muons (2), neutral hadrons (3), photons (4)
- Full kinematic coverage within |eta| < 2.7 and pT > 0.25 GeV
- Impact parameter generation (d0, z0 with errors)

The model uses learned variable transformations (standardization, min-max scaling, and
nonlinear pre-processing functions like log, asinh, sqrt) to normalize physics variables
before neural network processing. These transformations are stored in `var_transform.yaml`
and automatically applied/inverted during generation.

---

## 5. Installation

```bash
# Clone the repository
git clone https://github.com/parnassus-hep/parnassus.git
cd parnassus

# Install with pip (requires Python 3.12)
pip install .

# For development
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.12, < 3.13
- PyTorch >= 2.6.0
- Pythia8mc 8.316.0
- PyHepMC >= 2.14.0
- FastJet >= 3.4.3.1
- EnergyFlow >= 1.4.0
- Uproot >= 5.5.2

---

## 6. Quick Start

### Initialize a Configuration File

```bash
parnassus init .
```

This copies `default_config.yaml` to your current directory. The default configuration uses
the `cms_2011_flow_v00` neural generator with anti-kT jet clustering and lepton isolation.

### Run with a HepMC Input File

```bash
parnassus run \
  -c default_config.yaml \
  -i events.hepmc \
  -ne 1000 \
  -bs 2000 \
  -o output.root
```

| Flag | Description |
|---|---|
| `-c` / `--config` | Path to YAML configuration file |
| `-i` / `--input_path` | Input HepMC or Pythia .cmnd file (overrides config) |
| `-o` / `--output_path` | Output ROOT file path (overrides config) |
| `-ne` / `--num_events` | Number of events to process |
| `-bs` / `--batch_size` | Batch size for neural generation |
| `-n` / `--num_steps` | Number of diffusion sampler steps (default: 50) |

---

## 7. Examples: Changing Physics Processes

The key insight of Parnassus is that the neural models are **process-agnostic**. They learn the
detector response as a function of truth-level particle properties -- not the physics process
that produced those particles. This means you can apply the same pre-trained model to any
physics process by simply providing different truth-level input.

### Example 1: Starting from a HepMC File (Any External Generator)

If you have events from any MC generator (MadGraph, Sherpa, Herwig, Powheg, etc.) saved in
HepMC format, you can use them directly:

```bash
# Generate fast-sim output from externally produced HepMC events
parnassus run \
  -c default_config.yaml \
  -i /path/to/madgraph_ttbar.hepmc \
  -ne 10000 \
  -o ttbar_fastsim.root
```

No configuration changes are needed beyond pointing to your HepMC file. The framework
automatically:

1. Reads final-state particles (status == 1)
2. Applies selection cuts: |eta| < 2.7, pT > 0.25 GeV, excludes neutrinos
3. Maps PDG IDs to the 5 particle classes
4. Computes event-level context variables (HT, MET, n_particles)
5. Runs the neural fast simulation conditioned on these truth particles

**Supported processes include** (but are not limited to):
- H -> ZZ -> 4l (what the model was trained on)
- ttbar production
- W/Z + jets
- Diboson (WW, WZ, ZZ)
- Single top
- QCD multijet
- BSM signals (SUSY, extra dimensions, etc.)

### Example 2: Starting from Pythia8 with a Custom Process

To generate events on-the-fly with Pythia8, create a `.cmnd` steering card for your desired
process and pass it as the input file. Parnassus detects `.cmnd` files automatically and uses
`PythiaDataset` instead of `HepMCDataset`.

#### Step 1: Create a Pythia8 Steering Card

For example, to generate ttbar events, create `ttbar.cmnd`:

```
! ttbar.cmnd -- top pair production at 7 TeV

! Beam settings
Beams:eCM = 7000.          ! center-of-mass energy (GeV)
Beams:idA = 2212           ! proton
Beams:idB = 2212           ! proton

! Process: top pair production
Top:gg2ttbar = on          ! g g -> t tbar
Top:qqbar2ttbar = on       ! q qbar -> t tbar

! Top quark settings
6:m0 = 172.5               ! top mass

! Decays
Top:all = on               ! allow all top decays

! Parton shower / hadronization
PartonLevel:MPI = on
PartonLevel:ISR = on
PartonLevel:FSR = on
HadronLevel:Hadronize = on
```

#### Step 2: Run Parnassus

```bash
parnassus run \
  -c default_config.yaml \
  -i ttbar.cmnd \
  -ne 5000 \
  -bs 2000 \
  -o ttbar_fastsim.root
```

Pythia8 generates the truth-level events on-the-fly, and the neural fast simulation
produces detector-level particle-flow objects for each event.

#### More Pythia8 Process Examples

**W + jets:**

```
! wjets.cmnd
Beams:eCM = 7000.
Beams:idA = 2212
Beams:idB = 2212

WeakSingleBoson:ffbar2W = on

PartonLevel:MPI = on
PartonLevel:ISR = on
PartonLevel:FSR = on
HadronLevel:Hadronize = on
```

**QCD dijet production:**

```
! qcd_dijet.cmnd
Beams:eCM = 7000.
Beams:idA = 2212
Beams:idB = 2212

HardQCD:all = on
PhaseSpace:pTHatMin = 20.    ! minimum pT of the hard process

PartonLevel:MPI = on
PartonLevel:ISR = on
PartonLevel:FSR = on
HadronLevel:Hadronize = on
```

**Drell-Yan (Z/gamma* -> ll):**

```
! drell_yan.cmnd
Beams:eCM = 7000.
Beams:idA = 2212
Beams:idB = 2212

WeakSingleBoson:ffbar2gmZ = on
23:onMode = off              ! turn off all Z decays
23:onIfAny = 11 13           ! allow Z -> ee and Z -> mumu

PartonLevel:MPI = on
PartonLevel:ISR = on
PartonLevel:FSR = on
HadronLevel:Hadronize = on
```

### Example 3: Batch Generation with the HepMC3 Generator

For producing large HepMC datasets from Pythia8 in parallel (useful for preparing training
data or large-scale studies), use the `HepMC3Generator` class directly:

```python
from parnassus.pythia import HepMC3Generator

generator = HepMC3Generator(
    cmnd_file="ttbar.cmnd",
    output_dir="hepmc_output/",
    log_dir="hepmc_logs/",
)

# Generate 100k events using 8 parallel workers
merged_file = generator.generate(
    n_events=100_000,
    max_workers=8,
)

# Now run fast-sim on the merged file
# parnassus run -c config.yaml -i hepmc_output/events.hepmc -ne 100000 -o output.root
```

### Example 4: Using Parnassus as a Python Library

You can also use Parnassus programmatically for tighter integration with analysis workflows:

```python
from pathlib import Path
from parnassus.configs import Config
from parnassus.pipelines import GenerationPipeline, JetClusteringPipeline, IsolationPipeline
from parnassus.configs.pipeline import JetClusteringConfig, IsolationConfig
from parnassus.writers import RootWriter

# Load configuration
config = Config.from_yaml("my_config.yaml")

# Override settings programmatically
config.dataset_config.file_path = Path("my_events.hepmc").absolute()
config.dataset_config.num_events = 5000

# Run generation
pipeline = GenerationPipeline(config)
gen_events, accessors_dict = pipeline.run()

# Post-processing: jet clustering
for pipeline_config in config.pipeline_configs:
    if isinstance(pipeline_config, JetClusteringConfig):
        jet_pipeline = JetClusteringPipeline(pipeline_config)
        jet_pipeline.process(gen_events)
    elif isinstance(pipeline_config, IsolationConfig):
        iso_pipeline = IsolationPipeline(pipeline_config)
        iso_pipeline.process(gen_events)

# Access generated data
for event in gen_events[:5]:
    print(f"Event {event.event_number}:")
    print(f"  Truth particles: {len(event.truth_particles.pt)}")
    print(f"  Pflow particles: {len(event.pflow_particles.pt)}")
    print(f"  HT = {event.ht:.1f} GeV")

# Write to ROOT
writer = RootWriter(config.writer_config)
writer.write(gen_events)
```

---

## 8. Configuration Reference

The YAML configuration file has four top-level sections:

```yaml
# ── Dataset ──────────────────────────────────────────────────
dataset:
  file_path: ""              # Path to .hepmc or .cmnd file
  num_events: 1000           # Number of events to process
  entry_start: 0             # Starting event index (for HepMC files)

# ── Generator ────────────────────────────────────────────────
generator:
  type: "neural"             # "neural" or "parametric"
  name: "cms_2011_flow_v00"  # Registered model name
  num_steps: 50              # Diffusion sampler steps (neural only)
  batch_size: 2000           # Events per batch
  device: "cpu"              # "cpu", "cuda", or "mps"

# ── Post-processing Pipelines ────────────────────────────────
pipelines:
  TruthJetsAntiKt05:         # Pipeline name (user-defined)
    type: "cluster"          # "cluster" or "isolation"
    collection: truth        # "truth" or "pflow"
    dr: 0.5                  # Jet radius parameter
    algorithm: antikt        # "antikt" or "genkt"
    pt_min: 10               # Minimum jet pT (GeV)
    nconst_min: 2            # Minimum number of constituents

  PflowJetsAntiKt05:
    type: "cluster"
    collection: pflow
    dr: 0.5
    algorithm: antikt
    pt_min: 10
    nconst_min: 2

  ElectronIsolation:
    type: "isolation"
    collection: "electrons"  # "electrons" or "muons"
    dr: 0.4                  # Isolation cone radius

  MuonIsolation:
    type: "isolation"
    collection: "muons"
    dr: 0.4

# ── Output ───────────────────────────────────────────────────
output:
  file_path: ""              # Output ROOT file path
  format: default            # Output format
```

### Pipeline Configuration Options

**Jet clustering:**
- `algorithm`: `antikt` (anti-kT) or `genkt` (generalized kT)
- `dr`: Radius parameter (typical values: 0.4, 0.5, 0.8, 1.0)
- `pt_min`: Minimum jet pT in GeV
- `nconst_min`: Minimum number of jet constituents
- `collection`: Which particles to cluster (`truth` or `pflow`)

**Lepton isolation:**
- `collection`: `electrons` or `muons`
- `dr`: Isolation cone delta-R (typical: 0.3, 0.4)
- Computes: `iso_score`, `pt_sum`, `pt_sum_charged`, `pt_sum_neutral`

---

## 9. Extending Parnassus

### Registering a New Pre-trained Model

1. Create a directory under `pretrained_models/`:

```
pretrained_models/my_model/
├── metadata.yaml
├── var_transform.yaml
├── event.pt2
├── particle.pt2
└── impact.pt2          # optional
```

2. Write `metadata.yaml` following the CMS 2011 template:

```yaml
name: My Model v1
max_particles: 400

models:
  event:
    file_name: event.pt2
    version: v1
    sampler:
      type: euler
      num_steps: 50
      reverse_time: false
    variables:
      truth_vars_to_load: [pt, eta, phi, vx, vy, vz, class]
      fs_vars: [pflow_ht, pflow_met_x, pflow_met_y, npflow]
      ctxt_vars: [truth_ptrel, truth_eta, truth_phi, truth_vx, truth_vy, truth_vz, truth_class]
      ctxt_global_vars: [means, truth_ht, truth_met_x, truth_met_y, ntruth]
  particle:
    file_name: particle.pt2
    # ... (same structure as event)
  impact:     # optional
    file_name: impact.pt2
    # ...
```

3. Register in `src/parnassus/configs/generators/neural.py`:

```python
NEURAL_GENERATORS_REGISTRY: dict[str, GeneratorConfig] = {
    "cms_2011_flow_v00": NeuralGeneratorConfig.load_from_metadata(...),
    "my_model_v1": NeuralGeneratorConfig.load_from_metadata(
        Path(__file__).parent.parent.parent / "pretrained_models/my_model/metadata.yaml"
    ),
}
```

4. Reference in your configuration:

```yaml
generator:
  type: "neural"
  name: "my_model_v1"
```

### Adding Custom Post-processing Pipelines

Implement the pipeline protocol and register it in the configuration parser. See
`pipelines/cluster.py` and `pipelines/isolation.py` for reference implementations.

---

## 10. Particle Classification

Parnassus maps PDG particle IDs to 5 detector-level classes:

| Class ID | Particle Type | Examples |
|---|---|---|
| 0 | Charged hadron | pi+/-, K+/-, p/pbar |
| 1 | Electron | e+/- |
| 2 | Muon | mu+/- |
| 3 | Neutral hadron | n, K_L, K_S |
| 4 | Photon | gamma |

Neutrinos (PDG IDs 12, 14, 16) are excluded from the simulation as they are invisible
to the detector. The selection cuts applied to all input particles are:

- **|eta| < 2.7** -- central detector coverage
- **pT > 0.25 GeV** -- minimum transverse momentum
- **status == 1** -- final-state particles only

---

## 11. Output Format

The ROOT output file contains a flat TTree with branches for each physics object collection.
The accessor system provides a declarative way to define which variables are written for
each collection:

- **Truth particles**: pT, eta, phi, vx, vy, vz, class, PDG ID
- **Pflow particles**: pT, eta, phi, vx, vy, vz, class, PDG ID + impact parameters (d0, z0, errors)
- **Electrons/Muons**: pT, eta, phi + impact parameters + isolation variables
- **Jets**: pT, eta, phi, mass + substructure (D2, C2)

The output can be read with standard ROOT tools, uproot, or any framework that supports
the ROOT file format.

```python
import uproot

f = uproot.open("output.root")
tree = f["tree"]

# Access pflow jet pT
jet_pt = tree["PflowJetsAntiKt05_pt"].array()
```

---

## 12. Dependencies and Ecosystem

| Package | Role |
|---|---|
| **PyTorch** (>= 2.6.0) | Neural network inference, model loading via `torch.export` |
| **Pythia8mc** (8.316.0) | Monte Carlo event generation for truth-level particles |
| **PyHepMC** (>= 2.14.0) | HepMC3 event format I/O |
| **FastJet** (>= 3.4.3.1) | Jet clustering algorithms |
| **EnergyFlow** (>= 1.4.0) | Jet substructure variable computation (D2, C2) |
| **Uproot** (>= 5.5.2) | ROOT file I/O without requiring ROOT installation |
| **NumPy** | Array operations throughout |
| **Rich** | Terminal logging and progress bars |
| **joblib** | Parallel Pythia8 event generation |
| **particle** (>= 0.25.2) | PDG particle data |

---

## Appendix A: Variable Transformations

Neural networks work best with normalized inputs. Parnassus uses a transformation pipeline
stored in `var_transform.yaml` to map between physics space and model space:

1. **Pre-processing function** (optional): `log`, `log1p`, `sqrt`, `tanh`, `asinh`, `atan`, `pow`
2. **Scaling**: `std` (standardization), `min_max` (0 to 1), `min_max_sym` (-1 to 1)
3. **Post-processing function** (optional): inverse of pre-processing

During generation, the neural network outputs are automatically inverse-transformed back to
physics space. Special handling exists for:

- **phi**: Generated as sin/cos components, then converted back to angle via atan2
- **class**: Generated as one-hot logits, then converted to class ID via argmax
- **ptrel**: Relative pT, multiplied by event HT to recover absolute pT

## Appendix B: Diffusion Sampling

The neural models use continuous-time diffusion with Euler integration for sampling.
The `EulerSampler` class implements the following procedure:

1. Sample initial noise from a standard normal distribution
2. Create a schedule of timesteps from 0 to 1 (or 1 to 0 for reverse-time models)
3. At each step, evaluate the neural network to get the velocity field
4. Update the sample: `x_{t+dt} = x_t + v(x_t, t) * dt`
5. After all steps, apply inverse variable transformations

The number of steps controls the trade-off between generation quality and speed:
- **10 steps**: Fast but lower quality; useful for quick validation
- **50 steps** (default): Good balance of quality and speed
- **100+ steps**: Highest quality; diminishing returns beyond ~100

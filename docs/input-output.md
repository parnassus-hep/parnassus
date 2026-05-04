# Input & Output

## Input Formats

Parnassus supports three input formats, selected automatically based on file extension.

### HepMC3 (`.hepmc`)

Standard HEP event record format. Contains truth-level particle four-vectors, PDG IDs, and vertex information.

```bash
uv run parnassus run -i events.hepmc ...
```

### ROOT (`.root`)

Preprocessed ROOT files with truth particle data stored in TTree branches. Used by `parnassus-core` for model training and evaluation.

```bash
uv run parnassus run -i preprocessed.root ...
```

### Pythia8 card (`.cmnd`)

Pythia8 configuration file. Events are generated on-the-fly before passing to the simulation pipeline.

```bash
uv run parnassus run -i pythia_config.cmnd ...
```

> **Note:** Pythia8 (`pythia8mc`) is included in the standard installation via `uv sync`. No extra steps are required.

## Output Format

Parnassus writes output as ROOT files using [uproot](https://github.com/scikit-hep/uproot5).

### Contents by mode

| Collection | Neural | Parametric | Description |
|------------|--------|------------|-------------|
| `Truth` | yes | yes | Truth-level input particles |
| `PFlow` | yes | yes | Simulated detector-level particles |
| `Electrons` | yes | yes | Electron kinematics; isolation fields added by isolation pipeline |
| `Muons` | yes | yes | Muon kinematics; isolation fields added by isolation pipeline |
| `Track` | no | yes | Reconstructed charged tracks |
| `Tower` | no | yes | Calorimeter towers |
| `Event` | yes | yes | Per-event scalars: EventNumber, TruthHT, PFlowHT, TruthMET, PFlowMET, and x/y MET components |
| `<JetName>` | if cluster pipeline | if cluster pipeline | One collection per configured jet pipeline |

See [Output Reference](output-reference.md) for all field names per collection.

### Reading output

The output file contains a single tree named `Parnassus`. Collections are accessed using dot notation:

```python
import uproot

f = uproot.open("output.root")
tree = f["Parnassus"]

# List all branches (flat dot-separated names like 'PFlow.PT', 'Event.TruthHT')
print(tree.keys())

# PFlow particles — jagged arrays (variable length per event)
pflow_pt = tree["PFlow.PT"].array()
pflow_class = tree["PFlow.ClassID"].array()  # see Output Reference for class IDs

# Filter to electrons only (ClassID == 1)
import awkward as ak
electrons = pflow_pt[pflow_class == 1]

# Event-level HT scalar (one value per event, not jagged)
truth_ht = tree["Event.TruthHT"].array()

# Jet pT from a clustering pipeline named TruthJetsAntiKt05 in config
jet_pt = tree["TruthJetsAntiKt05.PT"].array()
```

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

## Output Format

Parnassus writes output as ROOT files using [uproot](https://github.com/scikit-hep/uproot5).

### Contents

The output file contains the following trees/branches depending on the generator mode and pipeline configuration:

**Common (both modes):**

- Truth particle collections (`truth_pt`, `truth_eta`, `truth_phi`, ...)
- Reconstructed particle collections (`pflow_pt`, `pflow_eta`, `pflow_phi`, `pflow_class`, ...)
- Jet collections (one per clustering pipeline, e.g., `TruthJetsAntiKt05_pt`, `PflowJetsAntiKt05_pt`, ...)
- Isolation variables (per isolation pipeline)

**Parametric mode additional collections:**

- `Track` -- reconstructed charged particle tracks
- `Tower` -- calorimeter towers (`e`, `et`, `eta`, `phi`, `t`)
- Energy flow objects (`EFlowTrack`, `EFlowPhoton`, `EFlowNeutralHadron`)

### Reading output

```python
import uproot

f = uproot.open("output.root")

# List all available keys
print(f.keys())

# Read a specific branch
tree = f["events"]
pt = tree["pflow_pt"].array()
```

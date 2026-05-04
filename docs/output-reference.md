# Output Reference

The output ROOT file contains a single tree named `Parnassus`. Branches use flat dot-separated names — access them in uproot as `tree["Collection.Field"].array()`.

For each jagged collection, uproot also writes a count branch (`nTruth`, `nPFlow`, `nTruthJetsAntiKt05`, etc.). These are internal bookkeeping branches and do not need to be read directly.

## Particle Class IDs

The `ClassID` field maps to particle type:

| ClassID | PDG ID | Particle |
|---------|--------|----------|
| 0 | 211 | Charged hadron (π±) |
| 1 | 11 | Electron |
| 2 | 13 | Muon |
| 3 | 111 / 130 | Neutral hadron (π⁰ in neural; K_L⁰ or original PID in parametric) |
| 4 | 22 | Photon |

## Neural mode

### Truth

| Branch | Description |
|--------|-------------|
| `Truth.PT` | Transverse momentum (GeV) |
| `Truth.Eta` | Pseudorapidity |
| `Truth.Phi` | Azimuthal angle (rad) |
| `Truth.X`, `Truth.Y`, `Truth.Z` | Production vertex (mm) |
| `Truth.ClassID` | Particle class (see table above) |
| `Truth.PID` | PDG ID |
| `Truth.<JetName>_idx` | Index of associated jet (one field per clustering pipeline, e.g. `Truth.TruthJetsAntiKt05_idx`) |

### PFlow

| Branch | Description |
|--------|-------------|
| `PFlow.PT` | Transverse momentum (GeV) |
| `PFlow.Eta` | Pseudorapidity |
| `PFlow.Phi` | Azimuthal angle (rad) |
| `PFlow.X`, `PFlow.Y`, `PFlow.Z` | Production vertex (mm) |
| `PFlow.ClassID` | Particle class |
| `PFlow.PID` | PDG ID |
| `PFlow.D0` | Transverse impact parameter (mm) |
| `PFlow.Z0` | Longitudinal impact parameter (mm) |
| `PFlow.ErrorD0` | Uncertainty on D0 (mm) |
| `PFlow.ErrorZ0` | Uncertainty on Z0 (mm) |
| `PFlow.<JetName>_idx` | Index of associated jet |

### Electrons / Muons (with isolation pipeline)

| Branch | Description |
|--------|-------------|
| `Electrons.PT` | Electron transverse momentum (GeV) |
| `Electrons.Eta` | Pseudorapidity |
| `Electrons.Phi` | Azimuthal angle (rad) |
| `Electrons.D0`, `Electrons.Z0` | Impact parameters (mm) |
| `Electrons.ErrorD0`, `Electrons.ErrorZ0` | Impact parameter uncertainties (mm) |
| `Electrons.IsolationVar` | Relative isolation: (ΣpT in cone) / pT |
| `Electrons.SumPt` | Total ΣpT of all particles in cone |
| `Electrons.SumPtCharged` | ΣpT of charged particles in cone |
| `Electrons.SumPtNeutral` | ΣpT of neutral particles in cone |

`Muons.*` has the same fields as `Electrons.*` in neural mode, including `Muons.D0`, `Muons.Z0`, `Muons.ErrorD0`, and `Muons.ErrorZ0`.

---

## Parametric mode

### Truth

| Branch | Description |
|--------|-------------|
| `Truth.P` | Total momentum (GeV) |
| `Truth.PT` | Transverse momentum (GeV) |
| `Truth.Eta` | Pseudorapidity |
| `Truth.Phi` | Azimuthal angle (rad) |
| `Truth.Mass` | Particle mass (GeV) |
| `Truth.X`, `Truth.Y`, `Truth.Z` | Production vertex (mm) |
| `Truth.T` | Time of flight (s) |
| `Truth.PID` | PDG ID |
| `Truth.ClassID` | Particle class |
| `Truth.Charge` | Electric charge |
| `Truth.Status` | HepMC status code |
| `Truth.<JetName>_idx` | Index of associated jet |

### PFlow

Same fields as Truth (P, PT, Eta, Phi, Mass, X, Y, Z, T, PID, ClassID, Charge, Status) plus `PFlow.<JetName>_idx` for each configured clustering pipeline. No impact parameters.

### Track

Reconstructed charged particle tracks after efficiency and smearing. Same fields as parametric Truth (P, PT, Eta, Phi, Mass, X, Y, Z, T, PID, ClassID, Charge, Status), accessed as `Track.*`. No jet indices.

### Tower

Calorimeter tower deposits.

| Branch | Description |
|--------|-------------|
| `Tower.E` | Energy (GeV) |
| `Tower.ET` | Transverse energy E / cosh(η) (GeV) |
| `Tower.Eta` | Tower centre η |
| `Tower.Phi` | Tower centre φ (rad) |
| `Tower.T` | Tower time (s) |

### Electrons / Muons (with isolation pipeline)

| Branch | Description |
|--------|-------------|
| `Electrons.PT` | Electron transverse momentum (GeV) |
| `Electrons.Eta` | Pseudorapidity |
| `Electrons.Phi` | Azimuthal angle (rad) |
| `Electrons.IsolationVar` | Relative isolation: (ΣpT in cone) / pT |
| `Electrons.SumPt` | Total ΣpT in cone |
| `Electrons.SumPtCharged` | ΣpT of charged particles in cone |
| `Electrons.SumPtNeutral` | ΣpT of neutral particles in cone |

Muons have the same fields under `Muons.*`. Parametric mode does not include impact parameters in Electrons/Muons.

---

## Event scalars (both modes)

One value per event (not jagged).

| Branch | Description |
|--------|-------------|
| `Event.EventNumber` | Original event number from the input file |
| `Event.TruthHT` | Truth-level scalar HT (GeV) |
| `Event.TruthMET` | Truth-level MET magnitude (GeV) |
| `Event.TruthMETx`, `Event.TruthMETy` | Truth MET x/y components (GeV) |
| `Event.PFlowHT` | PFlow-level scalar HT (GeV) |
| `Event.PFlowMET` | PFlow-level MET magnitude (GeV) |
| `Event.PFlowMETx`, `Event.PFlowMETy` | PFlow MET x/y components (GeV) |

---

## Jet collections (both modes)

One collection per clustering pipeline, named after the pipeline key in the config (e.g., `TruthJetsAntiKt05`, `PFlowJetsAntiKt05`).

| Branch | Description |
|--------|-------------|
| `<JetName>.PT` | Jet transverse momentum (GeV) |
| `<JetName>.Eta` | Jet pseudorapidity |
| `<JetName>.Phi` | Jet azimuthal angle (rad) |
| `<JetName>.D2` | Energy correlation ratio D2 (jet substructure) |
| `<JetName>.C2` | Energy correlation ratio C2 (jet substructure) |

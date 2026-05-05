# Output Reference

The output ROOT file contains a single tree named `Parnassus`. Branches use flat dot-separated names; access them in uproot as `tree["Collection.Field"].array()`.

For each jagged collection, uproot also writes a count branch (`nTruth`, `nPFlow`, `nTruthJetsAntiKt05`, etc.). These are internal bookkeeping branches and do not need to be read directly.

## Particle Class IDs
The `ClassID` field maps to particle type:

| ClassID | PDG ID | Particle |
|---------|--------|----------|
| 0 | 211 | Charged hadron ($\pi^\pm$) |
| 1 | 11 | Electron |
| 2 | 13 | Muon |
| 3 | 111 / 130 | Neutral hadron ($\pi^0$ in neural; $K_L^0$ or original PID in parametric) |
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
| `Truth.<JetName>_idx` | Index of associated jet, present for each configured clustering pipeline with `collection: truth` (e.g. `Truth.TruthJetsAntiKt05_idx`) |

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
| `PFlow.<JetName>_idx` | Index of associated jet, present for each configured clustering pipeline with `collection: pflow` |

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
| `Truth.<JetName>_idx` | Index of associated jet, present for each configured clustering pipeline with `collection: truth` |

### PFlow

Same fields as Truth (P, PT, Eta, Phi, Mass, X, Y, Z, T, PID, ClassID, Charge, Status) plus `PFlow.<JetName>_idx` for each configured clustering pipeline with `collection: pflow`. No impact parameters.

### Track

Reconstructed charged particle tracks after efficiency and smearing. Same fields as parametric Truth (P, PT, Eta, Phi, Mass, X, Y, Z, T, PID, ClassID, Charge, Status), accessed as `Track.*`. No jet indices.

### Tower

Calorimeter tower deposits.

| Branch | Description |
|--------|-------------|
| `Tower.E` | Energy (GeV) |
| `Tower.ET` | Transverse energy $E / \cosh(\eta)$ (GeV) |
| `Tower.Eta` | Tower centre $\eta$ |
| `Tower.Phi` | Tower centre $\phi$ (rad) |
| `Tower.T` | Tower time (s) |

### Parametric debug collections

When `generator.debug: true`, parametric mode also writes intermediate detector-stage collections. Particle-like debug collections use the same fields as parametric Truth (`P`, `PT`, `Eta`, `Phi`, `Mass`, `X`, `Y`, `Z`, `T`, `PID`, `ClassID`, `Charge`, `Status`). Tower-like debug collections use the same fields as Tower (`E`, `ET`, `Eta`, `Phi`, `T`).

| Collection | Field set | Description |
|------------|-----------|-------------|
| `ParticleBeforeProp` | particle-like | Stable particles before detector propagation |
| `ParticleAfterProp` | particle-like | Particles after detector propagation |
| `ChargedHadron`, `Electron`, `Muon`, `NeutralParticle` | particle-like | Particle categories split before efficiency and smearing stages |
| `ChargedHadronEfficiency`, `ElectronEfficiency`, `MuonEfficiency` | particle-like | Particles surviving tracking efficiency |
| `ChargedHadronSmeared`, `ElectronSmeared`, `MuonSmeared` | particle-like | Particles after momentum smearing |
| `ECal_EFlowTrack`, `EFlowTrack` | particle-like | Track-like energy-flow intermediates |
| `ECalTower`, `HCalTower`, `EFlowPhoton`, `EFlowNeutralHadron` | tower-like | Calorimeter and neutral energy-flow intermediates |

---

## Lepton collections (both modes)

`Electrons` and `Muons` are always present in both neural and parametric output. They are extracted from `PFlow` and use the same branch names, replacing `<Lepton>` with either `Electrons` or `Muons`.

### Kinematics

| Branch | Description |
|--------|-------------|
| `<Lepton>.PT` | Lepton transverse momentum (GeV) |
| `<Lepton>.Eta` | Pseudorapidity |
| `<Lepton>.Phi` | Azimuthal angle (rad) |

### Impact parameters

These fields are present only in neural mode when the impact model ran. Parametric mode does not include impact parameters.

| Branch | Description |
|--------|-------------|
| `<Lepton>.D0`, `<Lepton>.Z0` | Impact parameters (mm) |
| `<Lepton>.ErrorD0`, `<Lepton>.ErrorZ0` | Impact parameter uncertainties (mm) |

### Isolation fields

These fields are present when an isolation pipeline is configured for the lepton collection.

| Branch | Description |
|--------|-------------|
| `<Lepton>.IsolationVar` | Relative isolation: ($\sum p_T$ in cone) / $p_T$ |
| `<Lepton>.SumPt` | Total $\sum p_T$ in cone |
| `<Lepton>.SumPtCharged` | $\sum p_T$ of charged particles in cone |
| `<Lepton>.SumPtNeutral` | $\sum p_T$ of neutral particles in cone |

---

## Event scalars (both modes)

One value per event (not jagged).

| Branch | Description |
|--------|-------------|
| `Event.EventNumber` | Original event number from the input file |
| `Event.TruthHT` | Truth-level scalar $H_T$ (GeV) |
| `Event.TruthMET` | Truth-level $E_{T}^{\text{miss}}$ magnitude (GeV) |
| `Event.TruthMETx`, `Event.TruthMETy` | Truth-level $E_{x}^{\text{miss}}$ and $E_{y}^{\text{miss}}$ components (GeV) |
| `Event.PFlowHT` | PFlow-level scalar $H_T$ (GeV) |
| `Event.PFlowMET` | PFlow-level $E_{T}^{\text{miss}}$ magnitude (GeV) |
| `Event.PFlowMETx`, `Event.PFlowMETy` | PFlow-level $E_{x}^{\text{miss}}$ and $E_{y}^{\text{miss}}$ components (GeV) |

---

## Jet collections (both modes)

One collection per clustering pipeline, named after the pipeline key in the config (e.g., `TruthJetsAntiKt05`, `PFlowJetsAntiKt05`).

| Branch | Description |
|--------|-------------|
| `<JetName>.PT` | Jet transverse momentum (GeV) |
| `<JetName>.Eta` | Jet pseudorapidity |
| `<JetName>.Phi` | Jet azimuthal angle (rad) |
| `<JetName>.D2` | Energy correlation ratio $D_2$ (jet substructure) |
| `<JetName>.C2` | Energy correlation ratio $C_2$ (jet substructure) |

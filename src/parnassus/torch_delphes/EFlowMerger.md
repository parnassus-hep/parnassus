# EFlowMerger: Design Decisions and Validation

This document explains the design decisions in the `EFlowMerger` implementation and validates them against C++ Delphes source code.

## Overview

The `EFlowMerger` class merges three input streams into ParticleFlowCandidate objects:
1. **Track objects** (from HCal/eflowTracks): Charged particles with tracking information
2. **Photon towers** (from ECal/eflowPhotons): Electromagnetic calorimeter deposits
3. **Neutral hadron towers** (from HCal/eflowNeutralHadrons): Hadronic calorimeter deposits

The merger applies specific transformations to ensure consistency with C++ Delphes output format.

---

## Design Decision 1: Eta Field Uses Position Eta

### Implementation
For Track objects, we set: `Eta = EtaOuter` (position eta at tracker edge)

### Validation
**C++ Delphes Source:** `TreeWriter.cc` lines 485-490, 557

```cpp
// ProcessParticleFlowCandidates method
const TLorentzVector &position = candidate->Position;
cosTheta = TMath::Abs(position.CosTheta());
signz = (position.Pz() >= 0.0) ? 1.0 : -1.0;
eta = (cosTheta == 1.0 ? signz * 999.9 : position.Eta());
// ...
entry->Eta = eta;  // Position-based eta, NOT momentum eta
```

### Rationale
ParticleFlow algorithms need consistent position-based pseudorapidity for matching tracks with calorimeter deposits. Using the track's position at the calorimeter edge (EtaOuter) ensures:
- Tracks and towers use compatible eta coordinates
- Spatial matching between tracking and calorimeter systems is accurate
- Output format matches C++ Delphes exactly

**Result:** ✅ Histogram validation confirms perfect agreement

---

## Design Decision 2: Eem/Ehad Computation

### Implementation
```python
# In tensor_utils.py:
Eem = E if PID == 22 else 0  # Photons only
Ehad = E if PID == 0 else 0  # Neutral hadrons only
```

### Validation
**C++ Delphes Source:** `SimpleCalorimeter.cc` lines 462-463

```cpp
fTower->Eem = (!fIsEcal) ? 0 : energy;
fTower->Ehad = (fIsEcal) ? 0 : energy;
```

### Data Flow in delphes_card_CMS_6_1.tcl

1. **ECal** (SimpleCalorimeter with `IsEcal=true`):
   - Creates towers with `Eem = E, Ehad = 0`
   - Outputs to `ECal/eflowPhotons`

2. **HCal** (SimpleCalorimeter with `IsEcal=false`):
   - Creates towers with `Eem = 0, Ehad = E`
   - Outputs to `HCal/eflowNeutralHadrons`

3. **EFlowMerger** receives these towers and preserves their Eem/Ehad values

### PID Assignment
- **Photons:** `PID = 22` (photon PDG code)
- **Neutral hadrons:** `PID = 0` (C++ Delphes convention for neutral calorimeter deposits)
- **Tracks:** Keep original PID from particle propagation

**C++ Delphes Source:** `TreeWriter.cc` lines 575-576
```cpp
entry->Eem = candidate->Eem;   // Direct copy from Candidate
entry->Ehad = candidate->Ehad;
```

**Result:** ✅ Histogram validation confirms correct Eem/Ehad distributions

---

## Design Decision 3: Vertex Position (X, Y, Z)

### Implementation
- **Track objects:** Keep original vertex position (X, Y, Z non-zero)
- **Tower objects:** Set `X = Y = Z = 0` (no vertex information)

### Rationale
Calorimeter towers represent energy deposits in the calorimeter cells, not tracked particles with vertex positions. Setting X/Y/Z=0 for towers:
- Matches C++ Delphes behavior
- Allows distinguishing tracks from towers in merged output
- Correctly represents that towers lack vertex information

**Result:** ✅ Histogram validation shows towers at zero, tracks with distributed positions

---

## Design Decision 4: ET Field Does NOT Exist

### Finding
ParticleFlowCandidate class in C++ Delphes **does NOT have an ET field**.

**C++ Delphes Source:** `DelphesClasses.h` lines 532-613

The ParticleFlowCandidate class contains:
- `Float_t E` (energy)
- `Float_t PT` (transverse momentum)
- `Float_t P` (total momentum)
- **NO `Float_t ET` field**

**C++ Delphes Source:** `TreeWriter.cc` ProcessParticleFlowCandidates (lines 554-558)
```cpp
entry->E = e;
entry->P = p;
entry->PT = pt;
entry->Eta = eta;
entry->Phi = phi;
// NO ET assignment!
```

### Comparison with Tower Class
The `Tower` class DOES have ET (DelphesClasses.h line 506), but ParticleFlowCandidate does not.

**Result:** ✅ ET field removed from implementation to match C++ Delphes

---

## Summary Table

| Design Choice | C++ Delphes Reference | Validation Status |
|--------------|----------------------|-------------------|
| Eta = Position Eta (EtaOuter for tracks) | TreeWriter.cc:485-490, 557 | ✅ VALIDATED |
| Eem from ECal towers (PID=22) | SimpleCalorimeter.cc:462-463 | ✅ VALIDATED |
| Ehad from HCal towers (PID=0) | SimpleCalorimeter.cc:462-463 | ✅ VALIDATED |
| Tower X/Y/Z = 0 | Implicit in C++ (towers have no vertex) | ✅ VALIDATED |
| No ET field | DelphesClasses.h:532-613 | ✅ VALIDATED |

---

## Future Work

**TODO:** Consider computing Eem/Ehad directly in SimpleCalorimeter.py

Currently, Eem and Ehad are computed during ROOT file writing based on PID. A cleaner approach would be to:
1. Add Eem/Ehad columns to the tensor representation in SimpleCalorimeter
2. Set these values when creating Tower objects (matching C++ SimpleCalorimeter behavior)
3. EFlowMerger would then preserve these values without recomputation

This would:
- Make the tensor representation more complete
- Reduce logic in ROOT writing code
- Better match the internal C++ Candidate structure

**Decision:** Deferred until needed, as current implementation is validated and working correctly.

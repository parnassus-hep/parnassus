# Pipelines

Pipelines are post-generation processing stages defined in the `pipelines` section of the configuration file. They run after the generator produces detector-level particles.

Each pipeline has a user-chosen name (the YAML key) and a `type` field that determines its behavior. Multiple pipelines can be defined and they execute in order.

Cluster pipeline names (the YAML key, e.g., `TruthJetsAntiKt05`) become jet collection names in the output ROOT tree. Isolation pipeline names are not written as collections; isolation fields are appended to `Electrons` or `Muons`. Access them in uproot as `tree["TruthJetsAntiKt05.PT"].array()` and `tree["Electrons.IsolationVar"].array()`.

## Common Execution Fields

These fields are supported by both `cluster` and `isolation` pipelines:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | integer | `2000` | Number of events per postprocessing batch |
| `num_processes` | integer | `1` | Number of worker processes. Use `1` for synchronous execution; values above `1` use multiprocessing. |

## Cluster Pipeline

**Type:** `cluster`

Performs jet clustering using FastJet. Groups particles into jets based on a distance parameter.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | -- | Must be `"cluster"` |
| `collection` | string | `"pflow"` | Particle collection to cluster: `"truth"` or `"pflow"` |
| `algorithm` | string | `"antikt"` | Clustering algorithm: `"antikt"`, `"cambridge"`, `"genkt"`, or `"ee-genkt"` |
| `dr` | float | `0.5` | Jet radius parameter |
| `algorithm_param` | float | *none* | Extra algorithm parameter (the *p* exponent for `"genkt"` and `"ee-genkt"`: *p* = 1 for kt, 0 for Cambridge/Aachen, −1 for anti-kt). Required when using `"genkt"` or `"ee-genkt"`. |
| `pt_min` | float | `0` | Minimum jet transverse momentum in GeV |
| `nconst_min` | integer | `2` | Minimum number of jet constituents |

### Example

```yaml
pipelines:
  TruthJetsAntiKt05:
    type: "cluster"
    collection: truth
    dr: 0.5
    algorithm: antikt
    pt_min: 10
    nconst_min: 2
```

### Output fields

Each jet collection contains:

| Branch | Description |
|--------|-------------|
| `<JetName>.PT` | Jet transverse momentum (GeV) |
| `<JetName>.Eta` | Pseudorapidity |
| `<JetName>.Phi` | Azimuthal angle (rad) |
| `<JetName>.D2` | Energy correlation ratio D2 |
| `<JetName>.C2` | Energy correlation ratio C2 |

## Isolation Pipeline

**Type:** `isolation`

Computes lepton and photon isolation variables using a cone-based method with FSR (final state radiation) vetoing.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | -- | Must be `"isolation"` |
| `collection` | string | `"electrons"` | Particle collection: `"electrons"`, `"muons"`, or `"all"` |
| `dr` | float | `0.4` | Isolation cone radius (Delta R) |

### Example

```yaml
pipelines:
  ElectronIsolation:
    type: "isolation"
    collection: "electrons"
    dr: 0.4
```

### Output fields

The pipeline adds isolation fields to the `Electrons` or `Muons` collection (the kinematics `PT`, `Eta`, `Phi` are always present from the generator):

| Branch | Description |
|--------|-------------|
| `Electrons.IsolationVar` | Relative isolation: ($\sum p_T$ in cone) / $p_T$ |
| `Electrons.SumPt` | Total $\sum p_T$ in isolation cone |
| `Electrons.SumPtCharged` | $\sum p_T$ of charged particles in cone |
| `Electrons.SumPtNeutral` | $\sum p_T$ of neutral particles in cone |

Muons use `Muons.*` with the same isolation fields.

## Full Example

A typical configuration defines multiple pipelines for different jet collections and isolation calculations:

```yaml
pipelines:
  TruthJetsAntiKt05:
    type: "cluster"
    collection: truth
    dr: 0.5
    algorithm: antikt
    pt_min: 10
    nconst_min: 2
  TruthJetsAntiKt08:
    type: "cluster"
    collection: truth
    dr: 0.8
    algorithm: antikt
    pt_min: 10
    nconst_min: 2
  PFlowJetsAntiKt05:
    type: "cluster"
    collection: pflow
    dr: 0.5
    algorithm: antikt
    pt_min: 10
    nconst_min: 2
  ElectronIsolation:
    type: "isolation"
    collection: "electrons"
    dr: 0.4
  MuonIsolation:
    type: "isolation"
    collection: "muons"
    dr: 0.4
```

# Pipelines

Pipelines are post-generation processing stages defined in the `pipelines` section of the configuration file. They run after the generator produces detector-level particles.

Each pipeline has a user-chosen name (the YAML key) and a `type` field that determines its behavior. Multiple pipelines can be defined and they execute in order.

## Cluster Pipeline

**Type:** `cluster`

Performs jet clustering using FastJet. Groups particles into jets based on a distance parameter.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | -- | Must be `"cluster"` |
| `collection` | string | `"pflow"` | Particle collection to cluster: `"truth"` or `"pflow"` |
| `algorithm` | string | `"antikt"` | Clustering algorithm: `"antikt"` or `"genkt"` |
| `dr` | float | `0.5` | Jet radius parameter |
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
  PflowJetsAntiKt05:
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

Pipeline names (e.g., `TruthJetsAntiKt05`) are user-defined and appear as branch prefixes in the output ROOT file.

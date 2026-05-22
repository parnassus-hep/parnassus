# Configuration Reference

Parnassus uses YAML configuration files. This page documents the main user-facing settings.

## Dataset

Controls input data loading.

```yaml
dataset:
  file_path: "input.hepmc"
  num_events: 1000
  entry_start: 0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_path` | string | *required* | Path to the input file (`.hepmc`, `.root`, or `.cmnd`) |
| `num_events` | integer | `1` | Number of events to process |
| `entry_start` | integer | `0` | Starting entry index in the input file. Useful for skipping the first N events or for processing a file in parallel chunks by running multiple jobs with non-overlapping `entry_start` + `num_events` ranges. |

## Generator

Selects and configures the simulation backend.

### Common fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | *required* | Generator type: `"neural"` or `"parametric"` |
| `name` | string | *required* | Generator/model name from the registry |
| `batch_size` | integer | `2000` | Number of events per processing batch |
| `device` | string | `"cpu"` | Computation device: `"cpu"`, `"cuda"`, or `"mps"` |

### Neural-specific fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_steps` | integer | `50` | Number of ODE integration steps |
| `max_particles` | integer | from model | Maximum particles per event. Set from the model metadata; this is not a YAML setting for neural generators. Particles beyond this limit are dropped during preprocessing. |

### Parametric-specific fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed` | integer | *none* | Random seed for reproducibility |
| `debug` | boolean | `false` | Write intermediate detector-stage collections in addition to the standard parametric output |
| `pileup` | object | *none* | Optional pile-up merging configuration (see below) |

### Pile-up merging (`pileup`)

When the `pileup` section is present under `generator`, the parametric pipeline merges pile-up particles from a pre-generated MinBias file with the hard-scatter input **before** detector simulation. This matches the behaviour of the C++ Delphes `PileUpMerger` module.

```yaml
generator:
  type: "parametric"
  name: "cms"
  seed: 42
  pileup:
    file_path: "MinBias.pileup"
    mean_pileup: 50
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_path` | string | *required* | Path to a Delphes `.pileup` binary (XDR) file |
| `mean_pileup` | float | *required* | Average number of pile-up interactions per bunch crossing (Poisson-sampled) |
| `max_z_spread` | float | `0.25` | Vertex z truncation bound in meters |
| `max_t_spread` | float | `800e-12` | Vertex t truncation bound in seconds |
| `sigma_z` | float | `0.053` | Gaussian sigma for vertex z-smearing in meters |
| `sigma_t` | float | `160e-12` | Gaussian sigma for vertex t-smearing in seconds |
| `smear_hs_vertex` | boolean | `true` | Also smear the hard-scatter primary vertex position |

## Pipelines

Map of pipeline name to pipeline configuration. See [Pipelines](pipelines.md) for full details.

```yaml
pipelines:
  <PipelineName>:
    type: "cluster" | "isolation"
    batch_size: 2000
    num_processes: 1
    # ... type-specific fields
```

### Common pipeline fields

These execution controls are supported by both `cluster` and `isolation` pipelines.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | integer | `2000` | Number of events per postprocessing batch |
| `num_processes` | integer | `1` | Number of worker processes. Use `1` for synchronous execution; values above `1` use multiprocessing. |

### Cluster fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | -- | `"cluster"` |
| `collection` | string | `"pflow"` | `"truth"` or `"pflow"` |
| `algorithm` | string | `"antikt"` | `"antikt"` or `"genkt"` |
| `dr` | float | `0.5` | Jet radius |
| `pt_min` | float | `0` | Minimum jet pT (GeV) |
| `nconst_min` | integer | `2` | Minimum constituents |

### Isolation fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | -- | `"isolation"` |
| `collection` | string | `"electrons"` | `"electrons"`, `"muons"`, or `"all"` |
| `dr` | float | `0.4` | Cone radius |

## Output

```yaml
output:
  file_path: "output.root"
  format: default
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_path` | string | *required* | Path to the output ROOT file |
| `format` | string | `"default"` | Reserved for future output-format selection. Currently only ROOT output through the default writer is supported. |

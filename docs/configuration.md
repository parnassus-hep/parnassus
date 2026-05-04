# Configuration Reference

Parnassus uses YAML configuration files. This page documents all available settings.

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

## Pipelines

Map of pipeline name to pipeline configuration. See [Pipelines](pipelines.md) for full details.

```yaml
pipelines:
  <PipelineName>:
    type: "cluster" | "isolation"
    # ... type-specific fields
```

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
| `format` | string | `"default"` | Output format |

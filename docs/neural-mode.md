# Neural Mode

The neural generator uses flow-based generative models to simulate detector response. It runs a multi-stage pipeline:

1. **Event model** -- generates global event-level features
2. **Particle model** -- generates per-particle detector-level quantities
3. **Impact model** (optional) -- generates impact parameters (d0, z0)

Input particles are processed as padded sequences of shape `(batch_size, max_particles, n_features)`.

## Configuration

Set `generator.type` to `"neural"` in your config file:

```yaml
generator:
  type: "neural"
  name: "cms_2011_flow_v00"
  num_steps: 50
  batch_size: 2000
  device: "cpu"
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | -- | Must be `"neural"` |
| `name` | string | -- | Model identifier from the registry |
| `num_steps` | integer | `50` | Number of ODE integration steps for flow sampling |
| `batch_size` | integer | `2000` | Number of events per batch |
| `device` | string | `"cpu"` | Computation device: `"cpu"`, `"cuda"`, or `"mps"` |

### Choosing `num_steps`

Higher values produce better-quality samples at the cost of longer inference. `50` is the default and recommended. With values below ~20 output quality degrades noticeably. Inference time scales roughly linearly with `num_steps`.

## Available Models

| Name | Description |
|------|-------------|
| `cms_2011_flow_v00` | CMS 2011 era flow-based model. Includes event-level, particle-level, and optional impact parameter models. Uses model-defined max particles, variable transformations, and tunable sampling steps. |
| `aleph_flow_v00` | ALEPH flow-based model for LEP-era $e^+e^-$ collisions. Includes event-level and particle-level models. Selects truth particles above a 0.5 GeV pT cut (defined in the model metadata). |

## Example

```bash
uv run parnassus run \
  -c src/parnassus/configs/neural_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root
```

The `-n` CLI flag overrides the config value:

```bash
uv run parnassus run \
  -c src/parnassus/configs/neural_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root \
  -n 100
```

## Output Collections

| Collection | Description |
|------------|-------------|
| `Truth` | Truth-level input particles (PT, Eta, Phi, vertex, ClassID, PID) |
| `PFlow` | Generated detector-level particles (PT, Eta, Phi, vertex, ClassID, PID); impact parameters D0, Z0, ErrorD0, ErrorZ0 included if the impact model ran |
| `Electrons` / `Muons` | Lepton kinematics (always present); impact parameters if available; isolation fields added by isolation pipeline |
| `Event` | Per-event scalars: EventNumber, TruthHT, PFlowHT, TruthMET, PFlowMET |
| `<JetName>` | One collection per configured clustering pipeline (only if cluster pipeline defined) |

See [Output Reference](output-reference.md) for all branch names.

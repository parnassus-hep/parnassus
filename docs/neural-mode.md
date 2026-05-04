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
| `cms_2011_flow_v00` | CMS 2011 era flow-based model. Includes event-level, particle-level, and optional impact parameter models. Supports configurable max particles, variable transformations, and tunable sampling steps. |

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

## Output

Neural mode writes `Truth`, `PFlow`, and `Event` collections plus any configured jet and isolation collections. `PFlow` particles include impact parameters (`D0`, `Z0`, `ErrorD0`, `ErrorZ0`). See [Output Reference](output-reference.md) for all branch names and field descriptions.

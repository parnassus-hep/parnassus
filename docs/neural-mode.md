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

The `--num_steps` CLI flag overrides the config value:

```bash
uv run parnassus run \
  -c src/parnassus/configs/neural_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root \
  --num_steps 100
```

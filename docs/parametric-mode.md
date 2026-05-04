# Parametric Mode

The parametric generator uses `torch_delphes`, a PyTorch-based fast detector simulation that reproduces Delphes-like smearing and efficiencies. It applies particle propagation, tracking efficiency, momentum smearing, and calorimeter simulation.

Unlike neural mode, the parametric pipeline processes all particles independently as a flat `(N_particles, n_features)` tensor -- particles are not grouped into padded sequences.

## Configuration

Set `generator.type` to `"parametric"` in your config file:

```yaml
generator:
  type: "parametric"
  name: "cms"
  seed: 42
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | -- | Must be `"parametric"` |
| `name` | string | -- | Detector card name: `"cms"` or `"atlas"` |
| `seed` | integer | *none* | Random seed for reproducibility. Omit for non-deterministic output. |
| `debug` | boolean | `false` | Return intermediate processing stages (efficiency metrics, detector internals) |

## Available Detector Cards

| Card | Tracker Radius | Magnetic Field | Notes |
|------|---------------|----------------|-------|
| `cms` | 1.29 m | 3.8 T | Momentum smearing, ECal/HCal simulation |
| `atlas` | 1.15 m | 2.0 T | Muons included in calorimeter tower output |

## Output Collections

| Collection | Description |
|------------|-------------|
| `Truth` | Truth-level input particles (P, PT, Eta, Phi, Mass, vertex, T, PID, ClassID, Charge, Status) |
| `PFlow` | Simulated energy-flow particles after detector response |
| `Track` | Reconstructed charged tracks after efficiency and smearing |
| `Tower` | Calorimeter tower deposits (E, ET, Eta, Phi, T) |
| `Electrons` / `Muons` | Per-lepton kinematics and isolation (requires isolation pipeline) |
| `Event` | Per-event scalars: EventNumber, TruthHT, PFlowHT, TruthMET, PFlowMET |
| `<JetName>` | One collection per configured clustering pipeline |

See [Output Reference](output-reference.md) for all branch names.

## Example

```bash
uv run parnassus run \
  -c src/parnassus/configs/parametric_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root
```

The `--random_seed` CLI flag overrides the config value:

```bash
uv run parnassus run \
  -c src/parnassus/configs/parametric_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root \
  --random_seed 123
```

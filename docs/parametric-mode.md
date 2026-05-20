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
| `debug` | boolean | `false` | Write intermediate detector-stage collections; see [Output Reference](output-reference.md#parametric-debug-collections) |
| `pileup` | object | *none* | Optional pile-up merging; see [Pile-Up Merging](#pile-up-merging) below |

## Available Detector Cards

| Card | Tracker Radius | Magnetic Field | Notes |
|------|---------------|----------------|-------|
| `cms` | 1.29 m | 3.8 T | Momentum smearing, ECal/HCal simulation |
| `atlas` | 1.15 m | 2.0 T | Muons included in calorimeter tower output |

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

## Output Collections

| Collection | Description |
|------------|-------------|
| `Truth` | Truth-level input particles (P, PT, Eta, Phi, Mass, vertex, T, PID, ClassID, Charge, Status) |
| `PFlow` | Simulated energy-flow particles after detector response |
| `Track` | Reconstructed charged tracks after efficiency and smearing |
| `Tower` | Calorimeter tower deposits (E, ET, Eta, Phi, T) |
| `Electrons` / `Muons` | Lepton kinematics (PT, Eta, Phi); isolation fields added by isolation pipeline |
| `Event` | Per-event scalars: EventNumber, TruthHT, PFlowHT, TruthMET, PFlowMET |
| `<JetName>` | One collection per configured clustering pipeline (only if cluster pipeline defined) |

See [Output Reference](output-reference.md) for all branch names.

## Pile-Up Merging

The parametric pipeline supports Delphes-style pile-up merging. When enabled, pile-up particles from a pre-generated MinBias file are merged with hard-scatter particles **before** the detector simulation card runs. This means the calorimeter correctly sums energy deposits from both hard-scatter and pile-up sources, matching the physics of a real detector.

### How it works

1. For each hard-scatter event, the number of pile-up interactions is sampled from a Poisson distribution with the configured mean.
2. For each pile-up interaction, a MinBias event is randomly selected from the `.pileup` file.
3. Each pile-up event receives independent vertex smearing (z and t offsets from a truncated Gaussian) and a random azimuthal rotation.
4. Optionally, the hard-scatter primary vertex is also smeared (enabled by default), matching the C++ Delphes `PileUpMerger` behaviour.
5. The combined particle list is passed through the detector card.

The truth output contains only hard-scatter particles (with smeared vertex positions if enabled). Detector-level outputs (PFlow, Track, Tower) include contributions from both hard-scatter and pile-up.

### Configuration

Add a `pileup` section under `generator`:

```yaml
generator:
  type: "parametric"
  name: "cms"
  seed: 42
  pileup:
    file_path: "MinBias.pileup"    # Path to Delphes .pileup binary file
    mean_pileup: 50                # Average PU interactions per bunch crossing
    max_z_spread: 0.25             # Vertex z truncation in meters (default)
    max_t_spread: 800e-12          # Vertex t truncation in seconds (default)
    sigma_z: 0.053                 # Vertex z Gaussian sigma in meters (default)
    sigma_t: 160e-12               # Vertex t Gaussian sigma in seconds (default)
    smear_hs_vertex: true          # Also smear hard-scatter vertex (default)
```

Only `file_path` and `mean_pileup` are required; the remaining fields have defaults that match the standard CMS Delphes card.

See [Configuration Reference](configuration.md#pile-up-merging-pileup) for the full field reference.

### MinBias file format

The `.pileup` file uses the Delphes binary XDR (big-endian) format. These files can be generated using the Delphes `hepmc2pileup` or `root2pileup` conversion tools. Each file contains a set of pre-generated minimum-bias events that are randomly sampled during pile-up merging.

### Vertex smearing defaults

The default sigma and truncation values reproduce the standard CMS Delphes card vertex distribution:

| Parameter | Default | CMS card formula |
|-----------|---------|-----------------|
| `sigma_z` | 0.053 m | $\exp(-z^2/(2 \cdot 0.053^2))$ |
| `sigma_t` | 160 ps | $\exp(-t^2/(2 \cdot (160 \times 10^{-12})^2))$ |
| `max_z_spread` | 0.25 m | ${\sim}4.7\sigma$ truncation |
| `max_t_spread` | 800 ps | ${\sim}5\sigma$ truncation |

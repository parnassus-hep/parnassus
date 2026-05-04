# CLI Reference

Parnassus provides a command-line interface via the `parnassus` command.

## `parnassus init`

Copy the default configuration file to the current directory, or to a specified directory.

```bash
uv run parnassus init [dir]
```

| Argument | Required | Description |
|----------|----------|-------------|
| `dir` | no | Destination directory (default: current directory) |

## `parnassus run`

Run the detector simulation pipeline.

```bash
uv run parnassus run -c <config> [options]
```

### Arguments

| Flag | Long form | Required | Description |
|------|-----------|----------|-------------|
| `-c` | `--config` | yes | Path to the YAML configuration file |
| `-i` | `--input_path` | no | Input file path, overrides config |
| `-ne` | `--num_events` | no | Number of events to process, overrides config |
| `-bs` | `--batch_size` | no | Batch size for processing, overrides config |
| `-o` | `--output_path` | no | Output ROOT file path, overrides config |
| `-n` | `--num_steps` | no | Number of ODE steps (neural mode only), overrides config |
| | `--random_seed` | no | Random seed (parametric mode only), overrides config |

### Examples

Run with neural generator and custom step count:

```bash
uv run parnassus run \
  -c src/parnassus/configs/neural_config.yaml \
  -i events.hepmc \
  -ne 500 \
  -bs 50 \
  -o result.root \
  -n 100
```

Run with parametric generator:

```bash
uv run parnassus run \
  -c src/parnassus/configs/parametric_config.yaml \
  -i events.hepmc \
  -ne 500 \
  -bs 50 \
  -o result.root
```

Run parametric with fixed seed:

```bash
uv run parnassus run \
  -c src/parnassus/configs/parametric_config.yaml \
  -i events.hepmc \
  -ne 500 \
  -bs 50 \
  -o result.root \
  --random_seed 42
```

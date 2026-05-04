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
uv run parnassus run -c <config> -i <input> -ne <num_events> -bs <batch_size> -o <output>
```

### Arguments

| Flag | Long form | Required | Description |
|------|-----------|----------|-------------|
| `-c` | `--config` | yes | Path to the YAML configuration file |
| `-i` | `--input_path` | yes | Path to the input file (`.hepmc`, `.cmnd`, or `.root`) |
| `-ne` | `--num_events` | yes | Number of events to process |
| `-bs` | `--batch_size` | yes | Batch size for processing |
| `-o` | `--output_path` | yes | Path to the output ROOT file |
| `-n` | `--num_steps` | no | Number of ODE steps override (neural mode only) |
| | `--random_seed` | no | Random seed override (parametric mode only) |

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

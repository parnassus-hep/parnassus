# Quick Start

This guide walks through running your first detector simulation with Parnassus.

## 1. Prepare a configuration file

Parnassus ships with example configurations. Copy them to your working directory:

```bash
uv run parnassus init
```

This copies `neural_config.yaml` and `parametric_config.yaml` to your current directory. Alternatively, use one of the bundled configs directly:

- `src/parnassus/configs/neural_config.yaml` -- neural generator
- `src/parnassus/configs/parametric_config.yaml` -- parametric generator

## 2. Run the simulation

Run the parametric generator on a HepMC input file:

```bash
uv run parnassus run \
  -c src/parnassus/configs/parametric_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root
```

Or using the neural generator:

```bash
uv run parnassus run \
  -c src/parnassus/configs/neural_config.yaml \
  -i input.hepmc \
  -ne 100 \
  -bs 10 \
  -o output.root
```

## 3. Inspect the output

The output is a ROOT file that can be opened with [uproot](https://github.com/scikit-hep/uproot5) or any ROOT-compatible tool:

```python
import uproot

f = uproot.open("output.root")
print(f.keys())
```

The file contains truth and reconstructed particle collections, jet collections, and (for parametric mode) track and tower collections.

## Next steps

- Learn about [Neural mode](neural-mode.md) and [Parametric mode](parametric-mode.md)
- Configure [Pipelines](pipelines.md) for jet clustering and isolation
- See the full [Configuration reference](configuration.md)

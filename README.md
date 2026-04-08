[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)



# Parnassus

## Installation

To install the `Parnassus` package, ensure you have Python 3.10 or 3.11 and [uv](https://github.com/astral-sh/uv) installed. Then run:

```bash
# Clone the repository
git clone https://github.com/parnassus-hep/parnassus.git
cd parnassus

# Install package and all dependencies
uv sync --all-extras
```

## Development

Set up and run pre-commit hooks (ruff, mypy, basedpyright) with:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

To run the tests:

```bash
uv run pytest .
# or with coverage
uv run coverage run -m pytest
```

To download the testbench data for the pythia pytest:
```bash
gdown --folder https://drive.google.com/drive/folders/1W-V_rU6lRmtuaOclj3gYB1qJSn4J11qM?usp=sharing -O src/parnassus/tests/benchmark_data/
```

## Running the Package

To copy the default configuration file to the current directory:

```bash
uv run parnassus init
```

To run the `parnassus` package:

```bash
uv run parnassus run -c <config-file> -i <input-file> -ne <num-events> -bs <batch-size> -o <output-file>
```

### Generator Modes

Parnassus supports two simulation backends, selected via the `generator.type` field in the config:

**Neural** (`generator.type: "neural"`): uses flow-based generative models to simulate detector response.

```bash
uv run parnassus run -c src/parnassus/configs/neural_config.yaml -i input.hepmc -ne 100 -bs 10 -o output.root
```

**Parametric** (`generator.type: "parametric"`): uses `torch_delphes`, a PyTorch-based fast detector simulation that reproduces Delphes-like smearing and efficiencies. Supported detector cards: `"cms"` and `"atlas"`.

```bash
uv run parnassus run -c src/parnassus/configs/parametric_config.yaml -i input.hepmc -ne 100 -bs 10 -o output.root
```

### Input Formats

- **HepMC3** (`.hepmc`): standard HEP event record format.
- **ROOT** (`.root`): preprocessed input files used by `parnassus-core` to train/evaluate.
- **Pythia8 card** (`.cmnd`): Pythia8 configuration file; events are generated on-the-fly using the `pythia` module before passing to the simulation pipeline.

### Command-Line Arguments

- `-c, --config`: Path to the configuration file.
- `-i, --input-path`: Path to the input file (`.hepmc`, `.cmnd` or `.root`).
- `-ne, --num-events`: Number of events to process.
- `-bs, --batch-size`: Batch size for processing.
- `-o, --output-path`: Path to the output ROOT file.
- `--random_seed`: Random seed for reproducibility (parametric mode only; overrides config).
- `--num_steps`: Number of ODE steps (neural mode only; overrides config).

## Project Structure

- `src/parnassus/`
	- `configs/` - Dataclass configs with `from_yaml()`; neural and parametric YAML presets
		- `generators/` - Per-generator configs (neural, parametric, model/sampler)
		- `scheme.py` - Core data structures: `GenEvent`, `GenParticleCollection`, `GenTowerCollection`
	- `pipelines/` - `GenerationPipeline` orchestration, jet clustering, lepton isolation
		- `generators/` - `NeuralEventGenerator` and `ParametricEventGenerator`
	- `torch_delphes/` - PyTorch detector simulation (propagator, efficiency, smearing, calorimeter)
		- `defaults/` - `CMSEnergyFlowDefault` and `ATLASEnergyFlowDefault` detector cards
		- `validation/` - Comparison scripts against C++ Delphes
	- `nn/` - `ModelWrapper` (`.pt2` models) and `EulerSampler` (flow sampling)
	- `data/` - Dataset backends (HepMC3, ROOT, Pythia8) and adapters (neural/parametric)
	- `pythia/` - Parallel Pythia8 → HepMC3 event generation
	- `writers/` - ROOT output via uproot
	- `pretrained_models/cms_2011/` - Pretrained flow models (`event.pt2`, `particle.pt2`, `impact.pt2`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

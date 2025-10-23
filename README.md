[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)



# Parnassus

## Installation

To install the `Parnassus` package, ensure you have Python 3.10 or 3.11 installed. Then, run the following commands:

```bash
# Clone the repository
git clone https://github.com/parnassus-hep/parnassus.git
cd parnassus

# Install package
pip install -e .
```

To additionally install the development dependencies (for formatting and linting) use
```bash
pip install -e '.[dev]'
```

## Development

You can set up and run pre-commit hooks with

```bash
pre-commit install
pre-commmit run --all-files
```

To run the tests you can use the `pytest` or `coverage` command, for example

```bash
pytest .
```

## Running the Package
To init working directory, run the following command:

```bash
parnassus init
```
This will copy the default configuration file to the current directory.

To run the `parnassus` package, use the following command:

```bash
parnassus run -c <config-file> -i <input-file> -ne <num-events> -bs <batch-size> -o <output-file>
```

### Example Command

```bash
parnassus run -c src/parnassus/configs/default_config.yaml -i src/tests/h4lep_test_1k.hepmc -ne 4 -bs 2 -o test.root
```

### Command-Line Arguments

- `-c, --config`: Path to the configuration file (e.g., `src/parnassus/configs/default_config.yaml`).
- `-i, --input-path`: Path to the input file (e.g., `.hepmc` file).
- `-ne, --num-events`: Number of events to process.
- `-bs, --batch-size`: Batch size for processing.
- `-o, --output-path`: Path to the output file (e.g., `.root` file).

## Project Structure

This project is structured in the following way:

- `docs/`
- `src/`
	- `parnassus/`
		- `main.py`
		- `configs/`
		- `models/`
			- `modules/`
			- `utils/`
			- `sampler/`
		- `readers/`
		- `writers/`
		- `postprocessing/`
		- `utils/`
		- `tests/`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

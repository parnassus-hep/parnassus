# Installation

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager

## Install from source

Clone the repository and install all dependencies:

```bash
git clone https://github.com/parnassus-hep/parnassus.git
cd parnassus
uv sync --all-extras
```

## Verify installation

```bash
uv run parnassus --help
```

You should see the available CLI commands and options.

## Optional: Pythia8 benchmark data

To run the Pythia8-related tests, download benchmark data:

```bash
gdown --folder https://drive.google.com/drive/folders/1W-V_rU6lRmtuaOclj3gYB1qJSn4J11qM?usp=sharing -O src/parnassus/tests/benchmark_data/
```

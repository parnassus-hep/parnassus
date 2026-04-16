# Differentiable Delphes: JINST-style preprint

This directory holds the source of the preprint
`differentiable_delphes.tex` plus the figures and fit-history data
that go into it. The paper documents the `learnable=True` mode of
`CMSEnergyFlowDefault`, the Gumbel-ST tracking efficiency, the
autograd-safe numerics refactors, and an end-to-end Adam fit against
a Pythia-generated multi-knob pseudo-dataset.

## Layout

- `differentiable_delphes.tex` — the manuscript. Uses the standard
  `jinstpub.sty` package distributed by the Journal of Instrumentation
  (SISSA Medialab). The style file is committed at `doc/jinstpub.sty`.
- `figures/` — PDF figures included by the manuscript.
  Regenerate with `python -m parnassus.torch_delphes.plot_fit_results`
  (see below).
- `fit_results/` — JSON history files from `tune_cms_fullsim.py`.
  `all66_history.json` is the 120-step, all-66-parameter fit that the
  paper reports.

## Reproducing the numbers

All commands below assume you are at the repository root and have run
`uv sync --all-extras` at least once.

1. **Regenerate the pseudo-dataset** (takes ~3 min, writes ~20 MB to
   `src/parnassus/tests/benchmark_data/cms_pseudodata.root`):

   ```bash
   uv run python -m parnassus.torch_delphes.generate_pseudodata \
       --output src/parnassus/tests/benchmark_data/cms_pseudodata.root \
       --target-size-mb 20 \
       --pt-hat-min 100 \
       --seed 1
   ```

2. **Run the 120-step all-66-parameter fit** (takes ~5 min on a single
   CPU core). The history JSON is what the plotting script consumes:

   ```bash
   uv run python -m parnassus.torch_delphes.tune_cms_fullsim \
       --root-file src/parnassus/tests/benchmark_data/cms_pseudodata.root \
       --n-events 300 --n-steps 120 --n-passes-per-step 2 \
       --train-what all \
       --lr-scales 5e-2 --lr-resolution 5e-3 \
       --lr-efficiency 5e-2 --lr-fractions 5e-2 \
       --history-path doc/fit_results/all66_history.json
   ```

3. **Regenerate all six figures**:

   ```bash
   uv run python -m parnassus.torch_delphes.plot_fit_results \
       --history doc/fit_results/all66_history.json \
       --root-file src/parnassus/tests/benchmark_data/cms_pseudodata.root \
       --output-dir doc/figures
   ```

4. **Compile the paper** (`jinstpub.sty` is already committed in
   `doc/`):

   ```bash
   cd doc
   pdflatex differentiable_delphes.tex
   pdflatex differentiable_delphes.tex
   ```

## Scaling to the full Zenodo sample

The committed pseudo-dataset is deliberately tiny (2200 events,
20 MB) so that it can live in git. To reproduce the results on the
full `train_800_1000_filter.root` from [Zenodo record
11389651](https://zenodo.org/records/11389651), the only change is to
pass a different path to `--root-file` in step 2 above and to scale
`--n-events` up. Everything else — the harness, the loss, the plot
script, the paper tables — is agnostic to the input size.

A few parameters that the current pseudo-dataset fails to constrain
(barrel low-pT charged-hadron efficiency, `K0-short` ECal fraction,
barrel charged-hadron resolution `a`) are expected to pick up a
non-zero gradient on the larger sample because it covers more of
their native kinematic regions.

## Known limitations, in plain text

- **Scale-parameter degeneracy.** The charged-hadron pT scale, the
  ECal energy scale and the HCal energy scale all enter the PF object
  pT linearly, so a single particle-level pT histogram cannot
  distinguish them. Adding a per-species pT histogram (charged vs
  photon vs neutral hadron) to `DEFAULT_OBS_WEIGHTS` breaks the
  degeneracy.
- **Silent parameters.** A pT-hat > 100 GeV dijet sample has
  essentially no charged hadrons with pT < 1 GeV, no K0S particles,
  and few barrel-region tracks in the jet core. Parameters gated to
  those regions have numerically zero gradient and stay at their
  initial value. A broader kinematic selection (minimum bias, lower
  pT-hat) or a separate targeted sample is the right fix.
- **Peak memory.** The autograd graph of the calorimeter
  `scatter_add_` accumulator is currently the dominant memory cost
  (~6.5 GB per 300-event batch). Scaling to 2000-event batches
  requires gradient checkpointing of that block or splitting the
  batch into chunks.

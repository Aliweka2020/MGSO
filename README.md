# MGSO: Modified Glider Snake Optimizer

Official code for the paper **"<PAPER TITLE>"** (<Authors>, <Venue/Journal>, <Year>).

[![Paper](https://img.shields.io/badge/paper-<link>-blue)](#citation)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-<LICENSE>-green)](#license)

MGSO is a population-based metaheuristic for continuous optimisation. It extends the Glider Snake Optimizer (GSO) with
adaptive, mode-switching operators inspired by the flight of the flying snake. This repository contains:

- the **MGSO** implementation and a switchable-component version for **ablation studies**;
- a common benchmarking framework with **seven baseline optimisers** (PSO, GA, GWO, JAYA, DE, AOA, GSO);
- experiments on the **23 classical benchmark functions** (original, shifted and rotated variants);
- a **real-world application**: analog beamforming / beam refinement for mmWave massive MIMO (DeepMIMO ray-tracing channels);
- a deep-learning **beam-prediction** pipeline (CNN / LSTM / BiLSTM / GRU / Transformer) whose Transformer hyper-parameters can be tuned by any of the optimisers.

---

## Table of contents

1. [The MGSO algorithm](#the-mgso-algorithm)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Reproducing the experiments](#reproducing-the-experiments)
6. [Ablation study](#ablation-study)
7. [Using MGSO on your own problem](#using-mgso-on-your-own-problem)
8. [Adding a new optimiser](#adding-a-new-optimiser)
9. [Fair-comparison protocol](#fair-comparison-protocol)
10. [Outputs and crash safety](#outputs-and-crash-safety)
11. [Citation](#citation)
12. [License](#license)

---

## The MGSO algorithm

Each metaphor in MGSO maps to a concrete operator that targets a known weakness of metaheuristics
(`mgso.py`).

| Failure mode | MGSO remedy | Component |
|---|---|---|
| Premature convergence | **Undulation**: sinusoidal difference-vector moves plus Lévy jumps; probability decays with progress but stays above zero | `undulate` |
| Stagnation / lost individuals | **Dive**: an individual with no improvement for `stag_limit` steps is re-launched around the elite (Gaussian + Lévy) | `dive` |
| Hand-tuned control parameters | **Glide**: current-to-pbest/1 move with success-history adapted `F` and `CR` (learned online) | `pbest`, `adapt` |
| Poor scaling to high dimensions | Binomial dimension-mask crossover (only some coordinates move) | `binomial` |
| No fine convergence | **Landing**: in the last phase, part of the population searches locally around the best with a 1/5-success-rule step size | `landing` |
| Population degradation | Greedy elitist selection (offspring replaces parent only if not worse) and leader protection | `greedy`, `leader_protect` |

**Operating modes per iteration:** `0` undulate, `1` glide, `2` dive, `3` land. A shrinking elite pool (25% to 10% of the
population) drives the glide move. For high-dimensional problems, landing starts earlier and involves more individuals.

**Budget:** exactly one call of `f` on `pop` candidates per iteration, i.e. `pop * (iters + 1)` evaluations per run,
identical to every baseline in this repository.

Default hyper-parameters (class attributes in `mgso.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `stag_limit` | 6 | Steps without improvement before an individual dives |
| `land_start` | 0.80 | Fraction of the run after which landing starts (earlier when D is large) |
| `land_frac` | 0.20 | Fraction of the population doing landing (larger when D is large) |

---

## Repository structure

```
.
├── optimizers.py             # Base class + PSO, GA, GWO, JAYA, DE, AOA, GSO, MGSO and the optimizer registry
├── mgso.py                   # MGSO (standalone, documented implementation)
├── mgso_ablation.py          # MGSO with on/off switches for every component + ablation presets
├── gso_impl.py               # Adapter of the original function-style GSO to the Base interface
│
├── benchmark23.py            # 23 classical benchmark functions F1-F23 (vectorised)
├── benchmark23_shift.py      # Same functions with optional random shift and rotation
├── exp_bench23.py            # Experiment: benchmark suite, Mean/STD, ranks, Wilcoxon rank-sum, time/FEs
├── exp_bench23_shift.py      # Same, for shifted / rotated variants
│
├── exp3_beam_optimization.py # Experiment 3: beam optimisation (angle refinement, phase-shifter weights)
├── exp3_ablation.py          # Ablation of MGSO on the beam-optimisation problems (paired design)
│
├── exp2_beam_prediction.py   # Experiment 2: beam prediction with deep models + optimiser-tuned Transformer
├── models.py                 # CNN1D, RNNNet (LSTM/BiLSTM/GRU), TransformerNet, training/evaluation utilities
├── data.py                   # DeepMIMO / synthetic channels, features, spatial-block splits, trajectories
├── common.py                 # Seeding, codebook, steering vectors, shared CLI arguments, crash-safe CSV logger
├── diagnostics.py            # Diagnostic utilities
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate     # optional
pip install numpy scipy pandas scikit-learn matplotlib torch deepmimo
```

- The optimiser and benchmark experiments (`exp_bench23*.py`, MGSO itself) need only `numpy`, `scipy` and `pandas`
  (`matplotlib` for plots).
- `torch` is required only by `common.py`, `data.py`, `models.py` and the beam-prediction experiment.
- `deepmimo` is required only with `--source deepmimo`. The loader follows the **DeepMIMO v4** API. If your installed
  version differs, adapt only `deepmimo_channels()` in `data.py`.
- Without DeepMIMO, `--source synthetic` (default) uses a geometric LoS + 2-reflection fallback channel for pipeline
  testing. It is **not** a substitute for ray-tracing data.

---

## Quick start

Minimise a function with MGSO in a few lines:

```python
import numpy as np
from optimizers import MGSO

f = lambda X: (X ** 2).sum(1)          # X: (pop, dim) -> (pop,), minimisation
lb, ub = np.full(30, -100.0), np.full(30, 100.0)

x_best, f_best, history = MGSO(f, lb, ub, pop=30, iters=500, seed=0).run()
print(f_best)
```

Run a quick benchmark comparison:

```bash
python exp_bench23.py --funcs F1,F9,F14 --runs 5 --iters 100
```

---

## Reproducing the experiments

All experiments give every optimiser the same evaluation budget and the same seeds.

### 1. Classical benchmark functions (F1-F23)

```bash
# Full protocol: 30-D, pop 30, 500 iterations, 30 independent runs
python exp_bench23.py --opts PSO,GA,GWO,JAYA,DE,AOA,GSO,MGSO --dim 30 --pop 30 --iters 500 --runs 30

# Function groups: unimodal | multimodal | fixed | all
python exp_bench23.py --funcs unimodal

# Convergence curves
python exp_bench23.py --plots
```

Shifted and rotated variants (F1-F13 only; the same transformation is used for every optimiser and run):

```bash
python exp_bench23_shift.py --shift                 # optimum moved to a random off-centre point
python exp_bench23_shift.py --shift --rotate        # shift + random rotation (breaks separability)
python exp_bench23_shift.py --rotate                # rotation only
```

Results are written to `results/<orig|shift|rot|shift_rot>/` so runs never overwrite each other. Outputs include
Mean/STD tables (CSV, text, LaTeX), time and function-evaluation tables, average ranks, and Wilcoxon rank-sum
win/tie/loss counts of the reference method (`--ref`, default MGSO) against all others.

### 2. Beam optimisation for massive MIMO (Experiment 3)

Two problems per user channel `h` (normalised so `||h|| = 1`); the score is the **normalised gain** (1.0 = optimum):

- `--problem angle`: refine the steering angle `u = sin(theta)` in `[-1, 1]` (1-D, off-grid refinement of a DFT codebook).
- `--problem phase`: optimise `nt` analog phase-shifter weights (`nt`-D, harder; the analytic optimum is known).

```bash
python exp3_beam_optimization.py --source deepmimo --scenario asu_campus_3p5 --problem angle
python exp3_beam_optimization.py --source deepmimo --scenario asu_campus_3p5 --problem phase \
       --opts PSO,GA,GWO,JAYA,DE,AOA,GSO,MGSO --instances 50 --runs 5
```

Reports the DFT-codebook baseline, mean normalised gain per optimiser, paired Wilcoxon signed-rank tests versus a
reference optimiser, and a convergence plot.

### 3. Beam prediction with optimiser-tuned Transformer (Experiment 2)

Predicts the best beam from a sequence of past channel observations. A Transformer whose hyper-parameters
(model width, heads, layers, feed-forward ratio, dropout, learning rate) are searched by a metaheuristic is compared to
CNN, LSTM, BiLSTM, GRU and a default Transformer.

```bash
python exp2_beam_prediction.py --source deepmimo --n_train 200000 --n_val 20000 --n_test 20000 \
       --seeds 10 --tuners GSO,MGSO
```

Use `--resume` to continue an interrupted run. Train/validation/test sets are separated by **spatial blocks** (`--block`)
to avoid trajectory leakage.

---

## Ablation study

`mgso_ablation.py` exposes one switch per component. With all flags `True` it is the same algorithm as
`mgso.MGSO` (identical budget).

| Flag | `True` | `False` |
|---|---|---|
| `pbest` | glide = current-to-pbest/1 | glide = rand/1 |
| `shrink_elite` | elite pool shrinks 25% to 10% | fixed 25% |
| `adapt` | `F`, `CR` learned online | fixed `F ~ Cauchy(0.5)`, `CR ~ N(0.8)` |
| `undulate` | sinusoidal + Lévy mode | never used |
| `dive` | stagnant individuals re-launched around the best | never |
| `landing` | last-phase local search (1/5 rule) | never |
| `binomial` | dimension-wise crossover | all dimensions move |
| `greedy` | offspring replaces parent only if not worse | offspring always accepted |
| `leader_protect` | best individual never wanders | no protection |

Three study designs are provided:

| Study | Description |
|---|---|
| `loo` | Leave-one-out: full MGSO minus one component |
| `inc` | Incremental: start from a minimal DE/JADE-like core and add components cumulatively |
| `only` | Core plus exactly one component (stand-alone value of each component) |

```bash
python exp3_ablation.py --source deepmimo --scenario asu_campus_3p5 --problem phase \
       --instances 10 --runs 5 --study loo        # loo | inc | only | all
```

Seeds (`1000 * instance + run`) and budget are identical across variants, so comparisons are **paired**. The output
table contains mean, worst run, delta versus full MGSO, Wilcoxon p-value, and the fraction of iterations spent in each
mode (undulate / glide / dive / land).

> p-values are **not** corrected for multiple comparisons. Apply Holm or Bonferroni correction if you report them.

A single variant can also be used directly:

```python
from mgso_ablation import make_variant
NoDive = make_variant("MGSO-noDive", dive=False)
```

---

## Using MGSO on your own problem

All optimisers share one interface (`optimizers.Base`):

- the objective `f` is **vectorised**: it takes `X` of shape `(pop, dim)` and returns `(pop,)`;
- all optimisers **minimise** (maximisation problems: return the negative);
- `run()` returns `(best_x, best_f, history)` where `history` has `iters + 1` entries.

```python
from optimizers import get_optimizers
algos = get_optimizers(["PSO", "DE", "MGSO"])
for name, cls in algos.items():
    x, fv, hist = cls(f, lb, ub, pop=30, iters=100, seed=0).run()
```

---

## Adding a new optimiser

Subclass `Base`, implement `step(t)`, and perform exactly one `s.evaluate(...)` call per step to keep the budget equal:

```python
from optimizers import Base

class MyOpt(Base):
    name = "MyOpt"
    def step(s, t):
        Xn = s.clip(...)              # candidate population, shape (pop, dim)
        Fn = s.evaluate(Xn)           # tracks the global best automatically
        ...
```

Register it in `OPTIMIZERS` in `optimizers.py`.

**NiOA (optional).** If a file `nioa_impl.py` exposing `NiOA` (a `Base` subclass) is present, it is picked up
automatically and can be used as `--opts ...,NiOA`. Without it, NiOA is skipped with a warning.

---

## Fair-comparison protocol

- **Equal budget:** every method uses `pop * (iters + 1)` function evaluations. `exp_bench23.py` counts the actual
  evaluations with a wrapper and warns if any method deviates.
- **Equal seeds:** run `r` on function/instance `i` uses seed `1000 * i + r` for every method.
- **Identical landscapes:** shifted/rotated functions are generated from a fixed `--shift_seed` and shared by all methods.
- **Statistics:** Wilcoxon rank-sum over independent runs (benchmark functions); paired Wilcoxon signed-rank over
  instances (beam optimisation); average ranks across functions.
- **GSO baseline:** ported 1:1 from the original reference implementation. Differences forced by the framework are
  documented in `gso_impl.py` and in the `GSO` docstring in `optimizers.py`.

Notes: function `F7` contains additive uniform noise drawn from NumPy's global RNG, so it is not controlled by the
per-run seed. For shifted `F8`, the true minimum can lie below `-418.98 * d`, so compare algorithms against each other
rather than against the original optimum.

---

## Outputs and crash safety

Results are saved under `results/` (configurable with `--out`):

| File | Content |
|---|---|
| `bench23_mean_std.csv`, `bench23_runs.csv` | Benchmark Mean/STD and every individual run |
| `bench23_time_fes.csv`, `bench23_time_summary.csv` | Runtime and evaluation counts |
| `bench23_mean_std.tex` | LaTeX table (best mean marked `*` in the text version) |
| `bench23_convergence.npz`, `conv_F*.png` | Convergence histories / plots |
| `exp3_<problem>.csv`, `exp3_<problem>_runs.csv` | Beam-optimisation summary and per-run gains |
| `exp3_<problem>_convergence.{npz,png}` | Beam-optimisation convergence |
| `exp3_ablation_<problem>_<study>.csv` | Ablation table |
| `exp2_runs.csv`, `exp2_summary.csv`, `tuned_<opt>.json` | Beam-prediction runs, summary, tuned hyper-parameters |

Runs are appended to CSV and `fsync`ed immediately, so a crash loses nothing. `exp2_beam_prediction.py` accepts
`--resume`: rerun the same command and finished `(model, seed)` rows are skipped. Do not change settings between
resumes.

---

## Citation

If you use this code, please cite:

```bibtex
@article{<key>,
  title   = {<PAPER TITLE>},
  author  = {<Authors>},
  journal = {<Journal>},
  year    = {<Year>},
  volume  = {<Vol>},
  pages   = {<Pages>},
  doi     = {<DOI>}
}
```

---

## License

Released under the <LICENSE> license. See [`LICENSE`](LICENSE) for details.

## Contact

<Name> - <email> - Issues and pull requests are welcome.

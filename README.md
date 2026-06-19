# ElectroPolyBench

**A retrospective active-learning (AL) benchmark seed for electropolymerization / OMIEC materials.**

> **Research question:** *How few experiments do you need?* For electrodeposited
> conducting-polymer optimization, how do active-learning strategies compare against
> random and space-filling baselines when we **retrospectively replay finished
> datasets**?

ElectroPolyBench takes a *completed* experimental campaign, hides all the labels, and
lets different query strategies re-discover the good conditions one experiment at a time.
Because every label is already known, we can replay any strategy over many random restarts
and measure the only thing that matters for a self-driving lab: **data efficiency —
queries-to-target.**

---

## Why this exists (the gap)

As of mid-2026 there is **no shared, open AL benchmark for OMIEC / electropolymerization
materials** (confirmed — unlike, say, molecular property prediction or general BO, this
sub-field has no common retrospective-replay harness). Meanwhile the experimental oracles
in this space are *small*: the PANDA self-driving lab reports **48 + 20 experiments**.
Small-oracle, expensive-experiment regimes are exactly where data efficiency — not
asymptotic accuracy — decides whether a campaign succeeds. ElectroPolyBench is a seed for
filling that gap.

---

## v1 scope (this repo)

**Minimal and deliberately narrow:** prove the retrospective-replay harness end-to-end on
**one real dataset**, produce the headline *queries-to-target vs strategy* figure, and
leave a clean structure to extend. This is **not** the full multi-dataset / multi-fidelity
/ calibrated-UQ benchmark — that is the v2 roadmap below.

**Dataset (v1):** the **PANDA** self-driving-lab dataset of electrodeposited PEDOT:PSS
electrochromic films — Quinn et al., *Materials Horizons* **2024**, DOI
[10.1039/D4MH00797B](https://doi.org/10.1039/D4MH00797B) (arXiv
[2406.17725](https://arxiv.org/abs/2406.17725)). Data is loaded directly from the authors'
repo, [BU-KABlab/PANDA-SDL](https://github.com/BU-KABlab/PANDA-SDL).

- **Inputs (3):** EDOT/monomer concentration (M), deposition potential (V), deposition time (s)
- **Objective:** ΔE00 — CIEDE2000 electrochromic optical contrast (**maximize**)
- **Pool:** 48 (Latin-hypercube screening) + 20 (Bayesian AL) = **68 experiments**

If the real PANDA file cannot be downloaded/parsed, the harness **falls back to a clearly
labelled synthetic surface** (`make_synthetic_surface`) — a documented 3-input continuous
function with one known optimum — so the harness is provably correct and PANDA can be
slotted back in. **A synthetic surface is always named "synthetic" and is never presented
as PANDA data.** The active source is printed at runtime and stamped onto every figure.

**Strategies compared:** `random`, `space_filling` (max-min distance), `greedy_exploit`,
`ucb`, `ei`. **Surrogates:** sklearn `GaussianProcessRegressor` (Matern + input
normalization) and `RandomForestRegressor`, behind a common `fit / predict -> (mean, std)`
interface.

---

## Repository layout

```
ElectroPolyBench/
├── pyproject.toml                 # numpy, pandas, scikit-learn, scipy, matplotlib
├── data/
│   └── loaders.py                 # load_panda() (real) + make_synthetic_surface() (fallback)
├── src/electropolybench/
│   ├── surrogates.py              # GP / RF surrogates: fit(X,y), predict(X)->(mean,std)
│   ├── acquisitions.py            # random, space_filling, greedy_exploit, ucb, ei
│   ├── replay.py                  # retrospective pool-based AL replay engine
│   └── metrics.py                 # queries_to_target, simple_regret_curve, frac-to-top5
├── experiments/
│   └── run_panda_replay.py        # run all strategies × N restarts -> CSV + headline figure
├── tests/
│   └── test_replay.py             # harness sanity checks on a known-optimum surface
└── results/                       # replay_results.csv, queries_to_target.png (generated)
```

Everything in `replay.py`, `acquisitions.py`, `surrogates.py`, and `metrics.py` is
**dataset- and strategy-agnostic** — a new dataset only needs a loader returning the same
`(inputs..., objective)` schema.

---

## How to run

```bash
# 1. install (Python >= 3.9)
pip install -e .

# 2. unit tests (offline; uses the synthetic surface)
pytest -q

# 3. headline experiment: all strategies × 20 restarts on the real PANDA data
python experiments/run_panda_replay.py --restarts 20 --seed-size 4

#    force the synthetic surface instead:
python experiments/run_panda_replay.py --synthetic
```

Outputs land in `results/`:
- `replay_results.csv` — tidy per-(strategy, restart, step) log
- `queries_to_target.png` — **headline figure**: simple-regret curves (mean ± 95% CI) +
  queries-to-target bar chart, AL vs random vs space-filling
- `queries_to_target_summary.csv` — per-strategy mean ± CI summary

The repo works without `pip install` too (a `conftest.py` and a path shim in the
experiment script add `src/` and the root to `sys.path`).

---

## v2 roadmap

This seed is built to grow. Planned extensions, roughly in order:

1. **More datasets.** Add a **cortisol electrochemical molecularly-imprinted-polymer
   (e-MIP)** dataset and an **Olympus** benchmark surface as 2nd/3rd datasets, so results
   are not anecdotal to PANDA.
2. **Low-fidelity feature layer.** Add **OMG-pretrained features** as a low-fidelity layer
   to test whether cheap representations accelerate AL.
3. **Principled BO + calibrated UQ.** Upgrade from sklearn surrogates to **BoTorch / Ax**
   for principled **multi-objective** and **multi-fidelity** Bayesian optimization, and add
   a **calibrated-UQ** comparison (the `metrics.calibration()` stub is the placeholder).
4. **Multi-fidelity decision is gated.** Whether to invest in the multi-fidelity layer is
   **gated on the CombHTS xTB→DFT Fit-1 R²** — if cheap xTB does not predict DFT well
   enough there, a multi-fidelity AL layer here is not yet worth building.

> BoTorch/Ax are intentionally **not** v1 dependencies — the whole point of the seed is to
> prove the retrospective-replay harness with a minimal scikit-learn stack first.

---

## Honesty notes

- The real PANDA data is downloaded from the authors' public GitHub repo; the active
  dataset (real vs synthetic) is printed at runtime and labelled on every figure.
- The unit tests' "AL beats random" assertion is a **harness sanity check** on a synthetic
  unimodal surface — *not* a research claim about PANDA or about AL in general.

## License

MIT.

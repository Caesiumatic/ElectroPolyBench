# Project Status
_Last updated: 2026-06-19_

## Current phase
**v1 seed COMPLETE and clean.** The retrospective pool-based active-learning replay harness
is proven end-to-end on the REAL PANDA dataset, the headline "queries-to-target vs strategy"
figure is produced, and the test suite is green. Scope is deliberately minimal (one real
dataset, sklearn surrogates, a single-objective replay) with a dataset- and strategy-agnostic
structure to extend — NOT the full multi-dataset / multi-fidelity / calibrated-UQ benchmark,
which is the v2 roadmap (README + THINK).

The active dataset is the REAL PANDA self-driving-lab dataset of electrodeposited PEDOT:PSS
electrochromic films (Quinn et al., *Mater. Horiz.* 2024, DOI 10.1039/D4MH00797B; arXiv
2406.17725), downloaded directly from the authors' repo `BU-KABlab/PANDA-SDL`: 48 (LHS
screening) + 20 (Bayesian AL) = **68 experiments**, 3 inputs (EDOT concentration M,
deposition potential V, deposition time s) → objective ΔE00 (CIEDE2000 electrochromic
contrast, maximize). A documented synthetic-surface fallback exists and is wired in, but is
NOT the active source.

## Headline result (real PANDA, N=68, global best ΔE00 = 23.86; 20 random-seed restarts, GP surrogate, seed size 4)
Queries to reach (1 − 0.05) × global best, mean ± 95% CI (all strategies reached the target on every restart):

| strategy | queries-to-target | frac-budget-to-top5% |
| --- | --- | --- |
| greedy_exploit | **19.4 ± 4.8** | 0.10 |
| ucb | 21.1 ± 4.2 | 0.18 |
| ei | 21.6 ± 4.8 | 0.14 |
| random | 26.4 ± 7.3 | 0.17 |
| space_filling | 53.0 ± 9.6 | 0.43 |

All three surrogate-driven AL strategies beat the random baseline; pure-exploration
space-filling is worst (expected — it spends budget covering input space rather than chasing
the optimum). This is a HARNESS demonstration on one dataset, NOT a validated research claim
about AL in OMIEC generally (see Scientific caution).

## What works and is verified
- **Loaders** (`data/loaders.py`): `load_panda()` downloads + tidies the real PANDA CSVs
  (header newline normalized; canonical schema `edot_concentration_M`,
  `deposition_potential_V`, `deposition_time_s`, `delta_E00`, `source`); `make_synthetic_surface()`
  is a documented 3-input Gaussian-bump surface with ONE known optimum on PANDA-like ranges;
  `load_dataset()` tries real first and falls back to synthetic, returning an `info` dict that
  records which source is active and why. Never fabricates PANDA data.
- **Surrogates** (`src/electropolybench/surrogates.py`): a common `Surrogate` ABC with
  `fit` / `predict -> (mean, std)`. `GPSurrogate` = sklearn GaussianProcessRegressor with
  input standardization + a Matern(ν=2.5) × constant + white-noise kernel and `normalize_y`;
  `RFSurrogate` = RandomForestRegressor with per-tree-disagreement std. Registry +
  `make_surrogate(name)` factory.
- **Acquisitions** (`src/electropolybench/acquisitions.py`): `random`, `space_filling`
  (max-min distance in standardized input space), `greedy_exploit` (max predicted mean),
  `ucb` (mean + β·std, β=2), `ei` (expected improvement over best revealed). All share one
  signature; `NEEDS_SURROGATE` marks which consult the model so the engine can skip fitting
  for the surrogate-free baselines.
- **Replay engine** (`src/electropolybench/replay.py`): from a finished (X, y) pool, reveal a
  random seed subset (logged as ONE initial curve step at queries = seed_size), then iterate
  [fit surrogate on revealed → strategy picks next unrevealed index → reveal true y → log
  cumulative queries + best-so-far] until budget. Raises if a strategy ever re-queries a
  revealed index. Returns a `ReplayResult` with the full reveal order, the learning curve, and
  a `simple_regret` property.
- **Metrics** (`src/electropolybench/metrics.py`): `queries_to_target` (relative or absolute
  eps; ∞ if never reached), `simple_regret_curve`, `fraction_of_budget_to_top5`. `calibration()`
  is a documented v2 STUB that raises NotImplementedError.
- **Experiment** (`experiments/run_panda_replay.py`): runs all strategies × N restarts,
  writes `results/replay_results.csv` (tidy per-step log), `results/queries_to_target_summary.csv`,
  and the headline `results/queries_to_target.png` (simple-regret curves with 95% CI + a
  queries-to-target bar chart, stamped with the active dataset + REAL/SYNTHETIC tag).
  `--synthetic` forces the fallback; `--surrogate {gp,rf}`, `--restarts`, `--seed-size`,
  `--budget`, `--eps` are exposed.
- **Tests** (`tests/test_replay.py`): on the synthetic known-optimum surface — (a) exactly one
  new point per step and never a re-query; (b) budget > pool is clipped to the pool; (c) GP-EI
  and GP-UCB reach the optimum in fewer queries than random over repeated seeds (harness sanity
  check); (d) `calibration()` raises (v2 stub). **5 passed** (Python 3.14, sklearn 1.9).
- The package installs cleanly (`pip install -e .`) and also runs from a fresh checkout without
  install via `conftest.py` + the experiment-script path shim.

## Scientific caution
- This is an ENGINEERING / harness milestone on ONE dataset, not a validated benchmark result.
  The strategy ranking above is real but specific to PANDA's 68-point pool, the GP surrogate, a
  seed size of 4, and the relative-5% target definition — all of which are still open choices
  (THINK T1, T2).
- Retrospective pool replay can only ever pick from the MEASURED points. The "global best" is
  the pool maximum, not the true continuous-surface optimum; queries-to-target measures how
  fast a strategy reaches the best EXPERIMENT that was actually run, which is the honest
  retrospective question but is not the same as continuous BO regret (THINK T3).
- The unit tests' "AL beats random" assertion is a sanity check on a synthetic unimodal
  surface — NOT a claim about PANDA or about AL in general.
- sklearn GP-EI/UCB is a screening-grade stand-in for principled BO; it is not multi-objective,
  multi-fidelity, or calibrated-UQ-aware (all v2, THINK T5/T6).

## Placeholders / not yet validated
- `metrics.calibration()` is a v2 STUB (raises NotImplementedError) — the engine does not yet
  log per-step (mean, std) predictions needed for a reliability-diagram / ECE metric.
- Only ONE dataset is wired (PANDA). The cortisol e-MIP and Olympus surfaces named in the v2
  roadmap are NOT present (THINK T4).
- No principled BO: sklearn GP, fixed UCB β = 2, EI ξ = 0.01 — no acquisition-hyperparameter
  study, no BoTorch/Ax (THINK T5).
- No multi-fidelity layer and no OMG-pretrained low-fidelity features; the multi-fidelity
  decision is explicitly GATED on the CombHTS xTB→DFT Fit-1 R² (THINK T6).
- The synthetic fallback is for harness correctness only; its absolute landscape is not meant
  to resemble real electrodeposition physics beyond having PANDA-like input ranges and one
  optimum.

## Open debts
1. (v2, datasets) Add a 2nd and 3rd dataset (cortisol e-MIP; an Olympus benchmark surface) so
   the strategy comparison is not anecdotal to PANDA. Each is just a new loader returning the
   canonical schema. (THINK T4)
2. (v2, BO) Upgrade the surrogate/acquisition layer to BoTorch/Ax for principled
   multi-objective and multi-fidelity BO; keep the sklearn path as the lightweight default.
   (THINK T5)
3. (v2, UQ) Implement `metrics.calibration()` — first log per-step (mean, std) along the replay
   trajectory, then a reliability diagram / expected calibration error, and test whether
   calibrated UQ buys data efficiency.
4. (v2, multi-fidelity, GATED) Add OMG-pretrained features as a low-fidelity layer ONLY after
   the CombHTS xTB→DFT Fit-1 R² shows cheap fidelity predicts the expensive one well enough to
   be worth it. (THINK T6)
5. (protocol) Seed-size and target-definition sensitivity sweep — the headline ranking depends
   on seed_size=4 and the relative-5% target; report how robust it is. (THINK T1, T2)
6. (fairness) space_filling is evaluated as a pure batch-coverage baseline; decide whether a
   hybrid (space-filling seed → AL) is the fairer "baseline" comparison. (THINK T8)
7. (infra) No CI yet (CombHTS has GitHub Actions + ruff). Add a `pytest` + lint workflow when
   the repo gets a remote.
8. (data provenance) PANDA is downloaded live at runtime with no local cache or pinned commit;
   add an optional on-disk cache + record the source commit so a run is reproducible offline.

## Immediate next action
v1 is done. The first v2 step is to add a SECOND dataset (open debt #1) so the harness is
exercised on more than PANDA — this is the cheapest way to find dataset-agnosticism leaks in
the engine before investing in BoTorch (debt #2). The multi-fidelity work (debt #4) stays
parked until the CombHTS Fit-1 R² gate (THINK T6) is known.

## Architecture invariants
See AGENTS.md: dataset- and strategy-agnostic core; one surrogate interface (fit / predict →
mean,std); one acquisition signature; never fabricate PANDA data (synthetic is always named
synthetic); seeded, multi-restart, mean ± CI; minimal sklearn stack for v1.

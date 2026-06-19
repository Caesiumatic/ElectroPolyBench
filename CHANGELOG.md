# Changelog

## 2026-06-19 — seed v1: retrospective AL replay harness, proven on real PANDA
Initial commit of ElectroPolyBench — a minimal, dataset- and strategy-agnostic retrospective
active-learning replay benchmark for electropolymerization / OMIEC materials, answering "how
few experiments do you need?" by replaying finished campaigns and comparing AL strategies
against random / space-filling baselines.

- **Real dataset loaded (NOT the fallback).** `data/loaders.py::load_panda()` downloads the
  authors' PANDA CSVs from `BU-KABlab/PANDA-SDL` (paper: Quinn et al., *Mater. Horiz.* 2024,
  DOI 10.1039/D4MH00797B; arXiv 2406.17725) and tidies them to the canonical schema
  (`edot_concentration_M`, `deposition_potential_V`, `deposition_time_s`, `delta_E00`,
  `source`): 48 LHS-screening + 20 Bayesian-AL = **68 experiments**, 3 inputs → ΔE00 contrast
  (maximize). The raw EDOT header carries an embedded newline (parses as a double space); it is
  normalized on load. `make_synthetic_surface()` is a documented 3-input Gaussian-bump surface
  with ONE known optimum on PANDA-like ranges, used for tests + as a fallback; it is ALWAYS
  named "synthetic" and never presented as PANDA. `load_dataset()` tries real first, falls back
  to synthetic, and returns an `info` dict recording which source is active and why.
- **Surrogates** (`src/electropolybench/surrogates.py`): a `Surrogate` ABC with
  `fit(X,y)` / `predict(X)->(mean,std)`; `GPSurrogate` (sklearn GP, input standardization,
  Matern×constant+white-noise kernel, `normalize_y`) and `RFSurrogate` (per-tree-disagreement
  std). Registry + `make_surrogate()` factory.
- **Acquisitions** (`src/electropolybench/acquisitions.py`): `random`, `space_filling`
  (max-min distance in standardized space), `greedy_exploit`, `ucb` (β=2), `ei` (over best
  revealed) — one shared dataset-agnostic signature, `NEEDS_SURROGATE` marks the model-driven
  ones. Objective convention: maximize.
- **Replay engine** (`src/electropolybench/replay.py`): random seed subset (logged as one
  initial curve step at queries = seed_size) → iterate [fit → pick → reveal → log best-so-far];
  raises on any re-query; returns a `ReplayResult` with reveal order, learning curve, and a
  `simple_regret` property. Strategy- and dataset-agnostic.
- **Metrics** (`src/electropolybench/metrics.py`): `queries_to_target` (relative/absolute eps,
  ∞ if unreached), `simple_regret_curve`, `fraction_of_budget_to_top5`; `calibration()` is a
  documented v2 stub that raises NotImplementedError.
- **Experiment** (`experiments/run_panda_replay.py`): all strategies × 20 restarts → tidy
  `results/replay_results.csv` + `results/queries_to_target_summary.csv` + the headline
  `results/queries_to_target.png` (simple-regret curves with 95% CI + queries-to-target bars,
  stamped with the active dataset and a REAL/SYNTHETIC tag). CLI flags for surrogate, restarts,
  seed size, budget, eps, and `--synthetic` to force the fallback.
- **Tests** (`tests/test_replay.py`): on the synthetic known-optimum surface — exactly one new
  reveal per step / never a re-query; budget > pool is clipped; GP-EI and GP-UCB beat random in
  queries-to-target over repeated seeds (harness sanity check, not a research claim);
  `calibration()` raises. **5 passed** (Python 3.14, sklearn 1.9).
- **Packaging / docs**: `pyproject.toml` (numpy/pandas/scikit-learn/scipy/matplotlib; BoTorch/Ax
  in an optional `v2` extra, noted as the planned upgrade), `conftest.py` + an experiment-script
  path shim so the repo runs without an editable install, and a README stating the research
  question, the confirmed gap (no shared OMIEC/electropolymerization AL benchmark as of mid-2026),
  the v1 scope, how to run, and the v2 roadmap.
- **Engineering fix during build**: the replay engine first logged each seed point as its own
  curve step (so `queries[0]` was 1, not seed_size); changed to log the seed reveal as a single
  initial step at queries = seed_size (standard AL-curve convention), keeping the full reveal
  sequence separately for the no-requery invariant. Test now green.

**Headline (real PANDA, N=68, global best ΔE00 = 23.86; GP surrogate, seed size 4, 20 restarts):**
queries to reach 95% of global best — greedy_exploit 19.4 ± 4.8, ucb 21.1 ± 4.2, ei 21.6 ± 4.8
< random 26.4 ± 7.3 < space_filling 53.0 ± 9.6 (all reached the target on every restart). AL
beats random; pure-exploration space-filling is worst. Demonstration on one dataset, not a
validated research claim.

Added the project-maintenance docs (this CHANGELOG, STATUS.md, AGENTS.md, THINK.md) mirroring
the CombHTS conventions: STATUS is a mutable snapshot, CHANGELOG is append-only history, AGENTS
holds the architecture invariants + documentation-maintenance rules, THINK is the register of
open scientific/decision questions.

PROJECT: A retrospective active-learning (AL) benchmark for electropolymerization / OMIEC
materials. Take a FINISHED experimental campaign as a fixed pool of (X, y) rows, hide the
labels, and replay query strategies that re-discover the good conditions one experiment at
a time. Research question: "How few experiments do you need?" — compare AL strategies
against random / space-filling baselines by data efficiency (queries-to-target), in the
small-oracle regime where it matters (PANDA: 68 experiments).

ARCHITECTURE PRINCIPLES (do not violate):
1. DATASET- AND STRATEGY-AGNOSTIC CORE. The replay engine, acquisition functions,
   surrogates, and metrics only ever see an (X, y) table and a small interface. A new
   dataset is just a loader returning the same `(inputs..., objective)` schema; a new
   strategy is one function with the shared signature. Never special-case PANDA inside
   the engine.
2. ONE SURROGATE INTERFACE. Every surrogate exposes `fit(X, y)` and
   `predict(X) -> (mean, std)`. Acquisition functions touch only that interface, so a
   sklearn GP today and a BoTorch GP tomorrow are drop-in (v2).
3. ONE ACQUISITION SIGNATURE. Every strategy is
   `(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng) -> next_index`,
   registered in `acquisitions.STRATEGIES`. Objective convention: MAXIMIZE y.
4. NEVER FABRICATE "PANDA DATA". The real dataset is downloaded from the authors' public
   source and labelled "PANDA (real)". The synthetic fallback is ALWAYS named "synthetic"
   and is never presented as PANDA. The active source is printed at runtime and stamped
   onto every figure.
5. REPRODUCIBILITY. Every run is seeded (`numpy.random.Generator`); headline results are
   aggregated over many random-seed restarts as mean ± 95% CI, never a single lucky seed.
6. MINIMAL STACK FOR v1. numpy / pandas / scikit-learn / scipy / matplotlib only. BoTorch /
   Ax (principled multi-objective + multi-fidelity BO) and calibrated-UQ are the v2 upgrade;
   the whole point of the seed is to prove the replay harness without them first.
7. HONEST METRICS. Report queries-to-target, simple-regret curves, and fraction-of-budget
   summaries; never collapse to a single flattering number. The pool maximum is the only
   "global best" — regret and target are defined against it, not against an unmeasured
   surface optimum (see THINK T3).

STACK: Python 3.9+ (developed on 3.14), numpy, pandas, scikit-learn, scipy, matplotlib;
pytest for tests. src-layout package `electropolybench`; loaders in a top-level `data`
package. A root `conftest.py` + a path shim in the experiment script make both importable
without an editable install.
STYLE: type hints, docstrings stating the physical meaning + units of every quantity
(concentration M, potential V, time s, contrast ΔE00 dimensionless), small pure functions,
a pytest for every module. The objective convention (maximize) is stated wherever it matters.

DOCUMENTATION MAINTENANCE:
At the end of every work unit: (1) overwrite STATUS.md to reflect the new current state
(phase, what works, placeholders, open debts, next action); (2) prepend a dated entry to
CHANGELOG.md describing what changed and why. STATUS.md is a mutable snapshot; CHANGELOG.md
is append-only history. Keep both concise.
THINK.md is the register of open scientific/decision questions (the why/what-if layer). At
the end of a work unit, if a genuine research question, tradeoff, or sign-off need was
opened, advanced, or resolved, update the relevant THINK.md entry (open -> exploring ->
decided/parked) and its Decision log. Keep engineering debt in STATUS.md; THINK.md is only
for items needing scientific judgment. Do not duplicate STATUS content.

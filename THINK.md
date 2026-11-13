# THINK

THINK.md is the register of OPEN SCIENTIFIC / RESEARCH / DECISION questions for this project —
the "why and what-if" layer. It is distinct from STATUS.md (a mutable snapshot of current state)
and CHANGELOG.md (append-only history). THINK.md holds only items that require genuine scientific
judgment, a tradeoff, or a sign-off — NOT routine engineering debt (those stay in STATUS.md).
Entries are opened, updated as thinking evolves, and marked `decided`/`parked` with a resolution;
this file is neither a snapshot nor append-only.

_Research entries: 2026-06-19; repository names and navigation updated 2026-11-13._

## How to read this

Each entry is a question we have not fully resolved. The `Forum` field says who decides: self =
objective/rigor call we make and merely report; group-meeting = a tradeoff to surface for
discussion; PI sign-off = a resource/scope/policy call. The `Cost` field says how expensive
resolution is.

| ID | Title | Status | Forum | Cost |
| --- | --- | --- | --- | --- |
| T1 | Replay protocol: seed size and restart count | open | self (rigor) | one compute sweep |
| T2 | What "queries-to-target" should target | exploring | group-meeting | scope/policy |
| T3 | Pool replay measures the wrong optimum (measured pool vs continuous surface) | open | self (rigor) | conceptual |
| T4 | Which datasets to add for v2 | open | group-meeting | lit/data-curation |
| T5 | sklearn GP now vs BoTorch/Ax (principled + multi-objective BO) | exploring (staged) | group-meeting | compute/eng |
| T6 | Multi-fidelity low-fidelity layer — GATED on electropoly-jointopt-hts xTB→DFT Fit-1 R² | parked (gated) | PI sign-off | compute-heavy |
| T7 | Is a GP well-posed on a 4-point seed? surrogate choice for tiny data | open | self (rigor) | one compute sweep |
| T8 | Is space_filling a fair baseline, or a strawman? | exploring | group-meeting | scope/policy |

## T1 — Replay protocol: seed size and restart count
- **Status**: open
- **Forum**: self (rigor)
- **Cost**: one compute sweep
- **Question**: The headline ranking uses a random seed subset of size 4 and 20 random-seed
  restarts. How sensitive is the strategy ordering (and its CI overlap) to those two knobs?
- **Why it matters**: With only 68 pool points, a seed of 4 is ~6% of the pool; too small and
  the surrogate's first fit is degenerate (the GP optimizes a length-scale on 4 points), too
  large and the seed alone may already be near the target, masking the strategy's contribution.
  20 restarts gives visibly wide CIs (random ±7.3 queries) — the ei/ucb/greedy gap is real but
  the three are within each other's CI, so we cannot yet rank AMONG the AL strategies.
- **Thinking / options**: Sweep seed_size ∈ {2, 4, 8, 12} × restarts ∈ {20, 50, 100} and report
  the ranking stability + CI width. Expect: AL-vs-random separation robust; intra-AL ordering
  (greedy vs ucb vs ei) only resolvable at higher restart counts, if at all on this one dataset.
- **Current lean**: Keep seed_size=4 / 20 restarts as the cheap default for the seed, but treat
  any intra-AL ordering as NOT established until a restart sweep tightens the CIs. Report
  AL-vs-baseline as the only durable v1 claim.
- **Resolves when**: a seed/restart sweep is run and STATUS/README state the ranking only at the
  resolution the CIs actually support.
- **Links**: [Archived status debt #5](docs/history/status-before-2026-11-11.md#open-debts); [experiments/run_panda_replay.py](experiments/run_panda_replay.py).

## T2 — What "queries-to-target" should target
- **Status**: exploring
- **Forum**: group-meeting
- **Cost**: scope/policy
- **Question**: The primary metric is "queries until best-revealed y reaches (1 − eps) × global
  best", eps = 0.05 relative. Is relative-5%-of-the-pool-max the right target, vs an absolute
  ΔE00 tolerance, vs "reach the top-5% band" (already reported separately as
  `fraction_of_budget_to_top5`)?
- **Why it matters**: The target definition IS the benchmark's headline number; different
  definitions can reorder strategies. Relative-to-pool-max rewards finding the single best point;
  top-k-band rewards finding ANY very-good point (more forgiving, arguably closer to what a lab
  wants — "a good enough film fast"). A relative tolerance also behaves oddly if an objective can
  be near zero or negative (handled in code, but ΔE00 ≥ 0 here so it is moot for PANDA).
- **Thinking / options**: (a) keep relative-5%-of-max as headline, top-5%-band as secondary
  (current); (b) make absolute-ΔE00 the headline since contrast has a meaningful unit
  (a 1-unit ΔE00 difference is perceptually real); (c) report a small panel of targets and let
  the reader pick. For a cross-dataset benchmark (v2), an absolute objective tolerance does not
  transfer across datasets with different objectives, which argues for relative or top-k.
- **Current lean**: relative-to-max for cross-dataset comparability, top-k-band as the
  lab-relevant companion; revisit once a 2nd dataset exists and we see whether the two targets
  ever disagree on the ranking.
- **Resolves when**: the group picks the headline target definition (and whether it is
  dataset-relative), ideally after a 2nd dataset shows whether the choice changes conclusions.
- **Links**: [src/electropolybench/metrics.py](src/electropolybench/metrics.py); [T4](#t4--which-datasets-to-add-for-v2).

## T3 — Pool replay measures the wrong optimum (measured pool vs continuous surface)
- **Status**: open
- **Forum**: self (rigor)
- **Cost**: conceptual (no compute)
- **Question**: Retrospective pool replay can only ever query the 68 EXPERIMENTS THAT WERE
  ACTUALLY RUN. The "global best" is the pool maximum, not the true continuous-surface optimum.
  Is "queries to reach the best measured point" the honest question, and what does it NOT tell us?
- **Why it matters**: A strategy that is great at INTERPOLATING a continuous response surface and
  proposing a NEW unmeasured optimum is invisible in pool replay — it can only be credited for
  reaching points someone already measured. PANDA itself was collected BY a Bayesian-AL campaign
  (the 20 AL points cluster near the optimum), so the pool is not an unbiased grid; replaying AL
  on an AL-collected pool can flatter AL (the good region is densely sampled, so any
  exploitation strategy finds it fast) — a subtle selection effect.
- **Thinking / options**: (a) Accept pool replay as the honest retrospective question and state
  the caveat loudly (current). (b) Fit a surrogate "oracle" on ALL pool points and let strategies
  query a continuous space against the oracle's prediction — but then we are benchmarking against
  a model, not data (circular). (c) Report whether the AL-collected subset biases the result by
  also replaying on the 48 LHS-only screening points (an unbiased space-filling design) and
  comparing. Option (c) is cheap and directly tests the selection-effect worry.
- **Current lean**: keep pool replay (option a) as the v1 design, but add the LHS-only control
  (option c) as a robustness check before making any cross-dataset claim. Never describe pool
  queries-to-target as continuous-surface regret.
- **Resolves when**: the caveat is documented (done in STATUS) AND the LHS-only-vs-full-pool
  control is run to bound the AL-collected-pool selection effect.
- **Links**: [Archived scientific caution](docs/history/status-before-2026-11-11.md#scientific-caution); [data/loaders.py](data/loaders.py) (the `source` column distinguishes training/LHS vs AL points).

## T4 — Which datasets to add for v2
- **Status**: open
- **Forum**: group-meeting
- **Cost**: lit/data-curation
- **Question**: The v2 roadmap names a cortisol electrochemical-MIP (e-MIP) dataset and an
  Olympus benchmark surface as the 2nd/3rd datasets. Are those the right next two, and in what
  order?
- **Why it matters**: One dataset cannot tell dataset-agnosticism from PANDA-overfitting; the
  cheapest way to find leaks in the "agnostic" engine is a second loader. Olympus surfaces are
  analytic/emulated (clean, continuous, no pool-bias of T3) and good for stress-testing;
  cortisol e-MIP is another real small-oracle electro-materials campaign (on-topic for OMIEC) but
  needs the same provenance care as PANDA (real source, never fabricated).
- **Thinking / options**: Olympus first (synthetic-but-standard, removes the T3 pool-bias
  confound, fast to wire) to validate the engine; then cortisol e-MIP (real, on-domain) to make
  the comparison scientifically interesting. Each is just a loader returning the canonical
  `(inputs..., objective)` schema.
- **Current lean**: Olympus surface next for engine validation, cortisol e-MIP after for a 2nd
  REAL on-domain point; keep PANDA as the flagship real dataset.
- **Resolves when**: the group confirms the 2nd/3rd datasets and their order, and each lands as a
  loader with documented provenance.
- **Links**: [Archived status debt #1](docs/history/status-before-2026-11-11.md#open-debts); [Current remaining work](STATUS.md#remaining-work).

## T5 — sklearn GP now vs BoTorch/Ax (principled + multi-objective BO)
- **Status**: exploring (staged: sklearn now, BoTorch later)
- **Forum**: group-meeting
- **Cost**: compute/eng
- **Question**: v1 uses sklearn GP-EI/UCB with fixed acquisition hyperparameters (β=2, ξ=0.01).
  When do we upgrade to BoTorch/Ax for principled single-/multi-objective and multi-fidelity BO,
  and does the upgrade change the v1 conclusions?
- **Why it matters**: PANDA's real objective is arguably multi-objective (contrast AND switching
  speed / charge), which sklearn cannot do natively; BoTorch's qNEHVI etc. are the right tool.
  But BoTorch/torch is a heavy dependency that contradicts the v1 "minimal stack" principle, and
  the headline AL-beats-random result almost certainly does NOT need it.
- **Thinking / options**: Keep the sklearn path as the lightweight default and ALWAYS-available
  baseline; add BoTorch behind the optional `v2` extra so installing it is opt-in. First BoTorch
  deliverable: reproduce the v1 single-objective ranking (sanity), THEN add multi-objective
  (hypervolume regret) using PANDA's secondary measurements. The surrogate interface
  (`fit`/`predict→mean,std`) is already designed so a BoTorch GP is a drop-in for single-objective.
- **Current lean**: staged — sklearn is the v1 default and stays; BoTorch/Ax is the v2 upgrade,
  opt-in, justified primarily by multi-objective / multi-fidelity, not by single-objective AL.
- **Resolves when**: the group agrees BoTorch is warranted (i.e. the multi-objective question is
  in scope) and the BoTorch path first reproduces the sklearn single-objective result.
- **Links**: [Archived status debt #2](docs/history/status-before-2026-11-11.md#open-debts); [pyproject.toml](pyproject.toml) (current optional dependencies; the proposed `v2` extra is not implemented); [T6](#t6-multi-fidelity-gate).

<a id="t6-multi-fidelity-gate"></a>

## T6 — Multi-fidelity low-fidelity layer — GATED on electropoly-jointopt-hts xTB→DFT Fit-1 R²
- **Status**: parked (gated on an external result)
- **Forum**: PI sign-off
- **Cost**: compute-heavy
- **Question**: Should v2 add a multi-fidelity AL layer (a cheap low-fidelity signal — e.g.
  OMG-pretrained features, or a cheap computed descriptor — used to triage before the expensive
  experiment), and is it worth the engineering?
- **Why it matters**: Multi-fidelity AL only pays off if the cheap fidelity actually predicts the
  expensive one. That is exactly the question the SIBLING project electropoly-jointopt-hts answers for
  electropolymerization via its two-stage calibration: Fit-1 is xTB→DFT. If cheap xTB does NOT
  predict DFT well there (low Fit-1 R²), a multi-fidelity layer HERE is built on sand and should
  not be funded; if Fit-1 R² is high, the analogous low-fidelity→high-fidelity triage is
  promising and worth porting.
- **Thinking / options**: This is a deliberate cross-project gate, not an internal one. DECISION
  RULE: do not start the multi-fidelity layer until the electropoly-jointopt-hts xTB→DFT Fit-1 R² is known from
  its live g16 DFT-calibration batch; treat that R² as the go/no-go evidence. (As of 2026-06-19
  the electropoly-jointopt-hts live batch is still running — see electropoly-jointopt-hts STATUS/THINK T4 — so this stays parked.)
- **Current lean**: parked until the electropoly-jointopt-hts Fit-1 R² lands; then PI decides go/no-go on the
  multi-fidelity layer based on that R².
- **Resolves when**: the electropoly-jointopt-hts xTB→DFT Fit-1 R² is reported and the PI signs off on whether to
  build the multi-fidelity layer.
- **Links**: [Archived status debt #4](docs/history/status-before-2026-11-11.md#open-debts); [Current remaining work](STATUS.md#remaining-work); [electropoly-jointopt-hts calibration record](https://github.com/Caesiumatic/electropoly-jointopt-hts/blob/main/docs/runs/2026-06-19_dftcal-417442-gasphase.md) (the xTB→DFT calibration that supplies the gating R²); [T5](#t5--sklearn-gp-now-vs-botorchax-principled--multi-objective-bo).

## T7 — Is a GP well-posed on a 4-point seed? surrogate choice for tiny data
- **Status**: open
- **Forum**: self (rigor)
- **Cost**: one compute sweep
- **Question**: At seed_size=4 the GP optimizes a per-dimension Matern length-scale (plus
  constant + white-noise) on 4 points — borderline ill-posed. Does the marginal-likelihood
  optimizer just hit the kernel bounds early (the ConvergenceWarnings we see), and would a
  fixed-length-scale GP, a more constrained kernel, or the RF surrogate be more honest in the
  first few steps?
- **Why it matters**: If the early-step surrogate is essentially uninformative, then "AL" in the
  first handful of queries is closer to random/space-filling than the label suggests, and the
  data-efficiency credit may belong to the seed and to later steps. The sklearn ConvergenceWarnings
  (length-scale hitting bounds) are a symptom worth understanding, not suppressing blindly.
- **Thinking / options**: Compare GPSurrogate vs RFSurrogate vs a fixed-hyperparameter GP across
  the same restarts; inspect at which query count the surrogate starts to beat random. Possibly
  widen kernel bounds or use a prior on the length-scale rather than free MLE on 4 points.
- **Current lean**: acknowledge the small-n GP is screening-grade early on; run a surrogate
  comparison (GP vs RF vs fixed-kernel GP) before claiming the surrogate (not just the
  acquisition policy) is what drives the win.
- **Resolves when**: a surrogate-comparison sweep shows whether the conclusion is robust to the
  surrogate choice, and the kernel/bounds are set deliberately rather than by default.
- **Links**: [src/electropolybench/surrogates.py](src/electropolybench/surrogates.py); [pyproject.toml](pyproject.toml) (the pytest ConvergenceWarning filter); [T1](#t1--replay-protocol-seed-size-and-restart-count).

## T8 — Is space_filling a fair baseline, or a strawman?
- **Status**: exploring
- **Forum**: group-meeting
- **Cost**: scope/policy
- **Question**: space_filling finishes worst by a wide margin (53 vs ~20–26 queries). Is that the
  right comparison — a pure max-min-coverage design that never looks at the objective — or should
  the "space-filling baseline" be the more realistic "space-filling seed, THEN switch to AL"?
- **Why it matters**: A pure space-filling design is GUARANTEED to be slow at reaching a single
  optimum (it is optimizing coverage, not the objective) — beating it is almost a tautology, so it
  risks reading as a strawman. The honest baselines are (a) random (cheap, no model) and (b) a
  realistic two-phase "space-filling seed → exploit" that practitioners actually use. The current
  seed IS random; a space-filling seed feeding AL is untested.
- **Thinking / options**: Keep pure space_filling as a labelled REFERENCE (it shows the cost of
  ignoring the objective entirely), but add a two-phase "LHS/space-filling seed → AL" variant as
  the fairer baseline, and make sure random — not space_filling — is framed as the baseline AL has
  to beat.
- **Current lean**: pure space_filling stays as a reference point, not "the baseline"; random is
  the baseline of record; add a space-filling-seed→AL variant in v2 to avoid the strawman reading.
- **Resolves when**: the group agrees which baseline is the headline comparator and the two-phase
  variant is added (or explicitly declined).
- **Links**: [Archived status debt #6](docs/history/status-before-2026-11-11.md#open-debts); [src/electropolybench/acquisitions.py](src/electropolybench/acquisitions.py).

## Decision log

- 2026-06-19 (v1 seed) — built the dataset- and strategy-agnostic retrospective replay harness and
  proved it end-to-end on the REAL PANDA dataset (68 experiments). Opened T1–T8. Key honest
  framings recorded at birth: the AL-vs-random separation is the only durable v1 claim; intra-AL
  ordering is within-CI and NOT established (T1); pool replay measures queries-to-best-MEASURED-point,
  not continuous-surface regret, and PANDA being an AL-collected pool may flatter exploitation (T3);
  the multi-fidelity layer is deliberately GATED on the external CombHTS xTB→DFT Fit-1 R² (T6);
  BoTorch/Ax is staged as a v2 opt-in justified by multi-objective/multi-fidelity, not by
  single-objective AL (T5). No claim is made beyond a one-dataset harness demonstration.

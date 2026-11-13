# electropoly-active-compute status

## Current state

The replay package compares five acquisition strategies through shared GP and
random-forest surrogate interfaces. The recorded PANDA comparison covers 68
measured conditions, 20 restarts, and four initial observations. Its CSVs and
figure are available in `results/`.

New runs select PANDA or synthetic data explicitly and write to separate run
directories. Each run records its input CSV, SHA-256, settings, per-step
observations, and per-restart metrics. `--base-seed` controls the restart seeds.

The [README](README.md) presents the method and recorded query-efficiency
comparison. [The experiment guide](docs/experiments.md) documents setup, options,
outputs, confidence intervals, and source entry points.

## Repository name synchronization — 2026-11-13

- Updated the README introduction, module descriptions, contributor display name,
  and plot heading to `electropoly-active-compute`. Python imports and the package
  distribution name remain `electropolybench`.
- Updated the sibling-project references and repaired research-note navigation;
  the multi-fidelity decision remains parked.
- Regenerated the README figure with `plot_headline` in
  `experiments/run_panda_replay.py` from `results/replay_results.csv` and
  `results/queries_to_target_summary.csv`. Its source data,
  strategy summaries, and uncertainty calculations are unchanged; no replay or
  laboratory experiments were run.
- Checked current documentation links, Python syntax, the figure layout, and
  `git diff --check`. Old names remain in dated history and package paths.

## Documentation verification — 2026-11-13 (earlier revision)

- Renamed the repository to `electropoly-active-compute` and updated the README
  title; the Python package remains `electropolybench`.
- Revised the overview and comments to describe the fixed-pool comparison,
  explicit synthetic selection, and target-threshold definitions directly.
- Checked Python syntax trees with docstrings removed and token streams with
  comments/docstrings removed; executable code and runtime strings are unchanged.
- Checked local links and `git diff --check`; recorded results are unchanged.
- This revision ran no replay experiments or laboratory campaigns.

## Recorded verification — 2026-11-11

- 7 tests passed using the existing Python 3.14 / sklearn 1.9 environment.
- Offline RF replay: 5 strategies × 2 restarts, 8-query budget; input hash,
  explicit synthetic source, seed, and all 10 per-restart records verified.
- Recorded PANDA CSVs and figure were byte-identical to the source checkout.
- Wheel build passed with the existing Python 3.11 / setuptools environment.

## Remaining work

- PANDA is fetched from upstream `main`; new runs snapshot inputs, but the
  historical run predates that snapshot mechanism.
- Calibration metrics and multi-dataset/multi-fidelity extensions are not
  implemented. Scientific decisions remain in [THINK.md](THINK.md).
- Maintain the documented replay interface and experiment records.

[Previous status and research notes](docs/history/status-before-2026-11-11.md).

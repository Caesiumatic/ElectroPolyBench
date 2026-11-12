# ElectroPolyBench status

## Current state

The overview has been shortened to a technical repository introduction.

The replay package, five acquisition strategies, GP/RF surrogates, PANDA loader,
and recorded 68-condition comparison are present in this checkout. Source was
restored from local commit `1642518092c85cf762b56a949ed6ca1bac7e0f95`.
Commit metadata maintenance has since updated the existing remote `main` to
`29d33bd7bbc4e5518908926b5146816e97eb72b5`, preserving file trees, human authors,
timestamps, and commit topology. This checkout now uses that complete history
as its base; all pending working files were retained.

The local revision adds explicit dataset selection and separates new run outputs
from the recorded `results/`. Each run saves its input CSV, SHA-256, settings,
per-step records, and per-restart metrics. `--base-seed` controls restarts.

The README describes the method, recorded results, and documentation entry points. Setup,
options, outputs, confidence intervals, and the code map are collected in
[`docs/experiments.md`](docs/experiments.md). The final README revision changes documentation only; prior code validation
remains recorded below.

## Verification — 2026-11-11

- 7 tests passed using the existing Python 3.14 / sklearn 1.9 environment.
- Offline RF replay: 5 strategies × 2 restarts, 8-query budget; input hash,
  explicit synthetic source, seed, and all 10 per-restart records verified.
- Recorded PANDA CSVs and figure are byte-identical to the source checkout.
- Wheel build passed with the existing Python 3.11 / setuptools environment.
- Removed speculative optional dependencies; core acquisition/replay math is unchanged.

## Remaining work

- PANDA is fetched from upstream `main`; new runs snapshot inputs, but the
  historical run predates that snapshot mechanism.
- Calibration metrics and multi-dataset/multi-fidelity extensions are not
  implemented. Scientific decisions remain in `THINK.md`.
- Maintain the documented replay interface and experiment records.

[Previous status and research notes](docs/history/status-before-2026-11-11.md).

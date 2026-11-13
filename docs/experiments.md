# Running and configuring experiments

[Project overview](../README.md)

## Setup

Use Python 3.9 or later. Clone the repository and install its dependencies:

```bash
git clone https://github.com/Caesiumatic/electropoly-active-compute.git
cd electropoly-active-compute
python -m pip install -e '.[dev]'
python -m pytest -q
```

The package uses NumPy, pandas, scikit-learn, SciPy, and Matplotlib. Tests use a
synthetic surface and do not download PANDA data.

## PANDA replay

```bash
python experiments/run_panda_replay.py --restarts 20 --seed-size 4 --output-dir runs/panda
```

The loader downloads the authors' CSVs from
[BU-KABlab/PANDA-SDL](https://github.com/BU-KABlab/PANDA-SDL). The default pool
combines 48 training experiments and 20 active-learning experiments. The ten
post-AL validation replicates are included only with `--include-post-al`.

Inputs are monomer concentration (M), deposition potential (V), and deposition
time (s). The objective is optical contrast, ΔE00, which is maximized.
PANDA download or parsing errors stop the run.

## Offline example

```bash
python experiments/run_panda_replay.py --synthetic --surrogate rf \
  --restarts 2 --budget 8 --output-dir runs/example
```

`--synthetic` explicitly selects a 216-point synthetic surface. This small run
exercises data loading, all five strategies, metrics, and figure generation.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--restarts` | `20` | Number of restarts per strategy |
| `--seed-size` | `4` | Initially revealed observations |
| `--base-seed` | `0` | Restart `r` uses seed `base_seed + r` |
| `--budget` | Whole pool | Total revealed observations, including the initial set |
| `--surrogate` | `gp` | Gaussian process (`gp`) or random forest (`rf`) |
| `--eps` | `0.05` | Target tolerance; for PANDA, target = `(1 - eps) × pool maximum` |
| `--synthetic` | Off | Select the offline synthetic dataset |
| `--include-post-al` | Off | Include PANDA post-AL validation replicates |
| `--output-dir` | `runs/panda` or `runs/synthetic` | Destination for this run's files |

Use a separate output directory to keep multiple runs. Reusing a directory
replaces its output files. The default destinations are separate from the
recorded [`results/`](../results/) directory.

## Outputs

| File | Contents |
| --- | --- |
| `dataset.csv` | Input pool used by the run |
| `run.json` | Dataset identity, input SHA-256, strategy list, and run settings |
| `replay_results.csv` | Per-strategy, per-restart, per-step observations and regret |
| `queries_to_target_runs.csv` | Per-restart target and top-5%-band metrics |
| `queries_to_target_summary.csv` | Strategy means, interval half-widths, and reach fractions |
| `queries_to_target.png` | Regret curves and queries-to-target comparison |

The PANDA loader reads upstream `main`; the saved input and hash identify the
data used in each new run. The historical results predate input snapshots.

## Recorded comparison

The recorded PANDA comparison uses a GP surrogate, 20 restarts, four initial
observations, and the full 68-condition pool. Query counts include the initial
observations. All runs reached 95% of the best measured contrast.

| Strategy | Mean queries | 95% CI half-width |
| --- | ---: | ---: |
| Greedy exploitation | 19.35 | 4.81 |
| UCB | 21.10 | 4.16 |
| Expected improvement | 21.55 | 4.83 |
| Random sampling | 26.35 | 7.33 |
| Space filling | 52.95 | 9.56 |

Intervals use 1.96 × standard error across restarts. For runs with a shorter
budget, queries-to-target means summarize only the restarts that reached the
target; `reached_frac` reports their fraction. Simple regret is the difference
between the pool maximum and the best revealed outcome. The top-5% metric uses
the pool's 95th-percentile outcome as its threshold.

[Per-step results](../results/replay_results.csv) ·
[Summary with confidence intervals](../results/queries_to_target_summary.csv) ·
[Recorded figure](../results/queries_to_target.png)

## Code map

| Path | Purpose |
| --- | --- |
| [`replay.py`](../src/electropolybench/replay.py) | Label reveal sequence and replay records |
| [`acquisitions.py`](../src/electropolybench/acquisitions.py) | Query strategies |
| [`surrogates.py`](../src/electropolybench/surrogates.py) | GP and random-forest models |
| [`metrics.py`](../src/electropolybench/metrics.py) | Queries to target, regret, and top-5% metrics |
| [`loaders.py`](../data/loaders.py) | PANDA and synthetic input pools |
| [`run_panda_replay.py`](../experiments/run_panda_replay.py) | Experiment options, aggregation, and figures |

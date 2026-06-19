"""Harness sanity checks on a tiny synthetic surface with a KNOWN optimum.

These are correctness checks on the replay machinery, not research claims:

(a) the replay engine reveals exactly one new point per step and never re-queries;
(b) a GP-driven AL strategy (EI / UCB) reaches the optimum in fewer queries than random,
    averaged over repeated random seeds, on a simple unimodal surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.loaders import INPUT_COLUMNS, OBJECTIVE_COLUMN, make_synthetic_surface
from electropolybench import metrics
from electropolybench.replay import run_replay


@pytest.fixture(scope="module")
def surface():
    df = make_synthetic_surface(n_per_dim=6)  # 6**3 = 216 points, single optimum
    X = df[INPUT_COLUMNS].to_numpy(float)
    y = df[OBJECTIVE_COLUMN].to_numpy(float)
    return X, y


def test_reveals_one_new_point_per_step_no_requery(surface):
    X, y = surface
    res = run_replay(X, y, strategy="ei", surrogate="gp",
                     seed_size=4, budget=20, seed=0)

    order = res.revealed_order
    # never re-queries: all revealed indices are distinct
    assert len(order) == len(set(order)), "an index was revealed more than once"
    # exactly one new point per step after the seed
    assert len(order) == 20
    # cumulative query count increases by exactly 1 each logged step
    queries = np.asarray(res.queries)
    assert queries[0] == res.seed_size
    assert np.all(np.diff(queries) == 1)
    # best-so-far is monotonically non-decreasing (maximization)
    best = np.asarray(res.best_y)
    assert np.all(np.diff(best) >= -1e-12)


def test_budget_and_pool_bounds(surface):
    X, y = surface
    # budget larger than pool is clipped to the pool size; never exceeds it
    res = run_replay(X, y, strategy="random", seed_size=3,
                     budget=10_000, seed=1)
    assert len(res.revealed_order) == len(X)
    assert len(set(res.revealed_order)) == len(X)


@pytest.mark.parametrize("al_strategy", ["ei", "ucb"])
def test_al_beats_random_in_queries_to_target(surface, al_strategy):
    X, y = surface
    global_best = float(np.max(y))
    seeds = range(12)
    budget = 30
    eps = 0.05

    def mean_qtt(strategy):
        vals = []
        for s in seeds:
            res = run_replay(X, y, strategy=strategy, surrogate="gp",
                             seed_size=4, budget=budget, seed=s)
            q, b = res.as_arrays()
            qtt = metrics.queries_to_target(q, b, global_best, eps=eps)
            # cap unreached runs at the budget so the comparison stays finite
            vals.append(qtt if np.isfinite(qtt) else budget)
        return float(np.mean(vals))

    al_qtt = mean_qtt(al_strategy)
    rand_qtt = mean_qtt("random")
    assert al_qtt < rand_qtt, (
        f"{al_strategy} (mean QTT={al_qtt:.1f}) did not beat random "
        f"(mean QTT={rand_qtt:.1f}) on the synthetic surface"
    )


def test_calibration_is_v2_stub():
    with pytest.raises(NotImplementedError):
        metrics.calibration()

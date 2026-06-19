"""Data-efficiency metrics for the retrospective replay benchmark.

All metrics operate on the per-step logs produced by ``replay.run_replay`` (or directly
on ``queries`` / ``best_y`` arrays), so they are dataset- and strategy-agnostic.

Objective convention: y is MAXIMIZED (e.g. ΔE00 electrochromic contrast).
"""

from __future__ import annotations

import numpy as np


def queries_to_target(queries, best_y, global_best, eps: float = 0.05,
                      relative: bool = True) -> float:
    """Number of experiments until best-revealed-y first comes within ``eps`` of the best.

    Parameters
    ----------
    queries  : cumulative experiment count after each step (monotonically increasing).
    best_y   : best revealed objective after each step.
    global_best : the true pool maximum.
    eps : tolerance. If ``relative`` (default), the target is ``(1 - eps) * global_best``
          for non-negative objectives; otherwise the target is ``global_best - eps``
          (absolute units of y).
    relative : choose relative vs absolute tolerance.

    Returns
    -------
    The cumulative query count at the first step that reaches the target, or ``np.inf``
    if the target is never reached within the logged budget.
    """
    queries = np.asarray(queries, dtype=int)
    best_y = np.asarray(best_y, dtype=float)
    if relative:
        target = (1.0 - eps) * global_best if global_best >= 0 else (1.0 + eps) * global_best
    else:
        target = global_best - eps
    hit = np.flatnonzero(best_y >= target)
    if hit.size == 0:
        return float("inf")
    return float(queries[hit[0]])


def simple_regret_curve(best_y, global_best) -> np.ndarray:
    """Simple regret at each step: ``global_best - best_revealed_y`` (>= 0, lower better)."""
    return float(global_best) - np.asarray(best_y, dtype=float)


def fraction_of_budget_to_top5(queries, best_y, y_pool, budget=None,
                               top_frac: float = 0.05) -> float:
    """Fraction of the budget needed to reach the top-``top_frac`` band of the pool.

    The threshold is the ``(1 - top_frac)`` quantile of all pool y-values (default: the
    top-5% band). Returns ``queries_to_reach / budget`` in [0, 1], or ``np.inf`` if the
    band is never reached within the logged steps.
    """
    queries = np.asarray(queries, dtype=int)
    best_y = np.asarray(best_y, dtype=float)
    y_pool = np.asarray(y_pool, dtype=float).ravel()
    threshold = np.quantile(y_pool, 1.0 - top_frac)
    if budget is None:
        budget = int(queries[-1])
    hit = np.flatnonzero(best_y >= threshold)
    if hit.size == 0:
        return float("inf")
    return float(queries[hit[0]]) / float(budget)


def calibration(*args, **kwargs):
    """[v2 STUB] Uncertainty-calibration metric for probabilistic surrogates.

    Planned for v2: compare predicted vs. empirical coverage of the surrogate's
    predictive intervals along the replay trajectory (e.g. a reliability diagram /
    expected calibration error), to test whether well-calibrated UQ actually buys
    data efficiency. Requires logging per-step (mean, std) predictions, which the v1
    engine does not yet store.
    """
    raise NotImplementedError("calibration() is a v2 stub; see docstring and README roadmap.")

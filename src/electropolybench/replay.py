"""Retrospective pool-based active-learning replay engine.

The core idea of the benchmark: take a *finished* experimental campaign as a fixed pool
of (X, y) rows, hide all the labels, then let a strategy re-discover the good points one
query at a time. Because every label is already known, we can replay any strategy as many
times as we like and ask the only question that matters for a self-driving lab:

    "How few experiments do you need to reach (near) the best outcome?"

The engine is strategy- and dataset-agnostic: it only sees an (X, y) table, a strategy
function (see ``acquisitions``), a surrogate factory (see ``surrogates``), a seed size,
and a budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import acquisitions
from .surrogates import make_surrogate


@dataclass
class ReplayResult:
    """Log of one replay run.

    ``revealed_order`` / ``revealed_y`` record EVERY revealed point in reveal order
    (length == total experiments revealed). ``queries`` / ``best_y`` record the learning
    *curve*: the initial random seed is logged as a single step at ``queries == seed_size``,
    and each subsequent active-learning step adds exactly one experiment.
    """

    strategy: str
    seed_size: int
    budget: int
    global_best: float
    revealed_order: list = field(default_factory=list)  # pool indices, full reveal sequence
    revealed_y: list = field(default_factory=list)       # y aligned with revealed_order
    queries: list = field(default_factory=list)          # cumulative #experiments per curve step
    best_y: list = field(default_factory=list)           # best revealed y per curve step

    def as_arrays(self):
        return (
            np.asarray(self.queries, dtype=int),
            np.asarray(self.best_y, dtype=float),
        )

    @property
    def simple_regret(self) -> np.ndarray:
        """global_best - best_revealed_y at each step (>= 0, smaller is better)."""
        return self.global_best - np.asarray(self.best_y, dtype=float)


def _make_rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def run_replay(
    X,
    y,
    strategy: str = "ei",
    surrogate: str = "gp",
    seed_size: int = 4,
    budget: int | None = None,
    seed: int = 0,
    surrogate_kwargs: dict | None = None,
) -> ReplayResult:
    """Replay one strategy over a finished (X, y) pool.

    Parameters
    ----------
    X : (n_pool, n_dim) array-like of inputs.
    y : (n_pool,) array-like of objective values (MAXIMIZED).
    strategy : key into ``acquisitions.STRATEGIES``.
    surrogate : key into ``surrogates.SURROGATES`` (only fit when the strategy needs it).
    seed_size : number of randomly chosen points revealed before the AL loop starts.
    budget : total number of experiments to reveal (incl. the seed). Defaults to the
             whole pool, so every strategy is replayed to exhaustion.
    seed : int / Generator controlling the random seed subset and tie-breaking.

    Returns
    -------
    ReplayResult with the per-step reveal order, cumulative query count, and best-so-far.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_pool = X.shape[0]
    if seed_size < 1:
        raise ValueError("seed_size must be >= 1")
    if seed_size > n_pool:
        raise ValueError(f"seed_size ({seed_size}) exceeds pool size ({n_pool})")

    if budget is None:
        budget = n_pool
    budget = min(budget, n_pool)

    strat_fn = acquisitions.get_strategy(strategy)
    needs_surrogate = strategy in acquisitions.NEEDS_SURROGATE
    surrogate_kwargs = surrogate_kwargs or {}
    rng = _make_rng(seed)

    global_best = float(np.max(y))
    result = ReplayResult(
        strategy=strategy, seed_size=seed_size, budget=budget, global_best=global_best
    )

    # --- random seed subset (logged as ONE initial curve step) --------------
    revealed_mask = np.zeros(n_pool, dtype=bool)
    seed_idx = rng.choice(n_pool, size=seed_size, replace=False)
    for idx in seed_idx:
        revealed_mask[idx] = True
        result.revealed_order.append(int(idx))
        result.revealed_y.append(float(y[idx]))
    result.queries.append(int(revealed_mask.sum()))
    result.best_y.append(float(y[revealed_mask].max()))

    # --- active-learning loop ----------------------------------------------
    while revealed_mask.sum() < budget:
        revealed_idx = np.flatnonzero(revealed_mask)
        unrevealed_idx = np.flatnonzero(~revealed_mask)
        revealed_y = y[revealed_idx]

        surr = None
        if needs_surrogate:
            surr = make_surrogate(surrogate, **surrogate_kwargs).fit(X[revealed_idx], revealed_y)

        nxt = strat_fn(X, revealed_idx, revealed_y, unrevealed_idx, surr, rng)
        if revealed_mask[nxt]:
            raise RuntimeError(
                f"Strategy {strategy!r} re-queried already-revealed index {nxt}."
            )

        revealed_mask[nxt] = True
        result.revealed_order.append(int(nxt))
        result.revealed_y.append(float(y[nxt]))
        result.queries.append(int(revealed_mask.sum()))
        result.best_y.append(float(y[revealed_mask].max()))

    return result

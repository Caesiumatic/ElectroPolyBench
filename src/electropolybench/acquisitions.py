"""Acquisition strategies for pool-based active learning.

Strategies share a dataset-independent interface::

    next_index = strategy(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng)

where
  X             : (n_pool, n_dim) features for the complete experimental pool
  revealed_idx  : 1-D int array of pool indices already revealed
  revealed_y    : 1-D float array of revealed objective values (aligned with revealed_idx)
  unrevealed_idx: 1-D int array of candidate pool indices not yet revealed
  surrogate     : a fitted Surrogate (already fit on the revealed data), or None
  rng           : a numpy Generator for reproducible tie-breaking / random choices

Each strategy returns one global pool index from ``unrevealed_idx``.

The objective y is maximized (e.g. ΔE00 electrochromic contrast).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _argmax_with_tiebreak(scores: np.ndarray, candidates: np.ndarray, rng) -> int:
    """Return the candidate index with the max score, breaking ties uniformly at random."""
    best = np.flatnonzero(scores == scores.max())
    return int(candidates[rng.choice(best)])


def random(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng) -> int:
    """Pick a uniformly random unrevealed point. The pure-exploration baseline."""
    return int(rng.choice(unrevealed_idx))


def space_filling(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng) -> int:
    """Select by max-min distance in standardized feature space.

    Pick the unrevealed point whose distance to its nearest already-revealed point is
    largest, i.e. fill the most empty region of input space. Distances are computed in
    standardized input space so that physical units do not dominate. This baseline
    measures the value of feature-space coverage without fitting a surrogate.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    cand = Xs[unrevealed_idx]
    if len(revealed_idx) == 0:
        # No reference yet: pick the point closest to the centroid for a stable start.
        d = np.linalg.norm(cand - Xs.mean(axis=0), axis=1)
        return _argmax_with_tiebreak(-d, unrevealed_idx, rng)

    ref = Xs[revealed_idx]
    # nearest-revealed distance for each candidate
    dists = np.linalg.norm(cand[:, None, :] - ref[None, :, :], axis=2).min(axis=1)
    return _argmax_with_tiebreak(dists, unrevealed_idx, rng)


def greedy_exploit(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng) -> int:
    """Pure exploitation: pick the unrevealed point with the highest predicted mean."""
    X = np.asarray(X, dtype=float)
    mean, _ = surrogate.predict(X[unrevealed_idx])
    return _argmax_with_tiebreak(mean, unrevealed_idx, rng)


def ucb(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng, beta: float = 2.0) -> int:
    """Upper confidence bound: maximize mean + beta * std."""
    X = np.asarray(X, dtype=float)
    mean, std = surrogate.predict(X[unrevealed_idx])
    return _argmax_with_tiebreak(mean + beta * std, unrevealed_idx, rng)


def ei(X, revealed_idx, revealed_y, unrevealed_idx, surrogate, rng, xi: float = 0.01) -> int:
    """Expected improvement over the best revealed value (maximization)."""
    X = np.asarray(X, dtype=float)
    mean, std = surrogate.predict(X[unrevealed_idx])
    best = float(np.max(revealed_y)) if len(revealed_y) else 0.0
    std = np.maximum(std, 1e-9)
    imp = mean - best - xi
    z = imp / std
    ei_vals = imp * norm.cdf(z) + std * norm.pdf(z)
    ei_vals = np.maximum(ei_vals, 0.0)
    return _argmax_with_tiebreak(ei_vals, unrevealed_idx, rng)


# Registry of dataset-agnostic strategies.
STRATEGIES = {
    "random": random,
    "space_filling": space_filling,
    "greedy_exploit": greedy_exploit,
    "ucb": ucb,
    "ei": ei,
}

# Fit a surrogate only for strategies that use its predictions.
NEEDS_SURROGATE = {"greedy_exploit", "ucb", "ei"}


def get_strategy(name: str):
    key = name.lower()
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; options: {sorted(STRATEGIES)}")
    return STRATEGIES[key]

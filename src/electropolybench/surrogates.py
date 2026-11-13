"""Surrogate models with a shared fit and predictive mean/standard-deviation interface.

The Gaussian process uses standardized inputs and a Matern plus white-noise
kernel. The random forest estimates uncertainty from per-tree disagreement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class Surrogate(ABC):
    """Minimal surrogate interface used by the rest of the harness."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        """Fit on revealed (X, y). Returns self."""

    @abstractmethod
    def predict(self, X: np.ndarray):
        """Return (mean, std), each shape (n_samples,)."""


class GPSurrogate(Surrogate):
    """Gaussian-process surrogate with input normalization.

    Inputs are standardized (zero mean / unit variance) before fitting so that a
    single length-scale-per-dimension Matern kernel is well conditioned across the
    very different physical units of the inputs (concentration in M, potential in V,
    time in s). The target is normalized internally by sklearn (``normalize_y``).
    """

    def __init__(self, nu: float = 2.5, n_restarts_optimizer: int = 5,
                 random_state: int | None = 0):
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self._scaler: StandardScaler | None = None
        self._gp: GaussianProcessRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        n_dim = Xs.shape[1]
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e2), nu=self.nu)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self._gp.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray):
        if self._gp is None or self._scaler is None:
            raise RuntimeError("GPSurrogate.predict called before fit.")
        Xs = self._scaler.transform(np.asarray(X, dtype=float))
        mean, std = self._gp.predict(Xs, return_std=True)
        return mean.ravel(), std.ravel()


class RFSurrogate(Surrogate):
    """Random-forest surrogate; predictive std from per-tree disagreement.

    The mean averages the tree predictions; the standard deviation measures
    their disagreement. This uncertainty proxy can underestimate predictive
    error. A positive floor keeps acquisition denominators nonzero.
    """

    def __init__(self, n_estimators: int = 300, random_state: int | None = 0,
                 std_floor: float = 1e-6, **rf_kwargs):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.std_floor = std_floor
        self.rf_kwargs = rf_kwargs
        self._rf: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self._rf.fit(X, y)
        return self

    def predict(self, X: np.ndarray):
        if self._rf is None:
            raise RuntimeError("RFSurrogate.predict called before fit.")
        X = np.asarray(X, dtype=float)
        per_tree = np.stack([est.predict(X) for est in self._rf.estimators_], axis=0)
        mean = per_tree.mean(axis=0)
        std = per_tree.std(axis=0) + self.std_floor
        return mean.ravel(), std.ravel()


# Registry so callers can request a surrogate by name.
SURROGATES = {
    "gp": GPSurrogate,
    "rf": RFSurrogate,
}


def make_surrogate(name: str = "gp", **kwargs) -> Surrogate:
    """Factory: ``make_surrogate("gp")`` -> a fresh, unfitted surrogate."""
    key = name.lower()
    if key not in SURROGATES:
        raise KeyError(f"Unknown surrogate {name!r}; options: {sorted(SURROGATES)}")
    return SURROGATES[key](**kwargs)

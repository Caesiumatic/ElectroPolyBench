"""Dataset loaders for ElectroPolyBench.

Available sources:

1. ``load_panda()``  -- the REAL self-driving-lab dataset of electrodeposited
   PEDOT:PSS electrochromic films (Quinn et al., *Mater. Horiz.* 2024).
2. ``make_synthetic_surface()`` -- a DOCUMENTED synthetic 3-input surface with a single
   known optimum for offline replay checks.

``load_dataset()`` returns the requested source and its metadata. PANDA download
or parsing errors propagate to the caller.

Canonical schema (both sources expose exactly these columns):
    INPUT_COLUMNS  = ["edot_concentration_M", "deposition_potential_V", "deposition_time_s"]
    OBJECTIVE_COLUMN = "delta_E00"          # maximize
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

INPUT_COLUMNS = ["edot_concentration_M", "deposition_potential_V", "deposition_time_s"]
OBJECTIVE_COLUMN = "delta_E00"

# ---------------------------------------------------------------------------
# Real PANDA dataset
# ---------------------------------------------------------------------------
#
# Source paper:
#   J. A. Quinn et al., "Self-driving lab discovers electrodeposited PEDOT:PSS
#   electrochromic films" (PANDA), Materials Horizons, 2024.
#   DOI: 10.1039/D4MH00797B   |   arXiv: 2406.17725
#
# Code & data (authors' official repo, Keith A. Brown / BU-KABlab):
#   https://github.com/BU-KABlab/PANDA-SDL
#
# The experimental campaign is committed as plain CSVs under ``ExperimentalData/``:
#   PEDOT-data-TrainingData.csv        48 rows  (Latin-hypercube initial screening)
#   PEDOT-data-ALexperiments.csv       20 rows  (Bayesian active-learning campaign)
#   PEDOT-data-postALexperiments.csv   10 rows  (validation replicates at predicted optimum)
#
# Raw-file schema (note: the EDOT column header contains an embedded newline, so it
# parses with a double space "edot  concentration (M)"):
#   experiment_id, description,
#   edot  concentration (M)   -> input  (monomer / EDOT concentration, mol/L)
#   deposition potential (V)  -> input  (vs Ag/Ag+, volts)
#   deposition time (s)       -> input  (seconds)
#   delta E00                 -> OBJECTIVE: CIEDE2000 ΔE00 electrochromic contrast (maximize)
#   Bleach Charge Passed      -> auxiliary measurement (Coulombs) [dropped here]
#   PEDOT Predicted DeltaE00 / Uncertainty -> the paper's own model outputs (AL file only) [dropped]
#
# Default benchmark pool = TrainingData (48) + ALexperiments (20) = 68 distinct
# experiments, matching the paper's reported "48 + 20". The post-AL file is excluded by
# default because it is replicate validation runs at one near-optimal condition.
_PANDA_BASE = "https://raw.githubusercontent.com/BU-KABlab/PANDA-SDL/main/ExperimentalData/"
_PANDA_FILES = {
    "training": "PEDOT-data-TrainingData.csv",
    "al": "PEDOT-data-ALexperiments.csv",
    "post_al": "PEDOT-data-postALexperiments.csv",
}

# Mapping from raw (whitespace-normalized) headers -> canonical schema.
_PANDA_RENAME = {
    "edot concentration (M)": "edot_concentration_M",
    "deposition potential (V)": "deposition_potential_V",
    "deposition time (s)": "deposition_time_s",
    "delta E00": "delta_E00",
}


def _read_panda_csv(filename: str, timeout: float = 30.0) -> pd.DataFrame:
    url = _PANDA_BASE + filename
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (trusted host)
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    # Normalize headers: collapse embedded newlines / repeated whitespace.
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df


def load_panda(include_post_al: bool = False, timeout: float = 30.0) -> pd.DataFrame:
    """Load the real PANDA PEDOT:PSS electrodeposition dataset.

    Downloads the authors' CSVs from GitHub (``BU-KABlab/PANDA-SDL``) and returns a
    tidy DataFrame with exactly ``INPUT_COLUMNS + [OBJECTIVE_COLUMN]`` plus a
    ``source`` label column. Raises on any network/parse error so callers can fall back.

    Parameters
    ----------
    include_post_al : also append the 10 post-AL validation replicates (default False).

    Returns
    -------
    DataFrame with columns:
        edot_concentration_M, deposition_potential_V, deposition_time_s, delta_E00, source
    ``df.attrs["dataset"]`` is set to ``"PANDA (real)"`` with provenance.
    """
    keys = ["training", "al"] + (["post_al"] if include_post_al else [])
    frames = []
    for key in keys:
        raw = _read_panda_csv(_PANDA_FILES[key], timeout=timeout)
        raw = raw.rename(columns=_PANDA_RENAME)
        missing = [c for c in INPUT_COLUMNS + [OBJECTIVE_COLUMN] if c not in raw.columns]
        if missing:
            raise ValueError(f"PANDA file {key!r} missing expected columns: {missing}")
        sub = raw[INPUT_COLUMNS + [OBJECTIVE_COLUMN]].copy()
        sub["source"] = key
        frames.append(sub)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=INPUT_COLUMNS + [OBJECTIVE_COLUMN]).reset_index(drop=True)
    df.attrs["dataset"] = "PANDA (real)"
    df.attrs["doi"] = "10.1039/D4MH00797B"
    df.attrs["source_url"] = "https://github.com/BU-KABlab/PANDA-SDL"
    return df


# ---------------------------------------------------------------------------
# Synthetic fallback surface
# ---------------------------------------------------------------------------
def make_synthetic_surface(n_per_dim: int = 6, noise: float = 0.0,
                           seed: int = 0) -> pd.DataFrame:
    """Documented synthetic 3-input surface with ONE known global optimum.

    This is NOT PANDA — it is a deliberately simple, fully-specified function used to
    (a) prove the replay harness is correct in unit tests and (b) provide a deterministic
    fallback if the real PANDA download fails.

    The inputs span PANDA-like physical ranges:
        edot_concentration_M  in [0.01, 0.10]
        deposition_potential_V in [0.60, 1.80]
        deposition_time_s      in [1.0, 100.0]

    The objective is a single smooth Gaussian bump (anisotropic, on normalized inputs)
    with its maximum at the center of each range, so the global optimum is known exactly::

        x_norm = (x - lo) / (hi - lo)            # each input -> [0, 1]
        objective = AMP * exp(-sum_d ((x_norm_d - 0.5) / WIDTH_d)^2)

    giving a clean unimodal landscape that AL should solve in far fewer queries than random.

    Returns a DataFrame on a regular grid (``n_per_dim`` points per axis) with the same
    columns as ``load_panda`` (``source == "synthetic"``). ``df.attrs["optimum_row"]`` is
    the index of the grid point closest to the true optimum.
    """
    rng = np.random.default_rng(seed)
    bounds = {
        "edot_concentration_M": (0.01, 0.10),
        "deposition_potential_V": (0.60, 1.80),
        "deposition_time_s": (1.0, 100.0),
    }
    axes = [np.linspace(lo, hi, n_per_dim) for (lo, hi) in bounds.values()]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T

    los = np.array([b[0] for b in bounds.values()])
    his = np.array([b[1] for b in bounds.values()])
    x_norm = (grid - los) / (his - los)

    AMP = 20.0
    WIDTH = np.array([0.35, 0.30, 0.40])  # anisotropic so axes are not interchangeable
    obj = AMP * np.exp(-np.sum(((x_norm - 0.5) / WIDTH) ** 2, axis=1))
    if noise > 0:
        obj = obj + rng.normal(0.0, noise, size=obj.shape)

    df = pd.DataFrame(grid, columns=INPUT_COLUMNS)
    df[OBJECTIVE_COLUMN] = obj
    df["source"] = "synthetic"
    df.attrs["dataset"] = "synthetic surface"
    df.attrs["optimum_row"] = int(np.argmax(obj))
    return df


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def load_dataset(prefer_real: bool = True, include_post_al: bool = False,
                 verbose: bool = True):
    """Return ``(df, info)`` for the active dataset.

    ``prefer_real=True`` selects PANDA; ``False`` selects the synthetic surface.
    """
    if prefer_real:
        df = load_panda(include_post_al=include_post_al)
        info = {
            "name": "PANDA (real)",
            "is_real": True,
            "n": len(df),
            "source_url": df.attrs.get("source_url"),
            "doi": df.attrs.get("doi"),
            "reason": "Downloaded PANDA CSVs from BU-KABlab/PANDA-SDL.",
        }
        if verbose:
            print(f"[loaders] Loaded PANDA: {len(df)} experiments.")
        return df, info

    df = make_synthetic_surface()
    info = {
        "name": "synthetic surface",
        "is_real": False,
        "n": len(df),
        "source_url": None,
        "doi": None,
        "reason": "Synthetic surface selected explicitly.",
    }
    if verbose:
        print(f"[loaders] Loaded synthetic surface: {len(df)} grid points.")
    return df, info

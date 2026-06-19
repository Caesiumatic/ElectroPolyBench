"""Headline experiment: replay all strategies on the loaded dataset.

Loads the real PANDA dataset (or the documented synthetic fallback), replays every
acquisition strategy over many random-seed restarts, aggregates the mean +/- CI of the
simple-regret curve, and writes:

    results/replay_results.csv      -- tidy per-(strategy, restart, step) log
    results/queries_to_target.png   -- headline "queries-to-target vs strategy" figure

Run:
    python experiments/run_panda_replay.py
    python experiments/run_panda_replay.py --restarts 20 --seed-size 4 --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ and repo root importable when run as a plain script.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.loaders import INPUT_COLUMNS, OBJECTIVE_COLUMN, load_dataset  # noqa: E402
from electropolybench import metrics  # noqa: E402
from electropolybench.replay import run_replay  # noqa: E402

STRATEGIES = ["random", "space_filling", "greedy_exploit", "ucb", "ei"]
STRATEGY_COLORS = {
    "random": "#888888",
    "space_filling": "#1f77b4",
    "greedy_exploit": "#2ca02c",
    "ucb": "#ff7f0e",
    "ei": "#d62728",
}


def run_all(df, strategies, restarts, seed_size, budget, surrogate, base_seed=0):
    X = df[INPUT_COLUMNS].to_numpy(float)
    y = df[OBJECTIVE_COLUMN].to_numpy(float)
    global_best = float(np.max(y))

    rows = []
    for strat in strategies:
        for r in range(restarts):
            res = run_replay(
                X, y,
                strategy=strat,
                surrogate=surrogate,
                seed_size=seed_size,
                budget=budget,
                seed=base_seed + r,
            )
            queries, best_y = res.as_arrays()
            regret = res.simple_regret
            for step in range(len(queries)):
                rows.append({
                    "strategy": strat,
                    "restart": r,
                    "step": step,
                    "queries": int(queries[step]),
                    "best_y": float(best_y[step]),
                    "simple_regret": float(regret[step]),
                    "global_best": global_best,
                })
    return pd.DataFrame(rows), global_best


def summarize_queries_to_target(long_df, global_best, y_pool, eps=0.05):
    """Per (strategy, restart): queries to reach (1-eps)*global_best; aggregate mean/CI."""
    recs = []
    for (strat, r), g in long_df.groupby(["strategy", "restart"]):
        g = g.sort_values("queries")
        qtt = metrics.queries_to_target(
            g["queries"].to_numpy(), g["best_y"].to_numpy(), global_best, eps=eps
        )
        frac = metrics.fraction_of_budget_to_top5(
            g["queries"].to_numpy(), g["best_y"].to_numpy(), y_pool,
            budget=int(g["queries"].max()),
        )
        recs.append({"strategy": strat, "restart": r,
                     "queries_to_target": qtt, "frac_budget_to_top5": frac})
    qdf = pd.DataFrame(recs)

    summ = []
    for strat, g in qdf.groupby("strategy"):
        finite = g["queries_to_target"].replace(np.inf, np.nan).dropna()
        n_reached = int(finite.shape[0])
        mean = float(finite.mean()) if n_reached else float("nan")
        sem = float(finite.std(ddof=1) / np.sqrt(n_reached)) if n_reached > 1 else 0.0
        summ.append({
            "strategy": strat,
            "qtt_mean": mean,
            "qtt_ci95": 1.96 * sem,
            "reached_frac": n_reached / len(g),
            "frac_budget_to_top5_mean": float(g["frac_budget_to_top5"]
                                              .replace(np.inf, np.nan).dropna().mean()),
        })
    return qdf, pd.DataFrame(summ).sort_values("qtt_mean")


def plot_headline(long_df, summary, global_best, info, out_png, eps=0.05):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: mean +/- 95% CI simple-regret curves -----------------------
    for strat in STRATEGIES:
        g = long_df[long_df["strategy"] == strat]
        if g.empty:
            continue
        by_q = g.groupby("queries")["simple_regret"]
        q = by_q.mean().index.to_numpy()
        mean = by_q.mean().to_numpy()
        n = by_q.count().to_numpy()
        sd = by_q.std(ddof=1).fillna(0.0).to_numpy()
        ci = 1.96 * sd / np.sqrt(np.maximum(n, 1))
        c = STRATEGY_COLORS.get(strat, None)
        ax_left.plot(q, mean, label=strat, color=c, lw=2)
        ax_left.fill_between(q, mean - ci, mean + ci, color=c, alpha=0.15)
    ax_left.set_xlabel("cumulative experiments (queries)")
    ax_left.set_ylabel("simple regret  (global best − best revealed)")
    ax_left.set_title("Simple-regret curve  (mean ± 95% CI)")
    ax_left.legend(title="strategy")
    ax_left.grid(alpha=0.3)

    # ---- right: queries-to-target bar chart -------------------------------
    s = summary.copy()
    order = s.sort_values("qtt_mean")["strategy"].tolist()
    s = s.set_index("strategy").loc[order].reset_index()
    colors = [STRATEGY_COLORS.get(st, "#444") for st in s["strategy"]]
    ax_right.bar(s["strategy"], s["qtt_mean"], yerr=s["qtt_ci95"],
                 color=colors, capsize=4, alpha=0.9)
    for i, row in s.iterrows():
        if row["reached_frac"] < 1.0:
            ax_right.text(i, 0.5, f"{row['reached_frac']*100:.0f}% reached",
                          ha="center", va="bottom", fontsize=8, rotation=90, color="white")
    ax_right.set_ylabel(f"queries to reach (1−{eps:g})×global best")
    ax_right.set_title("Queries-to-target  (lower = more data-efficient)")
    ax_right.grid(axis="y", alpha=0.3)
    plt.setp(ax_right.get_xticklabels(), rotation=20, ha="right")

    tag = "REAL PANDA" if info["is_real"] else "SYNTHETIC (not PANDA)"
    fig.suptitle(f"ElectroPolyBench — retrospective AL replay  |  dataset: {info['name']} "
                 f"[{tag}], N={info['n']}, global best ΔE00={global_best:.2f}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--seed-size", type=int, default=4)
    ap.add_argument("--budget", type=int, default=None,
                    help="experiments to reveal (default: whole pool)")
    ap.add_argument("--surrogate", default="gp", choices=["gp", "rf"])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--synthetic", action="store_true",
                    help="force the synthetic surface instead of real PANDA")
    ap.add_argument("--include-post-al", action="store_true")
    args = ap.parse_args()

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    df, info = load_dataset(prefer_real=not args.synthetic,
                            include_post_al=args.include_post_al)
    print(f"[run] dataset={info['name']} (is_real={info['is_real']}), N={info['n']}, "
          f"reason={info['reason']}")

    y_pool = df[OBJECTIVE_COLUMN].to_numpy(float)
    long_df, global_best = run_all(
        df, STRATEGIES, args.restarts, args.seed_size, args.budget, args.surrogate
    )

    out_csv = results_dir / "replay_results.csv"
    long_df.to_csv(out_csv, index=False)
    print(f"[run] wrote {out_csv}  ({len(long_df)} rows)")

    qdf, summary = summarize_queries_to_target(long_df, global_best, y_pool, eps=args.eps)
    summary.to_csv(results_dir / "queries_to_target_summary.csv", index=False)

    out_png = results_dir / "queries_to_target.png"
    plot_headline(long_df, summary, global_best, info, out_png, eps=args.eps)
    print(f"[run] wrote {out_png}")

    print(f"\n=== Queries to reach (1-{args.eps:g}) x global best (ΔE00={global_best:.3f}) ===")
    print(f"{'strategy':<16}{'qtt_mean':>10}{'±95%CI':>10}{'reached':>10}{'frac_top5':>12}")
    for _, row in summary.iterrows():
        print(f"{row['strategy']:<16}{row['qtt_mean']:>10.2f}{row['qtt_ci95']:>10.2f}"
              f"{row['reached_frac']*100:>9.0f}%{row['frac_budget_to_top5_mean']:>12.3f}")


if __name__ == "__main__":
    main()

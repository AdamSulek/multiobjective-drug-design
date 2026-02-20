#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
k = 0.0
algorithm = "rectangle"
DATA_GLOB = f"data/labels/{algorithm}/k_{k}/iter_*.parquet"
X_COL = "score_3GVB"
Y_COL = "score_6D6P"
OUT = f"data/labels/{algorithm}/k_{k}/pareto_iterations_10.png"

BACKGROUND_SAMPLE = 200_000

# Global Pareto points (already in "higher better; docking inverted" convention)
GLOBAL_PARETO = np.array([
    [11.600000381469727, 18.799999237060547],
    [12.0,               18.100000381469727],
    [12.699999809265137, 17.600000381469727],
    [12.100000381469727, 17.700000762939453],
    [13.100000381469727, 17.299999237060547],
], dtype=np.float32)

# ======================
# LOGGING
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("pareto")

# ======================
# utils
# ======================
def get_iter_num(path):
    m = re.search(r"iter_(\d+)", path)
    return int(m.group(1)) if m else -1


def get_pareto(points):
    """
    Pareto for maximize in 2D.
    Returns Pareto-optimal points (subset of input).
    """
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = (
                np.any(points[is_pareto] > p, axis=1)
                | np.all(points[is_pareto] == p, axis=1)
            )
    return points[is_pareto]


def make_stairs(x, y):
    """
    Build step coordinates for Pareto staircase (where='post').
    Assumes x/y are already in "higher is better" convention.
    """
    if len(x) == 0:
        return np.array([]), np.array([])

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]

    X = [xs[0]]
    Y = [ys[0]]
    for i in range(1, len(xs)):
        X.append(xs[i])
        Y.append(Y[-1])   # horizontal
        X.append(xs[i])
        Y.append(ys[i])   # vertical

    return np.asarray(X), np.asarray(Y)


def load_scores_parquet(path, x_col, y_col):
    df = pd.read_parquet(path, columns=[x_col, y_col])
    return df[[x_col, y_col]].to_numpy(dtype=np.float32, copy=False)

# ======================
# MAIN
# ======================
def main():
    files = sorted(glob.glob(DATA_GLOB), key=get_iter_num)
    if not files:
        raise SystemExit("No files matched: %s" % DATA_GLOB)

    log.info("Found iterations: %d (from %s to %s)", len(files), files[0], files[-1])

    # Background from last iteration (docking -> invert sign -> maximize)
    log.info("Loading background from last iteration: %s", files[-1])
    bg = load_scores_parquet(files[-1], X_COL, Y_COL)
    bg = -bg  # docking -> maximize (your convention)

    if BACKGROUND_SAMPLE and len(bg) > BACKGROUND_SAMPLE:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(bg), size=BACKGROUND_SAMPLE, replace=False)
        bg = bg[idx]
        log.info("Background downsampled to %d points", len(bg))
    else:
        log.info("Background points: %d", len(bg))

    # Prepare global pareto (already inverted convention)
    g = GLOBAL_PARETO.copy()
    g = g[np.isfinite(g).all(axis=1)]
    if len(g) == 0:
        log.warning("GLOBAL_PARETO is empty after filtering.")
    else:
        g = np.unique(g, axis=0)
        log.info("Global Pareto points: %d", len(g))

    # Plot
    plt.figure(figsize=(9, 8))
    plt.scatter(bg[:, 0], bg[:, 1], s=2, alpha=0.05, label="All (last iter)")

    # Global front first
    if len(g) > 0:
        gxs, gys = make_stairs(g[:, 0], g[:, 1])
        plt.step(gxs, gys, where="post", linewidth=3, label="Global Pareto (3M pool)")
        plt.scatter(g[:, 0], g[:, 1], s=70, label="Global Pareto points")

    # Iteration fronts
    for f in files:
        it = get_iter_num(f)
        log.info("Processing iter %03d: %s", it, f)

        arr = load_scores_parquet(f, X_COL, Y_COL)
        arr = -arr  # docking -> maximize

        pareto = get_pareto(arr)
        log.info("  points=%d | pareto=%d", len(arr), len(pareto))

        xs, ys = make_stairs(pareto[:, 0], pareto[:, 1])
        plt.step(xs, ys, where="post", linewidth=2, label="iter %03d" % it)

    plt.xlabel("%s (higher better; docking inverted)" % X_COL)
    plt.ylabel("%s (higher better; docking inverted)" % Y_COL)
    plt.title("Pareto frontier evolution across iterations")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    log.info("Saved: %s", OUT)


if __name__ == "__main__":
    main()

# Run in background:
# nohup python -u plot_pareto_iterations.py > pareto_iterations_rectangle_k0.0.log 2>&1 &

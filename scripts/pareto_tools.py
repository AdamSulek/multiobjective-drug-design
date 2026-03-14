#!/usr/bin/env python3
"""Pareto front hypervolume computation, error metrics, and visualization."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from screen_2stage import pareto_front_2d, get_pareto_points, stair_y_at_x


# ======================================================
# Computation
# ======================================================

def hypervolume_2d(points: np.ndarray, ref: Tuple[float, float]) -> float:
    """Compute the 2-D hypervolume indicator (dominated area) for maximization.

    Parameters
    ----------
    points : (N, 2) array of objective values.
    ref : reference point (rx, ry); only points dominating ref contribute.

    Returns
    -------
    Hypervolume (float >= 0).
    """
    rx, ry = float(ref[0]), float(ref[1])
    pts = np.asarray(points, dtype=float).reshape(-1, 2)

    # keep only points that dominate the reference point
    mask = (pts[:, 0] > rx) & (pts[:, 1] > ry)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0

    front = pareto_front_2d(pts)  # sorted x desc, y strictly increasing
    if len(front) == 0:
        return 0.0

    xs = front[:, 0]
    ys = front[:, 1]

    hv = 0.0
    for i in range(len(front) - 1):
        hv += (xs[i] - xs[i + 1]) * (ys[i] - ry)
    # last (leftmost-x) strip extends to rx
    hv += (xs[-1] - rx) * (ys[-1] - ry)
    return float(hv)


def hypervolume_error(
    points_approx: np.ndarray,
    points_true: np.ndarray,
    ref: Tuple[float, float],
    mode: str = "absolute",
) -> float:
    """Error between two fronts' hypervolumes.

    Parameters
    ----------
    mode : ``"absolute"`` |HV_a - HV_t|,
           ``"relative"`` |HV_a - HV_t| / HV_t,
           ``"gap"``      1 - HV_a / HV_t.
    """
    hv_a = hypervolume_2d(points_approx, ref)
    hv_t = hypervolume_2d(points_true, ref)

    if mode == "absolute":
        return abs(hv_a - hv_t)
    if mode == "relative":
        if hv_t == 0.0:
            return 0.0 if hv_a == 0.0 else float("inf")
        return abs(hv_a - hv_t) / hv_t
    if mode == "gap":
        if hv_t == 0.0:
            return 0.0 if hv_a == 0.0 else float("inf")
        return 1.0 - hv_a / hv_t
    raise ValueError(f"Unknown mode: {mode!r}. Use 'absolute', 'relative', or 'gap'.")


def hypervolume_convergence(
    points_per_iteration: Sequence[np.ndarray],
    ref: Tuple[float, float],
) -> np.ndarray:
    """Compute hypervolume at each iteration (cumulative).

    Parameters
    ----------
    points_per_iteration : sequence of (N_i, 2) arrays, one per iteration.
    ref : reference point.

    Returns
    -------
    1-D array of HV values, one per iteration.
    """
    hvs = []
    accumulated = np.empty((0, 2), dtype=float)
    for pts in points_per_iteration:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        accumulated = np.vstack([accumulated, pts])
        hvs.append(hypervolume_2d(accumulated, ref))
    return np.array(hvs, dtype=float)


# ======================================================
# Visualization helpers
# ======================================================

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def build_stair_polygon(
    front: np.ndarray,
    ref: Tuple[float, float],
) -> np.ndarray:
    """Construct polygon vertices tracing the HV-dominated staircase region.

    Returns vertices (counterclockwise) suitable for ``matplotlib.patches.Polygon``.
    Front must be sorted x descending / y increasing (output of ``pareto_front_2d``).
    """
    rx, ry = float(ref[0]), float(ref[1])
    if len(front) == 0:
        return np.empty((0, 2))

    xs = front[:, 0]  # x descending
    ys = front[:, 1]  # y increasing

    # Trace counterclockwise starting at bottom-right:
    #   (x0, ry) -> up to (x0, y0) -> stair steps left/up -> (rx, yk) -> (rx, ry)
    verts = [(xs[0], ry), (xs[0], ys[0])]
    for i in range(1, len(front)):
        verts.append((xs[i], ys[i - 1]))  # horizontal step left
        verts.append((xs[i], ys[i]))       # vertical step up
    # extend leftmost step to reference x, then drop to ry
    verts.append((rx, ys[-1]))
    verts.append((rx, ry))

    return np.array(verts, dtype=float)


def _prepare_ax(ax, figsize=(8, 6)):
    plt = _import_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _maybe_save(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_pareto_front(
    points: np.ndarray,
    ref: Tuple[float, float],
    *,
    ax=None,
    show_dominated: bool = True,
    show_stair: bool = True,
    show_hv_area: bool = True,
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    title: str = "Pareto Front",
    save_path: Optional[str] = None,
):
    """Plot a 2-D Pareto front with optional HV area shading."""
    plt = _import_matplotlib()
    fig, ax = _prepare_ax(ax)

    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    front = pareto_front_2d(pts)
    front_set = set(map(tuple, front.tolist()))

    if show_dominated:
        dom_mask = np.array([tuple(p) not in front_set for p in pts])
        if dom_mask.any():
            ax.scatter(pts[dom_mask, 0], pts[dom_mask, 1],
                       c="gray", alpha=0.4, s=15, label="Dominated")

    # Pareto-optimal points
    if len(front) > 0:
        ax.scatter(front[:, 0], front[:, 1],
                   c="red", s=40, zorder=5, label="Pareto-optimal")

    # stair-step line
    if show_stair and len(front) > 0:
        rx, ry = float(ref[0]), float(ref[1])
        stair_x = [rx, front[-1, 0]]
        stair_y = [front[-1, 1], front[-1, 1]]
        for i in range(len(front) - 2, -1, -1):
            stair_x.append(front[i + 1, 0])
            stair_y.append(front[i, 1])
            stair_x.append(front[i, 0])
            stair_y.append(front[i, 1])
        stair_x.append(front[0, 0])
        stair_y.append(ry)
        ax.plot(stair_x, stair_y, "r-", linewidth=1.2, zorder=4)

    # HV shaded area
    if show_hv_area and len(front) > 0:
        poly = build_stair_polygon(front, ref)
        if len(poly) > 0:
            from matplotlib.patches import Polygon
            patch = Polygon(poly, closed=True, facecolor="lightblue",
                            edgecolor="none", alpha=0.4, label="HV area")
            ax.add_patch(patch)

    # reference point
    rx, ry = float(ref[0]), float(ref[1])
    ax.plot(rx, ry, "kx", markersize=10, markeredgewidth=2, label="Reference")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


def plot_pareto_comparison(
    points_a: np.ndarray,
    points_b: np.ndarray,
    ref: Tuple[float, float],
    *,
    labels: Tuple[str, str] = ("Approx", "True"),
    ax=None,
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    title: str = "Pareto Front Comparison",
    save_path: Optional[str] = None,
):
    """Overlay two Pareto fronts with HV annotations."""
    plt = _import_matplotlib()
    fig, ax = _prepare_ax(ax)

    pts_a = np.asarray(points_a, dtype=float).reshape(-1, 2)
    pts_b = np.asarray(points_b, dtype=float).reshape(-1, 2)

    front_a = pareto_front_2d(pts_a)
    front_b = pareto_front_2d(pts_b)

    hv_a = hypervolume_2d(pts_a, ref)
    hv_b = hypervolume_2d(pts_b, ref)

    if len(front_b) > 0:
        ax.scatter(front_b[:, 0], front_b[:, 1],
                   c="tab:blue", s=40, zorder=4, marker="s", label=labels[1])
    if len(front_a) > 0:
        ax.scatter(front_a[:, 0], front_a[:, 1],
                   c="tab:orange", s=40, zorder=5, marker="o", label=labels[0])

    # stair lines
    for front, color in [(front_b, "tab:blue"), (front_a, "tab:orange")]:
        if len(front) == 0:
            continue
        rx, ry = float(ref[0]), float(ref[1])
        sx = [rx, front[-1, 0]]
        sy = [front[-1, 1], front[-1, 1]]
        for i in range(len(front) - 2, -1, -1):
            sx.append(front[i + 1, 0])
            sy.append(front[i, 1])
            sx.append(front[i, 0])
            sy.append(front[i, 1])
        sx.append(front[0, 0])
        sy.append(ry)
        ax.plot(sx, sy, color=color, linewidth=1.2, zorder=3)

    # reference point
    rx, ry = float(ref[0]), float(ref[1])
    ax.plot(rx, ry, "kx", markersize=10, markeredgewidth=2, label="Reference")

    # annotate HV values
    err_pct = 0.0
    if hv_b > 0:
        err_pct = abs(hv_a - hv_b) / hv_b * 100
    text = f"HV({labels[0]})={hv_a:.4f}\nHV({labels[1]})={hv_b:.4f}\nError={err_pct:.2f}%"
    ax.annotate(text, xy=(0.02, 0.98), xycoords="axes fraction",
                verticalalignment="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


def plot_hypervolume_convergence(
    hv_values: np.ndarray,
    *,
    hv_true: Optional[float] = None,
    ax=None,
    xlabel: str = "Iteration",
    ylabel: str = "Hypervolume",
    title: str = "Hypervolume Convergence",
    save_path: Optional[str] = None,
):
    """Line plot of hypervolume over iterations."""
    plt = _import_matplotlib()
    fig, ax = _prepare_ax(ax)

    hvs = np.asarray(hv_values, dtype=float)
    iterations = np.arange(1, len(hvs) + 1)

    ax.plot(iterations, hvs, "o-", markersize=4, label="HV")

    if hv_true is not None:
        ax.axhline(hv_true, color="green", linestyle="--", linewidth=1.2,
                    label=f"Target HV = {hv_true:.4f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig, ax


# ======================================================
# CLI
# ======================================================

def _read_points(csv_path: str, col_x: str, col_y: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[[col_x, col_y]].dropna().values.astype(float)


def main():
    ap = argparse.ArgumentParser(
        description="Pareto front hypervolume tools",
    )
    sub = ap.add_subparsers(dest="command")

    # --- hv ---
    p_hv = sub.add_parser("hv", help="Compute hypervolume from CSV")
    p_hv.add_argument("csv", help="Path to CSV file")
    p_hv.add_argument("--col_x", default="score_3GVB")
    p_hv.add_argument("--col_y", default="score_6D6P")
    p_hv.add_argument("--ref", nargs=2, type=float, required=True,
                       metavar=("RX", "RY"), help="Reference point")

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare HV of two fronts")
    p_cmp.add_argument("csv_approx", help="Approximate front CSV")
    p_cmp.add_argument("csv_true", help="True/reference front CSV")
    p_cmp.add_argument("--col_x", default="score_3GVB")
    p_cmp.add_argument("--col_y", default="score_6D6P")
    p_cmp.add_argument("--ref", nargs=2, type=float, required=True,
                        metavar=("RX", "RY"))
    p_cmp.add_argument("--mode", default="absolute",
                        choices=["absolute", "relative", "gap"])

    # --- plot ---
    p_plot = sub.add_parser("plot", help="Plot Pareto front from CSV")
    p_plot.add_argument("csv", help="Path to CSV file")
    p_plot.add_argument("--col_x", default="score_3GVB")
    p_plot.add_argument("--col_y", default="score_6D6P")
    p_plot.add_argument("--ref", nargs=2, type=float, required=True,
                         metavar=("RX", "RY"))
    p_plot.add_argument("--out", default="pareto_front.png",
                         help="Output image path")
    p_plot.add_argument("--title", default="Pareto Front")

    # --- convergence ---
    p_conv = sub.add_parser("convergence",
                            help="Plot HV convergence from iteration files")
    p_conv.add_argument("csvs", nargs="+", help="Iteration CSV files in order")
    p_conv.add_argument("--col_x", default="score_3GVB")
    p_conv.add_argument("--col_y", default="score_6D6P")
    p_conv.add_argument("--ref", nargs=2, type=float, required=True,
                          metavar=("RX", "RY"))
    p_conv.add_argument("--hv_true", type=float, default=None,
                          help="True/target HV for reference line")
    p_conv.add_argument("--out", default="hv_convergence.png",
                          help="Output image path")

    args = ap.parse_args()

    if args.command is None:
        ap.print_help()
        sys.exit(1)

    if args.command == "hv":
        pts = _read_points(args.csv, args.col_x, args.col_y)
        ref = tuple(args.ref)
        hv = hypervolume_2d(pts, ref)
        front = pareto_front_2d(pts)
        print(f"Points: {len(pts)}")
        print(f"Pareto front size: {len(front)}")
        print(f"Reference: {ref}")
        print(f"Hypervolume: {hv:.6f}")

    elif args.command == "compare":
        pts_a = _read_points(args.csv_approx, args.col_x, args.col_y)
        pts_t = _read_points(args.csv_true, args.col_x, args.col_y)
        ref = tuple(args.ref)
        hv_a = hypervolume_2d(pts_a, ref)
        hv_t = hypervolume_2d(pts_t, ref)
        err = hypervolume_error(pts_a, pts_t, ref, mode=args.mode)
        print(f"HV (approx): {hv_a:.6f}")
        print(f"HV (true):   {hv_t:.6f}")
        print(f"Error ({args.mode}): {err:.6f}")

    elif args.command == "plot":
        pts = _read_points(args.csv, args.col_x, args.col_y)
        ref = tuple(args.ref)
        plot_pareto_front(pts, ref,
                          xlabel=args.col_x, ylabel=args.col_y,
                          title=args.title, save_path=args.out)
        print(f"Saved plot to {args.out}")

    elif args.command == "convergence":
        ref = tuple(args.ref)
        points_per_iter = [
            _read_points(f, args.col_x, args.col_y) for f in args.csvs
        ]
        hvs = hypervolume_convergence(points_per_iter, ref)
        for i, h in enumerate(hvs, 1):
            print(f"Iteration {i}: HV = {h:.6f}")
        plot_hypervolume_convergence(hvs, hv_true=args.hv_true,
                                     save_path=args.out)
        print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

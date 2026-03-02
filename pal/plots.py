# pal/plots.py
from __future__ import annotations

import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import StrategyResult
from .pareto import pareto_front_2d, build_stair_polygon  # still 2D for now


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _pick_median_replicate(res: StrategyResult):
    final_hvs = [h[-1] for h in res.hv_histories]
    median_hv = np.median(final_hvs)
    idx = int(np.argmin(np.abs(np.array(final_hvs) - median_hv)))
    return res.states[idx]


def _plot_2d_front(ax, Y_pool, Y_labeled, ref_point, obj_names, title):
    ax.scatter(
        Y_pool[:, 0], Y_pool[:, 1],
        s=5, alpha=0.15, color="gray", label="Pool"
    )

    ax.scatter(
        Y_labeled[:, 0], Y_labeled[:, 1],
        s=15, alpha=0.6, color="tab:blue", label="Labeled"
    )

    disc_front = pareto_front_2d(Y_labeled)
    if len(disc_front) > 0:
        poly = build_stair_polygon(disc_front, ref_point)
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color="tab:blue")
        ax.plot(disc_front[:, 0], disc_front[:, 1], "o-", color="tab:blue", markersize=4)

    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _plot_2d_projection(ax, Y_pool, Y_labeled, dims: Tuple[int, int], obj_names, title):
    i, j = dims
    ax.scatter(
        Y_pool[:, i], Y_pool[:, j],
        s=5, alpha=0.1, color="gray"
    )
    ax.scatter(
        Y_labeled[:, i], Y_labeled[:, j],
        s=15, alpha=0.6, color="tab:blue"
    )

    ax.set_xlabel(obj_names[i])
    ax.set_ylabel(obj_names[j])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# -------------------------------------------------
# HV Convergence
# -------------------------------------------------
def _extract_hv_matrix(strategy_result: Any) -> np.ndarray:
    """
    Try to extract HV history as a 2D matrix: (n_replicates, n_steps).

    Supported common shapes:
    - strategy_result["hv"] is:
        * list of 1D arrays (one per replicate)
        * 2D np.ndarray (n_reps, n_steps)
        * 1D np.ndarray (n_steps,)  -> treated as single replicate
    - strategy_result is itself a list of replicate dicts, each having ["hv"]
    """
    # Case A: dict with key "hv"
    if isinstance(strategy_result, dict) and "hv" in strategy_result:
        hv = strategy_result["hv"]
        if isinstance(hv, list):
            # list of arrays
            hv_rows = [np.asarray(x, dtype=float).ravel() for x in hv]
            return np.vstack(hv_rows)
        hv = np.asarray(hv, dtype=float)
        if hv.ndim == 1:
            return hv[None, :]
        if hv.ndim == 2:
            return hv
        raise ValueError(f"Unsupported hv ndim={hv.ndim} in strategy_result['hv'].")

    # Case B: list of replicate dicts
    if isinstance(strategy_result, list) and len(strategy_result) > 0:
        if isinstance(strategy_result[0], dict) and "hv" in strategy_result[0]:
            hv_rows = [np.asarray(rep["hv"], dtype=float).ravel() for rep in strategy_result]
            return np.vstack(hv_rows)

    raise KeyError(
        "Could not find HV history. Expected results[strategy]['hv'] "
        "or results[strategy] as list of replicates each with ['hv']."
    )


def plot_hv_convergence(
    results: Dict[str, Any],
    oracle_hv: Optional[float],
    config: Any,
    save_path: str,
) -> None:
    """
    Plot HV convergence curves for multiple strategies.

    - If oracle_hv is not None and >0: plot normalized HV (hv / oracle_hv).
    - Else: plot raw HV.

    Expects `results` to be a dict: strategy_name -> strategy_result.
    strategy_result must contain HV history per replicate (see _extract_hv_matrix()).
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    normalize = oracle_hv is not None and float(oracle_hv) > 0.0
    y_label = "HV / oracle" if normalize else "HV"

    plt.figure()
    for name, strat_res in results.items():
        hv_mat = _extract_hv_matrix(strat_res)  # (n_reps, n_steps)

        # Guard: drop NaNs per column (if any)
        hv_mat = np.asarray(hv_mat, dtype=float)
        if normalize:
            hv_mat = hv_mat / float(oracle_hv)

        mean = np.nanmean(hv_mat, axis=0)
        std = np.nanstd(hv_mat, axis=0)

        x = np.arange(mean.shape[0])
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.title("HV convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------------------------------------------------
# Pareto Snapshots
# -------------------------------------------------

def plot_pareto_snapshots(results, Y_pool, config, save_path=None):
    dim = Y_pool.shape[1]
    n_strategies = len(results)

    if dim == 2:
        fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5), squeeze=False)

        for idx, (_, res) in enumerate(results.items()):
            ax = axes[0, idx]
            state = _pick_median_replicate(res)
            _plot_2d_front(
                ax,
                Y_pool,
                state.Y_labeled,
                config.al.ref_point,
                config.obj_names,
                res.name,
            )

    elif dim == 3:
        fig, axes = plt.subplots(3, n_strategies, figsize=(5 * n_strategies, 12), squeeze=False)

        projections = [(0, 1), (0, 2), (1, 2)]

        for col, (_, res) in enumerate(results.items()):
            state = _pick_median_replicate(res)

            for row, dims in enumerate(projections):
                ax = axes[row, col]
                _plot_2d_projection(
                    ax,
                    Y_pool,
                    state.Y_labeled,
                    dims,
                    config.obj_names,
                    f"{res.name} ({config.obj_names[dims[0]]} vs {config.obj_names[dims[1]]})",
                )
    else:
        raise ValueError("plot_pareto_snapshots supports only 2D or 3D objectives.")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logging.info(f"Saved Pareto snapshots -> {save_path}")
    plt.close(fig)


# -------------------------------------------------
# Iteration selections (2D only for clarity)
# -------------------------------------------------

def plot_iteration_selections(results, Y_pool, config, save_path=None):
    if Y_pool.shape[1] != 2:
        logging.info("Skipping iteration selection plot (only implemented for 2D).")
        return

    n_strategies = len(results)
    fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5), squeeze=False)

    for idx, (_, res) in enumerate(results.items()):
        ax = axes[0, idx]
        state = _pick_median_replicate(res)

        ax.scatter(Y_pool[:, 0], Y_pool[:, 1], s=5, alpha=0.1, color="gray")

        n_iters = len(state.selections_per_iter)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=max(n_iters - 1, 1))

        for it, sel_indices in enumerate(state.selections_per_iter):
            pts = Y_pool[sel_indices]
            color = cmap(norm(it))
            ax.scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.7, color=color)

        ax.set_xlabel(config.obj_names[0])
        ax.set_ylabel(config.obj_names[1])
        ax.set_title(res.name)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logging.info(f"Saved iteration selections -> {save_path}")
    plt.close(fig)


# -------------------------------------------------
# Acquisition timing
# -------------------------------------------------

def plot_acq_timing(results, config, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for _, res in results.items():
        timing = np.array([state.acq_time_per_iter for state in res.states])
        iters = np.arange(timing.shape[1])

        mean_t = timing.mean(axis=0)
        std_t = timing.std(axis=0)

        ax1.plot(iters, mean_t, marker="o", markersize=3, label=res.name)
        ax1.fill_between(iters, mean_t - std_t, mean_t + std_t, alpha=0.2)

        cum = np.cumsum(timing, axis=1)
        cum_mean = cum.mean(axis=0)
        cum_std = cum.std(axis=0)

        ax2.plot(iters, cum_mean, marker="o", markersize=3, label=res.name)
        ax2.fill_between(iters, cum_mean - cum_std, cum_mean + cum_std, alpha=0.2)

    ax1.set_title("Per-iteration acquisition time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Cumulative acquisition time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logging.info(f"Saved acquisition timing -> {save_path}")
    plt.close(fig)


# -------------------------------------------------
# Validation metrics (dimension-agnostic)
# -------------------------------------------------

def plot_validation_metrics(results, config, save_path=None):
    n_obj = len(config.obj_names)
    metric_keys = ["mse", "mae", "r2"]
    metric_labels = ["MSE", "MAE", "R²"]

    fig, axes = plt.subplots(n_obj, 3, figsize=(18, 5 * n_obj))

    for obj in range(n_obj):
        for col, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[obj, col] if n_obj > 1 else axes[col]

            for _, res in results.items():
                train_vals = []
                for state in res.states:
                    train_vals.append([m[key][obj] for m in state.train_metrics])

                train_arr = np.array(train_vals)
                iters = np.arange(train_arr.shape[1])

                mean = np.nanmean(train_arr, axis=0)
                std = np.nanstd(train_arr, axis=0)

                ax.plot(iters, mean, label=res.name)
                ax.fill_between(iters, mean - std, mean + std, alpha=0.2)

            ax.set_title(f"{config.obj_names[obj]} - {label}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logging.info(f"Saved validation metrics -> {save_path}")
    plt.close(fig)
    

def plot_iteration_selections_3d(results, Y_pool, config, save_path=None):
    """
    3D scatter of iteration selections (median replicate) for each strategy.
    Pure visualization. No training. No HV.
    """
    if Y_pool.shape[1] != 3:
        logging.info("Skipping 3D iteration selections plot (requires exactly 3 objectives).")
        return

    n_strategies = len(results)
    fig = plt.figure(figsize=(6 * n_strategies, 6))

    for idx, (_, res) in enumerate(results.items()):
        ax = fig.add_subplot(1, n_strategies, idx + 1, projection="3d")
        state = _pick_median_replicate(res)

        # background pool
        ax.scatter(
            Y_pool[:, 0], Y_pool[:, 1], Y_pool[:, 2],
            s=4, alpha=0.06, c="gray"
        )

        # selections colored by iteration
        n_iters = len(state.selections_per_iter)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=max(n_iters - 1, 1))

        for it, sel_indices in enumerate(state.selections_per_iter):
            pts = Y_pool[sel_indices]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=18, alpha=0.75, color=cmap(norm(it))
            )

        ax.set_xlabel(config.obj_names[0])
        ax.set_ylabel(config.obj_names[1])
        ax.set_zlabel(config.obj_names[2])
        ax.set_title(res.name)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        logging.info(f"Saved 3D iteration selections -> {save_path}")
    plt.close(fig)
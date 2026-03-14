"""Explanatory visualizations for UCB and Ellipse acquisition strategies."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .acquisition import get_acquisition
from .config import ExperimentConfig
from .model import MLP, build_model, mc_predict, train_model
from .pareto import build_stair_polygon, delta_hv_contribution, pareto_front_2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_pareto_background(
    ax: plt.Axes,
    front: np.ndarray,
    ref_point: Tuple[float, float],
) -> None:
    """Draw the current Pareto stair polygon and front points."""
    if len(front) == 0:
        return
    poly = build_stair_polygon(front, ref_point)
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.10, color="tab:green", label="Current HV")
    ax.plot(
        front[:, 0], front[:, 1], "s-",
        color="tab:green", markersize=4, linewidth=1, label="Current front",
    )


def _draw_confidence_ellipse(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    k: float,
    n_points: int = 128,
    **kwargs,
) -> np.ndarray:
    """Draw a k-sigma confidence ellipse and return the boundary points.

    Returns
    -------
    points : np.ndarray, shape ``(n_points, 2)``
    """
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=True)
    circle = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)  # (n_points, 2)

    try:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(2))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov).clip(1e-9)))

    points = mean + k * (circle @ L.T)
    style = dict(color="tab:orange", linewidth=1.2, linestyle="--")
    style.update(kwargs)
    ax.plot(points[:, 0], points[:, 1], **style)
    return points


def _draw_mc_cloud(
    ax: plt.Axes,
    samples: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    k: float,
) -> None:
    """Scatter MC samples, draw mean star, and overlay dashed confidence ellipse."""
    ax.scatter(
        samples[:, 0], samples[:, 1],
        s=8, alpha=0.35, color="tab:blue", label="MC samples",
    )
    ax.plot(
        mean[0], mean[1], "*",
        color="tab:red", markersize=14, zorder=5, label="Mean",
    )
    _draw_confidence_ellipse(ax, mean, cov, k, label=f"{k}\u03c3 ellipse")


# ---------------------------------------------------------------------------
# Main figure: UCB explained
# ---------------------------------------------------------------------------

def plot_ucb_explained(
    samples: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    cov: np.ndarray,
    current_labels: np.ndarray,
    ref_point: Tuple[float, float],
    k: float = 2.0,
    save_path: str | None = None,
    obj_names: Tuple[str, str] = ("Objective 0", "Objective 1"),
) -> None:
    """Two-panel figure explaining the UCB acquisition for one molecule.

    Parameters
    ----------
    samples : (n_passes, 2)  MC dropout predictions for this molecule
    mean    : (2,)
    std     : (2,)
    cov     : (2, 2)
    current_labels : (M, 2)  already-labeled objectives
    ref_point : HV reference point
    k       : confidence multiplier
    save_path : optional file path
    """
    front = pareto_front_2d(current_labels)
    optimistic = mean + k * std

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: MC sample cloud + fitted ellipse ---
    ax = axes[0]
    _draw_pareto_background(ax, front, ref_point)
    _draw_mc_cloud(ax, samples, mean, cov, k)
    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("Model uncertainty (MC dropout)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # --- Right panel: UCB rectangle + optimistic corner ---
    ax = axes[1]
    _draw_pareto_background(ax, front, ref_point)

    # mean point
    ax.plot(mean[0], mean[1], "*", color="tab:red", markersize=14, zorder=5, label="Mean")

    # error bars (k*std per axis)
    ax.errorbar(
        mean[0], mean[1],
        xerr=k * std[0], yerr=k * std[1],
        fmt="none", ecolor="tab:purple", elinewidth=1.5, capsize=4,
        label=f"\u00b1{k}\u03c3 bars",
    )

    # dashed rectangle: from (mean - k*std) to (mean + k*std)
    lo = mean - k * std
    hi = mean + k * std
    rect_x = [lo[0], hi[0], hi[0], lo[0], lo[0]]
    rect_y = [lo[1], lo[1], hi[1], hi[1], lo[1]]
    ax.plot(rect_x, rect_y, "--", color="tab:purple", linewidth=1, label="UCB rectangle")

    # optimistic corner (diamond)
    dhv = delta_hv_contribution(front, optimistic[0], optimistic[1], ref_point)
    ax.plot(
        optimistic[0], optimistic[1], "D",
        color="tab:orange", markersize=10, zorder=5,
        label=f"Optimistic corner",
    )
    ax.annotate(
        f"\u0394HV = {dhv:.4f}",
        xy=(optimistic[0], optimistic[1]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=8, color="tab:orange",
        arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.8),
    )

    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("UCB acquisition")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("UCB Acquisition Explained", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved UCB explanation -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main figure: Ellipse explained
# ---------------------------------------------------------------------------

def plot_ellipse_explained(
    samples: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    cov: np.ndarray,
    current_labels: np.ndarray,
    ref_point: Tuple[float, float],
    k: float = 2.0,
    n_angles: int = 64,
    save_path: str | None = None,
    obj_names: Tuple[str, str] = ("Objective 0", "Objective 1"),
) -> None:
    """Two-panel figure explaining the Ellipse acquisition for one molecule.

    Parameters
    ----------
    samples : (n_passes, 2)  MC dropout predictions for this molecule
    mean    : (2,)
    std     : (2,)
    cov     : (2, 2)
    current_labels : (M, 2)  already-labeled objectives
    ref_point : HV reference point
    k       : confidence multiplier
    n_angles: number of sampled points on the ellipse boundary
    save_path : optional file path
    """
    front = pareto_front_2d(current_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: MC sample cloud + fitted ellipse ---
    ax = axes[0]
    _draw_pareto_background(ax, front, ref_point)
    _draw_mc_cloud(ax, samples, mean, cov, k)
    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("Model uncertainty (MC dropout)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # --- Right panel: Ellipse boundary + sampled points + best ---
    ax = axes[1]
    _draw_pareto_background(ax, front, ref_point)

    # mean point
    ax.plot(mean[0], mean[1], "*", color="tab:red", markersize=14, zorder=5, label="Mean")

    # full ellipse boundary (solid)
    ellipse_pts = _draw_confidence_ellipse(
        ax, mean, cov, k, n_points=128,
        linestyle="-", color="tab:orange", linewidth=1.5, label=f"{k}\u03c3 ellipse",
    )

    # sampled evaluation points on the boundary
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    circle = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    try:
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(2))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov).clip(1e-9)))
    eval_points = mean + k * (circle @ L.T)  # (n_angles, 2)

    ax.scatter(
        eval_points[:, 0], eval_points[:, 1],
        s=18, color="tab:blue", zorder=4, alpha=0.7, label=f"{n_angles} eval points",
    )

    # find the max-delta-HV point
    best_dhv = -np.inf
    best_idx = 0
    for j in range(n_angles):
        dhv = delta_hv_contribution(front, eval_points[j, 0], eval_points[j, 1], ref_point)
        if dhv > best_dhv:
            best_dhv = dhv
            best_idx = j

    best_pt = eval_points[best_idx]
    ax.plot(
        best_pt[0], best_pt[1], "D",
        color="tab:orange", markersize=10, zorder=5,
        label="Max \u0394HV point",
    )
    ax.annotate(
        f"\u0394HV = {best_dhv:.4f}",
        xy=(best_pt[0], best_pt[1]),
        xytext=(8, 8), textcoords="offset points",
        fontsize=8, color="tab:orange",
        arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.8),
    )

    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    ax.set_title("Ellipse acquisition")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Ellipse Acquisition Explained", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved Ellipse explanation -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_acquisition_explanations(
    model: MLP,
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    labeled_indices: list[int],
    unlabeled_indices: list[int],
    ref_point: Tuple[float, float],
    config: ExperimentConfig,
    output_dir: str,
    k: float = 2.0,
) -> None:
    """Generate explanatory plots for UCB and Ellipse on their top-1 molecule.

    Parameters
    ----------
    model : MLP
        Already-trained model (will be used for MC-dropout inference).
    X_pool, Y_pool : full pool arrays
    labeled_indices, unlabeled_indices : current AL partition
    ref_point : HV reference point
    config : experiment config (for mc_passes etc.)
    output_dir : where to write the PNG files
    k : confidence multiplier
    """
    X_unlabeled = X_pool[unlabeled_indices]
    current_labels = Y_pool[labeled_indices]

    # MC-dropout with raw samples
    means, stds, covs, all_samples = mc_predict(
        model,
        X_unlabeled,
        n_passes=config.model.mc_passes,
        device=config.device,
        return_samples=True,
    )

    # Score with UCB
    ucb_fn = get_acquisition("ucb", k_ucb=k)
    ucb_scores = ucb_fn.score(means, stds, current_labels, ref_point, covs=covs)
    ucb_top = int(np.argmax(ucb_scores))

    # Score with Ellipse
    ell_fn = get_acquisition("ellipse", k=k)
    ell_scores = ell_fn.score(means, stds, current_labels, ref_point, covs=covs)
    ell_top = int(np.argmax(ell_scores))

    import os

    # UCB explanation for top-1 UCB molecule
    plot_ucb_explained(
        samples=all_samples[ucb_top],       # (n_passes, 2)
        mean=means[ucb_top],
        std=stds[ucb_top],
        cov=covs[ucb_top],
        current_labels=current_labels,
        ref_point=ref_point,
        k=k,
        save_path=os.path.join(output_dir, "acq_ucb_explained.png"),
        obj_names=config.obj_names,
    )

    # Ellipse explanation for top-1 Ellipse molecule
    plot_ellipse_explained(
        samples=all_samples[ell_top],        # (n_passes, 2)
        mean=means[ell_top],
        std=stds[ell_top],
        cov=covs[ell_top],
        current_labels=current_labels,
        ref_point=ref_point,
        k=k,
        save_path=os.path.join(output_dir, "acq_ellipse_explained.png"),
        obj_names=config.obj_names,
    )

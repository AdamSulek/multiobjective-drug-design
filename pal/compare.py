"""Run multiple acquisition strategies and plot convergence (CLI entry)."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .acquisition import get_acquisition
from .acquisition.base import AcquisitionFunction
from .config import ExperimentConfig
from .data import generate_zinc_dataset, load_dataset_from_file
from .featurizer import compute_ecfp
from .loop import ALState, run_al_loop
from .model import build_model, mc_predict, predict_eval, train_model
from .pareto import build_stair_polygon, hypervolume_2d, pareto_front_2d
from .visualize import generate_acquisition_explanations


@dataclass
class StrategyResult:
    name: str
    hv_histories: List[List[float]]
    states: List[ALState]


def run_comparison(
    strategies: Dict[str, AcquisitionFunction],
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    config: ExperimentConfig,
) -> Dict[str, StrategyResult]:
    """Run each strategy with multiple replicates on pre-computed pool data."""
    base_seed = config.data.seed
    n_replicates = config.al.n_replicates

    results: Dict[str, StrategyResult] = {}
    for sname, acq_fn in strategies.items():
        results[sname] = StrategyResult(
            name=acq_fn.name, hv_histories=[], states=[]
        )

    for r in range(n_replicates):
        rep_seed = base_seed + r
        print(f"--- Replicate {r} (seed={rep_seed}) ---")
        for sname, acq_fn in strategies.items():
            print(f"=== Running strategy: {acq_fn.name} ===")
            state = run_al_loop(
                X_pool, Y_pool, acq_fn, config, seed=rep_seed
            )
            results[sname].hv_histories.append(state.hv_history)
            results[sname].states.append(state)
            print()

    return results


def plot_hv_convergence(
    results: Dict[str, StrategyResult],
    oracle_hv: float,
    config: ExperimentConfig,
    save_path: str | None = None,
) -> None:
    """Plot HV convergence with mean +/- 1 std shading across replicates."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for sname, res in results.items():
        hv_arr = np.array(res.hv_histories)  # (n_replicates, n_iterations+1)
        mean = hv_arr.mean(axis=0)
        std = hv_arr.std(axis=0)
        n_labeled = np.array([
            config.al.seed_size + i * config.al.batch_size
            for i in range(hv_arr.shape[1])
        ])
        ax.plot(n_labeled, mean, marker="o", markersize=3, label=res.name)
        ax.fill_between(n_labeled, mean - std, mean + std, alpha=0.2)

    ax.axhline(oracle_hv, color="k", linestyle="--", linewidth=1, label="Oracle HV")
    ax.set_xlabel("Number of labeled compounds")
    ax.set_ylabel("Hypervolume")
    ax.set_title("HV Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved HV convergence plot -> {save_path}")
    plt.close(fig)


def _pick_median_replicate(res: StrategyResult) -> ALState:
    """Return the replicate whose final HV is closest to the median."""
    final_hvs = [h[-1] for h in res.hv_histories]
    median_hv = np.median(final_hvs)
    best_idx = int(np.argmin(np.abs(np.array(final_hvs) - median_hv)))
    return res.states[best_idx]


def plot_pareto_snapshots(
    results: Dict[str, StrategyResult],
    Y_pool: np.ndarray,
    config: ExperimentConfig,
    save_path: str | None = None,
) -> None:
    """Plot final discovered Pareto fronts vs oracle front (median replicate)."""
    oracle_front = pareto_front_2d(Y_pool)

    n_strategies = len(results)
    fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5), squeeze=False)

    for idx, (sname, res) in enumerate(results.items()):
        ax = axes[0, idx]
        state = _pick_median_replicate(res)

        # pool background
        ax.scatter(
            Y_pool[:, 0], Y_pool[:, 1],
            s=5, alpha=0.15, color="gray", label="Pool",
        )

        # discovered points
        Y_labeled = state.Y_labeled
        ax.scatter(
            Y_labeled[:, 0], Y_labeled[:, 1],
            s=15, alpha=0.6, color="tab:blue", label="Labeled",
        )

        # discovered front
        disc_front = pareto_front_2d(Y_labeled)
        if len(disc_front) > 0:
            poly = build_stair_polygon(disc_front, config.al.ref_point)
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color="tab:blue")
            ax.plot(disc_front[:, 0], disc_front[:, 1], "o-", color="tab:blue", markersize=4)

        # oracle front
        ax.plot(
            oracle_front[:, 0], oracle_front[:, 1],
            "s--", color="tab:red", markersize=3, label="Oracle front",
        )

        ax.set_xlabel(config.obj_names[0])
        ax.set_ylabel(config.obj_names[1])
        ax.set_title(res.name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved Pareto snapshots -> {save_path}")
    plt.close(fig)


def plot_iteration_selections(
    results: Dict[str, StrategyResult],
    Y_pool: np.ndarray,
    config: ExperimentConfig,
    save_path: str | None = None,
) -> None:
    """Plot which points each strategy selected at each iteration (median replicate)."""
    oracle_front = pareto_front_2d(Y_pool)

    n_strategies = len(results)
    fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 5), squeeze=False)

    for idx, (sname, res) in enumerate(results.items()):
        ax = axes[0, idx]
        state = _pick_median_replicate(res)

        # pool background
        ax.scatter(
            Y_pool[:, 0], Y_pool[:, 1],
            s=5, alpha=0.10, color="gray", zorder=1,
        )

        # oracle Pareto front
        ax.plot(
            oracle_front[:, 0], oracle_front[:, 1],
            "s--", color="tab:red", markersize=3, label="Oracle front", zorder=4,
        )

        # color selections by iteration
        n_iters = len(state.selections_per_iter)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=max(n_iters - 1, 1))

        for it, sel_indices in enumerate(state.selections_per_iter):
            pts = Y_pool[sel_indices]
            color = cmap(norm(it))
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=15, alpha=0.7, color=color, zorder=3,
            )

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label("Iteration")

        ax.set_xlabel(config.obj_names[0])
        ax.set_ylabel(config.obj_names[1])
        ax.set_title(res.name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved iteration selections plot -> {save_path}")
    plt.close(fig)


def plot_acq_timing(
    results: Dict[str, StrategyResult],
    config: ExperimentConfig,
    save_path: str | None = None,
) -> None:
    """Plot per-iteration and cumulative acquisition timing across replicates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for sname, res in results.items():
        # Collect timing arrays: (n_replicates, n_iterations+1)
        timing_arrs = []
        for state in res.states:
            timing_arrs.append(state.acq_time_per_iter)
        timing = np.array(timing_arrs)
        iters = np.arange(timing.shape[1])

        mean_t = timing.mean(axis=0)
        std_t = timing.std(axis=0)

        # Per-iteration timing
        ax1.plot(iters, mean_t, marker="o", markersize=3, label=res.name)
        ax1.fill_between(iters, mean_t - std_t, mean_t + std_t, alpha=0.2)

        # Cumulative timing
        cum = np.cumsum(timing, axis=1)
        cum_mean = cum.mean(axis=0)
        cum_std = cum.std(axis=0)
        ax2.plot(iters, cum_mean, marker="o", markersize=3, label=res.name)
        ax2.fill_between(iters, cum_mean - cum_std, cum_mean + cum_std, alpha=0.2)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Acquisition time (s)")
    ax1.set_title("Per-iteration acquisition time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cumulative time (s)")
    ax2.set_title("Cumulative acquisition time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved acquisition timing plot -> {save_path}")
    plt.close(fig)


def plot_validation_metrics(
    results: Dict[str, StrategyResult],
    config: ExperimentConfig,
    save_path: str | None = None,
) -> None:
    """Plot regression metrics: 2x3 grid (rows=objectives, cols=MSE/MAE/R2)."""
    metric_keys = ["mse", "mae", "r2"]
    metric_labels = ["MSE", "MAE", "R$^2$"]
    obj_labels = list(config.obj_names)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for obj in range(2):
        for col, (key, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[obj, col]
            for sname, res in results.items():
                # training metrics: collect across replicates
                train_vals = []
                val_vals = []
                for state in res.states:
                    train_vals.append([
                        m[key][obj] for m in state.train_metrics
                    ])
                    val_vals.append([
                        m[key][obj] for m in state.val_metrics
                    ])

                iters = np.arange(len(train_vals[0]))
                train_arr = np.array(train_vals)  # (n_rep, n_iter+1)
                val_arr = np.array(val_vals)

                # training: line with mean +/- std shading
                t_mean = np.nanmean(train_arr, axis=0)
                t_std = np.nanstd(train_arr, axis=0)
                ax.plot(iters, t_mean, label=f"{res.name} (train)")
                ax.fill_between(iters, t_mean - t_std, t_mean + t_std, alpha=0.15)

                # validation: line with mean +/- std shading (at non-NaN iterations)
                v_mean = np.nanmean(val_arr, axis=0)
                v_std = np.nanstd(val_arr, axis=0)
                valid_mask = ~np.isnan(v_mean)
                if valid_mask.any():
                    v_iters = iters[valid_mask]
                    ax.plot(v_iters, v_mean[valid_mask], '--', label=f"{res.name} (val)")
                    ax.fill_between(v_iters, (v_mean - v_std)[valid_mask],
                                    (v_mean + v_std)[valid_mask], alpha=0.10)

            ax.set_xlabel("Iteration")
            ax.set_ylabel(mlabel)
            ax.set_title(f"{obj_labels[obj]} - {mlabel}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved validation metrics plot -> {save_path}")
    plt.close(fig)


def plot_al_diagnostics(
    results: Dict[str, StrategyResult],
    strategies: Dict[str, AcquisitionFunction],
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    config: ExperimentConfig,
    save_path: str | None = None,
) -> None:
    """Diagnostic figure: pred-vs-actual + acquisition scores in Y-space.

    For each strategy's median replicate, re-trains a model on its final
    labeled set, gets MC predictions + acquisition scores on the unlabeled
    pool, and produces a 3-row x n_strategies-column figure.
    """
    from .loop import compute_regression_metrics

    oracle_front = pareto_front_2d(Y_pool)
    n_strategies = len(results)
    obj_names = list(config.obj_names)

    fig, axes = plt.subplots(
        3, n_strategies, figsize=(5 * n_strategies, 14), squeeze=False,
    )

    for col, (sname, res) in enumerate(results.items()):
        state = _pick_median_replicate(res)
        acq_fn = strategies[sname]

        # Re-train model on final labeled set
        model = build_model(config.model, device=config.device)

        X_train = X_pool[state.labeled_indices]
        Y_train = state.Y_labeled

        # Normalize targets (same as loop.py)
        Y_mean = Y_train.mean(axis=0)
        Y_std_arr = np.maximum(Y_train.std(axis=0), 1e-8)
        Y_train_norm = (Y_train - Y_mean) / Y_std_arr

        train_model(
            model, X_train, Y_train_norm,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
            device=config.device,
            patience=config.model.patience,
            min_epochs=config.model.min_epochs,
            val_fraction=config.model.val_fraction,
            lr_scheduler_patience=config.model.lr_scheduler_patience,
            lr_scheduler_factor=config.model.lr_scheduler_factor,
        )

        # Training predictions (denormalized)
        Y_train_pred = predict_eval(model, X_train, device=config.device)
        Y_train_pred = Y_train_pred * Y_std_arr + Y_mean

        # MC predictions on unlabeled pool (denormalized)
        X_unlabeled = X_pool[state.unlabeled_indices]
        means, stds, covs = mc_predict(
            model, X_unlabeled,
            n_passes=config.model.mc_passes,
            device=config.device,
        )
        means = means * Y_std_arr + Y_mean
        stds = stds * Y_std_arr
        covs = covs * np.outer(Y_std_arr, Y_std_arr)[None, :, :]

        Y_true_unlabeled = Y_pool[state.unlabeled_indices]

        # Acquisition scores
        scores = acq_fn.score(
            means, stds, state.Y_labeled, config.al.ref_point, covs=covs,
        )

        # Row 0-1: Pred vs Actual for each objective
        for obj in range(2):
            ax = axes[obj, col]

            ax.scatter(
                Y_true_unlabeled[:, obj], means[:, obj],
                s=3, alpha=0.3, c="gray", label="Unlabeled", rasterized=True,
            )
            ax.scatter(
                Y_train[:, obj], Y_train_pred[:, obj],
                s=8, alpha=0.6, c="tab:blue", label="Train",
            )

            # y=x reference line
            all_vals = np.concatenate([
                Y_true_unlabeled[:, obj], Y_train[:, obj],
            ])
            lo, hi = all_vals.min(), all_vals.max()
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "k--", lw=1)

            # R² annotations
            train_r2 = compute_regression_metrics(
                Y_train, Y_train_pred,
            )["r2"][obj]
            unlabeled_r2 = compute_regression_metrics(
                Y_true_unlabeled, means,
            )["r2"][obj]

            ax.set_xlabel(f"True {obj_names[obj]}")
            ax.set_ylabel(f"Predicted {obj_names[obj]}")
            ax.set_title(
                f"{res.name} — {obj_names[obj]}\n"
                f"Train R²={train_r2:.3f}  Unlabeled R²={unlabeled_r2:.3f}"
            )
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Row 2: Acquisition scores in objective space
        ax = axes[2, col]
        ranks = scores.argsort().argsort().astype(np.float64)
        ranks /= max(len(ranks) - 1, 1)
        sc = ax.scatter(
            Y_true_unlabeled[:, 0], Y_true_unlabeled[:, 1],
            c=ranks, cmap="viridis", s=5, alpha=0.5, rasterized=True,
        )
        ax.scatter(
            Y_train[:, 0], Y_train[:, 1],
            c="tab:blue", s=3, alpha=0.2, label="Labeled",
        )
        ax.plot(
            oracle_front[:, 0], oracle_front[:, 1],
            "s-", color="tab:red", markersize=3, label="Oracle front",
        )
        fig.colorbar(sc, ax=ax, shrink=0.8, label="Acq. score (rank)")
        ax.set_xlabel(config.obj_names[0])
        ax.set_ylabel(config.obj_names[1])
        ax.set_title(f"{res.name} — Acquisition scores")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved AL diagnostics -> {save_path}")
    plt.close(fig)


def save_hv_csv(
    results: Dict[str, StrategyResult],
    config: ExperimentConfig,
    save_path: str,
) -> None:
    """Save HV convergence data as CSV (one row per strategy/replicate/iteration)."""
    rows = []
    for sname, res in results.items():
        for r, (hv_history, state) in enumerate(
            zip(res.hv_histories, res.states)
        ):
            for i, hv in enumerate(hv_history):
                acq_time = state.acq_time_per_iter[i] if i < len(state.acq_time_per_iter) else 0.0
                row = {
                    "strategy": res.name,
                    "replicate": r,
                    "iteration": i,
                    "n_labeled": config.al.seed_size + i * config.al.batch_size,
                    "hypervolume": hv,
                    "acq_time_s": acq_time,
                }
                for prefix, metric_list in [
                    ("train", state.train_metrics),
                    ("val", state.val_metrics),
                    ("sel", state.sel_metrics),
                ]:
                    if i < len(metric_list):
                        m = metric_list[i]
                        for key in ("mse", "mae", "r2"):
                            for obj in range(2):
                                row[f"{prefix}_{key}_{obj}"] = m[key][obj]
                    else:
                        for key in ("mse", "mae", "r2"):
                            for obj in range(2):
                                row[f"{prefix}_{key}_{obj}"] = float("nan")
                rows.append(row)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Saved HV data -> {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL: Compare AL strategies")
    parser.add_argument("--n_compounds", type=int, default=5000)
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--seed_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--k_ucb", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mc_passes", type=int, default=50)
    parser.add_argument("--n_replicates", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="pal_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run validation split every N iterations")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience (0 to disable)")
    parser.add_argument("--min_epochs", type=int, default=10,
                        help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64],
                        help="Hidden layer sizes (e.g. --hidden_sizes 128 64)")
    parser.add_argument("--visualize-acq", action="store_true",
                        help="Generate explanatory acquisition visualizations")
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=["exploitation", "ucb", "random", "ellipse_fast"],
        help="Acquisition functions to compare. "
             "Available: exploitation, ucb, random, ellipse, ellipse_fast"
    )
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to CSV or parquet file with custom dataset")
    parser.add_argument("--smiles_col", type=str, default="smiles",
                        help="SMILES column name (default: smiles)")
    parser.add_argument("--property_cols", type=str, nargs=2, default=None,
                        help="Two column names for objectives")
    parser.add_argument("--fingerprint_col", type=str, default=None,
                        help="Precomputed fingerprint column (optional)")
    parser.add_argument("--ref_point", type=float, nargs=2, default=[0.0, 0.0],
                        help="HV reference point (default: 0.0 0.0)")
    args = parser.parse_args()

    config = ExperimentConfig()
    config.data.n_compounds = args.n_compounds
    config.data.seed = args.seed
    config.al.n_iterations = args.n_iterations
    config.al.n_replicates = args.n_replicates
    config.al.seed_size = args.seed_size
    config.al.batch_size = args.batch_size
    config.model.epochs = args.epochs
    config.model.mc_passes = args.mc_passes
    config.model.patience = args.patience
    config.model.min_epochs = args.min_epochs
    config.model.weight_decay = args.weight_decay
    config.model.dropout = args.dropout
    config.model.hidden_sizes = tuple(args.hidden_sizes)
    config.al.val_every = args.val_every
    config.output_dir = args.output_dir
    config.device = args.device

    config.al.ref_point = tuple(args.ref_point)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load dataset ---
    if args.data_file is not None:
        if args.property_cols is None:
            parser.error("--property_cols required when --data_file is provided")
        print(f"Loading dataset from {args.data_file} ...")
        df, X_precomputed = load_dataset_from_file(
            args.data_file,
            args.property_cols,
            smiles_col=args.smiles_col if not args.fingerprint_col else None,
            fingerprint_col=args.fingerprint_col,
        )
        Y_pool = df[args.property_cols].values.astype(np.float32)
        config.obj_names = tuple(args.property_cols)
        if X_precomputed is not None:
            X_pool = X_precomputed
            config.model.in_features = X_pool.shape[1]
        else:
            print("Computing ECFP fingerprints ...")
            X_pool = compute_ecfp(
                df[args.smiles_col].tolist(),
                radius=config.ecfp_radius,
                n_bits=config.ecfp_nbits,
            )
    else:
        print("Generating ZINC dataset ...")
        df = generate_zinc_dataset(
            n_compounds=config.data.n_compounds, seed=config.data.seed
        )
        Y_pool = df[["sa_score", "qed"]].values.astype(np.float32)
        config.obj_names = ("SA score (10 - raw)", "QED")
        print("Computing ECFP fingerprints ...")
        X_pool = compute_ecfp(
            df["smiles"].tolist(),
            radius=config.ecfp_radius,
            n_bits=config.ecfp_nbits,
        )

    oracle_hv = hypervolume_2d(Y_pool, config.al.ref_point)
    print(f"Oracle HV = {oracle_hv:.4f}  (pool size = {len(Y_pool)})\n")

    acq_kwargs = {
        "ucb": {"k_ucb": args.k_ucb},
        "ellipse": {"k": args.k_ucb},
        "ellipse_fast": {"k": args.k_ucb},
    }
    strategies = {
        name: get_acquisition(name, **acq_kwargs.get(name, {}))
        for name in args.strategies
    }

    results = run_comparison(strategies, X_pool, Y_pool, config)

    plot_hv_convergence(
        results,
        oracle_hv,
        config,
        save_path=os.path.join(args.output_dir, "hv_convergence.png"),
    )
    plot_pareto_snapshots(
        results,
        Y_pool,
        config,
        save_path=os.path.join(args.output_dir, "pareto_fronts.png"),
    )
    plot_iteration_selections(
        results,
        Y_pool,
        config,
        save_path=os.path.join(args.output_dir, "iteration_selections.png"),
    )
    plot_acq_timing(
        results,
        config,
        save_path=os.path.join(args.output_dir, "acq_timing.png"),
    )
    plot_validation_metrics(
        results,
        config,
        save_path=os.path.join(args.output_dir, "validation_metrics.png"),
    )
    plot_al_diagnostics(
        results,
        strategies,
        X_pool,
        Y_pool,
        config,
        save_path=os.path.join(args.output_dir, "diagnostics.png"),
    )
    save_hv_csv(
        results,
        config,
        save_path=os.path.join(args.output_dir, "hv_convergence.csv"),
    )

    # --- Acquisition explanation visualizations ---
    if args.visualize_acq:
        print("\nGenerating acquisition explanation plots ...")
        # Pick the median replicate from the ellipse strategy
        ell_res = results["ellipse"]
        state = _pick_median_replicate(ell_res)

        # Re-train model on that replicate's labeled set (with normalization)
        Y_acq_train = state.Y_labeled
        Y_acq_mean = Y_acq_train.mean(axis=0)
        Y_acq_std = np.maximum(Y_acq_train.std(axis=0), 1e-8)
        Y_acq_train_norm = (Y_acq_train - Y_acq_mean) / Y_acq_std

        model = build_model(config.model, device=config.device)
        train_model(
            model,
            X_pool[state.labeled_indices],
            Y_acq_train_norm,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            lr=config.model.lr,
            weight_decay=config.model.weight_decay,
            device=config.device,
            patience=config.model.patience,
            min_epochs=config.model.min_epochs,
            val_fraction=config.model.val_fraction,
            lr_scheduler_patience=config.model.lr_scheduler_patience,
            lr_scheduler_factor=config.model.lr_scheduler_factor,
        )

        generate_acquisition_explanations(
            model=model,
            X_pool=X_pool,
            Y_pool=Y_pool,
            labeled_indices=state.labeled_indices,
            unlabeled_indices=state.unlabeled_indices,
            ref_point=config.al.ref_point,
            config=config,
            output_dir=args.output_dir,
            k=args.k_ucb,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()

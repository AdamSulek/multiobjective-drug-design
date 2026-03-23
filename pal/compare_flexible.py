"""Dimension-aware CLI: uses 2D or 3D acquisition stack based on objective count."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import generate_zinc_dataset
from .featurizer import LazyECFP, compute_ecfp
from .pareto import hypervolume_2d, hypervolume_3d_max_fast
from .pipeline import run_comparison, save_hv_csv, save_iteration_selections
from .plots import (
    plot_acq_timing,
    plot_hv_convergence,
    plot_iteration_selections,
    plot_iteration_selections_3d,
    plot_pareto_snapshots,
    plot_validation_metrics,
)
from .utils import load_seed_indices, seed_everything, setup_logging


def auto_ref_point(Y: np.ndarray, margin_frac: float = 0.01) -> tuple[float, ...]:
    Y = np.asarray(Y, dtype=float)
    y_min = np.nanmin(Y, axis=0)
    y_max = np.nanmax(Y, axis=0)
    span = np.maximum(y_max - y_min, 1e-9)
    ref = y_min - margin_frac * span
    return tuple(ref.tolist())


def _read_tabular(path: str) -> pd.DataFrame:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "parquet":
        return pd.read_parquet(path)
    if ext == "csv":
        return pd.read_csv(path)
    if ext == "tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file extension: .{ext}")


def _dedupe_by_id(df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in df.columns:
        return df.drop_duplicates(subset="ID").reset_index(drop=True)
    return df.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in {where}: {missing}")


def _extract_objectives(
    df: pd.DataFrame,
    property_cols: list[str],
    negate_cols: list[str] | None,
) -> np.ndarray:
    _require_columns(df, property_cols, where="data_file")

    Y_pool = df[property_cols].to_numpy(dtype=np.float32)
    if negate_cols:
        for col in negate_cols:
            if col not in property_cols:
                raise ValueError(f"Column '{col}' from --negate_cols not in --property_cols")
            Y_pool[:, property_cols.index(col)] *= -1.0
    return Y_pool


def _load_pool_data(args: argparse.Namespace, config: ExperimentConfig) -> tuple[pd.DataFrame, Any, np.ndarray]:
    if getattr(args, "x_npy", None) is not None:
        if args.data_file is None:
            raise SystemExit("--data_file is required when using --x_npy.")
        if not args.property_cols:
            raise SystemExit("--property_cols is required when using --x_npy.")

        logging.info(f"Loading labels/metadata from {args.data_file} ...")
        df = _dedupe_by_id(_read_tabular(args.data_file))
        Y_pool = _extract_objectives(df, args.property_cols, args.negate_cols)

        config.obj_names = tuple(args.property_cols)
        logging.info(f"Loading X from {args.x_npy} (mmap_mode='r') ...")
        X_pool = np.load(args.x_npy, mmap_mode="r")
        if X_pool.ndim != 2:
            raise ValueError(f"x_npy must be 2D (N,D), got {X_pool.shape}")
        if len(df) != int(X_pool.shape[0]):
            raise ValueError(f"Row mismatch: df={len(df)} vs X={X_pool.shape[0]}")
        config.model.in_features = int(X_pool.shape[1])
        return df, X_pool, Y_pool

    if args.data_file is not None:
        if not args.property_cols:
            raise SystemExit("--property_cols required when --data_file is provided")

        logging.info(f"Loading dataset from {args.data_file} ...")
        df = _dedupe_by_id(_read_tabular(args.data_file))
        _require_columns(df, args.property_cols, where="data_file")

        df = df.dropna(subset=args.property_cols).reset_index(drop=True)
        Y_pool = _extract_objectives(df, args.property_cols, args.negate_cols)

        config.obj_names = tuple(args.property_cols)
        if args.fingerprint_col is not None:
            _require_columns(df, [args.fingerprint_col], where="data_file")
            X_pool = np.stack(df[args.fingerprint_col].values).astype(np.float32)
            config.model.in_features = int(X_pool.shape[1])
        else:
            smiles = df[args.smiles_col].tolist()
            if args.lazy_fingerprints:
                X_pool = LazyECFP(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
            else:
                X_pool = compute_ecfp(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
        return df, X_pool, Y_pool

    logging.info("Generating ZINC dataset ...")
    df = generate_zinc_dataset(n_compounds=config.data.n_compounds, seed=config.data.seed)
    Y_pool = df[["sa_score", "qed"]].to_numpy(dtype=np.float32)
    config.obj_names = ("SA score (10 - raw)", "QED")
    smiles = df["smiles"].tolist()
    if args.lazy_fingerprints:
        X_pool = LazyECFP(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
    else:
        X_pool = compute_ecfp(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
    return df, X_pool, Y_pool


def _build_strategies(args: argparse.Namespace, n_obj: int) -> Dict[str, Any]:
    if n_obj == 2:
        from .acquisition_2D import get_acquisition
    elif n_obj == 3:
        from .acquisition import get_acquisition
    else:
        raise ValueError(f"Only 2D/3D objectives supported, got {n_obj}")

    strategies: Dict[str, Any] = {}

    if "random" in args.strategies:
        strategies["random"] = get_acquisition("random")

    if n_obj == 2 and "exploitation" in args.strategies:
        strategies["exploitation"] = get_acquisition("exploitation")

    if "ucb" in args.strategies:
        ucb_ks = ([0] if args.ucb_include_k0 else []) + list(args.k_list)
        for k in ucb_ks:
            kwargs = {"k_ucb": float(k)}
            if n_obj == 2:
                kwargs["zero_negative_hv"] = args.zero_negative_hv
            else:
                kwargs["max_exact_candidates"] = args.ucb_max_exact_candidates
                kwargs["clip_negative_hv"] = args.zero_negative_hv
            strategies[f"ucb_k{k}"] = get_acquisition("ucb", **kwargs)

    if n_obj == 2 and "ellipse" in args.strategies:
        for k in args.k_list:
            strategies[f"ellipse_k{k}"] = get_acquisition("ellipse", k=float(k))

    if "ellipse_fast" in args.strategies:
        for k in args.k_list:
            kwargs = {"k": float(k)}
            if n_obj == 2:
                kwargs["zero_negative_hv"] = args.zero_negative_hv
            else:
                kwargs["clip_negative_hv"] = args.zero_negative_hv
            strategies[f"ellipse_fast_k{k}"] = get_acquisition("ellipse_fast", **kwargs)

    if "ellipse_directions" in args.strategies:
        for k in args.k_list:
            kwargs = {"k": float(k), "use_front_penalty": args.direction_use_front_penalty}
            if n_obj == 3:
                kwargs["clip_negative_hv"] = args.zero_negative_hv
            strategies[f"ellipse_directions_k{k}"] = get_acquisition("ellipse_directions", **kwargs)

    return strategies


def _fmt_stage_stats(vals: list[float]) -> str:
    if not vals:
        return "sum=0.000s mean=0.000s std=0.000s min=0.000s max=0.000s"
    arr = np.asarray(vals, dtype=float)
    return (
        f"sum={arr.sum():.3f}s mean={arr.mean():.3f}s std={arr.std(ddof=0):.3f}s "
        f"min={arr.min():.3f}s max={arr.max():.3f}s"
    )


def _log_global_timing_summary(results: Dict[str, Any], logger: logging.Logger) -> None:
    logger.info("[TIME_SUMMARY_GLOBAL] start")

    all_rep_totals: list[float] = []
    all_iter_totals: list[float] = []

    for strategy_key, result in results.items():
        rep_totals: list[float] = []
        rep_iter_means: list[float] = []
        stage_vals: dict[str, list[float]] = {}

        for state in result.states:
            iter_rows = getattr(state, "iter_stage_times", [])
            if not iter_rows:
                continue

            totals = [float(r.get("total", 0.0)) for r in iter_rows]
            rep_totals.append(float(np.sum(totals)))
            rep_iter_means.append(float(np.mean(totals)))
            all_rep_totals.append(float(np.sum(totals)))
            all_iter_totals.extend(totals)

            for row in iter_rows:
                for k, v in row.items():
                    stage_vals.setdefault(k, []).append(float(v))

        if not rep_totals:
            logger.info(f"[TIME_SUMMARY_GLOBAL] strategy={result.name} key={strategy_key} no_timing_rows")
            continue

        logger.info(
            f"[TIME_SUMMARY_GLOBAL] strategy={result.name} key={strategy_key} "
            f"n_rep={len(rep_totals)} "
            f"rep_total_mean={np.mean(rep_totals):.3f}s rep_total_std={np.std(rep_totals, ddof=0):.3f}s "
            f"iter_mean_over_reps={np.mean(rep_iter_means):.3f}s"
        )

        for stage in ["total", "build", "train", "pred_train", "pred_unlab", "cov_reconstruct", "acq", "hv"]:
            vals = stage_vals.get(stage, [])
            if vals:
                logger.info(
                    f"[TIME_SUMMARY_GLOBAL] strategy={result.name} key={strategy_key} "
                    f"stage={stage} {_fmt_stage_stats(vals)}"
                )

    if all_rep_totals:
        logger.info(
            f"[TIME_SUMMARY_GLOBAL] overall_replicate_totals "
            f"n={len(all_rep_totals)} {_fmt_stage_stats(all_rep_totals)}"
        )
    if all_iter_totals:
        logger.info(
            f"[TIME_SUMMARY_GLOBAL] overall_iteration_totals "
            f"n={len(all_iter_totals)} {_fmt_stage_stats(all_iter_totals)}"
        )

def _parse_k_from_key(strategy_key: str) -> float | None:
    marker = "_k"
    if marker not in strategy_key:
        return None
    tail = strategy_key.rsplit(marker, 1)[1]
    try:
        return float(tail)
    except ValueError:
        return None


def _base_method_from_key(strategy_key: str) -> str:
    if "_k" in strategy_key:
        return strategy_key.rsplit("_k", 1)[0]
    return strategy_key


def _build_final_results_df(results: Dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for strategy_key, result in results.items():
        k_val = _parse_k_from_key(strategy_key)
        method = _base_method_from_key(strategy_key)

        for rep_idx, state in enumerate(result.states):
            hv_hist = np.asarray(getattr(state, "hv_history", []), dtype=float)
            if hv_hist.size == 0:
                continue

            xs = np.arange(hv_hist.size, dtype=float)
            hv_auc = float(np.trapz(hv_hist, xs)) if hv_hist.size > 1 else float(hv_hist[0])
            hv_final = float(hv_hist[-1])

            iter_rows = getattr(state, "iter_stage_times", [])
            totals = [float(r.get("total", 0.0)) for r in iter_rows]
            acq_vals = [float(r.get("acq", 0.0)) for r in iter_rows]

            rows.append(
                {
                    "strategy_key": strategy_key,
                    "strategy": result.name,
                    "method": method,
                    "k": np.nan if k_val is None else float(k_val),
                    "replicate": int(rep_idx),
                    "n_iters": int(len(totals)),
                    "hv_final": hv_final,
                    "hv_auc": hv_auc,
                    "iter_total_sum_s": float(np.sum(totals)) if totals else 0.0,
                    "iter_total_mean_s": float(np.mean(totals)) if totals else 0.0,
                    "acq_sum_s": float(np.sum(acq_vals)) if acq_vals else 0.0,
                    "acq_mean_s": float(np.mean(acq_vals)) if acq_vals else 0.0,
                }
            )

    return pd.DataFrame(rows)


def _log_and_save_final_results(results: Dict[str, Any], output_dir: str, logger: logging.Logger) -> None:
    rep_df = _build_final_results_df(results)
    if rep_df.empty:
        logger.info("[RESULT_SUMMARY] no rows")
        return

    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    rep_sorted = rep_df.sort_values(["method", "k", "replicate"], na_position="last")
    rep_csv = os.path.join(out_dir, "final_results_by_replicate.csv")
    rep_sorted.to_csv(rep_csv, index=False)

    logger.info("[RESULT_SUMMARY] per_replicate start")
    for r in rep_sorted.itertuples(index=False):
        k_txt = "-" if pd.isna(r.k) else f"{float(r.k):.1f}"
        logger.info(
            f"[RESULT_REP] method={r.method} key={r.strategy_key} k={k_txt} rep={int(r.replicate)} "
            f"HV_final={float(r.hv_final):.6f} HV_AUC={float(r.hv_auc):.6f} "
            f"iter_total={float(r.iter_total_sum_s):.3f}s iter_mean={float(r.iter_total_mean_s):.3f}s "
            f"acq_total={float(r.acq_sum_s):.3f}s acq_mean={float(r.acq_mean_s):.3f}s"
        )

    grp = (
        rep_df.groupby(["method", "k"], dropna=False, as_index=False)
        .agg(
            n_rep=("replicate", "count"),
            hv_final_mean=("hv_final", "mean"),
            hv_final_std=("hv_final", "std"),
            hv_auc_mean=("hv_auc", "mean"),
            hv_auc_std=("hv_auc", "std"),
            iter_total_mean_s=("iter_total_sum_s", "mean"),
            iter_total_std_s=("iter_total_sum_s", "std"),
            iter_mean_mean_s=("iter_total_mean_s", "mean"),
            iter_mean_std_s=("iter_total_mean_s", "std"),
            acq_total_mean_s=("acq_sum_s", "mean"),
            acq_total_std_s=("acq_sum_s", "std"),
        )
        .sort_values(["method", "k"], na_position="last")
    )

    grp = grp.fillna(0.0)
    k_csv = os.path.join(out_dir, "final_results_by_k.csv")
    grp.to_csv(k_csv, index=False)

    logger.info("[RESULT_SUMMARY] per_k start")
    for r in grp.itertuples(index=False):
        k_txt = "-" if pd.isna(getattr(r, "k")) else f"{float(getattr(r, 'k')):.1f}"
        logger.info(
            f"[RESULT_K] method={r.method} k={k_txt} n_rep={int(r.n_rep)} "
            f"HV_final={float(r.hv_final_mean):.6f}+/-{float(r.hv_final_std):.6f} "
            f"HV_AUC={float(r.hv_auc_mean):.6f}+/-{float(r.hv_auc_std):.6f} "
            f"iter_total={float(r.iter_total_mean_s):.3f}+/-{float(r.iter_total_std_s):.3f}s "
            f"iter_mean={float(r.iter_mean_mean_s):.3f}+/-{float(r.iter_mean_std_s):.3f}s"
        )

    logger.info(f"[RESULT_SUMMARY] saved per_replicate={rep_csv}")
    logger.info(f"[RESULT_SUMMARY] saved per_k={k_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL: flexible compare for 2D/3D")
    parser.add_argument("--n_compounds", type=int, default=5000)
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--seed_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed_indices_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mc_passes", type=int, default=50)
    parser.add_argument("--n_replicates", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="pal_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64])
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["exploitation", "ucb", "random", "ellipse_fast", "ellipse_directions"],
        help="For 2D: exploitation, ucb, random, ellipse, ellipse_fast, ellipse_directions. For 3D: ucb, random, ellipse_fast, ellipse_directions.",
    )
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--x_npy", type=str, default=None)
    parser.add_argument("--fingerprint_col", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--property_cols", type=str, nargs="+", default=None)
    parser.add_argument("--ref_point", type=float, nargs="+", default=None)
    parser.add_argument("--lazy-fingerprints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negate_cols", type=str, nargs="*", default=None)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--zero-negative-hv", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--direction-use-front-penalty", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ucb_include_k0", action="store_true")
    parser.add_argument("--ucb_max_exact_candidates", type=int, default=None)
    parser.add_argument("--global_pareto_file", type=str, default=None)
    args = parser.parse_args()

    t_all = time.perf_counter()
    seed_everything(args.seed)
    logger = setup_logging(args.output_dir)
    logger.info("Starting PAL compare_flexible")

    config = ExperimentConfig()
    config.data.n_compounds = args.n_compounds
    config.data.seed = args.seed
    config.al.n_iterations = args.n_iterations
    config.al.n_replicates = args.n_replicates
    config.al.seed_size = args.seed_size
    config.al.batch_size = args.batch_size
    config.al.val_every = args.val_every
    config.model.epochs = args.epochs
    config.model.mc_passes = args.mc_passes
    config.model.patience = args.patience
    config.model.min_epochs = args.min_epochs
    config.model.weight_decay = args.weight_decay
    config.model.dropout = args.dropout
    config.model.hidden_sizes = tuple(args.hidden_sizes)
    config.output_dir = args.output_dir
    config.device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    df, X_pool, Y_pool = _load_pool_data(args, config)
    n_obj = int(Y_pool.shape[1])
    if n_obj not in (2, 3):
        raise ValueError(f"Only 2D/3D objectives are supported, got {n_obj}")
    logger.info(f"Objective dimension detected: {n_obj}D")

    if args.ref_point is None:
        args.ref_point = list(auto_ref_point(Y_pool, margin_frac=0.01))
        logger.info(f"Auto ref_point = {args.ref_point}")
    if len(args.ref_point) != n_obj:
        raise ValueError(f"--ref_point must have dimension {n_obj}, got {len(args.ref_point)}")
    config.al.ref_point = tuple(args.ref_point)

    oracle_hv = None
    if args.global_pareto_file is not None:
        df_gp = _read_tabular(args.global_pareto_file)
        missing = [c for c in args.property_cols if c not in df_gp.columns]
        if missing:
            raise ValueError(f"Columns not found in global_pareto_file: {missing}")
        Y_gp = df_gp[args.property_cols].to_numpy(dtype=np.float32)
        if args.negate_cols:
            for col in args.negate_cols:
                Y_gp[:, args.property_cols.index(col)] *= -1.0
        oracle_hv = hypervolume_2d(Y_gp, config.al.ref_point) if n_obj == 2 else hypervolume_3d_max_fast(Y_gp, config.al.ref_point)
        logger.info(f"Global Pareto HV = {oracle_hv:.6f} (|GP|={len(Y_gp)})")

    strategies = _build_strategies(args, n_obj=n_obj)
    if len(strategies) == 0:
        raise SystemExit("No strategies selected.")

    seed_indices_files = None
    if args.seed_indices_file is not None:
        seed_indices_files = [args.seed_indices_file]
        logger.info(f"Using single seed file for this run: {args.seed_indices_file}")

    t_run = time.perf_counter()
    results = run_comparison(
        strategies,
        X_pool,
        Y_pool,
        config,
        seed_indices_files=seed_indices_files,
        load_seed_indices_fn=load_seed_indices,
    )
    logger.info(f"[TIME] run_comparison total={time.perf_counter() - t_run:.3f}s")

    save_iteration_selections(
        results=results,
        df=df,
        output_dir=args.output_dir,
        id_col="ID",
        smiles_col=args.smiles_col,
    )

    plot_jobs = [
        (plot_hv_convergence, {"oracle_hv": oracle_hv, "config": config, "save_path": os.path.join(args.output_dir, "hv_convergence.png")}),
        (plot_pareto_snapshots, {"Y_pool": Y_pool, "config": config, "save_path": os.path.join(args.output_dir, "pareto_fronts.png")}),
        (plot_iteration_selections, {"Y_pool": Y_pool, "config": config, "save_path": os.path.join(args.output_dir, "iteration_selections.png")}),
        (plot_iteration_selections_3d, {"Y_pool": Y_pool, "config": config, "save_path": os.path.join(args.output_dir, "iteration_selections_3d.png")}),
        (plot_acq_timing, {"config": config, "save_path": os.path.join(args.output_dir, "acq_timing.png")}),
        (plot_validation_metrics, {"config": config, "save_path": os.path.join(args.output_dir, "validation_metrics.png")}),
    ]
    for fn, kwargs in plot_jobs:
        fn(results, **kwargs)

    save_hv_csv(results=results, config=config, save_path=os.path.join(args.output_dir, "hv_convergence.csv"))

    _log_global_timing_summary(results, logger)

    _log_and_save_final_results(results, args.output_dir, logger)

    logger.info(f"[TIME] full_pipeline total={time.perf_counter() - t_all:.3f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()
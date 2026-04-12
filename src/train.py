"""Unified 2D/3D training CLI with optional Weights & Biases logging."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from pal.config import ExperimentConfig
from pal.data import generate_zinc_dataset
from pal.featurizer import LazyECFP, compute_ecfp
from pal.pareto import hypervolume_2d, hypervolume_3d_max_fast
from pal.pipeline import save_hv_csv, save_iteration_selections
from pal.plots import (
    plot_acq_timing,
    plot_hv_convergence,
    plot_iteration_selections,
    plot_iteration_selections_3d,
    plot_pareto_snapshots,
    plot_validation_metrics,
)
from pal.log_prefs import pal_log_timer, reset_pal_log_prefs, set_pal_log_prefs
from pal.utils import load_seed_indices, seed_everything, setup_logging

from .loop import build_strategies, run_loop_matrix


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


def _log_train_infer_stage_totals(results: dict[str, Any], logger: logging.Logger) -> None:
    """Aggregate GPU-side training and prediction times from each AL iteration."""
    for strategy_key, result in results.items():
        for rep_idx, state in enumerate(result.states):
            rows = getattr(state, "iter_stage_times", [])
            tr = sum(float(r.get("train", 0.0)) for r in rows)
            pt = sum(float(r.get("pred_train", 0.0)) for r in rows)
            pu = sum(float(r.get("pred_unlab", 0.0)) for r in rows)
            logger.info(
                "[TIME_SUMMARY_GPU] key=%s rep=%d sum_train_s=%.3f sum_pred_train_s=%.3f sum_pred_unlab_s=%.3f",
                strategy_key,
                rep_idx,
                tr,
                pt,
                pu,
            )


def _log_global_timing_summary(results: dict[str, Any], logger: logging.Logger) -> None:
    for strategy_key, result in results.items():
        rep_totals = []
        for state in result.states:
            iter_rows = getattr(state, "iter_stage_times", [])
            totals = [float(r.get("total", 0.0)) for r in iter_rows]
            if totals:
                rep_totals.append(float(np.sum(totals)))
        if rep_totals:
            logger.info(
                f"[TIME_SUMMARY_GLOBAL] key={strategy_key} n_rep={len(rep_totals)} "
                f"rep_total_mean={np.mean(rep_totals):.3f}s rep_total_std={np.std(rep_totals, ddof=0):.3f}s"
            )


def _log_and_save_final_results(results: dict[str, Any], output_dir: str, logger: logging.Logger) -> None:
    rows = []
    for strategy_key, result in results.items():
        for rep_idx, state in enumerate(result.states):
            hv_hist = np.asarray(getattr(state, "hv_history", []), dtype=float)
            if hv_hist.size == 0:
                continue
            rows.append(
                {
                    "strategy_key": strategy_key,
                    "strategy": result.name,
                    "replicate": int(rep_idx),
                    "hv_final": float(hv_hist[-1]),
                    "hv_auc": float(np.trapz(hv_hist, np.arange(hv_hist.size))),
                }
            )

    if not rows:
        logger.info("[RESULT_SUMMARY] no rows")
        return

    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "final_results_by_replicate.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"[RESULT_SUMMARY] saved per_replicate={out_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified PAL train entrypoint for 2D and 3D.")
    p.add_argument("--n_compounds", type=int, default=5000)
    p.add_argument("--n_iterations", type=int, default=50)
    p.add_argument("--seed_size", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--seed_indices_file", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="PyTorch batch size for MLP training (not acquisition batch).",
    )
    p.add_argument(
        "--predict_eval_batch_size",
        type=int,
        default=4096,
        help="Batch size for predict_eval (labeled + unlabeled pool inference).",
    )
    p.add_argument(
        "--mc_predict_batch_size",
        type=int,
        default=4096,
        help="Batch size for MC-dropout inference (uncertainty / covariance path).",
    )
    p.add_argument("--mc_passes", type=int, default=50)
    p.add_argument("--n_replicates", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="pal_results")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="PyTorch DataLoader workers; -1 = auto from --device (8 if CUDA device string, else 0).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_sizes", type=int, nargs="+", default=[64])
    p.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["random", "ucb", "ellipse_fast", "ellipse_directions"],
        help="Common strategy list for both 2D and 3D.",
    )
    p.add_argument("--data_file", type=str, default=None)
    p.add_argument("--x_npy", type=str, default=None)
    p.add_argument("--fingerprint_col", type=str, default=None)
    p.add_argument("--smiles_col", type=str, default="smiles")
    p.add_argument("--property_cols", type=str, nargs="+", default=None)
    p.add_argument("--ref_point", type=float, nargs="+", default=None)
    p.add_argument("--lazy-fingerprints", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--negate_cols", type=str, nargs="*", default=None)
    p.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--zero-negative-hv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--direction-use-front-penalty", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ucb_include_k0", action="store_true")
    p.add_argument("--ucb_max_exact_candidates", type=int, default=None)
    p.add_argument("--global_pareto_file", type=str, default=None)

    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_project", type=str, default="multiobjective-drug-design")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument(
        "--diag-logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit [DIAG] lines (GPU / dataloader detail). Use --no-diag-logging to disable.",
    )
    p.add_argument(
        "--timer-logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit [TIMER] lines. Use --no-timer-logging to disable.",
    )
    return p.parse_args()


def _init_wandb(args: argparse.Namespace, config: ExperimentConfig, n_obj: int):
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        logging.getLogger("pal").warning(f"wandb disabled: import failed: {exc}")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=os.path.abspath(args.output_dir),
        config={
            **vars(args),
            "n_obj": n_obj,
            "ref_point": list(config.al.ref_point),
        },
    )
    # Chart axes: AL iteration vs training epoch (global across AL iters).
    wandb.define_metric("al/iteration")
    wandb.define_metric("train/epoch")
    _al_metrics = (
        "loop/hv",
        "loop/delta_hv",
        "timing/iteration_s",
        "timing/fit_s",
        "timing/acquisition_s",
        "data/n_labeled",
        "strategy/k_ucb",
        "strategy/key",
        "strategy/name",
        "al/replicate",
    )
    for name in _al_metrics:
        try:
            wandb.define_metric(name, step_metric="al/iteration")
        except Exception:
            pass
    for name in ("train/train_loss", "train/val_loss", "timing/train_epoch_s"):
        try:
            wandb.define_metric(name, step_metric="train/epoch")
        except Exception:
            pass
    return run


def _log_wandb_summary(wandb_run, results: dict[str, Any], t_total_s: float) -> None:
    if wandb_run is None:
        return

    wandb_run.summary["runtime_total_s"] = float(t_total_s)

    for key, result in results.items():
        finals = [float(s.hv_history[-1]) for s in result.states if len(s.hv_history) > 0]
        if finals:
            wandb_run.summary[f"hv_final/{key}_mean"] = float(np.mean(finals))
            wandb_run.summary[f"hv_final/{key}_std"] = float(np.std(finals, ddof=0))
    wandb_run.finish()


def main() -> None:
    args = _parse_args()
    t_all = time.perf_counter()

    seed_everything(args.seed)
    logger = setup_logging(args.output_dir)
    logger.info("Starting unified train.py")

    config = ExperimentConfig()
    config.data.n_compounds = args.n_compounds
    config.data.seed = args.seed
    config.al.n_iterations = args.n_iterations
    config.al.n_replicates = args.n_replicates
    config.al.seed_size = args.seed_size
    config.al.batch_size = args.batch_size
    config.al.val_every = args.val_every
    config.model.epochs = args.epochs
    config.model.batch_size = int(args.train_batch_size)
    config.model.predict_eval_batch_size = int(args.predict_eval_batch_size)
    config.model.mc_predict_batch_size = int(args.mc_predict_batch_size)
    config.model.mc_passes = args.mc_passes
    config.model.patience = args.patience
    config.model.min_epochs = args.min_epochs
    config.model.weight_decay = args.weight_decay
    config.model.dropout = args.dropout
    config.model.hidden_sizes = tuple(args.hidden_sizes)
    config.output_dir = args.output_dir
    config.device = args.device
    if int(args.num_workers) < 0:
        nw = 8 if str(config.device).startswith("cuda") else 0
        logger.info(f"DataLoader num_workers={nw} (auto from device={config.device})")
    else:
        nw = max(0, int(args.num_workers))
        logger.info(f"DataLoader num_workers={nw} (from --num_workers)")
    config.model.num_workers = nw
    os.makedirs(args.output_dir, exist_ok=True)

    config.log_diag = bool(args.diag_logging)
    config.log_timer = bool(args.timer_logging)
    _lp_tok = set_pal_log_prefs(log_diag=config.log_diag, log_timer=config.log_timer)
    try:
        t_pool = time.perf_counter()
        df, X_pool, Y_pool = _load_pool_data(args, config)
        if pal_log_timer():
            logger.info(
                "[TIMER] pool_load: %.3fs N=%d n_obj=%d x_mmap=%s",
                time.perf_counter() - t_pool,
                len(Y_pool),
                int(Y_pool.shape[1]),
                isinstance(X_pool, np.memmap),
            )
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

        strategies = build_strategies(
            n_obj=n_obj,
            strategy_names=list(args.strategies),
            k_list=list(args.k_list),
            ucb_include_k0=bool(args.ucb_include_k0),
            zero_negative_hv=bool(args.zero_negative_hv),
            direction_use_front_penalty=bool(args.direction_use_front_penalty),
            ucb_max_exact_candidates=args.ucb_max_exact_candidates,
        )

        seed_indices_files = [args.seed_indices_file] if args.seed_indices_file is not None else None
        run = _init_wandb(args, config, n_obj)
        config.wandb_log = run is not None

        t_run = time.perf_counter()
        loop_out = run_loop_matrix(
            config=config,
            X_pool=X_pool,
            Y_pool=Y_pool,
            strategies=strategies,
            seed_indices_files=seed_indices_files,
            load_seed_indices_fn=load_seed_indices,
        )
        logger.info(f"[TIME] run_loop_matrix total={time.perf_counter() - t_run:.3f}s")
        _log_train_infer_stage_totals(loop_out.results, logger)

        save_iteration_selections(
            results=loop_out.results,
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
            fn(loop_out.results, **kwargs)

        save_hv_csv(results=loop_out.results, config=config, save_path=os.path.join(args.output_dir, "hv_convergence.csv"))
        _log_global_timing_summary(loop_out.results, logger)
        _log_and_save_final_results(loop_out.results, args.output_dir, logger)

        t_total = time.perf_counter() - t_all
        _log_wandb_summary(run, loop_out.results, t_total)
        logger.info(f"[TIME] full_pipeline total={t_total:.3f}s")
        logger.info("Done!")
    finally:
        reset_pal_log_prefs(_lp_tok)


if __name__ == "__main__":
    main()

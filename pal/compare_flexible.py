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


def _load_pool_data(args: argparse.Namespace, config: ExperimentConfig) -> tuple[pd.DataFrame, Any, np.ndarray]:
    if getattr(args, "x_npy", None) is not None:
        if args.data_file is None:
            raise SystemExit("--data_file is required when using --x_npy.")
        if not args.property_cols:
            raise SystemExit("--property_cols is required when using --x_npy.")

        logging.info(f"Loading labels/metadata from {args.data_file} ...")
        df = _read_tabular(args.data_file)
        if "ID" in df.columns:
            df = df.drop_duplicates(subset="ID").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        missing = [c for c in args.property_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in data_file: {missing}")

        Y_pool = df[args.property_cols].to_numpy(dtype=np.float32)
        if args.negate_cols:
            for col in args.negate_cols:
                if col not in args.property_cols:
                    raise ValueError(f"Column '{col}' from --negate_cols not in --property_cols")
                Y_pool[:, args.property_cols.index(col)] *= -1.0

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
        df = _read_tabular(args.data_file)
        if "ID" in df.columns:
            df = df.drop_duplicates(subset="ID").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        missing = [c for c in args.property_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in data_file: {missing}")

        df = df.dropna(subset=args.property_cols).reset_index(drop=True)
        Y_pool = df[args.property_cols].to_numpy(dtype=np.float32)
        if args.negate_cols:
            for col in args.negate_cols:
                if col not in args.property_cols:
                    raise ValueError(f"Column '{col}' from --negate_cols not in --property_cols")
                Y_pool[:, args.property_cols.index(col)] *= -1.0

        config.obj_names = tuple(args.property_cols)
        if args.fingerprint_col is not None:
            if args.fingerprint_col not in df.columns:
                raise ValueError(f"Fingerprint column '{args.fingerprint_col}' not found")
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
    X_pool = LazyECFP(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits) if args.lazy_fingerprints else compute_ecfp(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL: flexible compare for 2D/3D")
    parser.add_argument("--n_compounds", type=int, default=5000)
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--seed_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed_indices_files", type=str, nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mc_passes", type=int, default=50)
    parser.add_argument("--n_replicates", type=int, default=3)
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

    t_run = time.perf_counter()
    results = run_comparison(
        strategies,
        X_pool,
        Y_pool,
        config,
        seed_indices_files=args.seed_indices_files,
        load_seed_indices_fn=load_seed_indices,
    )
    logger.info(f"[TIME] run_comparison total={time.perf_counter() - t_run:.3f}s")

    save_iteration_selections(results=results, df=df, output_dir=args.output_dir, id_col="ID", smiles_col=args.smiles_col)
    plot_hv_convergence(results, oracle_hv, config, save_path=os.path.join(args.output_dir, "hv_convergence.png"))
    plot_pareto_snapshots(results, Y_pool, config, save_path=os.path.join(args.output_dir, "pareto_fronts.png"))
    plot_iteration_selections(results, Y_pool, config, save_path=os.path.join(args.output_dir, "iteration_selections.png"))
    plot_iteration_selections_3d(results, Y_pool, config, save_path=os.path.join(args.output_dir, "iteration_selections_3d.png"))
    plot_acq_timing(results, config, save_path=os.path.join(args.output_dir, "acq_timing.png"))
    plot_validation_metrics(results, config, save_path=os.path.join(args.output_dir, "validation_metrics.png"))
    save_hv_csv(results=results, config=config, save_path=os.path.join(args.output_dir, "hv_convergence.csv"))

    logger.info(f"[TIME] full_pipeline total={time.perf_counter() - t_all:.3f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()

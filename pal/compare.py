# pal/compare.py
"""
Run multiple acquisition strategies and plot convergence (CLI entry).

This file is intentionally kept "thin":
- parses CLI args
- prepares ExperimentConfig + loads dataset
- builds acquisition strategies
- calls pipeline to run AL and save artifacts
- calls plotting helpers

HV/Pareto math lives elsewhere (currently .pareto + inside run_al_loop()).
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

from .acquisition import get_acquisition
from .acquisition.base import AcquisitionFunction
from .config import ExperimentConfig
from .data import generate_zinc_dataset, load_dataset_from_file
from .featurizer import LazyECFP, compute_ecfp
from .pareto import hypervolume_2d, hypervolume_3d_max_fast  # will be swapped later (2D->3D) centrally

from .pipeline import (
    run_comparison,
    save_iteration_selections,
    save_hv_csv,
)
from .plots import (
    plot_hv_convergence,
    plot_pareto_snapshots,
    plot_iteration_selections,
    plot_iteration_selections_3d,   # <-- DODAJ
    plot_acq_timing,
    plot_validation_metrics,
)

# Keep these helpers local for now (you said you don't want too many files).
# Later you can move them into pal/utils.py if you want.
import sys
import re
from pathlib import Path

from .utils import load_seed_indices, seed_everything, setup_logging

def auto_ref_point(Y: np.ndarray, margin_frac: float = 0.01) -> tuple[float, ...]:
    Y = np.asarray(Y, dtype=float)
    y_min = np.nanmin(Y, axis=0)
    y_max = np.nanmax(Y, axis=0)
    span = np.maximum(y_max - y_min, 1e-9)
    ref = y_min - margin_frac * span
    return tuple(ref.tolist())

def _build_strategies(args: argparse.Namespace) -> Dict[str, AcquisitionFunction]:
    strategies: Dict[str, AcquisitionFunction] = {}

    # random
    if "random" in args.strategies:
        strategies["random"] = get_acquisition("random")

    # UCB
    if "ucb" in args.strategies:
        ucb_ks = ([0] if args.ucb_include_k0 else []) + list(args.k_list)
        for k in ucb_ks:
            strategies[f"ucb_k{k}"] = get_acquisition(
                "ucb",
                k_ucb=float(k),
                max_exact_candidates=args.ucb_max_exact_candidates,
                clip_negative_hv=args.clip_negative_hv,
            )

    # ellipse_fast
    if "ellipse_fast" in args.strategies:
        for k in args.k_list:
            strategies[f"ellipse_fast_k{k}"] = get_acquisition(
                "ellipse_fast",
                k=float(k),
                clip_negative_hv=args.clip_negative_hv,
            )

    # ellipse_directions
    if "ellipse_directions" in args.strategies:
        for k in args.k_list:
            strategies[f"ellipse_directions_k{k}"] = get_acquisition(
                "ellipse_directions",
                k=float(k),
                clip_negative_hv=args.clip_negative_hv,
                use_front_penalty=args.direction_use_front_penalty,
            )

    return strategies


def _load_pool_data(
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns (df, X_pool, Y_pool).

    NEW fast path (recommended for 3M):
      - if args.x_npy is provided:
          * df is loaded from args.data_file (parquet/csv) WITHOUT fingerprints (you already removed X_ecfp_2)
          * X_pool is loaded from args.x_npy via np.load(..., mmap_mode="r")
          * Y_pool from args.property_cols
          * optionally negate selected objectives by name via args.negate_cols

    Legacy paths:
      - if args.data_file is provided and args.x_npy is None:
          * uses load_dataset_from_file() which may load fingerprint_col or smiles_col
      - else:
          * generates ZINC benchmark
    """
    # ------------------------------------------------------------
    # FAST PATH: X from .npy (memmap), labels/metadata from data_file
    # ------------------------------------------------------------
    if getattr(args, "x_npy", None) is not None:
        if args.data_file is None:
            raise SystemExit("--data_file is required when using --x_npy (labels/metadata live there).")
        if args.property_cols is None or len(args.property_cols) == 0:
            raise SystemExit("--property_cols is required when using --x_npy.")

        logging.info(f"Loading labels/metadata from {args.data_file} ...")
        ext = args.data_file.rsplit(".", 1)[-1].lower()
        if ext == "parquet":
            df = pd.read_parquet(args.data_file)
        elif ext in ("csv", "tsv"):
            df = pd.read_csv(args.data_file)
        else:
            raise ValueError(f"Unsupported data_file extension: .{ext}")

        if "ID" in df.columns:
            df = df.drop_duplicates(subset="ID").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        missing = [c for c in args.property_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in data_file: {missing}")

        Y_pool = df[args.property_cols].values.astype(np.float32)

        # Negate selected objective columns by NAME
        if args.negate_cols is not None and len(args.negate_cols) > 0:
            logging.info(f"Negating objective columns: {args.negate_cols}")
            for col in args.negate_cols:
                if col not in args.property_cols:
                    raise ValueError(
                        f"Column '{col}' requested in --negate_cols is not in --property_cols {args.property_cols}"
                    )
                j = args.property_cols.index(col)
                Y_pool[:, j] = -Y_pool[:, j]

        config.obj_names = tuple(args.property_cols)

        logging.info(f"Loading X from {args.x_npy} (mmap_mode='r') ...")
        X_pool = np.load(args.x_npy, mmap_mode="r")
        if X_pool.ndim != 2:
            raise ValueError(f"x_npy must be a 2D matrix (N,D), got shape {X_pool.shape}")

        config.model.in_features = int(X_pool.shape[1])

        # Sanity check alignment
        if len(df) != int(X_pool.shape[0]):
            raise ValueError(
                f"Row mismatch: df has {len(df)} rows, X has {X_pool.shape[0]} rows. "
                "They must be aligned in the same row order."
            )

        return df, X_pool, Y_pool

    # ------------------------------------------------------------
    # LEGACY PATH: data_file via load_dataset_from_file()
    # ------------------------------------------------------------
    if args.data_file is not None:
        if args.property_cols is None:
            raise SystemExit("--property_cols required when --data_file is provided")

        logging.info(f"Loading dataset from {args.data_file} ...")
        df, X_precomputed = load_dataset_from_file(
            args.data_file,
            args.property_cols,
            smiles_col=args.smiles_col if not args.fingerprint_col else None,
            fingerprint_col=args.fingerprint_col,
        )

        if "ID" in df.columns:
            df = df.drop_duplicates(subset="ID").reset_index(drop=True)

        Y_pool = df[args.property_cols].values.astype(np.float32)

        # Negate selected objective columns by NAME
        if args.negate_cols is not None and len(args.negate_cols) > 0:
            logging.info(f"Negating objective columns: {args.negate_cols}")
            for col in args.negate_cols:
                if col not in args.property_cols:
                    raise ValueError(
                        f"Column '{col}' requested in --negate_cols is not in --property_cols {args.property_cols}"
                    )
                j = args.property_cols.index(col)
                Y_pool[:, j] = -Y_pool[:, j]

        config.obj_names = tuple(args.property_cols)

        if X_precomputed is not None:
            X_pool = X_precomputed
            config.model.in_features = int(X_pool.shape[1])
        else:
            smiles = df[args.smiles_col].tolist()
            if args.lazy_fingerprints:
                logging.info("Using lazy ECFP fingerprints (computed on demand) ...")
                X_pool = LazyECFP(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
            else:
                logging.info("Computing ECFP fingerprints ...")
                X_pool = compute_ecfp(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)

        return df, X_pool, Y_pool

    # ------------------------------------------------------------
    # DEFAULT: ZINC
    # ------------------------------------------------------------
    logging.info("Generating ZINC dataset ...")
    df = generate_zinc_dataset(n_compounds=config.data.n_compounds, seed=config.data.seed)
    Y_pool = df[["sa_score", "qed"]].values.astype(np.float32)
    config.obj_names = ("SA score (10 - raw)", "QED")

    smiles = df["smiles"].tolist()
    if args.lazy_fingerprints:
        logging.info("Using lazy ECFP fingerprints (computed on demand) ...")
        X_pool = LazyECFP(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)
    else:
        logging.info("Computing ECFP fingerprints ...")
        X_pool = compute_ecfp(smiles, radius=config.ecfp_radius, n_bits=config.ecfp_nbits)

    return df, X_pool, Y_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="PAL: Compare AL strategies")

    parser.add_argument("--n_compounds", type=int, default=5000)
    parser.add_argument("--n_iterations", type=int, default=50)
    parser.add_argument("--seed_size", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument(
        "--seed_indices_files",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of files with initial seed indices (one file per replicate). "
            "Each file can be .npy (numpy array) or a text file with integers "
            "(one per line or comma/space separated). If provided, overrides random seed selection."
        ),
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mc_passes", type=int, default=50)
    parser.add_argument("--n_replicates", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="pal_results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--val_every", type=int, default=1, help="Run validation split every N iterations")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience (0 to disable)")
    parser.add_argument("--min_epochs", type=int, default=10, help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="AdamW weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64], help="Hidden layer sizes")

    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["ucb", "random", "ellipse_fast", "ellipse_directions"],
        help="Acquisition functions to compare. Available: ucb, random, ellipse_fast, ellipse_directions",
    )

    parser.add_argument("--data_file", type=str, default=None, help="Path to CSV or parquet file with custom dataset")
    parser.add_argument("--x_npy", type=str, default=None,
                    help="Path to prebuilt feature matrix .npy (memmap-friendly), e.g. data/3D/X_uint8.npy")
    parser.add_argument("--ids_npy", type=str, default=None,
                    help="Path to ids.npy aligned with x_npy (optional, for debugging/mapping)")
    
    parser.add_argument("--smiles_col", type=str, default="smiles", help="SMILES column name (default: smiles)")
    parser.add_argument(
        "--property_cols",
        type=str,
        nargs="+",
        default=None,
        help="Objective column names (2 or 3). Example: --property_cols score_3GVB score_6D6P score_6GQP",
    )
    #parser.add_argument("--fingerprint_col", type=str, default=None, help="Precomputed fingerprint column (optional)")

    parser.add_argument("--ref_point", type=float, nargs="+", default=None,
                    help="HV reference point. Provide 2 or 3 values depending on objectives.")
    parser.add_argument(
        "--lazy-fingerprints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute ECFP fingerprints lazily on demand (default: on)",
    )
    parser.add_argument(
        "--negate_cols",
        type=str,
        nargs="*",
        default=None,
        help="Example: --negate_cols score_3GVB score_6D6P"
    )

    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 3, 4], help="List of k values for UCB/ellipse.")
    parser.add_argument(
        "--clip-negative-hv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip negative acquisition/HV deltas to 0. Default: enabled.",
    )
    parser.add_argument(
        "--direction-use-front-penalty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For ellipse_directions, include the Pareto-front penalty term max_p w^T p. Default: enabled.",
    )
    parser.add_argument("--ucb_include_k0", action="store_true", help="Also run UCB with k=0 (pure exploitation).")
    parser.add_argument(
        "--ucb_max_exact_candidates",
        type=int,
        default=None,
        help=(
            "Limit exact UCB HV scoring to top-K prefiltered candidates. "
            "If omitted, UCB with k>=2 uses 50000 by default."
        ),
    )
    parser.add_argument(
        "--global_pareto_file",
        type=str,
        default=None,
        help="CSV/parquet z globalnym Pareto frontem (musi zawierać property_cols).",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    logger = setup_logging(args.output_dir)
    logger.info("Starting PAL comparison")

    # Build config
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

    # Load dataset (df, X_pool, Y_pool) and set config.obj_names
    df, X_pool, Y_pool = _load_pool_data(args, config)
    
    if args.ref_point is None:
        args.ref_point = list(auto_ref_point(Y_pool, margin_frac=0.01))
        logging.info(f"Auto ref_point = {args.ref_point}")

    # walidacja wymiaru
    if len(args.ref_point) != Y_pool.shape[1]:
        raise ValueError(
            f"--ref_point must have dimension {Y_pool.shape[1]} (same as objectives), "
            f"got {len(args.ref_point)}"
        )

    if not np.all(np.isfinite(args.ref_point)):
        raise ValueError(f"Invalid ref_point computed: {args.ref_point}")

    config.al.ref_point = tuple(args.ref_point)
    
    oracle_hv = None
    if args.global_pareto_file is not None:
        ext = args.global_pareto_file.rsplit(".", 1)[-1].lower()
        if ext == "parquet":
            df_gp = pd.read_parquet(args.global_pareto_file)
        elif ext in ("csv", "tsv"):
            df_gp = pd.read_csv(args.global_pareto_file)
        else:
            raise ValueError(f"Unsupported global_pareto_file extension: .{ext}")

        missing = [c for c in args.property_cols if c not in df_gp.columns]
        if missing:
            raise ValueError(f"Columns not found in global_pareto_file: {missing}")

        Y_gp = df_gp[args.property_cols].to_numpy(dtype=np.float32)

        # neguj te same cele co w _load_pool_data()
        if args.negate_cols:
            for col in args.negate_cols:
                j = args.property_cols.index(col)  # zakłada pełne nazwy kolumn
                Y_gp[:, j] = -Y_gp[:, j]

        if Y_gp.shape[1] == 2:
            oracle_hv = hypervolume_2d(Y_gp, config.al.ref_point)
        elif Y_gp.shape[1] == 3:
            oracle_hv = hypervolume_3d_max_fast(Y_gp, config.al.ref_point)
        else:
            raise ValueError("Only 2D/3D supported")

        logging.info(f"Global Pareto HV = {oracle_hv:.6f} (|GP|={len(Y_gp)})")

    # Strategies
    strategies = _build_strategies(args)
    if len(strategies) == 0:
        raise SystemExit("No strategies selected. Check --strategies.")

    # Run AL comparison
    results = run_comparison(
        strategies,
        X_pool,
        Y_pool,
        config,
        seed_indices_files=args.seed_indices_files,
        load_seed_indices_fn=load_seed_indices,
    )

    # Save selections and stats
    save_iteration_selections(
        results=results,
        df=df,
        output_dir=args.output_dir,
        id_col="ID",
        smiles_col=args.smiles_col,
    )
    
    # 3D
    plot_iteration_selections_3d(
        results,
        Y_pool,
        config,
        save_path=os.path.join(args.output_dir, "iteration_selections_3d.png"),
    )

    # Plots + CSV
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
    
    save_hv_csv(
        results=results,
        config=config,
        save_path=os.path.join(args.output_dir, "hv_convergence.csv"),
    )

    # NOTE: visualize-acq block removed from the thin CLI on purpose.
    # It was also inconsistent (referenced results["ellipse"] which doesn't exist).
    # If you still need it, we can move it into a separate script or add it back cleanly.

    logging.info("\nDone!")


if __name__ == "__main__":
    main()

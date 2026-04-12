# pal/pipeline.py
"""
Pipeline utilities for running multi-strategy AL comparisons + saving artifacts.

This module intentionally does NOT implement Pareto/HV math.
HV is produced inside `run_al_loop()` (state.hv_history), so swapping 2D->3D HV
later only requires changing the pareto/HV implementation used by the loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .acquisition.base import AcquisitionFunction
from .config import ExperimentConfig
from .loop import ALState, run_al_loop


# ----------------------------
# Result container
# ----------------------------

@dataclass
class StrategyResult:
    """Aggregated results for one strategy key across replicates."""
    name: str
    hv_histories: List[List[float]]
    states: List[ALState]


# ----------------------------
# Core pipeline: run comparison
# ----------------------------

def run_comparison(
    strategies: Dict[str, AcquisitionFunction],
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    config: ExperimentConfig,
    *,
    seed_indices_files: Optional[List[str]] = None,
    load_seed_indices_fn=None,
) -> Dict[str, StrategyResult]:
    """
    Run each strategy for multiple replicates on precomputed pool data.

    Parameters
    ----------
    strategies:
        dict mapping strategy_key -> AcquisitionFunction instance
        e.g. {"ucb_k2": acq_fn, "random": acq_fn}
    X_pool, Y_pool:
        pool features/targets
    config:
        ExperimentConfig (must contain al.* fields)
    seed_indices_files:
        optional list of file paths, one per replicate, used to override random seeding
    load_seed_indices_fn:
        optional function(path)->np.ndarray, injected to avoid circular imports.
        If you already have `load_seed_indices` in compare.py or utils.py, pass it here.

    Returns
    -------
    dict strategy_key -> StrategyResult
    """
    base_seed = config.data.seed
    n_replicates = config.al.n_replicates

    if seed_indices_files is not None:
        if len(seed_indices_files) != n_replicates:
            raise ValueError(
                f"--seed_indices_files must have exactly n_replicates={n_replicates} paths, "
                f"got {len(seed_indices_files)}"
            )
        if load_seed_indices_fn is None:
            raise ValueError(
                "seed_indices_files provided but load_seed_indices_fn is None. "
                "Pass your load_seed_indices() function into run_comparison(...)."
            )

    results: Dict[str, StrategyResult] = {}
    for strategy_key, acq_fn in strategies.items():
        results[strategy_key] = StrategyResult(
            name=acq_fn.name,
            hv_histories=[],
            states=[],
        )

    for r in range(n_replicates):
        rep_seed = base_seed + r
        seed_indices = None

        if seed_indices_files is not None:
            seed_indices = load_seed_indices_fn(seed_indices_files[r])

            # Validate seed indices
            if len(seed_indices) != config.al.seed_size:
                raise ValueError(
                    f"Seed indices in {seed_indices_files[r]} have length {len(seed_indices)}, "
                    f"expected seed_size={config.al.seed_size}"
                )
            if seed_indices.min() < 0 or seed_indices.max() >= len(Y_pool):
                raise ValueError(
                    f"Seed indices in {seed_indices_files[r]} out of range "
                    f"[0, {len(Y_pool)-1}]"
                )
            if len(np.unique(seed_indices)) != len(seed_indices):
                raise ValueError(f"Seed indices in {seed_indices_files[r]} contain duplicates")

        logging.info(f"--- Replicate {r} (seed={rep_seed}) ---")

        for strategy_key, acq_fn in strategies.items():
            logging.info(f"=== Running strategy: {acq_fn.name} ({strategy_key}) ===")

            state = run_al_loop(
                X_pool,
                Y_pool,
                acq_fn,
                config,
                seed=rep_seed,
                seed_indices=seed_indices,
                strategy_key=strategy_key,
                replicate_idx=int(r),
            )

            results[strategy_key].hv_histories.append(state.hv_history)
            results[strategy_key].states.append(state)

    return results


# ----------------------------
# Saving artifacts
# ----------------------------

def save_iteration_selections(
    *,
    results: Dict[str, StrategyResult],
    df: pd.DataFrame,
    output_dir: str,
    id_col: str,
    smiles_col: str = "smiles",
) -> None:
    """
    Save selected compounds per iteration for each strategy/replicate.

    Writes one parquet per strategy_key in <output_dir>/selections/, e.g.:
      ucb_k2_selections.parquet
      ellipse_fast_k3_selections.parquet
      random_selections.parquet

    Columns:
      strategy, strategy_key, k, replicate, iteration, pool_index, mol_id, smiles (optional)
    """
    out_dir = Path(output_dir) / "selections"
    out_dir.mkdir(parents=True, exist_ok=True)

    if id_col not in df.columns:
        raise KeyError(f"id_col='{id_col}' not in df.columns")

    mol_ids = df[id_col].to_numpy()
    smiles = df[smiles_col].to_numpy() if smiles_col in df.columns else None

    def _extract_k(strategy_key: str) -> float:
        m = re.search(r"_k(\d+)$", strategy_key)
        return float(m.group(1)) if m else float("nan")

    for strategy_key, res in results.items():
        k_val = _extract_k(strategy_key)

        rows = []
        for r, state in enumerate(res.states):
            for it, sel_pool in enumerate(state.selections_per_iter):
                for pool_idx in sel_pool:
                    row = {
                        "strategy": res.name,            # pretty name
                        "strategy_key": strategy_key,    # stable key
                        "k": k_val,
                        "replicate": r,
                        "iteration": it,                 # 0 = seed
                        "pool_index": int(pool_idx),
                        "mol_id": mol_ids[pool_idx],
                    }
                    if smiles is not None:
                        row["smiles"] = smiles[pool_idx]
                    rows.append(row)

        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", strategy_key)
        out_path = out_dir / f"{safe_key}_selections.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        logging.info(f"Saved iteration selections -> {out_path}")


def save_hv_csv(
    *,
    results: Dict[str, StrategyResult],
    config: ExperimentConfig,
    save_path: str,
) -> None:
    """
    Save HV convergence data as CSV (one row per strategy/replicate/iteration).

    This is dimension-agnostic: it just stores state.hv_history and metrics arrays.
    """
    n_obj = len(config.obj_names)

    rows = []
    for _, res in results.items():
        for r, (hv_history, state) in enumerate(zip(res.hv_histories, res.states)):
            for i, hv in enumerate(hv_history):
                acq_time = state.acq_time_per_iter[i] if i < len(state.acq_time_per_iter) else 0.0
                stage = state.iter_stage_times[i - 1] if (i > 0 and (i - 1) < len(state.iter_stage_times)) else {}
                row = {
                    "strategy": res.name,
                    "replicate": r,
                    "iteration": i,
                    "n_labeled": config.al.seed_size + i * config.al.batch_size,
                    "hypervolume": float(hv),
                    "acq_time_s": float(acq_time),
                    "iter_total_s": float(stage.get("total", 0.0)) if i > 0 else 0.0,
                    "build_s": float(stage.get("build", 0.0)) if i > 0 else 0.0,
                    "train_s": float(stage.get("train", 0.0)) if i > 0 else 0.0,
                    "pred_train_s": float(stage.get("pred_train", 0.0)) if i > 0 else 0.0,
                    "pred_unlab_s": float(stage.get("pred_unlab", 0.0)) if i > 0 else 0.0,
                    "cov_reconstruct_s": float(stage.get("cov_reconstruct", 0.0)) if i > 0 else 0.0,
                    "hv_eval_s": float(stage.get("hv", 0.0)) if i > 0 else 0.0,
                }

                # metrics lists may be shorter (e.g. val computed every N)
                for prefix, metric_list in [
                    ("train", state.train_metrics),
                    ("val", state.val_metrics),
                    ("sel", state.sel_metrics),
                ]:
                    if i < len(metric_list):
                        m = metric_list[i]
                        for key in ("mse", "mae", "r2"):
                            vals = m.get(key, None)
                            if vals is None:
                                for obj in range(n_obj):
                                    row[f"{prefix}_{key}_{obj}"] = float("nan")
                            else:
                                for obj in range(n_obj):
                                    row[f"{prefix}_{key}_{obj}"] = float(vals[obj])
                    else:
                        for key in ("mse", "mae", "r2"):
                            for obj in range(n_obj):
                                row[f"{prefix}_{key}_{obj}"] = float("nan")

                rows.append(row)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    logging.info(f"Saved HV data -> {save_path}")
"""Unified loop helpers for 2D/3D experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pal.acquisition import get_acquisition
from pal.config import ExperimentConfig
from pal.pipeline import run_comparison


@dataclass
class LoopRun:
    """Container for raw loop outputs used by train/report layers."""

    results: dict[str, Any]
    n_obj: int


def build_strategies(
    *,
    n_obj: int,
    strategy_names: list[str],
    k_list: list[int],
    ucb_include_k0: bool,
    zero_negative_hv: bool,
    direction_use_front_penalty: bool,
    ucb_max_exact_candidates: int | None,
) -> dict[str, Any]:
    """Build a unified strategy registry for both 2D and 3D."""
    if n_obj not in (2, 3):
        raise ValueError(f"Only 2D/3D objectives are supported, got {n_obj}")

    strategies: dict[str, Any] = {}
    names = set(strategy_names)

    if "random" in names:
        strategies["random"] = get_acquisition("random")

    if "ucb" in names:
        ucb_ks = ([0] if ucb_include_k0 else []) + list(k_list)
        for k in ucb_ks:
            strategies[f"ucb_k{k}"] = get_acquisition(
                "ucb_flexible",
                k_ucb=float(k),
                clip_negative_hv=bool(zero_negative_hv),
                max_exact_candidates=ucb_max_exact_candidates,
            )

    if "ellipse_fast" in names:
        for k in k_list:
            strategies[f"ellipse_fast_k{k}"] = get_acquisition(
                "ellipse_fast_flexible",
                k=float(k),
                clip_negative_hv=bool(zero_negative_hv),
            )

    if "ellipse_directions" in names:
        for k in k_list:
            strategies[f"ellipse_directions_k{k}"] = get_acquisition(
                "ellipse_directions_flexible",
                k=float(k),
                use_front_penalty=bool(direction_use_front_penalty),
                clip_negative_hv=bool(zero_negative_hv),
            )

    if len(strategies) == 0:
        raise ValueError("No strategies selected after parsing --strategies.")

    return strategies


def run_loop_matrix(
    *,
    config: ExperimentConfig,
    X_pool: np.ndarray | Any,
    Y_pool: np.ndarray,
    strategies: dict[str, Any],
    seed_indices_files: list[str] | None,
    load_seed_indices_fn,
) -> LoopRun:
    """Run all strategies/replicates in a dimension-agnostic way."""
    results = run_comparison(
        strategies,
        X_pool,
        Y_pool,
        config,
        seed_indices_files=seed_indices_files,
        load_seed_indices_fn=load_seed_indices_fn,
    )
    return LoopRun(results=results, n_obj=int(Y_pool.shape[1]))


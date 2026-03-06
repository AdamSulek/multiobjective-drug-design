"""UCB exploration: optimistic HV improvement (mean + k*std)."""

from typing import Tuple

import numpy as np

from .base import AcquisitionFunction
from ..pareto_3D import (
    pareto_front_3d_max,
    hv_3d_max,
    build_dominance_index_3d,
    classify_dominated_offline,
    score_point,
)


class UCBExplorationAcquisition(AcquisitionFunction):
    """Upper Confidence Bound: score = delta-HV of optimistic point."""

    def __init__(self, k_ucb: float = 2.0, max_exact_candidates: int | None = None):
        self.k_ucb = float(k_ucb)
        # For larger k, optimistic sets can get very large.
        # Limit exact HV scoring to a prefiltered subset.
        if max_exact_candidates is None and self.k_ucb >= 2.0:
            max_exact_candidates = 50_000
        self.max_exact_candidates = (
            int(max_exact_candidates)
            if max_exact_candidates is not None and int(max_exact_candidates) > 0
            else None
        )

    @property
    def name(self) -> str:
        return f"UCB(k={self.k_ucb})"

    def score(
        self,
        means: np.ndarray,                          # (N,3)
        stds: np.ndarray,                           # (N,3)
        current_labels: np.ndarray,                 # (M,3)
        ref_point: Tuple[float, ...],               # len=3
        covs: np.ndarray | None = None,
        *,
        pareto_front: np.ndarray | None = None,
        pareto_dom_index=None,
        pareto_hv: float | None = None,
    ) -> np.ndarray:

        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)

        # 1) optimistic points
        optimistic = means + self.k_ucb * stds  # (N,3)

        # 2) reuse precomputed front/hv/index if provided
        if pareto_front is None:
            front = pareto_front_3d_max(np.asarray(current_labels, dtype=np.float64))
        else:
            front = np.asarray(pareto_front, dtype=np.float64)

        if pareto_hv is None:
            hv_front = hv_3d_max(front, ref_point)
        else:
            hv_front = float(pareto_hv)

        if pareto_dom_index is None:
            dom_index = build_dominance_index_3d(front)
        else:
            dom_index = pareto_dom_index

        # 3) screening: dominated vs front?
        dominated = classify_dominated_offline(optimistic, dom_index)  # bool (N,)

        # 4) scoring:
        #    - only NONDOMINATED (sticks out) can improve HV
        scores = np.zeros((optimistic.shape[0],), dtype=np.float64)
        nd_idx = np.where(~dominated)[0]
        if self.max_exact_candidates is not None and nd_idx.size > self.max_exact_candidates:
            ref = np.asarray(ref_point, dtype=np.float64).reshape(1, 3)
            slack = np.clip(optimistic[nd_idx] - ref, a_min=0.0, a_max=None)
            cheap = np.prod(slack, axis=1)
            top_local = np.argpartition(cheap, -self.max_exact_candidates)[-self.max_exact_candidates:]
            nd_idx = nd_idx[top_local]

        for i in nd_idx:
            scores[i] = score_point(
                optimistic[i],
                front,
                hv_front,
                ref_point,
                dominated_by_front=False,
            )

        return scores.astype(np.float32)

"""UCB exploration: optimistic HV improvement (mean + k*std)."""

from typing import Tuple

import numpy as np

from .base import AcquisitionFunction
from ..pareto import batch_delta_hv_2d, batch_delta_hv_3d, pareto_front_2d
from ..pareto_3D import (
    pareto_front_3d_max,
    build_dominance_index_3d,
    classify_dominated_offline,
)


class UCBExplorationAcquisition(AcquisitionFunction):
    """Upper Confidence Bound: score = delta-HV of optimistic point."""

    def __init__(
        self,
        k_ucb: float = 2.0,
        max_exact_candidates: int | None = None,
        clip_negative_hv: bool = True,
    ):
        self.k_ucb = float(k_ucb)
        self.clip_negative_hv = bool(clip_negative_hv)
        # For larger k, optimistic sets can get very large.
        # Limit exact HV scoring to a prefiltered subset (clip=True path only).
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

    @property
    def needs_uncertainty(self) -> bool:
        # k=0 is greedy on predictive mean; MC uncertainty is unnecessary.
        return self.k_ucb > 0.0

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, ...],
        covs: np.ndarray | None = None,
        *,
        pareto_front: np.ndarray | None = None,
        pareto_dom_index=None,
        pareto_hv: float | None = None,
    ) -> np.ndarray:
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        n_obj = int(means.shape[1])

        # optimistic points
        optimistic = means + self.k_ucb * stds

        if n_obj == 2:
            front2 = pareto_front_2d(np.asarray(current_labels, dtype=np.float64))
            scores2 = batch_delta_hv_2d(
                front2,
                optimistic,
                ref_point,
                compute_negative=not self.clip_negative_hv,
            ).astype(np.float32)
            if self.clip_negative_hv:
                np.maximum(scores2, 0.0, out=scores2)
            return scores2

        if n_obj != 3:
            raise ValueError(f"{self.name} supports only 2D/3D, got {n_obj} objectives")

        # reuse precomputed front if provided
        if pareto_front is None:
            front = pareto_front_3d_max(np.asarray(current_labels, dtype=np.float64))
        else:
            front = np.asarray(pareto_front, dtype=np.float64)

        # clip=False: evaluate full batch with compute_negative=True
        # so dominated / below-ref candidates get negative scores instead of zeros.
        if not self.clip_negative_hv:
            return batch_delta_hv_3d(
                front,
                optimistic,
                ref_point,
                compute_negative=True,
            ).astype(np.float32)

        # clip=True fast path: we only need potentially positive candidates.
        if pareto_dom_index is None:
            dom_index = build_dominance_index_3d(front)
        else:
            dom_index = pareto_dom_index

        dominated = classify_dominated_offline(optimistic, dom_index)
        ref_arr = np.asarray(ref_point, dtype=np.float64).reshape(1, 3)
        valid = np.all(optimistic > ref_arr, axis=1)

        nd_idx = np.where(~dominated)[0]
        nd_idx = nd_idx[valid[nd_idx]]

        if self.max_exact_candidates is not None and nd_idx.size > self.max_exact_candidates:
            slack = np.clip(optimistic[nd_idx] - ref_arr, a_min=0.0, a_max=None)
            cheap = np.prod(slack, axis=1)
            top_local = np.argpartition(cheap, -self.max_exact_candidates)[-self.max_exact_candidates:]
            nd_idx = nd_idx[top_local]

        scores = np.zeros((optimistic.shape[0],), dtype=np.float32)
        if nd_idx.size > 0:
            deltas = batch_delta_hv_3d(
                front,
                optimistic[nd_idx],
                ref_point,
                compute_negative=False,
            ).astype(np.float32)
            scores[nd_idx] = deltas

        np.maximum(scores, 0.0, out=scores)
        return scores

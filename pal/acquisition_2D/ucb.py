"""UCB exploration: optimistic HV improvement (mean + k*std)."""

from typing import Tuple

import numpy as np

from ..pareto import batch_delta_hv_2d, pareto_front_2d
from .base import AcquisitionFunction


class UCBExplorationAcquisition(AcquisitionFunction):
    """Upper Confidence Bound: score = delta-HV of optimistic point."""

    def __init__(self, k_ucb: float = 2.0, zero_negative_hv: bool = False):
        self.k_ucb = k_ucb
        self.zero_negative_hv = bool(zero_negative_hv)

    @property
    def name(self) -> str:
        return f"UCB(k={self.k_ucb})"

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
        covs: np.ndarray | None = None,
    ) -> np.ndarray:
        front = pareto_front_2d(current_labels)
        optimistic = means + self.k_ucb * stds
        return batch_delta_hv_2d(
            front,
            optimistic,
            ref_point,
            compute_negative=not self.zero_negative_hv,
        ).astype(np.float32)

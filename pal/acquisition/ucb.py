"""UCB exploration: optimistic HV improvement (mean + k*std)."""

from typing import Tuple

import numpy as np

from ..pareto import batch_delta_hv_3d, pareto_front  # <- 3D versions
from .base import AcquisitionFunction


class UCBExplorationAcquisition(AcquisitionFunction):
    """Upper Confidence Bound: score = delta-HV of optimistic point."""

    def __init__(self, k_ucb: float = 2.0):
        self.k_ucb = float(k_ucb)

    @property
    def name(self) -> str:
        return f"UCB(k={self.k_ucb})"

    def score(
        self,
        means: np.ndarray,                     # (N,3)
        stds: np.ndarray,                      # (N,3)
        current_labels: np.ndarray,            # (M,3)
        ref_point: Tuple[float, float, float], # 3D
        covs: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:

        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)

        # Pareto front in 3D
        front = pareto_front(np.asarray(current_labels, dtype=np.float32))

        # Optimistic point
        optimistic = means + self.k_ucb * stds   # (N,3)

        # ΔHV in 3D
        scores = batch_delta_hv_3d(front, optimistic, ref_point)

        return scores.astype(np.float32)
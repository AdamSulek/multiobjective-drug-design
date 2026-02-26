"""Random baseline acquisition."""

from typing import Tuple

import numpy as np

from .base import AcquisitionFunction


class RandomAcquisition(AcquisitionFunction):
    """Baseline: uniform random scores (no model information used)."""

    @property
    def name(self) -> str:
        return "Random"

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
        covs: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.random.rand(len(means)).astype(np.float32)

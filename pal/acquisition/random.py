from typing import Tuple

import numpy as np
from .base import AcquisitionFunction


class RandomAcquisition(AcquisitionFunction):
    @property
    def name(self) -> str:
        return "Random"

    @property
    def needs_uncertainty(self) -> bool:
        # Random baseline does not use uncertainty estimates.
        return False

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, ...],
        covs: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        return np.random.rand(len(means)).astype(np.float32)

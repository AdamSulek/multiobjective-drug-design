"""Abstract base class for acquisition functions."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AcquisitionFunction(ABC):
    """Interface that every acquisition strategy must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for plots / logs."""

    @abstractmethod
    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
        covs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a score for each candidate (higher = more desirable).

        Parameters
        ----------
        means : np.ndarray, shape ``(N, 2)``
            Predicted objective means for the unlabeled pool.
        stds : np.ndarray, shape ``(N, 2)``
            Predicted objective standard deviations.
        current_labels : np.ndarray, shape ``(M, 2)``
            Objective values of the already-labeled set.
        ref_point : tuple of float
            Reference point for hypervolume computation.
        covs : np.ndarray or None, shape ``(N, 2, 2)``
            Full covariance matrices (optional, used by ellipse strategy).

        Returns
        -------
        np.ndarray, shape ``(N,)``
            Acquisition scores.
        """

    def select(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
        k: int,
        covs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the indices of the top-*k* candidates by score."""
        scores = self.score(means, stds, current_labels, ref_point, covs=covs)
        # top-k indices (descending score)
        if k >= len(scores):
            return np.argsort(scores)[::-1].copy()
        top_k_idx = np.argpartition(scores, -k)[-k:]
        return top_k_idx[np.argsort(scores[top_k_idx])[::-1]].copy()

"""Abstract base class for acquisition functions."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any

import numpy as np


class AcquisitionFunction(ABC):
    """Interface that every acquisition strategy must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for plots / logs."""
        
    @property
    def needs_full_cov(self) -> bool:
        return False

    @abstractmethod
    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, ...],
        covs: np.ndarray | None = None,
        *,
        # Pareto/HV helpers (optional; only some strategies will use them)
        pareto_front: np.ndarray | None = None,
        pareto_dom_index: Any | None = None,
        pareto_hv: float | None = None,
    ) -> np.ndarray:
        """Return a score for each candidate (higher = more desirable).

        Parameters
        ----------
        means : np.ndarray, shape (N, m)
            Predicted objective means for the unlabeled pool.
        stds : np.ndarray, shape (N, m)
            Predicted objective standard deviations.
        current_labels : np.ndarray, shape (M, m)
            Objective values of the already-labeled set.
        ref_point : tuple of float
            Reference point for hypervolume computation (len=m).
        covs : np.ndarray or None, shape (N, m, m)
            Full covariance matrices (optional).
        pareto_front : np.ndarray or None
            Optional precomputed Pareto front of current_labels (only if m==3 in our use-case).
        pareto_dom_index : Any or None
            Optional prebuilt dominance screening index (e.g., Fenwick index) for pareto_front.
        pareto_hv : float or None
            Optional precomputed HV(pareto_front).

        Returns
        -------
        np.ndarray, shape (N,)
            Acquisition scores.
        """

    def select(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, ...],
        k: int,
        covs: np.ndarray | None = None,
        *,
        pareto_front: np.ndarray | None = None,
        pareto_dom_index: Any | None = None,
        pareto_hv: float | None = None,
    ) -> np.ndarray:
        """Return the indices of the top-*k* candidates by score."""
        scores = self.score(
            means,
            stds,
            current_labels,
            ref_point,
            covs=covs,
            pareto_front=pareto_front,
            pareto_dom_index=pareto_dom_index,
            pareto_hv=pareto_hv,
        )

        # top-k indices (descending score)
        if k >= len(scores):
            return np.argsort(scores)[::-1].copy()

        top_k_idx = np.argpartition(scores, -k)[-k:]
        return top_k_idx[np.argsort(scores[top_k_idx])[::-1]].copy()
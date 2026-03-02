"""Ellipse acquisition: exploit 2D Gaussian confidence isolines."""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np

from ..pareto import pareto_front_2d, batch_delta_hv_2d
from .base import AcquisitionFunction


class EllipseAcquisition(AcquisitionFunction):
    """Exact ellipse acquisition, faster implementation.

    Behavior preserved:
      - still uses Pareto front of current_labels
      - still evaluates k-sigma ellipse boundary
      - still returns max delta-HV on boundary (exact delta-HV)

    Speedups:
      - batch_delta_hv_2d for angles (no per-angle Python loop + no HV recompute)
      - caches unit circle points
      - optional top-K preselect (disabled by default)
    """

    def __init__(self, k: float = 2.0, n_angles: int = 64, preselect_k: int = 0):
        self.k = float(k)
        self.n_angles = int(n_angles)
        self.preselect_k = int(preselect_k)

        thetas = np.linspace(0.0, 2.0 * np.pi, self.n_angles, endpoint=False)
        self._circle = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1).astype(np.float32)  # (A,2)
        self._eye2 = np.eye(2, dtype=np.float32)

    @property
    def name(self) -> str:
        if self.preselect_k and self.preselect_k > 0:
            return f"Ellipse(k={self.k}, angles={self.n_angles}, topK={self.preselect_k})"
        return f"Ellipse(k={self.k}, angles={self.n_angles})"

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
        covs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Front must stay: compute once per call
        front = pareto_front_2d(current_labels)

        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        N = means.shape[0]
        scores = np.zeros(N, dtype=np.float32)

        # If covs missing: exact delta-HV of mean points (vectorized)
        if covs is None:
            return batch_delta_hv_2d(front, means, ref_point).astype(np.float32)

        covs = np.asarray(covs, dtype=np.float32)

        # Optional preselect (DISABLED by default to preserve behavior exactly)
        if self.preselect_k and 0 < self.preselect_k < N:
            proxy = batch_delta_hv_2d(front, means, ref_point).astype(np.float32)
            K = self.preselect_k
            idx = np.argpartition(-proxy, K - 1)[:K]
            # leave others at 0 or proxy (choose 0 to preserve "only ellipse" scoring)
        else:
            idx = np.arange(N, dtype=np.int64)

        circle = self._circle

        for i in idx:
            mu = means[i]          # (2,)
            cov = covs[i]          # (2,2)

            # Cholesky decomposition with jitter; fallback to diagonal stds
            try:
                L = np.linalg.cholesky(cov + 1e-9 * self._eye2).astype(np.float32, copy=False)
            except np.linalg.LinAlgError:
                L = np.diag(stds[i])

            # Points on ellipse: (A,2)
            pts = mu[None, :] + self.k * (circle @ L.T)

            # Exact delta-HV for all angles at once
            dhv = batch_delta_hv_2d(front, pts, ref_point)  # (A,)
            scores[i] = float(np.max(dhv)) if dhv.size else 0.0

        return scores
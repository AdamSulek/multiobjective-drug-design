"""Ellipse acquisition: exploit 2D Gaussian confidence isolines."""

from typing import Tuple

import numpy as np

from ..pareto import delta_hv_contribution, pareto_front_2d
from .base import AcquisitionFunction


class EllipseAcquisition(AcquisitionFunction):
    """Score each candidate by the best HV improvement on its confidence ellipse.

    For each candidate, the full 2x2 covariance matrix defines a confidence
    ellipse (isoline of the 2D Gaussian).  We sample points on the k-sigma
    ellipse boundary and return the maximum delta-HV over those points.

    Parameters
    ----------
    k : float
        Confidence level (number of standard deviations). Default 2.0.
    n_angles : int
        Number of angles to sample on the ellipse boundary. Default 64.
    """

    def __init__(self, k: float = 2.0, n_angles: int = 64):
        self.k = k
        self.n_angles = n_angles

    @property
    def name(self) -> str:
        return f"Ellipse(k={self.k})"

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, float],
        covs: np.ndarray | None = None,
    ) -> np.ndarray:
        front = pareto_front_2d(current_labels)
        N = len(means)
        scores = np.zeros(N, dtype=np.float32)

        if covs is None:
            # Fallback: treat objectives as independent (diagonal covariance)
            for i in range(N):
                scores[i] = delta_hv_contribution(
                    front, means[i, 0], means[i, 1], ref_point
                )
            return scores

        # Pre-compute unit circle points
        thetas = np.linspace(0, 2 * np.pi, self.n_angles, endpoint=False)
        circle = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)  # (n_angles, 2)

        for i in range(N):
            mu = means[i]          # (2,)
            cov = covs[i]          # (2, 2)

            # Cholesky decomposition with jitter for numerical safety
            try:
                L = np.linalg.cholesky(cov + 1e-9 * np.eye(2))
            except np.linalg.LinAlgError:
                # If Cholesky fails, fall back to diagonal
                L = np.diag(stds[i])

            # Points on k-sigma ellipse: mu + k * L @ [cos(theta), sin(theta)]
            points = mu + self.k * (circle @ L.T)  # (n_angles, 2)

            best = -np.inf
            for j in range(self.n_angles):
                dhv = delta_hv_contribution(
                    front, points[j, 0], points[j, 1], ref_point
                )
                if dhv > best:
                    best = dhv

            scores[i] = best

        return scores

"""Fast vectorized ellipse acquisition — no Python loops."""

from typing import Tuple

import numpy as np

from ..pareto import batch_delta_hv_2d, pareto_front_2d
from .base import AcquisitionFunction


class FastEllipseAcquisition(AcquisitionFunction):
    """Fully vectorized version of EllipseAcquisition.

    Same semantics: score each candidate by the best HV improvement on its
    k-sigma confidence ellipse boundary.  Uses batch Cholesky, ``einsum``,
    dominated-point filtering, and ``batch_delta_hv_2d`` to eliminate all
    Python loops.

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
        return f"FastEllipse(k={self.k})"

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
        A = self.n_angles

        if covs is None:
            # No covariance — just score the means via batch delta-HV
            return batch_delta_hv_2d(front, means, ref_point).astype(np.float32)

        # --- 1. Unit circle ---
        thetas = np.linspace(0, 2 * np.pi, A, endpoint=False)
        circle = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)  # (A, 2)

        # --- 2. Batch Cholesky on (N, 2, 2) ---
        cov_arr = np.array(covs, dtype=np.float64)
        cov_arr += 1e-9 * np.eye(2)
        try:
            L = np.linalg.cholesky(cov_arr)  # (N, 2, 2)
        except np.linalg.LinAlgError:
            # Fallback: per-candidate Cholesky with diagonal backup
            L = np.zeros((N, 2, 2), dtype=np.float64)
            for i in range(N):
                try:
                    L[i] = np.linalg.cholesky(cov_arr[i])
                except np.linalg.LinAlgError:
                    L[i] = np.diag(stds[i].astype(np.float64))

        # --- 3. All ellipse points at once: (N, A, 2) ---
        # points[n, a] = means[n] + k * L[n] @ circle[a]
        ellipse_offsets = self.k * np.einsum("nij,aj->nai", L, circle)  # (N, A, 2)
        points = means[:, None, :] + ellipse_offsets  # (N, A, 2)

        # --- 4. Dominated-by-mean filter ---
        # If an ellipse point has both objectives <= the mean, by symmetry
        # the opposite point dominates it and has >= delta-HV.  Skip it.
        dominated = np.all(points <= means[:, None, :], axis=2)  # (N, A)

        # Build a flat array of non-dominated points with an index map
        keep = ~dominated  # (N, A)
        # Record which candidate each kept point belongs to
        cand_idx = np.broadcast_to(np.arange(N)[:, None], (N, A))

        flat_points = points[keep]       # (K, 2)
        flat_cand = cand_idx[keep]       # (K,)

        # --- 5. Batch delta-HV ---
        if len(flat_points) == 0:
            return np.zeros(N, dtype=np.float32)

        flat_dhv = batch_delta_hv_2d(front, flat_points, ref_point)  # (K,)

        # --- 6. Max delta-HV per candidate ---
        scores = np.zeros(N, dtype=np.float64)
        np.maximum.at(scores, flat_cand, flat_dhv)

        return scores.astype(np.float32)

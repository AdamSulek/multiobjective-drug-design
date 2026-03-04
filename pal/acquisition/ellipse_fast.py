"""Fast vectorized 3D ellipse acquisition — no Python loops over directions."""

from __future__ import annotations
from typing import Tuple

import numpy as np

from ..pareto import batch_delta_hv_3d, pareto_front  # <- ND/3D versions
from .base import AcquisitionFunction


def fibonacci_sphere_directions(n: int) -> np.ndarray:
    """
    Deterministic, approximately uniform directions on S^2.
    Returns (n,3) unit vectors.
    """
    if n <= 1:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        z = 1.0 - (2.0 * i) / (n - 1)  # from +1 to -1
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        points.append([x, y, z])

    W = np.asarray(points, dtype=np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


class FastEllipseAcquisition(AcquisitionFunction):
    """Fully vectorized 3D version of ellipse acquisition.

    Semantics: score each candidate by the best HV improvement on its
    k-sigma confidence ellipsoid boundary. Uses batch Cholesky + einsum,
    dominated-by-mean filtering, and batch delta-HV in 3D.

    Parameters
    ----------
    k : float
        Confidence level (number of standard deviations). Default 2.0.
    n_directions : int
        Number of directions sampled on the sphere. Default 50.
    """

    def __init__(self, k: float = 2.0, n_directions: int = 50):
        self.k = float(k)
        self.n_directions = int(n_directions)
        self._dirs = fibonacci_sphere_directions(self.n_directions)  # (D,3)

    @property
    def name(self) -> str:
        return f"FastEllipse3D(k={self.k}, D={self.n_directions})"
    
    @property
    def needs_full_cov(self) -> bool:
        return True

    def score(
        self,
        means: np.ndarray,                     # (N,3)
        stds: np.ndarray,                      # (N,3)
        current_labels: np.ndarray,            # (M,3)
        ref_point: Tuple[float, float, float],
        covs: np.ndarray | None = None,        # (N,3,3)
        **kwargs,
    ) -> np.ndarray:
        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        front = pareto_front(np.asarray(current_labels, dtype=np.float32))  # (M',3)

        N = means.shape[0]
        D = self._dirs.shape[0]

        if covs is None:
            # No covariance — just score the means via batch delta-HV (3D)
            return batch_delta_hv_3d(front, means, ref_point).astype(np.float32)

        # --- 1) Directions on unit sphere ---
        dirs = self._dirs  # (D,3)

        # --- 2) Batch Cholesky on (N,3,3) ---
        cov_arr = np.asarray(covs, dtype=np.float64)
        cov_arr = cov_arr + 1e-9 * np.eye(3, dtype=np.float64)[None, :, :]

        try:
            L = np.linalg.cholesky(cov_arr)  # (N,3,3)
        except np.linalg.LinAlgError:
            # Fallback: per-candidate Cholesky with diagonal backup
            L = np.zeros((N, 3, 3), dtype=np.float64)
            for i in range(N):
                try:
                    L[i] = np.linalg.cholesky(cov_arr[i])
                except np.linalg.LinAlgError:
                    L[i] = np.diag(stds[i].astype(np.float64))

        # --- 3) All ellipsoid points at once: (N,D,3) ---
        # points[n,d] = means[n] + k * L[n] @ dirs[d]
        offsets = self.k * np.einsum("nij,dj->ndi", L, dirs.astype(np.float64))  # (N,D,3)
        points = means[:, None, :] + offsets.astype(np.float32)                  # (N,D,3)

        # --- 4) Dominated-by-mean filter ---
        # If a boundary point has all coords <= mean, the opposite boundary point
        # dominates it by symmetry and has >= delta-HV.
        dominated = np.all(points <= means[:, None, :], axis=2)  # (N,D)
        keep = ~dominated

        cand_idx = np.broadcast_to(np.arange(N)[:, None], (N, D))
        flat_points = points[keep]        # (K,3)
        flat_cand = cand_idx[keep]        # (K,)

        if flat_points.shape[0] == 0:
            return np.zeros(N, dtype=np.float32)

        # --- 5) Batch delta-HV in 3D ---
        flat_dhv = batch_delta_hv_3d(front, flat_points, ref_point)  # (K,)

        # --- 6) Max delta-HV per candidate ---
        scores = np.full(N, -np.inf, dtype=np.float64)
        np.maximum.at(scores, flat_cand, flat_dhv.astype(np.float64))

        return scores.astype(np.float32)
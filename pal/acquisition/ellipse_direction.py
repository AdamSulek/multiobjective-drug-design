from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from ..pareto import pareto_front  # ND version
from .base import AcquisitionFunction


def fibonacci_sphere_directions(n: int) -> np.ndarray:
    """
    Deterministic, approximately uniform directions on S^2.
    Returns (n,3) unit vectors.
    """
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        z = 1 - (2 * i) / (n - 1)          # z from 1 to -1
        radius = np.sqrt(max(0.0, 1 - z*z))
        theta = golden_angle * i

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points.append([x, y, z])

    W = np.array(points, dtype=np.float32)
    # already unit length, but normalize defensively
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


class EllipseDirectionAcquisition(AcquisitionFunction):
    def __init__(
        self,
        k: float = 2.0,
        n_directions: int = 50,
        eps: float = 1e-9,
    ):
        self.k = float(k)
        self.eps = float(eps)
        self.d = 3
        self.W = fibonacci_sphere_directions(n_directions)
        self.last_w_idx_max: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"EllipseDirections3D(k={self.k}, W={self.W.shape[0]})"

    def score(
        self,
        means: np.ndarray,                 # (N,3)
        stds: np.ndarray,                  # (N,3)
        current_labels: np.ndarray,        # (M,3)
        ref_point: Tuple[float, float, float],  # unused
        covs: np.ndarray | None = None,    # (N,3,3)
        **kwargs,
    ) -> np.ndarray:

        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        W = self.W

        # --- Pareto front (ND) ---
        front = pareto_front(np.asarray(current_labels, dtype=np.float32))

        # penalty(w) = max_p w^T p
        if front.shape[0] == 0:
            penalties = np.zeros((W.shape[0],), dtype=np.float32)
        else:
            penalties = (front @ W.T).max(axis=0).astype(np.float32)

        N = means.shape[0]

        # --- Covariance handling ---
        if covs is None:
            covs_use = np.zeros((N, 3, 3), dtype=np.float32)
            covs_use[:, 0, 0] = stds[:, 0] ** 2
            covs_use[:, 1, 1] = stds[:, 1] ** 2
            covs_use[:, 2, 2] = stds[:, 2] ** 2
        else:
            covs_use = np.asarray(covs, dtype=np.float32)

        # v(i,w) = Σ_i w   -> (N,W,3)
        v = np.einsum("nij,wj->nwi", covs_use, W)

        # denom^2 = w^T Σ_i w
        denom2 = np.einsum("nwi,wi->nw", v, W)
        denom = np.sqrt(np.clip(denom2, self.eps, None)).astype(np.float32)

        # optimistic boundary point on ellipsoid
        pts = means[:, None, :] + self.k * (v / denom[:, :, None])

        # alpha(i,w) = w^T pts - penalty(w)
        proj = np.einsum("nwi,wi->nw", pts, W)
        alpha = proj - penalties[None, :]

        best_w = np.argmax(alpha, axis=1).astype(np.int32)
        scores = alpha[np.arange(N), best_w].astype(np.float32)

        self.last_w_idx_max = best_w
        return scores
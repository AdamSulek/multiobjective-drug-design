from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from ..pareto import pareto_front_2d
from .base import AcquisitionFunction


class EllipseDirectionAcquisition(AcquisitionFunction):
    def __init__(
        self,
        k: float = 2.0,
        w_directions: Optional[np.ndarray] = None,  # (W,2)
        eps: float = 1e-9,
    ):
        self.k = float(k)
        self.eps = float(eps)

        if w_directions is None:
            w_directions = np.column_stack([
                np.linspace(1.0, 0.0, 22),
                np.linspace(0.0, 1.0, 22),
            ]).astype(np.float32)

        self.W = np.asarray(w_directions, dtype=np.float32)
        assert self.W.ndim == 2 and self.W.shape[1] == 2, "w_directions must be (W,2)"
        self.last_w_idx_max: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"EllipseDirections(k={self.k}, W={self.W.shape[0]})"

    def score(
        self,
        means: np.ndarray,                 # (N,2)
        stds: np.ndarray,                  # (N,2)
        current_labels: np.ndarray,        # (M,2)
        ref_point: Tuple[float, float],    # unused
        covs: np.ndarray | None = None,    # (N,2,2)
        **kwargs,
    ) -> np.ndarray:
        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        W = self.W  # (W,2)

        # Pareto front (must stay)
        front = pareto_front_2d(np.asarray(current_labels, dtype=np.float32))  # (M',2)

        # penalties(w) = max_p w^T p
        if front.shape[0] == 0:
            penalties = np.zeros((W.shape[0],), dtype=np.float32)
        else:
            penalties = (front @ W.T).max(axis=0).astype(np.float32)  # (W,)

        N = means.shape[0]
        nW = W.shape[0]

        # Build covs if missing
        if covs is None:
            covs_use = np.zeros((N, 2, 2), dtype=np.float32)
            covs_use[:, 0, 0] = stds[:, 0] ** 2
            covs_use[:, 1, 1] = stds[:, 1] ** 2
        else:
            covs_use = np.asarray(covs, dtype=np.float32)

        # v(i,w) = Σ_i w  -> shape (N,W,2)
        # einsum: (N,2,2) x (W,2) -> (N,W,2)
        v = np.einsum("nij,wj->nwi", covs_use, W)  # (N,W,2)

        # denom2(i,w) = w^T Σ_i w = sum_j w_j * v_j
        denom2 = np.einsum("nwi,wi->nw", v, W)     # (N,W)
        denom = np.sqrt(np.clip(denom2, self.eps, None)).astype(np.float32)  # (N,W)

        # pts(i,w,2) = mu_i + k * v(i,w)/denom(i,w)
        pts = means[:, None, :] + self.k * (v / denom[:, :, None])  # (N,W,2)

        # alpha(i,w) = w^T pts(i,w) - penalties(w)
        proj = np.einsum("nwi,wi->nw", pts, W)  # (N,W)
        alpha = proj - penalties[None, :]       # (N,W)

        # score(i) = max_w alpha(i,w)
        best_w = np.argmax(alpha, axis=1).astype(np.int32)          # (N,)
        scores = alpha[np.arange(N), best_w].astype(np.float32)     # (N,)

        self.last_w_idx_max = best_w
        return scores
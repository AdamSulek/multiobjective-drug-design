from typing import Tuple, Optional
import numpy as np

from ..pareto import pareto_front_2d
from .base import AcquisitionFunction


class EllipseDirectionAcquisition(AcquisitionFunction):
    """
    Directional ellipse (your old ellipsoid logic, but inside PAL):

    - Build Pareto front from current_labels.
    - penalties(w) = max_{p in front} w^T p
    - For each candidate i and direction w:
        x*(i,w) = mu_i + k * (Sigma_i w) / sqrt(w^T Sigma_i w)
        alpha(i,w) = w^T x*(i,w) - penalties(w)
      score(i) = max_w alpha(i,w)

    Uses full covs if provided; otherwise falls back to diagonal from stds.
    """

    def __init__(
        self,
        k: float = 2.0,
        w_directions: Optional[np.ndarray] = None,  # (W,2)
        eps: float = 1e-9,
    ):
        self.k = float(k)
        self.eps = float(eps)

        # 22 directions exactly like you gave
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

        # --- Pareto + penalties exactly like your torch code ---
        front = pareto_front_2d(np.asarray(current_labels, dtype=np.float32))  # (M',2)

        # penalties(w) = max_p w^T p
        # if front empty (edge case), penalties=0
        if front.shape[0] == 0:
            penalties = np.zeros((self.W.shape[0],), dtype=np.float32)
        else:
            penalties = (front @ self.W.T).max(axis=0).astype(np.float32)  # (W,)

        N = means.shape[0]

        # --- covariance per candidate ---
        if covs is None:
            covs_use = np.zeros((N, 2, 2), dtype=np.float32)
            covs_use[:, 0, 0] = stds[:, 0] ** 2
            covs_use[:, 1, 1] = stds[:, 1] ** 2
        else:
            covs_use = np.asarray(covs, dtype=np.float32)

        scores = np.full((N,), -np.inf, dtype=np.float32)
        best_w = np.zeros((N,), dtype=np.int32)

        # loop over 22 directions
        for wi in range(self.W.shape[0]):
            w = self.W[wi]  # (2,)

            # denom(i)=sqrt(w^T Σ_i w)
            denom2 = np.einsum("iab,a,b->i", covs_use, w, w)
            denom = np.sqrt(np.clip(denom2, self.eps, None))

            # v(i)=Σ_i w
            v = np.einsum("iab,b->ia", covs_use, w)

            # boundary point in direction w
            pts = means + self.k * (v / denom[:, None])  # (N,2)

            # project boundary point onto w, subtract baseline penalty(w)
            alpha = (pts @ w) - penalties[wi]  # (N,)

            mask = alpha > scores
            scores[mask] = alpha[mask]
            best_w[mask] = wi

        self.last_w_idx_max = best_w
        return scores
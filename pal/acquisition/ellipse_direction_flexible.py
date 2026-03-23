from __future__ import annotations
from typing import Tuple, Optional
import time
import logging

import numpy as np

from ..pareto import (
    pareto_front_2d,
    pareto_front_max_3d_fast,
    pareto_skyline,
)
from .base import AcquisitionFunction

logger = logging.getLogger(__name__)


def circle_directions(n: int) -> np.ndarray:
    """Deterministic directions on S^1. Returns (n,2)."""
    if n <= 1:
        return np.array([[1.0, 0.0]], dtype=np.float32)

    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)
    W = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    W = W.astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


def fibonacci_sphere_directions(n: int) -> np.ndarray:
    """Deterministic, approximately uniform directions on S^2. Returns (n,3)."""
    if n <= 1:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        z = 1.0 - (2.0 * i) / (n - 1)
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i

        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points.append([x, y, z])

    W = np.asarray(points, dtype=np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


def random_sphere_directions(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Approximately uniform random directions on S^(d-1). Returns (n,d)."""
    if n <= 1:
        out = np.zeros((1, d), dtype=np.float32)
        out[0, 0] = 1.0
        return out

    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, d)).astype(np.float64)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    W /= norms
    return W.astype(np.float32)


def generate_directions(n: int, d: int, seed: int = 0) -> np.ndarray:
    if d == 2:
        return circle_directions(n)
    if d == 3:
        return fibonacci_sphere_directions(n)
    return random_sphere_directions(n, d, seed=seed)


class EllipseDirectionAcquisitionFlexible(AcquisitionFunction):
    """
    ND directional ellipsoid acquisition.

    Score candidate i by:
        max_w [ w^T mu_i + k * sqrt(w^T Sigma_i w) - max_{y in front} w^T y ]

    For 2D uses pareto_front_2d,
    for 3D uses pareto_front_max_3d_fast,
    for 4D+ uses pareto_skyline.
    """

    def __init__(
        self,
        k: float = 2.0,
        n_directions: int = 50,
        use_front_penalty: bool = True,
        clip_negative_hv: bool = True,
        eps: float = 1e-9,
        random_seed: int = 0,
    ):
        self.k = float(k)
        self.eps = float(eps)
        self.n_directions = int(n_directions)
        self.use_front_penalty = bool(use_front_penalty)
        self.clip_negative_hv = bool(clip_negative_hv)
        self.random_seed = int(random_seed)

        self._W_cache: dict[int, np.ndarray] = {}
        self.last_w_idx_max: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"EllipseDirections(k={self.k}, W={self.n_directions})"

    @property
    def needs_full_cov(self) -> bool:
        return True

    def _get_directions(self, n_obj: int) -> np.ndarray:
        t0 = time.perf_counter()

        if n_obj not in self._W_cache:
            self._W_cache[n_obj] = generate_directions(
                self.n_directions,
                n_obj,
                seed=self.random_seed,
            )
            logger.info(
                "Ellipse.score: direction generation took %.6fs | n_obj=%d n_directions=%d cache_hit=False",
                time.perf_counter() - t0,
                n_obj,
                self.n_directions,
            )
        else:
            logger.info(
                "Ellipse.score: direction fetch took %.6fs | n_obj=%d n_directions=%d cache_hit=True",
                time.perf_counter() - t0,
                n_obj,
                self.n_directions,
            )

        return self._W_cache[n_obj]

    def _get_front(self, current_labels: np.ndarray, n_obj: int) -> np.ndarray:
        t0 = time.perf_counter()

        labels = np.asarray(current_labels, dtype=np.float32)

        if labels.ndim != 2 or labels.shape[1] != n_obj:
            raise ValueError(
                f"current_labels must have shape (M,{n_obj}), got {labels.shape}"
            )

        if labels.shape[0] == 0:
            logger.info(
                "Ellipse.score: front construction took %.6fs | n_obj=%d front_size=0",
                time.perf_counter() - t0,
                n_obj,
            )
            return labels

        if n_obj == 2:
            front = pareto_front_2d(labels)
        elif n_obj == 3:
            front = pareto_front_max_3d_fast(labels)
        else:
            front = pareto_skyline(labels)

        logger.info(
            "Ellipse.score: front construction took %.6fs | n_obj=%d labels=%d front_size=%d",
            time.perf_counter() - t0,
            n_obj,
            labels.shape[0],
            front.shape[0],
        )
        return front

    def score(
        self,
        means: np.ndarray,                 # (N,d)
        stds: np.ndarray,                  # (N,d)
        current_labels: np.ndarray,        # (M,d)
        ref_point: Tuple[float, ...],      # unused, kept for API compatibility
        covs: np.ndarray | None = None,    # (N,d,d)
        **kwargs,
    ) -> np.ndarray:
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        logger.info(
            "Ellipse.score: input cast took %.6fs | means_shape=%s stds_shape=%s",
            time.perf_counter() - t0,
            means.shape,
            stds.shape,
        )

        if means.ndim != 2:
            raise ValueError(f"means must have shape (N,d), got {means.shape}")
        if stds.shape != means.shape:
            raise ValueError(f"stds must have shape {means.shape}, got {stds.shape}")

        N, n_obj = means.shape

        if n_obj < 2:
            raise ValueError(f"{self.name} requires at least 2 objectives, got {n_obj}")

        t0 = time.perf_counter()
        W = self._get_directions(n_obj)            # (K,d)
        logger.info(
            "Ellipse.score: directions ready in %.6fs | W_shape=%s",
            time.perf_counter() - t0,
            W.shape,
        )

        t0 = time.perf_counter()
        front = self._get_front(current_labels, n_obj)
        logger.info(
            "Ellipse.score: front ready in %.6fs | front_shape=%s",
            time.perf_counter() - t0,
            front.shape,
        )

        t0 = time.perf_counter()
        if self.use_front_penalty and front.shape[0] > 0:
            penalties = (front @ W.T).max(axis=0).astype(np.float32)   # (K,)
        else:
            penalties = np.zeros((W.shape[0],), dtype=np.float32)
        logger.info(
            "Ellipse.score: penalty computation took %.6fs | use_front_penalty=%s penalty_shape=%s",
            time.perf_counter() - t0,
            self.use_front_penalty,
            penalties.shape,
        )

        t0 = time.perf_counter()
        if covs is None:
            covs_use = np.zeros((N, n_obj, n_obj), dtype=np.float32)
            idx = np.arange(n_obj)
            covs_use[:, idx, idx] = stds ** 2
            logger.info(
                "Ellipse.score: covariance fallback from stds took %.6fs | covs_shape=%s",
                time.perf_counter() - t0,
                covs_use.shape,
            )
        else:
            covs_use = np.asarray(covs, dtype=np.float32)
            if covs_use.shape != (N, n_obj, n_obj):
                raise ValueError(
                    f"covs must have shape {(N, n_obj, n_obj)}, got {covs_use.shape}"
                )
            logger.info(
                "Ellipse.score: covariance input cast took %.6fs | covs_shape=%s",
                time.perf_counter() - t0,
                covs_use.shape,
            )

        t0 = time.perf_counter()
        covs_use = 0.5 * (covs_use + np.swapaxes(covs_use, 1, 2))
        logger.info(
            "Ellipse.score: covariance symmetrization took %.6fs",
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        v = np.einsum("nij,wj->nwi", covs_use, W)
        logger.info(
            "Ellipse.score: einsum v=Sigma*w took %.6fs | v_shape=%s",
            time.perf_counter() - t0,
            v.shape,
        )

        t0 = time.perf_counter()
        denom2 = np.einsum("nwi,wi->nw", v, W)
        denom = np.sqrt(np.clip(denom2, self.eps, None)).astype(np.float32)
        logger.info(
            "Ellipse.score: denom computation took %.6fs | denom_shape=%s",
            time.perf_counter() - t0,
            denom.shape,
        )

        t0 = time.perf_counter()
        pts = means[:, None, :] + self.k * (v / denom[:, :, None])
        logger.info(
            "Ellipse.score: optimistic ellipsoid boundary points took %.6fs | pts_shape=%s",
            time.perf_counter() - t0,
            pts.shape,
        )

        t0 = time.perf_counter()
        proj = np.einsum("nwi,wi->nw", pts, W)
        alpha = proj - penalties[None, :]
        logger.info(
            "Ellipse.score: projection and alpha took %.6fs | alpha_shape=%s",
            time.perf_counter() - t0,
            alpha.shape,
        )

        t0 = time.perf_counter()
        best_w = np.argmax(alpha, axis=1).astype(np.int32)
        scores = alpha[np.arange(N), best_w].astype(np.float32)
        self.last_w_idx_max = best_w
        logger.info(
            "Ellipse.score: argmax and score gather took %.6fs",
            time.perf_counter() - t0,
        )

        if self.clip_negative_hv:
            t0 = time.perf_counter()
            np.maximum(scores, 0.0, out=scores)
            logger.info(
                "Ellipse.score: clip_negative_hv took %.6fs",
                time.perf_counter() - t0,
            )

        logger.info(
            "Ellipse.score: total took %.6fs | N=%d n_obj=%d n_directions=%d",
            time.perf_counter() - t_total,
            N,
            n_obj,
            self.n_directions,
        )

        return scores
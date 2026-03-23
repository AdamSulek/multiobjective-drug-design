"""Fast vectorized ND ellipse acquisition — no Python loops over directions."""

from __future__ import annotations
from typing import Tuple

import numpy as np

from ..pareto import (
    pareto_front_2D,
    pareto_front_max_3d_fast,
    pareto_skyline,
    batch_delta_hv_2d,
    batch_delta_hv_3d,
    hypervolume_nd,
)

from .base import AcquisitionFunction


# ---------------------------------------------------------
# Direction generators
# ---------------------------------------------------------

def circle_directions(n: int) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    return dirs.astype(np.float32)


def fibonacci_sphere_directions(n: int) -> np.ndarray:
    """Deterministic directions on S²."""
    if n <= 1:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        z = 1.0 - (2.0 * i) / (n - 1)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        points.append([x, y, z])

    W = np.asarray(points, dtype=np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W


def random_sphere_directions(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Random directions on S^(d-1)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    return W.astype(np.float32)


def generate_directions(n: int, d: int) -> np.ndarray:
    if d == 2:
        return circle_directions(n)
    if d == 3:
        return fibonacci_sphere_directions(n)
    return random_sphere_directions(n, d)


# ---------------------------------------------------------
# Acquisition
# ---------------------------------------------------------

class FastEllipseAcquisitionFlexible(AcquisitionFunction):
    """
    Fully vectorized ND ellipse acquisition.

    Scores each candidate by the best HV improvement on its
    k-sigma confidence ellipsoid boundary.
    """

    def __init__(self, k: float = 2.0, n_directions: int = 8, clip_negative_hv: bool = True):
        self.k = float(k)
        self.n_directions = int(n_directions)
        self.clip_negative_hv = bool(clip_negative_hv)

        self._dirs_cache: dict[int, np.ndarray] = {}

    @property
    def name(self) -> str:
        return f"FastEllipse(k={self.k}, D={self.n_directions})"

    @property
    def needs_full_cov(self) -> bool:
        return True

    def _get_dirs(self, d: int) -> np.ndarray:
        if d not in self._dirs_cache:
            self._dirs_cache[d] = generate_directions(self.n_directions, d)
        return self._dirs_cache[d]

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, ...],
        covs: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:

        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)

        N, n_obj = means.shape

        if n_obj < 2:
            raise ValueError("Need at least 2 objectives")

        # -------------------------------------------------
        # Pareto front
        # -------------------------------------------------

        if n_obj == 2:
            labels = np.asarray(current_labels, dtype=np.float32)
            front = labels[pareto_front_2D(labels)]
        elif n_obj == 3:
            front = pareto_front_max_3d_fast(np.asarray(current_labels, dtype=np.float32))
        else:
            labels = np.asarray(current_labels, dtype=np.float32)
            front = labels[pareto_skyline(labels)]

        # -------------------------------------------------
        # Directions
        # -------------------------------------------------

        dirs = self._get_dirs(n_obj)
        D = dirs.shape[0]

        # -------------------------------------------------
        # No covariance → just evaluate means
        # -------------------------------------------------

        if covs is None:

            if n_obj == 2:
                scores = batch_delta_hv_2d(
                    front,
                    means,
                    ref_point,
                    compute_negative=not self.clip_negative_hv,
                )

            elif n_obj == 3:
                scores = batch_delta_hv_3d(
                    front,
                    means,
                    ref_point,
                    compute_negative=not self.clip_negative_hv,
                )

            else:
                base_hv = hypervolume_nd(front, ref_point)

                scores = np.zeros(N, dtype=np.float32)

                for i in range(N):
                    union = np.vstack((front, means[i]))
                    new_front = union[pareto_skyline(union)]
                    hv = hypervolume_nd(new_front, ref_point)
                    scores[i] = hv - base_hv

            scores = scores.astype(np.float32)

            if self.clip_negative_hv:
                np.maximum(scores, 0.0, out=scores)

            return scores

        # -------------------------------------------------
        # Batch Cholesky
        # -------------------------------------------------

        cov_arr = np.asarray(covs, dtype=np.float64)
        cov_arr = cov_arr + 1e-9 * np.eye(n_obj)[None, :, :]

        try:
            L = np.linalg.cholesky(cov_arr)

        except np.linalg.LinAlgError:

            L = np.zeros((N, n_obj, n_obj), dtype=np.float64)

            for i in range(N):
                try:
                    L[i] = np.linalg.cholesky(cov_arr[i])
                except np.linalg.LinAlgError:
                    L[i] = np.diag(stds[i].astype(np.float64))

        # -------------------------------------------------
        # Ellipsoid points
        # -------------------------------------------------

        offsets = self.k * np.einsum("nij,dj->ndi", L, dirs.astype(np.float64))
        points = means[:, None, :] + offsets.astype(np.float32)

        # -------------------------------------------------
        # Dominated-by-mean filter
        # -------------------------------------------------

        dominated = np.all(points <= means[:, None, :], axis=2)
        keep = ~dominated

        cand_idx = np.broadcast_to(np.arange(N)[:, None], (N, D))

        flat_points = points[keep]
        flat_cand = cand_idx[keep]

        if flat_points.shape[0] == 0:
            return np.zeros(N, dtype=np.float32)

        # -------------------------------------------------
        # Delta HV
        # -------------------------------------------------

        if n_obj == 2:

            flat_dhv = batch_delta_hv_2d(
                front,
                flat_points,
                ref_point,
                compute_negative=not self.clip_negative_hv,
            )

        elif n_obj == 3:

            flat_dhv = batch_delta_hv_3d(
                front,
                flat_points,
                ref_point,
                compute_negative=not self.clip_negative_hv,
            )

        else:

            base_hv = hypervolume_nd(front, ref_point)

            flat_dhv = np.zeros(flat_points.shape[0], dtype=np.float32)

            for i, p in enumerate(flat_points):
                union = np.vstack((front, p))
                new_front = union[pareto_skyline(union)]
                hv = hypervolume_nd(new_front, ref_point)
                flat_dhv[i] = hv - base_hv

        # -------------------------------------------------
        # Max per candidate
        # -------------------------------------------------

        scores = np.full(N, -np.inf, dtype=np.float64)

        np.maximum.at(scores, flat_cand, flat_dhv.astype(np.float64))

        scores = scores.astype(np.float32)

        if self.clip_negative_hv:
            np.maximum(scores, 0.0, out=scores)

        return scores
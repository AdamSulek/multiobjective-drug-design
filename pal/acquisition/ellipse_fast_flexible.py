"""Fast vectorized ND ellipse acquisition — no Python loops over directions."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np

from ..log_prefs import pal_log_diag, pal_log_timer
from ..pareto import (
    pareto_front_2D,
    pareto_front_max_3d_fast,
    pareto_skyline,
    batch_delta_hv_2d,
    batch_delta_hv_3d,
    hypervolume_nd,
)

from .base import AcquisitionFunction

logger = logging.getLogger(__name__)

# --- ND (>3) multiprocessing: module-level for spawn/fork pickling ---
_ELL_ND_F: np.ndarray | None = None
_ELL_ND_REF: tuple | None = None
_ELL_ND_BH: float | None = None


def _ellipse_nd_mp_init(front: np.ndarray, ref_point: tuple, base_hv: float) -> None:
    global _ELL_ND_F, _ELL_ND_REF, _ELL_ND_BH
    _ELL_ND_F = np.asarray(front, dtype=np.float32)
    _ELL_ND_REF = ref_point
    _ELL_ND_BH = float(base_hv)


def _ellipse_nd_mp_scores_chunk(chunk: np.ndarray) -> np.ndarray:
    """Child process: score one chunk of candidate rows (must stay picklable)."""
    from ..pareto import hypervolume_nd, pareto_skyline

    f = _ELL_ND_F
    ref = _ELL_ND_REF
    bh = _ELL_ND_BH
    if f is None or ref is None or bh is None:
        raise RuntimeError("FastEllipse ND worker not initialized")
    chunk = np.asarray(chunk, dtype=np.float32)
    m = int(chunk.shape[0])
    out = np.zeros(m, dtype=np.float32)
    for i in range(m):
        union = np.vstack((f, chunk[i]))
        new_front = union[pareto_skyline(union)]
        out[i] = float(hypervolume_nd(new_front, ref)) - bh
    return out


def _ellipse_nd_scores_sequential_with_base(
    front: np.ndarray,
    points: np.ndarray,
    ref_point: tuple,
    base_hv: float,
) -> np.ndarray:
    m = int(points.shape[0])
    out = np.zeros(m, dtype=np.float32)
    f = np.asarray(front, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    for i in range(m):
        union = np.vstack((f, pts[i]))
        new_front = union[pareto_skyline(union)]
        out[i] = float(hypervolume_nd(new_front, ref_point)) - float(base_hv)
    return out


def _ellipse_nd_scores_parallel_or_seq(
    front: np.ndarray,
    points: np.ndarray,
    ref_point: tuple,
    *,
    context: str,
) -> np.ndarray:
    """ND hypervolume scoring: optional ProcessPoolExecutor (env FAST_ELLIPSE_ND_*)."""
    M = int(points.shape[0])
    if M == 0:
        return np.zeros(0, dtype=np.float32)
    front = np.asarray(front, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    base_hv = float(hypervolume_nd(front, ref_point))
    pe = os.environ.get("FAST_ELLIPSE_ND_PARALLEL", "0").strip().lower() in ("1", "true", "yes")
    nw = max(1, int(os.environ.get("FAST_ELLIPSE_ND_WORKERS", str(min(32, (os.cpu_count() or 1))))))
    min_chunk = max(1, int(os.environ.get("FAST_ELLIPSE_ND_MIN_CHUNK", "64")))
    if not pe or nw <= 1 or M < min_chunk * 2:
        t0 = time.perf_counter()
        out = _ellipse_nd_scores_sequential_with_base(front, points, ref_point, base_hv)
        if pal_log_diag():
            logger.info(
                "[FAST_ELLIPSE] ND %s sequential rows=%d wall_s=%.4f backend=main",
                context,
                M,
                time.perf_counter() - t0,
            )
        return out
    chunk_sz = max(min_chunk, (M + nw - 1) // nw)
    chunks = [points[i : i + chunk_sz] for i in range(0, M, chunk_sz)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=nw,
        initializer=_ellipse_nd_mp_init,
        initargs=(front, ref_point, base_hv),
    ) as ex:
        parts = list(ex.map(_ellipse_nd_mp_scores_chunk, chunks))
    out = np.concatenate(parts, axis=0)
    if pal_log_diag():
        logger.info(
            "[FAST_ELLIPSE] ND %s multiprocessing workers=%d n_chunks=%d rows=%d wall_s=%.4f "
            "backend=ProcessPoolExecutor",
            context,
            nw,
            len(chunks),
            M,
            time.perf_counter() - t0,
        )
    return out


def _log_fast_ellipse_3d_delta_hv_config() -> None:
    if not pal_log_diag():
        return
    logger.info(
        "[FAST_ELLIPSE] 3D scoring uses batch_delta_hv_3d | "
        "BATCH_DELTA_HV_3D_PARALLEL=%s WORKERS=%s USE_SHM=%s MIN_PER_CHUNK=%s "
        "(same knobs as UCB; set env before launch)",
        os.environ.get("BATCH_DELTA_HV_3D_PARALLEL", "1"),
        os.environ.get("BATCH_DELTA_HV_3D_WORKERS", "20"),
        os.environ.get("BATCH_DELTA_HV_3D_USE_SHM", "1"),
        os.environ.get("BATCH_DELTA_HV_3D_MIN_PER_CHUNK", "50"),
    )


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

        _t_score = time.perf_counter()

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

        if n_obj == 3:
            _log_fast_ellipse_3d_delta_hv_config()

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
                scores = _ellipse_nd_scores_parallel_or_seq(
                    front, means, ref_point, context="means"
                )

            scores = scores.astype(np.float32)

            if self.clip_negative_hv:
                np.maximum(scores, 0.0, out=scores)

            if pal_log_timer():
                logger.info(
                    "[TIMER] FastEllipse.score n_cand=%d n_obj=%d covs=False wall_s=%.4f",
                    N,
                    n_obj,
                    time.perf_counter() - _t_score,
                )
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
            if pal_log_timer():
                logger.info(
                    "[TIMER] FastEllipse.score n_cand=%d n_obj=%d covs=True wall_s=%.4f (empty flat)",
                    N,
                    n_obj,
                    time.perf_counter() - _t_score,
                )
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
            flat_dhv = _ellipse_nd_scores_parallel_or_seq(
                front, flat_points, ref_point, context="ellipse_flat"
            )

        # -------------------------------------------------
        # Max per candidate
        # -------------------------------------------------

        scores = np.full(N, -np.inf, dtype=np.float64)

        np.maximum.at(scores, flat_cand, flat_dhv.astype(np.float64))

        scores = scores.astype(np.float32)

        if self.clip_negative_hv:
            np.maximum(scores, 0.0, out=scores)

        if pal_log_timer():
            logger.info(
                "[TIMER] FastEllipse.score n_cand=%d n_obj=%d covs=True wall_s=%.4f",
                N,
                n_obj,
                time.perf_counter() - _t_score,
            )
        return scores
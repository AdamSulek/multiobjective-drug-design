"""UCB exploration: optimistic HV improvement (mean + k*std)."""

from typing import Tuple
import time
import logging

import numpy as np

from .base import AcquisitionFunction
from ..pareto import (
    pareto_front_2D,
    pareto_front_max_3d_fast,
    pareto_skyline,
    batch_delta_hv_2d,
    batch_delta_hv_3d,
    hypervolume_nd,
)

logger = logging.getLogger(__name__)


def _is_dominated_by_front(point: np.ndarray, front: np.ndarray) -> bool:
    if front.shape[0] == 0:
        return False
    ge = np.all(front >= point, axis=1)
    gt = np.any(front > point, axis=1)
    return bool(np.any(ge & gt))


def _filter_above_ref(points: np.ndarray, ref_point: Tuple[float, ...]) -> np.ndarray:
    ref_arr = np.asarray(ref_point, dtype=np.float64).reshape(1, -1)
    return np.all(points > ref_arr, axis=1)


def _front_indices(points: np.ndarray) -> np.ndarray:
    n_obj = int(points.shape[1])
    if n_obj == 2:
        return pareto_front_2D(points)
    if n_obj == 3:
        return pareto_front_max_3d_fast(points)
    if n_obj in (4, 5):
        return pareto_skyline(points)
    raise ValueError(f"Unsupported number of objectives: {n_obj}")


def _get_front(points: np.ndarray) -> np.ndarray:
    front_or_idx = _front_indices(points)
    arr = np.asarray(front_or_idx)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        return points[arr]
    return np.asarray(front_or_idx, dtype=np.float64)


class UCBExplorationAcquisitionFlexible(AcquisitionFunction):
    """Upper Confidence Bound: score = delta-HV of optimistic point."""

    def __init__(
        self,
        k_ucb: float = 2.0,
        max_exact_candidates: int | None = None,
        clip_negative_hv: bool = True,
    ):
        self.k_ucb = float(k_ucb)
        self.clip_negative_hv = bool(clip_negative_hv)

        if max_exact_candidates is None and self.k_ucb >= 2.0:
            max_exact_candidates = 50_000

        self.max_exact_candidates = (
            int(max_exact_candidates)
            if max_exact_candidates is not None and int(max_exact_candidates) > 0
            else None
        )

    @property
    def name(self) -> str:
        return f"UCB(k={self.k_ucb})"

    @property
    def needs_uncertainty(self) -> bool:
        return self.k_ucb > 0.0

    def score(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        current_labels: np.ndarray,
        ref_point: Tuple[float, ...],
        covs: np.ndarray | None = None,
        *,
        pareto_front: np.ndarray | None = None,
        pareto_dom_index=None,
        pareto_hv: float | None = None,
    ) -> np.ndarray:
        t_total = time.perf_counter()

        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        current_labels = np.asarray(current_labels, dtype=np.float64)

        n_obj = int(means.shape[1])

        t0 = time.perf_counter()
        optimistic = means + self.k_ucb * stds
        logger.info(
            "UCB.score: optimistic construction took %.6fs | n_candidates=%d n_obj=%d",
            time.perf_counter() - t0,
            optimistic.shape[0],
            n_obj,
        )

        scores = np.zeros((optimistic.shape[0],), dtype=np.float32)

        if n_obj == 2:
            t0 = time.perf_counter()
            front = (
                np.asarray(pareto_front, dtype=np.float64)
                if pareto_front is not None
                else current_labels[pareto_front_2D(current_labels)]
            )
            logger.info(
                "UCB.score: 2D front construction took %.6fs | front_size=%d",
                time.perf_counter() - t0,
                len(front),
            )

            t0 = time.perf_counter()
            scores = batch_delta_hv_2d(
                front,
                optimistic,
                ref_point,
                compute_negative=not self.clip_negative_hv,
            ).astype(np.float32)
            logger.info(
                "UCB.score: batch_delta_hv_2d took %.6fs | n_candidates=%d front_size=%d",
                time.perf_counter() - t0,
                len(optimistic),
                len(front),
            )

            if self.clip_negative_hv:
                t0 = time.perf_counter()
                np.maximum(scores, 0.0, out=scores)
                logger.info(
                    "UCB.score: 2D clip_negative_hv took %.6fs",
                    time.perf_counter() - t0,
                )

            logger.info("UCB.score: total 2D took %.6fs", time.perf_counter() - t_total)
            return scores

        if n_obj == 3:
            t0 = time.perf_counter()
            front = (
                np.asarray(pareto_front, dtype=np.float64)
                if pareto_front is not None
                else pareto_front_max_3d_fast(current_labels)
            )
            logger.info(
                "UCB.score: 3D front construction took %.6fs | front_size=%d",
                time.perf_counter() - t0,
                len(front),
            )

            t0 = time.perf_counter()
            scores = batch_delta_hv_3d(
                front,
                optimistic,
                ref_point,
                compute_negative=not self.clip_negative_hv,
            ).astype(np.float32)
            logger.info(
                "UCB.score: batch_delta_hv_3d took %.6fs | n_candidates=%d front_size=%d",
                time.perf_counter() - t0,
                len(optimistic),
                len(front),
            )

            if self.clip_negative_hv:
                t0 = time.perf_counter()
                np.maximum(scores, 0.0, out=scores)
                logger.info(
                    "UCB.score: 3D clip_negative_hv took %.6fs",
                    time.perf_counter() - t0,
                )

            logger.info("UCB.score: total 3D took %.6fs", time.perf_counter() - t_total)
            return scores

        if n_obj not in (4, 5):
            raise ValueError(f"{self.name} supports only 2D/3D/4D/5D, got {n_obj} objectives")

        t0 = time.perf_counter()
        front = (
            np.asarray(pareto_front, dtype=np.float64)
            if pareto_front is not None
            else current_labels[pareto_skyline(current_labels)]
        )
        logger.info(
            "UCB.score: %dD front construction took %.6fs | front_size=%d",
            n_obj,
            time.perf_counter() - t0,
            len(front),
        )

        t0 = time.perf_counter()
        hv_front = float(hypervolume_nd(front, ref_point)) if pareto_hv is None else float(pareto_hv)
        logger.info(
            "UCB.score: %dD base hypervolume took %.6fs | front_size=%d hv_front=%.6g",
            n_obj,
            time.perf_counter() - t0,
            len(front),
            hv_front,
        )

        t0 = time.perf_counter()
        above_ref = _filter_above_ref(optimistic, ref_point)
        cand_idx = np.where(above_ref)[0]
        logger.info(
            "UCB.score: %dD filter_above_ref took %.6fs | kept=%d/%d",
            n_obj,
            time.perf_counter() - t0,
            cand_idx.size,
            optimistic.shape[0],
        )

        if self.clip_negative_hv:
            t0 = time.perf_counter()
            keep = []
            for i in cand_idx:
                if not _is_dominated_by_front(optimistic[i], front):
                    keep.append(i)
            cand_idx = np.asarray(keep, dtype=np.int64)
            logger.info(
                "UCB.score: %dD dominated-filter took %.6fs | kept=%d",
                n_obj,
                time.perf_counter() - t0,
                cand_idx.size,
            )

        if self.max_exact_candidates is not None and cand_idx.size > self.max_exact_candidates:
            t0 = time.perf_counter()
            ref_arr = np.asarray(ref_point, dtype=np.float64).reshape(1, n_obj)
            slack = np.clip(optimistic[cand_idx] - ref_arr, a_min=0.0, a_max=None)
            cheap = np.prod(slack, axis=1)
            top_local = np.argpartition(cheap, -self.max_exact_candidates)[-self.max_exact_candidates:]
            cand_idx = cand_idx[top_local]
            logger.info(
                "UCB.score: %dD max_exact_candidates reduction took %.6fs | reduced_to=%d",
                n_obj,
                time.perf_counter() - t0,
                cand_idx.size,
            )

        t0 = time.perf_counter()
        for i in cand_idx:
            union = np.vstack([front, optimistic[i].reshape(1, -1)])
            union_front = _get_front(union)
            hv_new = float(hypervolume_nd(union_front, ref_point))
            scores[i] = hv_new - hv_front
        logger.info(
            "UCB.score: %dD exact candidate loop took %.6fs | processed=%d",
            n_obj,
            time.perf_counter() - t0,
            cand_idx.size,
        )

        if self.clip_negative_hv:
            t0 = time.perf_counter()
            np.maximum(scores, 0.0, out=scores)
            logger.info(
                "UCB.score: %dD final clip_negative_hv took %.6fs",
                n_obj,
                time.perf_counter() - t0,
            )

        logger.info("UCB.score: total %dD took %.6fs", n_obj, time.perf_counter() - t_total)
        return scores
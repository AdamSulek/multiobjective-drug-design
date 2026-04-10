"""
Self-contained Pareto front and hypervolume utilities (MAXIMIZATION).

Supports:
- 2D exact HV + fast batch delta-HV (staircase formula)
- 3D exact HV (slicing over x + 2D HV on yz)
- 3D batch delta-HV (exact, but per-candidate HV recompute)
  plus negative "gap volume" score for dominated points (Option B).

Conventions:
- Maximization in every objective (bigger = better).
- Reference point `ref` must be WORSE than all interesting points (in all dims)
  in the same transformed space (after any column negations).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Parallel batch_delta_hv_3d: max worker processes (coarse chunks, not one task per row).
_BATCH_DELTA_HV_3D_MAX_WORKERS = max(
    1, int(os.environ.get("BATCH_DELTA_HV_3D_WORKERS", "20"))
)
# Minimum candidates per chunk target; fewer chunks => coarser grain.
_BATCH_DELTA_HV_3D_MIN_PER_CHUNK = max(
    1, int(os.environ.get("BATCH_DELTA_HV_3D_MIN_PER_CHUNK", "50"))
)

# -----------------------------
# Pareto (MAX): keep nondominated
# -----------------------------
def pareto_front_max(Y: np.ndarray) -> np.ndarray:
    """
    Return nondominated points for MAX objectives.
    O(n^2) worst-case, but n here should be Pareto-size (small vs pool).
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return Y.reshape(0, Y.shape[1])

    n, m = Y.shape
    keep = np.ones(n, dtype=bool)

    # Simple dominance check
    for i in range(n):
        if not keep[i]:
            continue
        # If there exists j that dominates i => drop i
        # j dominates i if all >= and any >
        ge = (Y >= Y[i]).all(axis=1)
        gt = (Y >  Y[i]).any(axis=1)
        dom_i = ge & gt
        dom_i[i] = False
        if dom_i.any():
            keep[i] = False
            continue

        # i dominates j => drop j
        ge2 = (Y[i] >= Y).all(axis=1)
        gt2 = (Y[i] >  Y).any(axis=1)
        dom_j = ge2 & gt2
        dom_j[i] = False
        keep[dom_j] = False

    return Y[keep]


# -----------------------------
# 2D hypervolume for MAX
# -----------------------------
def hypervolume_2d_max(P: np.ndarray, ref: tuple[float, float]) -> float:
    """
    HV for MAX in 2D relative to ref (ref is "worse": <= all points).
    Uses standard staircase: sort by y desc, keep increasing z.
    """
    P = np.asarray(P, dtype=float)
    if P.size == 0:
        return 0.0
    ry, rz = float(ref[0]), float(ref[1])

    # Keep only points above ref (defensive)
    P = P[(P[:, 0] >= ry) & (P[:, 1] >= rz)]
    if P.size == 0:
        return 0.0

    # Sort by y descending, then z descending
    idx = np.lexsort((-P[:, 1], -P[:, 0]))
    P = P[idx]

    # Build 2D front in (y,z): as y goes down, z must go strictly up to contribute
    z_cummax = np.maximum.accumulate(P[:, 1])
    # Keep points that increase z (first always kept)
    inc = np.empty(P.shape[0], dtype=bool)
    inc[0] = True
    inc[1:] = z_cummax[1:] > z_cummax[:-1]
    F = P[inc]
    if F.size == 0:
        return 0.0

    # Now F has y strictly decreasing (or non-increasing with ties),
    # and z strictly increasing.
    y = F[:, 0]
    z = F[:, 1]

    # Area = sum over steps: (y_i - y_{i+1}) * (z_i - rz), with last y_{k+1}=ry
    y_next = np.r_[y[1:], ry]
    dy = y - y_next
    dz = z - rz

    # Clip for safety
    dy = np.maximum(dy, 0.0)
    dz = np.maximum(dz, 0.0)

    return float(np.sum(dy * dz))


# -----------------------------
# 3D fast HV for MAX (sweep x)
# -----------------------------
def hypervolume_3d_max_fast(Y: np.ndarray, ref: tuple[float, float, float]) -> float:
    """
    Fast-ish HV 3D for MAX using:
      1) filter to Pareto front (MAX)
      2) sort by x desc
      3) sweep unique x slabs, maintain yz front and compute 2D HV in yz

    Complexity ~ O(p * log p + u * cost(front_update)), where p = Pareto size,
    u = #unique x on Pareto. Front update here is vectorized + small loops on removed points.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return 0.0
    rx, ry, rz = map(float, ref)

    # Keep only points above ref (defensive)
    Y = Y[(Y[:, 0] >= rx) & (Y[:, 1] >= ry) & (Y[:, 2] >= rz)]
    if Y.size == 0:
        return 0.0

    # Pareto reduce in 3D (MAX)
    P = pareto_front_max(Y)
    if P.size == 0:
        return 0.0

    # Sort by x descending; group by x
    order = np.argsort(-P[:, 0], kind="mergesort")
    P = P[order]

    x_vals = P[:, 0]
    # unique x in descending order + start indices
    ux, start = np.unique(x_vals, return_index=True)
    # ux returned is ascending by default; we want descending
    # since P is sorted desc, start aligns with desc order already only if we take unique on reversed.
    # Fix robustly:
    ux = np.unique(x_vals)              # ascending
    ux = ux[::-1]                       # descending
    # For each unique x, find slice in P
    # We'll use boolean mask each step based on x threshold (vectorized)
    # but still only u steps where u = unique x on Pareto.

    hv = 0.0
    active_yz = np.empty((0, 2), dtype=float)

    # Sweep slabs: from current x to next x (or ref_x)
    for i, x in enumerate(ux):
        x_next = ux[i + 1] if i + 1 < len(ux) else rx
        dx = x - x_next
        if dx <= 0:
            continue

        # Add all points with this x into active set (yz)
        # Find points with x == current
        pts = P[P[:, 0] == x][:, 1:3]  # (y,z)

        if pts.size:
            active_yz = np.vstack([active_yz, pts])

            # Reduce active_yz to 2D Pareto front in (y,z) for MAX
            # Efficient staircase method:
            # sort by y desc, keep increasing z
            idx = np.lexsort((-active_yz[:, 1], -active_yz[:, 0]))
            S = active_yz[idx]
            z_cummax = np.maximum.accumulate(S[:, 1])
            inc = np.empty(S.shape[0], dtype=bool)
            inc[0] = True
            inc[1:] = z_cummax[1:] > z_cummax[:-1]
            active_yz = S[inc]

        # Area of union in yz for this slab
        area = hypervolume_2d_max(active_yz, (ry, rz))
        hv += dx * area

    return float(hv)


# -----------------------------
# Simple tracker for AL loop
# -----------------------------
class HV3DFastTracker:
    """
    Keep a growing set of labeled points and compute 3D MAX HV quickly
    by recomputing on Pareto only.
    This is already a huge win vs recomputing on all labeled if labeled gets big.
    """
    def __init__(self, ref_point: tuple[float, float, float]):
        self.ref = tuple(map(float, ref_point))
        self._points = np.empty((0, 3), dtype=float)

    def reset_and_compute(self, Y_labeled: np.ndarray) -> float:
        self._points = np.asarray(Y_labeled, dtype=float).copy()
        return hypervolume_3d_max_fast(self._points, self.ref)

    def add_points(self, new_Y: np.ndarray) -> float:
        new_Y = np.asarray(new_Y, dtype=float)
        if new_Y.size == 0:
            return hypervolume_3d_max_fast(self._points, self.ref)
        self._points = np.vstack([self._points, new_Y])
        return hypervolume_3d_max_fast(self._points, self.ref)
    
# ============================================================
# Generic ND Pareto front (works for 2D/3D/Nd)
# ============================================================

def pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Return the Pareto front for MAXIMIZATION for points with shape (N, d).

    Output is NOT "stair-sorted" (that's only meaningful for 2D).
    Output is the set of nondominated points, order not guaranteed.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.size == 0:
        return np.empty((0, pts.shape[1] if pts.ndim == 2 else 0), dtype=float)

    # drop non-finite
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return np.empty((0, points.shape[1]), dtype=float)

    # Remove duplicates early (helps performance)
    pts = np.unique(pts, axis=0)

    # Nondominated check: for each i, see if any j dominates i.
    # Dominates (max): j dominates i if all(j >= i) and any(j > i).
    N, d = pts.shape
    dominated = np.zeros(N, dtype=bool)

    # For moderate N (<= few thousands) this is fine.
    for i in range(N):
        if dominated[i]:
            continue
        pi = pts[i]
        # candidates that are >= in all dims
        ge_all = np.all(pts >= pi, axis=1)
        # and strictly > in at least one dim
        gt_any = np.any(pts > pi, axis=1)
        dom = ge_all & gt_any
        # ignore self if equal (dom requires strict somewhere, so self won't match)
        if np.any(dom):
            dominated[i] = True

    return pts[~dominated]


# ============================================================
# 2D Pareto front in "stair" order (your original)
# ============================================================

def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Compute the 2-D Pareto front for MAXIMIZATION of both objectives.

    Returns the front sorted by x descending, y strictly increasing
    (the "stair" representation).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)

    # sort by x desc, then y desc
    order = np.lexsort((-pts[:, 1], -pts[:, 0]))
    pts = pts[order]

    front = []
    best_y = -np.inf
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y
    return np.asarray(front, dtype=float)


# Compatibility helper used by older flexible acquisitions:
# returns indices of 2D nondominated points (MAX/MAX).
def pareto_front_2D(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return np.empty((0,), dtype=int)

    idx = np.flatnonzero(finite)
    P = pts[idx]
    n = P.shape[0]
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        ge_all = np.all(P >= P[i], axis=1)
        gt_any = np.any(P > P[i], axis=1)
        if np.any(ge_all & gt_any):
            dominated[i] = True

    return idx[~dominated].astype(int)


# ============================================================
# 2D HV + helpers (your original)
# ============================================================

def hypervolume_2d(points: np.ndarray, ref: Tuple[float, float]) -> float:
    """Compute the 2-D hypervolume indicator (dominated area) for maximization."""
    rx, ry = float(ref[0]), float(ref[1])
    pts = np.asarray(points, dtype=float).reshape(-1, 2)

    mask = (pts[:, 0] > rx) & (pts[:, 1] > ry)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0

    front = pareto_front_2d(pts)
    if len(front) == 0:
        return 0.0

    xs = front[:, 0]  # desc
    ys = front[:, 1]  # inc

    hv = 0.0
    for i in range(len(front) - 1):
        hv += (xs[i] - xs[i + 1]) * (ys[i] - ry)
    hv += (xs[-1] - rx) * (ys[-1] - ry)
    return float(hv)


def build_stair_polygon(front: np.ndarray, ref: Tuple[float, float]) -> np.ndarray:
    """Build the stair-step polygon vertices for plotting the dominated area (2D only)."""
    rx, ry = float(ref[0]), float(ref[1])
    if len(front) == 0:
        return np.array([[rx, ry]], dtype=float)

    f = pareto_front_2d(front)
    if len(f) == 0:
        return np.array([[rx, ry]], dtype=float)

    xs = f[:, 0]  # desc
    ys = f[:, 1]  # inc

    verts = []
    verts.append((xs[0], ry))
    for i in range(len(f)):
        verts.append((xs[i], ys[i]))
        if i + 1 < len(f):
            verts.append((xs[i + 1], ys[i]))
    verts.append((rx, ys[-1]))
    verts.append((rx, ry))
    return np.asarray(verts, dtype=float)


# ============================================================
# 2D batch delta-HV (your original, kept)
# ============================================================

def batch_delta_hv_2d(
    front: np.ndarray,
    candidates: np.ndarray,
    ref: Tuple[float, float],
    *,
    compute_negative: bool = True,
) -> np.ndarray:
    """Vectorized delta-HV for a batch of 2D candidate points.

    Returns:
    - positive exact ΔHV for nondominated candidates
    - if ``compute_negative=True``: negative "gap area" for dominated/behind-ref (Option B)
    - if ``compute_negative=False``: zeros for dominated/behind-ref
    """
    rx, ry = float(ref[0]), float(ref[1])
    cands = np.asarray(candidates, dtype=float).reshape(-1, 2)
    B = len(cands)

    if B == 0:
        return np.empty(0, dtype=float)

    px = cands[:, 0]
    py = cands[:, 1]

    valid = (px > rx) & (py > ry)

    if len(front) == 0 or front.shape[0] == 0:
        delta = np.zeros(B, dtype=float)
        delta[valid] = (px[valid] - rx) * (py[valid] - ry)

        need_dist = ~valid
        if compute_negative and np.any(need_dist):
            dist_v = ry - py
            dist_h = rx - px
            dv_pos = dist_v > 0
            dh_pos = dist_h > 0
            both = dv_pos & dh_pos
            only_v = dv_pos & ~dh_pos
            only_h = dh_pos & ~dv_pos
            area = np.zeros(B, dtype=float)
            area[both] = dist_v[both] * dist_h[both]
            area[only_v] = dist_v[only_v] ** 2
            area[only_h] = dist_h[only_h] ** 2
            delta[need_dist] = -area[need_dist]
        return delta

    front = np.asarray(front, dtype=float)
    M = len(front)
    fx = front[:, 0]  # descending
    fy = front[:, 1]  # ascending

    j = np.searchsorted(-fx, -px, side="right")  # (B,)
    y_below = np.where(j > 0, fy[np.clip(j - 1, 0, M - 1)], ry)
    y_below = np.where(px <= rx, ry, y_below)

    dominated = py <= y_below
    k = np.searchsorted(fy, py, side="left")  # (B,)

    fx_ext = np.append(fx, rx)  # (M+1,)
    strip_areas = (fx - fx_ext[1:]) * (fy - ry)
    prefix_sum = np.empty(M + 1, dtype=float)
    prefix_sum[0] = 0.0
    prefix_sum[1:] = np.cumsum(strip_areas)

    fx_at_k = fx_ext[k]
    fx_at_j = fx_ext[j]

    delta = (
        (px - fx_at_k) * (py - ry)
        - (px - fx_at_j) * (y_below - ry)
        - (prefix_sum[k] - prefix_sum[j])
    )

    non_dom = valid & ~dominated
    delta[~non_dom] = 0.0
    delta = np.maximum(delta, 0.0)

    need_dist = dominated | ~valid
    if compute_negative and np.any(need_dist):
        dist_v = y_below - py
        stair_x = np.where(py <= ry, rx, fx_at_k)
        dist_h = stair_x - px
        dv_pos = dist_v > 0
        dh_pos = dist_h > 0
        both = dv_pos & dh_pos
        only_v = dv_pos & ~dh_pos
        only_h = dh_pos & ~dv_pos
        area = np.zeros(B, dtype=float)
        area[both] = dist_v[both] * dist_h[both]
        area[only_v] = dist_v[only_v] ** 2
        area[only_h] = dist_h[only_h] ** 2
        delta[need_dist] = -area[need_dist]

    return delta


# ============================================================
# 3D Hypervolume (exact) for maximization
# ============================================================

def hypervolume_3d(points: np.ndarray, ref: Tuple[float, float, float]) -> float:
    """
    Exact 3D hypervolume for maximization.

    Uses slicing over x:
      HV = sum_{i} (x_i - x_{i+1}) * HV_2D( {(y,z) of points with x >= x_{i+1}} )

    Notes:
    - Works well for moderate number of points (front ~ 70).
    - Internally filters points that do not dominate the reference.
    """
    rx, ry, rz = float(ref[0]), float(ref[1]), float(ref[2])

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return 0.0

    # Keep only points that dominate ref in all dims
    mask = (pts[:, 0] > rx) & (pts[:, 1] > ry) & (pts[:, 2] > rz)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0

    # Reduce to nondominated set (ND in 3D) to speed up
    pts = pareto_front(pts)
    if len(pts) == 0:
        return 0.0

    # Sort by x descending
    order = np.argsort(-pts[:, 0], kind="mergesort")
    pts = pts[order]
    xs = pts[:, 0]

    # Unique x-levels (descending)
    x_levels = np.unique(xs)[::-1]  # ascending unique then reverse -> descending
    # Make sure descending (if numeric issues)
    x_levels = np.sort(x_levels)[::-1]

    hv = 0.0
    # For each slab between x_levels[i] and x_levels[i+1] (and last to rx)
    # define x_next = next lower x threshold; take points with x >= x_next
    for i in range(len(x_levels)):
        x_hi = x_levels[i]
        x_lo = x_levels[i + 1] if i + 1 < len(x_levels) else rx
        if x_hi <= x_lo:
            continue

        # points with x >= x_hi are at least those at this level; but slab should use set dominating x_lo
        # In standard slicing: use set of points with x >= x_hi (or >= x_lo).
        # Correct: for slab (x_hi -> x_lo), use points with x >= x_hi (since those dominate all x in [x_lo, x_hi])
        slab_pts = pts[pts[:, 0] >= x_hi]
        yz = slab_pts[:, 1:3]  # (K,2)

        hv_yz = hypervolume_2d(yz, (ry, rz))
        hv += (x_hi - x_lo) * hv_yz

    return float(hv)


# ============================================================
# 3D batch delta-HV (exact, with negative "gap volume" for dominated)
# ============================================================

def _gap_volume_3d(front_nd: np.ndarray, p: np.ndarray, ref: Tuple[float, float, float]) -> float:
    """
    "How far under the front" score (Option B) as a volume gap.

    Returns a POSITIVE number representing the smallest axis-aligned gap box
    from p to a dominating front point:
        min_{f dominates p} (f_x - p_x)(f_y - p_y)(f_z - p_z)
    If p is behind ref, returns a gap to ref-corner in the same spirit.

    Caller typically returns -gap as the score (more under -> more negative).
    """
    rx, ry, rz = float(ref[0]), float(ref[1]), float(ref[2])
    px, py, pz = float(p[0]), float(p[1]), float(p[2])

    # behind ref: use "distance volume" to ref corner (kept in volume-ish units)
    if (px <= rx) or (py <= ry) or (pz <= rz):
        dv = max(0.0, rx - px)
        dh = max(0.0, ry - py)
        dz = max(0.0, rz - pz)
        # mimic your 2D behavior: if multiple positive, multiply; else square
        positives = [dv > 0, dh > 0, dz > 0]
        cnt = sum(positives)
        if cnt >= 2:
            return dv * dh * dz
        if cnt == 1:
            # pick the nonzero and square (keep scale comparable)
            if dv > 0:
                return dv * dv
            if dh > 0:
                return dh * dh
            return dz * dz
        return 0.0

    if len(front_nd) == 0:
        return 0.0

    F = front_nd
    dominates = (F[:, 0] >= px) & (F[:, 1] >= py) & (F[:, 2] >= pz) & (
        (F[:, 0] > px) | (F[:, 1] > py) | (F[:, 2] > pz)
    )
    if not np.any(dominates):
        return 0.0

    D = F[dominates]
    gaps = (D[:, 0] - px) * (D[:, 1] - py) * (D[:, 2] - pz)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 0.0
    return float(np.min(gaps))


def _delta_hv_3d_dense_block(
    F_nd: np.ndarray,
    hv_old: float,
    cands: np.ndarray,
    ref: Tuple[float, float, float],
    compute_negative: bool,
) -> np.ndarray:
    """Per-candidate ΔHV for rows of *cands*; same math as the original batch_delta_hv_3d loop."""
    rx, ry, rz = float(ref[0]), float(ref[1]), float(ref[2])
    cands = np.asarray(cands, dtype=float).reshape(-1, 3)
    B = len(cands)
    out = np.zeros(B, dtype=float)
    for i in range(B):
        p = cands[i]
        if not np.isfinite(p).all():
            out[i] = 0.0
            continue

        hv_new = hypervolume_3d(np.vstack([F_nd, p]) if len(F_nd) else np.array([p]), (rx, ry, rz))
        delta = hv_new - hv_old

        if delta > 0.0:
            out[i] = float(delta)
        elif compute_negative:
            gap = _gap_volume_3d(F_nd, p, (rx, ry, rz))
            out[i] = -float(gap)
        else:
            out[i] = 0.0
    return out


def _batch_delta_hv_3d_process_chunk(
    payload: tuple[np.ndarray, float, np.ndarray, tuple[float, float, float], bool],
) -> np.ndarray:
    """Picklable entry point for ProcessPoolExecutor (must be top-level)."""
    F_nd, hv_old, cands_chunk, ref, compute_negative = payload
    return _delta_hv_3d_dense_block(F_nd, hv_old, cands_chunk, ref, compute_negative)


def batch_delta_hv_3d(
    front: np.ndarray,
    candidates: np.ndarray,
    ref: Tuple[float, float, float],
    *,
    compute_negative: bool = True,
) -> np.ndarray:
    """
    Exact delta-HV for a batch of 3D candidate points.

    Returns:
    - positive ΔHV if candidate increases HV (i.e., is "above" front)
    - if ``compute_negative=True``: negative "gap volume" (Option B) for dominated/behind-ref
    - if ``compute_negative=False``: zeros for dominated/behind-ref.

    Notes:
    - This implementation computes HV(front) once, and HV(front ∪ {p}) per candidate.
      For front size ~70 it's correct and usually acceptable.
      If you need faster later, we'll add a specialized 3D delta algorithm.

    Parallelism (optional): when ``B`` is large enough, splits candidates into coarse
    chunks and runs ``_delta_hv_3d_dense_block`` in worker processes. Disable with
    ``BATCH_DELTA_HV_3D_PARALLEL=0``. Tune with ``BATCH_DELTA_HV_3D_WORKERS`` (default 20)
    and ``BATCH_DELTA_HV_3D_MIN_PER_CHUNK`` (default 50).

    Inputs:
      front: (M,3) current labeled points OR already ND set (either ok)
      candidates: (B,3)
      ref: (rx,ry,rz)
    """
    rx, ry, rz = float(ref[0]), float(ref[1]), float(ref[2])

    cands = np.asarray(candidates, dtype=float).reshape(-1, 3)
    B = len(cands)
    if B == 0:
        return np.empty(0, dtype=float)

    # Ensure front is ND and finite
    F = np.asarray(front, dtype=float).reshape(-1, 3)
    F = F[np.isfinite(F).all(axis=1)]
    F_nd = pareto_front(F) if len(F) else np.empty((0, 3), dtype=float)

    hv_old = hypervolume_3d(F_nd, (rx, ry, rz)) if len(F_nd) else 0.0

    parallel_env = os.environ.get("BATCH_DELTA_HV_3D_PARALLEL", "1").strip().lower()
    use_parallel = parallel_env not in ("0", "false", "no")

    n_chunks = min(
        _BATCH_DELTA_HV_3D_MAX_WORKERS,
        max(1, (B + _BATCH_DELTA_HV_3D_MIN_PER_CHUNK - 1) // _BATCH_DELTA_HV_3D_MIN_PER_CHUNK),
    )

    if not use_parallel or n_chunks <= 1:
        t0 = time.perf_counter()
        out = _delta_hv_3d_dense_block(F_nd, hv_old, cands, (rx, ry, rz), compute_negative)
        t_serial = time.perf_counter() - t0
        logger.info(
            "[TIMER] batch_delta_hv_3d mode=serial total_s=%.6f B=%d F_nd=%d",
            t_serial,
            B,
            len(F_nd),
        )
        return out

    t_dispatch0 = time.perf_counter()
    chunks = np.array_split(cands, n_chunks)
    packs = [
        (F_nd, hv_old, np.ascontiguousarray(ch, dtype=float), (rx, ry, rz), compute_negative)
        for ch in chunks
    ]
    t_after_build = time.perf_counter()
    dispatch_build_s = t_after_build - t_dispatch0

    pool_workers = min(_BATCH_DELTA_HV_3D_MAX_WORKERS, n_chunks)
    t_pool0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=pool_workers) as ex:
        results = list(ex.map(_batch_delta_hv_3d_process_chunk, packs))
    t_pool1 = time.perf_counter()
    pool_wall_s = t_pool1 - t_pool0

    t_merge0 = time.perf_counter()
    out = np.concatenate(results, axis=0)
    merge_s = time.perf_counter() - t_merge0

    logger.info(
        "[TIMER] batch_delta_hv_3d mode=parallel pool_wall_s=%.6f B=%d n_chunks=%d pool_workers=%d "
        "F_nd=%d dispatch_build_packs_s=%.6f merge_concat_s=%.6f",
        pool_wall_s,
        B,
        n_chunks,
        pool_workers,
        len(F_nd),
        dispatch_build_s,
        merge_s,
    )
    return out


def pareto_front_max_3d_fast(Y: np.ndarray) -> np.ndarray:
    """
    Pareto front for MAXIMIZATION in 3D.
    Zwykle dużo szybsze niż naiwny O(n^2).

    Zwraca punkty niezdominowane.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return Y.reshape(0, 3)

    Y = Y[np.isfinite(Y).all(axis=1)]
    if Y.size == 0:
        return Y.reshape(0, 3)

    # sort: x desc, y desc, z desc
    order = np.lexsort((-Y[:, 2], -Y[:, 1], -Y[:, 0]))
    Y = Y[order]

    front = []
    yz_front = []  # lista punktów (y, z), utrzymywana jako schodek: y rośnie, z maleje

    i = 0
    n = len(Y)

    while i < n:
        x_val = Y[i, 0]
        j = i
        while j < n and Y[j, 0] == x_val:
            j += 1

        block = Y[i:j]

        # najpierw sprawdzamy dominację tylko przez wcześniejsze (większe x)
        keep_local = np.ones(len(block), dtype=bool)
        for k, (_, y, z) in enumerate(block):
            dominated = False
            # sprawdź czy istnieje punkt z y' >= y i z' >= z
            for yf, zf in yz_front:
                if yf >= y and zf >= z:
                    dominated = True
                    break
            keep_local[k] = not dominated

        block_kept = block[keep_local]

        if len(block_kept):
            # wewnątrz bloku tego samego x jeszcze 2D Pareto po (y,z)
            # MAX/MAX, sort y desc, z desc
            o2 = np.lexsort((-block_kept[:, 2], -block_kept[:, 1]))
            bz = block_kept[o2]

            best_z = -np.inf
            block_front = []
            for p in bz:
                if p[2] > best_z:
                    block_front.append(p)
                    best_z = p[2]

            block_front = np.asarray(block_front, dtype=float)
            front.extend(block_front)

            # zaktualizuj yz_front i zrób z niego schodek
            yz_front.extend((p[1], p[2]) for p in block_front)
            yz_front.sort(key=lambda t: t[0])  # y asc

            new_yz = []
            best_z = -np.inf
            # usuwamy punkty zdominowane w 2D, idąc od dużego y
            for y, z in reversed(yz_front):
                if z > best_z:
                    new_yz.append((y, z))
                    best_z = z
            yz_front = list(reversed(new_yz))

        i = j

    return np.asarray(front, dtype=float)


class FenwickMax:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.full(n + 1, -np.inf, dtype=np.float64)

    def update(self, i: int, v: float) -> None:
        while i <= self.n:
            if v > self.bit[i]:
                self.bit[i] = v
            i += i & -i

    def query(self, i: int) -> float:
        out = -np.inf
        while i > 0:
            if self.bit[i] > out:
                out = self.bit[i]
            i -= i & -i
        return out


def pareto_front_3d_fenwick(points: np.ndarray) -> np.ndarray:
    """
    3D Pareto front for MAXIMIZATION.
    Zwykle dużo szybsze niż naiwny O(n^2).
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # sort: x desc, then y desc, then z desc
    order = np.lexsort((-P[:, 2], -P[:, 1], -P[:, 0]))
    P = P[order]

    # compress y (ascending), reversed index => suffix(y>=py) becomes prefix query
    ys = np.unique(P[:, 1])
    M = len(ys)

    def rev_idx(y: float) -> int:
        pos = np.searchsorted(ys, y, side="left")
        if pos == M:
            return 0
        return M - pos

    bit = FenwickMax(M)
    front_blocks = []

    i = 0
    n = len(P)

    while i < n:
        x = P[i, 0]
        j = i
        while j < n and P[j, 0] == x:
            j += 1

        block = P[i:j]

        # 1) screen vs points with strictly larger x
        keep = np.ones(len(block), dtype=bool)
        for k, (_, y, z) in enumerate(block):
            ridx = rev_idx(y)
            if ridx > 0 and bit.query(ridx) >= z:
                keep[k] = False

        block = block[keep]
        if len(block) == 0:
            i = j
            continue

        # 2) local Pareto in (y,z) for equal x
        # sort y desc, z desc; keep only z-improving points
        o2 = np.lexsort((-block[:, 2], -block[:, 1]))
        block = block[o2]

        local = []
        best_z = -np.inf
        for p in block:
            z = p[2]
            if z > best_z:
                local.append(p)
                best_z = z

        local = np.asarray(local, dtype=np.float64)
        front_blocks.append(local)

        # 3) update BIT after whole x-block
        for _, y, z in local:
            bit.update(rev_idx(y), z)

        i = j

    if not front_blocks:
        return np.empty((0, 3), dtype=np.float64)

    return np.vstack(front_blocks)


def pareto_skyline(X):
    # X: (N, m)
    idx = np.argsort(-X[:, 0])     # sort malejąco po pierwszym wymiarze
    front = []

    for i in idx:
        x = X[i]
        dominated = False
        for j in front:
            if np.all(X[j] >= x) and np.any(X[j] > x):
                dominated = True
                break
        if not dominated:
            front.append(i)

    return np.array(front, dtype=int)


def hypervolume_nd(front, ref):
    front = np.asarray(front, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    m = front.shape[1]
    hv = 0.0

    stack = [(front, ref, m)]

    while stack:
        P, r, dim = stack.pop()

        if P.size == 0:
            continue

        if dim == 1:
            hv += np.max(P[:,0]) - r[0]
            continue

        order = np.argsort(P[:,dim-1])
        P = P[order]

        prev = r[dim-1]

        for i in range(len(P)):
            level = P[i,dim-1]
            width = level - prev
            if width > 0:
                active = P[i:,:dim-1]
                stack.append((active, r[:dim-1], dim-1))
                hv += width * 0  # contribution computed in lower dims
            prev = level

    return hv
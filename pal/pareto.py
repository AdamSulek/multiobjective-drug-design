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

from typing import Tuple

import numpy as np


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

def batch_delta_hv_2d(front: np.ndarray, candidates: np.ndarray, ref: Tuple[float, float]) -> np.ndarray:
    """Vectorized delta-HV for a batch of 2D candidate points.

    Returns:
    - positive exact ΔHV for nondominated candidates
    - negative "gap area" for dominated/behind-ref (Option B)
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
        if np.any(need_dist):
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
    if np.any(need_dist):
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


def batch_delta_hv_3d(front: np.ndarray, candidates: np.ndarray, ref: Tuple[float, float, float]) -> np.ndarray:
    """
    Exact delta-HV for a batch of 3D candidate points.

    Returns:
    - positive ΔHV if candidate increases HV (i.e., is "above" front)
    - negative "gap volume" (Option B) if candidate is dominated / under front / behind ref
      so you can rank by "less negative = closer to front".

    Notes:
    - This implementation computes HV(front) once, and HV(front ∪ {p}) per candidate.
      For front size ~70 it's correct and usually acceptable.
      If you need faster later, we'll add a specialized 3D delta algorithm.

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

    out = np.zeros(B, dtype=float)

    for i in range(B):
        p = cands[i]
        if not np.isfinite(p).all():
            out[i] = 0.0
            continue

        # Exact ΔHV
        hv_new = hypervolume_3d(np.vstack([F_nd, p]) if len(F_nd) else np.array([p]), (rx, ry, rz))
        delta = hv_new - hv_old

        if delta > 0.0:
            out[i] = float(delta)
        else:
            # Option B: negative "gap volume" under the front
            gap = _gap_volume_3d(F_nd, p, (rx, ry, rz))
            out[i] = -float(gap)

    return out
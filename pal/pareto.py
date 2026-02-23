"""Self-contained 2-D Pareto front and hypervolume utilities (maximization)."""

from typing import Tuple

import numpy as np


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


def hypervolume_2d(points: np.ndarray, ref: Tuple[float, float]) -> float:
    """Compute the 2-D hypervolume indicator (dominated area) for maximization."""
    rx, ry = float(ref[0]), float(ref[1])
    pts = np.asarray(points, dtype=float).reshape(-1, 2)

    # keep only points that dominate the reference
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


def stair_y_at_x(front: np.ndarray, x: float, ref: Tuple[float, float]) -> float:
    """Return baseline y (stair height) at coordinate *x*.

    Used by rectangle-based acquisition to find the existing front height.
    """
    rx, ry = float(ref[0]), float(ref[1])
    if x <= rx or len(front) == 0:
        return ry

    xs = front[:, 0]  # desc
    ys = front[:, 1]  # inc

    # x is to the right of the leftmost (largest-x) front point
    if x >= xs[0]:
        return ry

    # binary search: find where x falls in the descending-x stair
    i = int(np.searchsorted(-xs, -x, side="right") - 1)
    i = max(0, min(i, len(xs) - 1))
    return float(ys[i])


def stair_x_at_y(front: np.ndarray, y: float, ref: Tuple[float, float]) -> float:
    """Return the x-coordinate of the staircase boundary at coordinate *y*.

    Mirror of ``stair_y_at_x``.  For a dominated point at height *y* this
    gives the horizontal distance it must travel rightward to reach the front.
    """
    rx, ry = float(ref[0]), float(ref[1])
    if y <= ry or len(front) == 0:
        return rx

    fy = front[:, 1]  # ascending
    fx = front[:, 0]  # descending

    # y is above the entire front
    if y >= fy[-1]:
        return rx

    # first front index whose fy >= y
    k = int(np.searchsorted(fy, y, side="left"))
    return float(fx[k])


def delta_hv_contribution(
    front: np.ndarray, x: float, y: float, ref: Tuple[float, float]
) -> float:
    """Exact hypervolume improvement of a single candidate point (x, y).

    Returns a positive delta for non-dominated points and a negative
    area-gap score for dominated / behind-ref points so that candidates
    closer to the front rank higher (less negative).

    Dominated scoring uses ``-(dist_v * dist_h)`` (gap-rectangle area)
    when both axes are positive, or ``-(d²)`` when only one axis is
    positive, keeping the result in area units.
    """
    rx, ry = float(ref[0]), float(ref[1])

    # Non-dominated path: positive HV delta
    if x > rx and y > ry:
        hv_old = hypervolume_2d(front, ref) if len(front) > 0 else 0.0
        new_points = (np.vstack([front, [[x, y]]]) if len(front) > 0
                      else np.array([[x, y]]))
        hv_new = hypervolume_2d(new_points, ref)
        delta = float(hv_new - hv_old)
        if delta > 0.0:
            return delta

    # Dominated or behind-ref: negative distance to staircase
    dist_v = stair_y_at_x(front, x, ref) - y
    dist_h = stair_x_at_y(front, y, ref) - x
    dv_pos = dist_v > 0
    dh_pos = dist_h > 0
    if dv_pos and dh_pos:
        return -(dist_v * dist_h)
    if dv_pos:
        return -(dist_v * dist_v)
    if dh_pos:
        return -(dist_h * dist_h)
    return 0.0


def batch_delta_hv_2d(
    front: np.ndarray, candidates: np.ndarray, ref: Tuple[float, float]
) -> np.ndarray:
    """Vectorized delta-HV for a batch of 2D candidate points.

    Computes the hypervolume improvement each candidate would contribute
    if added to the current Pareto *front*, without any Python loops.

    Uses the staircase-integral formula with prefix sums so that the
    result is exact even when the candidate dominates existing front points.

    Parameters
    ----------
    front : np.ndarray, shape ``(M, 2)``
        Current Pareto front sorted by x descending / y ascending
        (as returned by ``pareto_front_2d``).
    candidates : np.ndarray, shape ``(B, 2)``
        Candidate points ``(x, y)`` to evaluate.
    ref : tuple of float
        Reference point ``(rx, ry)`` for HV computation.

    Returns
    -------
    np.ndarray, shape ``(B,)``
        Delta-HV for each candidate (negative area-gap score if dominated
        or behind ref).
    """
    rx, ry = float(ref[0]), float(ref[1])
    cands = np.asarray(candidates, dtype=float).reshape(-1, 2)
    B = len(cands)

    if B == 0:
        return np.empty(0, dtype=float)

    px = cands[:, 0]
    py = cands[:, 1]

    # Candidates behind the reference contribute nothing
    valid = (px > rx) & (py > ry)

    if len(front) == 0 or front.shape[0] == 0:
        delta = np.zeros(B, dtype=float)
        delta[valid] = (px[valid] - rx) * (py[valid] - ry)
        # Behind-ref: negative distance to reference corner
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

    # j[b] = number of front points with fx >= px[b]
    j = np.searchsorted(-fx, -px, side="right")  # (B,)

    # y_below[b] = fy[j-1] if j > 0 else ry  (stair height at x = px)
    y_below = np.where(j > 0, fy[np.clip(j - 1, 0, M - 1)], ry)
    # Fix: points left of / at ref should see ry, not fy[-1]
    y_below = np.where(px <= rx, ry, y_below)

    # Dominated: py <= stair height at x = px
    dominated = py <= y_below

    # k[b] = first index where fy >= py[b]
    k = np.searchsorted(fy, py, side="left")  # (B,)

    # Extended fx with sentinel: fx_ext[M] = rx
    fx_ext = np.append(fx, rx)  # (M+1,)

    # Prefix sums of strip areas:
    #   strip_areas[i] = (fx[i] - fx[i+1]) * (fy[i] - ry)
    # prefix_sum[0] = 0, prefix_sum[i] = sum(strip_areas[0..i-1])
    strip_areas = (fx - fx_ext[1:]) * (fy - ry)
    prefix_sum = np.empty(M + 1, dtype=float)
    prefix_sum[0] = 0.0
    prefix_sum[1:] = np.cumsum(strip_areas)

    # Vectorized delta using the staircase-integral formula:
    #   delta = (px - fx[k]) * (py - ry)
    #         - (px - fx[j]) * (y_below - ry)
    #         - (prefix_sum[k] - prefix_sum[j])
    fx_at_k = fx_ext[k]  # fx[k] if k < M else rx
    fx_at_j = fx_ext[j]  # fx[j] if j < M else rx

    delta = (
        (px - fx_at_k) * (py - ry)
        - (px - fx_at_j) * (y_below - ry)
        - (prefix_sum[k] - prefix_sum[j])
    )

    # Non-dominated: keep positive delta
    non_dom = valid & ~dominated
    delta[~non_dom] = 0.0
    delta = np.maximum(delta, 0.0)

    # Dominated/behind-ref: negative distance to stair
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


def build_stair_polygon(
    front: np.ndarray, ref: Tuple[float, float]
) -> np.ndarray:
    """Build the stair-step polygon vertices for plotting the dominated area.

    Returns an (M, 2) array of polygon vertices (closed) suitable for
    ``matplotlib.patches.Polygon`` or ``plt.fill``.
    """
    rx, ry = float(ref[0]), float(ref[1])
    if len(front) == 0:
        return np.array([[rx, ry]], dtype=float)

    f = pareto_front_2d(front)
    if len(f) == 0:
        return np.array([[rx, ry]], dtype=float)

    xs = f[:, 0]  # desc
    ys = f[:, 1]  # inc

    verts = []
    # start at top-left corner of the first stair step
    verts.append((xs[0], ry))
    for i in range(len(f)):
        verts.append((xs[i], ys[i]))
        if i + 1 < len(f):
            # horizontal step down to next x
            verts.append((xs[i + 1], ys[i]))
    # close along ref edges
    verts.append((rx, ys[-1]))
    verts.append((rx, ry))

    return np.asarray(verts, dtype=float)

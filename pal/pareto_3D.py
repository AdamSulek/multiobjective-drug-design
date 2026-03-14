"""
pareto_3D.py

Utilities for 3-objective Pareto front handling (maximization) and
hypervolume scoring in 3D with a fixed reference point.

Design target:
- Small Pareto front F per iteration (typically tens of points, derived from ~700).
- Very large candidate pool (e.g., 2.9M points).
- Fast dominance screening of candidates vs F using an offline sweep + Fenwick (BIT) trick.
- Scoring:
    * If candidate is NONDOMINATED vs F (it "sticks out"): score = HV(F') - HV(F),
      where F' = (F \ R) ∪ {p}, R = {f in F : p dominates f}.
    * If candidate is DOMINATED vs F (under the front): score = -HV_removed,
      where HV_removed = HV(F) - HV(F \ R), with the same R.

Notes:
- All objectives are maximized.
- Reference point 'ref' must be dominated by all points you want to count in HV
  (i.e., ref is the "worst" corner). Typically ref is component-wise <= all points.
- Hypervolume implementation is exact for 3D and expects (ideally) a nondominated set.
  If you feed dominated points, HV may be overcounted; keep sets nondominated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np


# ---------------------------
# Basic dominance (3D, max)
# ---------------------------

def dominates_3d_max(q: np.ndarray, p: np.ndarray) -> bool:
    """
    True if q dominates p in maximization:
    q >= p component-wise and strictly > in at least one component.
    q, p shape: (3,)
    """
    ge = (q[0] >= p[0]) and (q[1] >= p[1]) and (q[2] >= p[2])
    if not ge:
        return False
    return (q[0] > p[0]) or (q[1] > p[1]) or (q[2] > p[2])


def dominates_3d_max_ge(q: np.ndarray, p: np.ndarray) -> bool:
    """
    Non-strict dominance check (>= in all dims).
    Useful for screening.
    """
    return (q[0] >= p[0]) and (q[1] >= p[1]) and (q[2] >= p[2])


# ---------------------------
# Pareto front (small N)
# ---------------------------

def pareto_front_3d_max(points: np.ndarray) -> np.ndarray:
    """
    Compute nondominated set (Pareto front) for 3D maximization.

    points: np.ndarray shape (N,3)

    Returns: np.ndarray shape (M,3), M<=N, nondominated points.
    Practical algorithm for N~700: sort by x desc and maintain a small front list.
    """
    if points.size == 0:
        return points.reshape(0, 3)

    pts = np.asarray(points, dtype=np.float64)
    order = np.lexsort((-pts[:, 2], -pts[:, 1], -pts[:, 0]))  # x desc, then y desc, z desc
    pts = pts[order]

    front: List[np.ndarray] = []

    for p in pts:
        # if dominated by any current front point, skip
        dom = False
        for q in front:
            if dominates_3d_max(q, p):
                dom = True
                break
        if dom:
            continue

        # otherwise add p, and remove those dominated by p
        new_front = []
        for q in front:
            if not dominates_3d_max(p, q):
                new_front.append(q)
        new_front.append(p.copy())
        front = new_front

    return np.vstack(front) if front else np.empty((0, 3), dtype=np.float64)


# ---------------------------
# Fenwick (BIT) for MAX
# ---------------------------

class FenwickMax:
    """
    Fenwick tree supporting:
      - update(i, val): bit[i] = max(bit[i], val)
      - query(i): max over prefix [1..i]
    1-indexed.
    """
    __slots__ = ("n", "bit")

    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.full(self.n + 1, -np.inf, dtype=np.float64)

    def update(self, idx: int, value: float) -> None:
        n = self.n
        bit = self.bit
        i = int(idx)
        v = float(value)
        while i <= n:
            if v > bit[i]:
                bit[i] = v
            i += i & -i

    def query(self, idx: int) -> float:
        bit = self.bit
        i = int(idx)
        res = -np.inf
        while i > 0:
            v = bit[i]
            if v > res:
                res = v
            i -= i & -i
        return float(res)


@dataclass(frozen=True)
class DominanceIndex3D:
    """
    Data needed to screen dominance of candidates vs a fixed front F in 3D max.

    We do offline sweep by x descending.
    We compress y from the front and use reversed indexing so that condition y>=py becomes a Fenwick prefix query.
    """
    Fx: np.ndarray  # front x sorted desc
    Fy: np.ndarray  # front y sorted desc in same order as Fx
    Fz: np.ndarray  # front z sorted desc in same order as Fx
    ys_sorted: np.ndarray  # unique y values sorted ascending (for bisect)
    M: int  # len(ys_sorted)


def build_dominance_index_3d(front: np.ndarray) -> DominanceIndex3D:
    """
    Build screening index from front points.
    front: (M,3) nondominated points
    Returns DominanceIndex3D
    """
    F = np.asarray(front, dtype=np.float64)
    if F.size == 0:
        return DominanceIndex3D(
            Fx=np.empty((0,), dtype=np.float64),
            Fy=np.empty((0,), dtype=np.float64),
            Fz=np.empty((0,), dtype=np.float64),
            ys_sorted=np.empty((0,), dtype=np.float64),
            M=0,
        )

    order = np.argsort(-F[:, 0])  # x desc
    F = F[order]
    ys = np.unique(F[:, 1])
    ys.sort()  # ascending
    return DominanceIndex3D(
        Fx=F[:, 0].copy(),
        Fy=F[:, 1].copy(),
        Fz=F[:, 2].copy(),
        ys_sorted=ys,
        M=int(ys.shape[0]),
    )


def _y_to_rev_idx(ys_sorted: np.ndarray, y: float) -> int:
    """
    Map a threshold y (need y_front >= y) to a Fenwick prefix index using reversed indexing.

    ys_sorted is ascending unique y from the front.
    We find the first index y_idx such that ys_sorted[y_idx] >= y.
    The valid y's are y_idx..M-1 (a suffix). Reversed index converts it to a prefix length.

    Returns:
      0 if no y in front satisfies y>=threshold
      otherwise an integer in [1..M]
    """
    import bisect
    M = ys_sorted.shape[0]
    y_pos = bisect.bisect_left(ys_sorted, y)  # 0..M
    if y_pos == M:
        return 0
    # suffix [y_pos..M-1] -> reversed prefix length = M - y_pos
    return M - y_pos


def classify_dominated_offline(
    points: np.ndarray,
    index: DominanceIndex3D,
    assume_sorted_by_x_desc: bool = False,
) -> np.ndarray:
    """
    Screen whether each point is dominated by the fixed front F using offline sweep + Fenwick.

    points: (N,3)
    index: built from front
    assume_sorted_by_x_desc:
        If True, points must already be sorted by x descending; output aligns to this order.
        If False, we sort indices and return in original order.

    Returns:
      dominated: np.ndarray bool shape (N,), dominated[i] True if ∃f∈F: f>=p (non-strict).
    """
    P = np.asarray(points, dtype=np.float64)
    N = P.shape[0]

    if index.M == 0 or index.Fx.size == 0 or N == 0:
        return np.zeros((N,), dtype=bool)

    bit = FenwickMax(index.M)

    if assume_sorted_by_x_desc:
        order = np.arange(N)
    else:
        order = np.argsort(-P[:, 0])  # x desc

    dominated_sorted = np.zeros((N,), dtype=bool)

    iF = 0
    nF = index.Fx.shape[0]
    Fx, Fy, Fz = index.Fx, index.Fy, index.Fz
    ys_sorted = index.ys_sorted

    for k in range(N):
        qi = int(order[k])
        px, py, pz = P[qi, 0], P[qi, 1], P[qi, 2]

        # add all front points with x >= px
        while iF < nF and Fx[iF] >= px:
            ridx = _y_to_rev_idx(ys_sorted, Fy[iF])
            if ridx > 0:
                bit.update(ridx, Fz[iF])
            iF += 1

        ridx_p = _y_to_rev_idx(ys_sorted, py)
        if ridx_p == 0:
            dominated_sorted[k] = False
        else:
            best_z = bit.query(ridx_p)
            dominated_sorted[k] = (best_z >= pz)

    if assume_sorted_by_x_desc:
        return dominated_sorted
    else:
        dominated = np.zeros((N,), dtype=bool)
        dominated[order] = dominated_sorted
        return dominated


# ---------------------------
# Hypervolume 3D (exact, max)
# ---------------------------

def hv_2d_max(rects_yz: np.ndarray, ref_y: float, ref_z: float) -> float:
    """
    Exact 2D hyperarea for maximization in (y,z) with reference (ref_y, ref_z).

    Computes area of union of rectangles [ref_y..y] × [ref_z..z].
    Guaranteed monotone: adding points cannot decrease area.
    """
    if rects_yz.size == 0:
        return 0.0

    yz = np.asarray(rects_yz, dtype=np.float64)

    # keep only points above ref (strictly)
    yz = yz[(yz[:, 0] > ref_y) & (yz[:, 1] > ref_z)]
    if yz.size == 0:
        return 0.0

    # sort by y descending; for a 2D-max Pareto set, z should strictly increase as y decreases
    order = np.argsort(-yz[:, 0], kind="mergesort")
    yz = yz[order]

    # 2D Pareto filter (max): keep points that improve z
    nd = []
    best_z = -np.inf
    for y, z in yz:
        if z > best_z:
            nd.append((y, z))
            best_z = z
    yz = np.asarray(nd, dtype=np.float64)

    # area = sum over slabs in y: (y_i - y_{i+1}) * (z_i - ref_z), last down to ref_y
    area = 0.0
    for i in range(len(yz)):
        y_i, z_i = yz[i]
        y_next = yz[i + 1, 0] if i + 1 < len(yz) else ref_y
        if y_i > y_next:
            area += (y_i - y_next) * (z_i - ref_z)

    return float(area)


def hv_3d_max(points: np.ndarray, ref: Tuple[float, float, float]) -> float:
    """
    Exact 3D hypervolume for maximization with reference point ref.

    points: (M,3) should be nondominated for correctness (union of boxes from ref).
    ref: (rx, ry, rz)

    Algorithm: sort by x desc; for each x-slab compute 2D hyperarea in (y,z) of points with x>=current.
    Complexity: O(M^2 log M) worst-case due to recomputing 2D set per slab, but M is small (front size).
    For M<=100 it's fine and simple/robust.

    If you need faster, we can replace with incremental 2D maintenance.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.size == 0:
        return 0.0
    
    P = pareto_front_3d_max(P)
    
    rx, ry, rz = map(float, ref)

    # Filter points not above ref
    mask = (P[:, 0] > rx) & (P[:, 1] > ry) & (P[:, 2] > rz)
    P = P[mask]
    if P.size == 0:
        return 0.0

    # Sort by x desc
    order = np.argsort(-P[:, 0])
    P = P[order]

    hv = 0.0
    # Active set grows as we move to smaller x
    active_yz = []

    prev_x = P[0, 0]
    for i in range(P.shape[0]):
        x, y, z = P[i]
        # slab thickness from prev_x down to current x
        dx = prev_x - x
        if dx > 0 and active_yz:
            area = hv_2d_max(np.asarray(active_yz), ry, rz)
            hv += dx * area
        active_yz.append((y, z))
        prev_x = x

    # last slab down to ref_x
    dx = prev_x - rx
    if dx > 0 and active_yz:
        area = hv_2d_max(np.asarray(active_yz), ry, rz)
        hv += dx * area

    return float(hv)


# ---------------------------
# Per-point utilities
# ---------------------------

def dominated_points_by_p(front: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Return boolean mask over front: True where p dominates that front point (>= in all dims).
    front: (M,3), p: (3,)
    """
    F = np.asarray(front, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return (F[:, 0] <= p[0]) & (F[:, 1] <= p[1]) & (F[:, 2] <= p[2])


def score_point(
    p: np.ndarray,
    front: np.ndarray,
    hv_front: float,
    ref: Tuple[float, float, float],
    dominated_by_front: bool,
) -> float:
    """
    Score ONE point p according to your agreed logic:

    - If p is NONDOMINATED vs front (dominant / sticks out):
        F' = (front \ R) ∪ {p}, where R are front points dominated by p.
        score = HV(F') - HV(front)  (positive)
    - If p is DOMINATED vs front:
        R = front points dominated by p.
        HV_removed = HV(front) - HV(front \ R)
        score = -HV_removed

    dominated_by_front: result from Fenwick screening (non-strict).
    """
    p = np.asarray(p, dtype=np.float64)
    F = np.asarray(front, dtype=np.float64)

    if F.size == 0:
        # If no front, p "sticks out"; HV increase equals HV({p})
        return hv_3d_max(p.reshape(1, 3), ref) - 0.0

    mask_R = dominated_points_by_p(F, p)  # front points dominated by p (>= in all dims)

    if not dominated_by_front:
        # sticks out: compute HV of updated front
        F_keep = F[~mask_R]
        F_prime = np.vstack([F_keep, p.reshape(1, 3)])
        hv_new = hv_3d_max(F_prime, ref)
        return float(hv_new - hv_front)

    # dominated: only penalty based on removed HV
    if not np.any(mask_R):
        return 0.0  # removes nothing -> no HV removed
    F_keep = F[~mask_R]
    hv_keep = hv_3d_max(F_keep, ref)
    hv_removed = hv_front - hv_keep
    return float(-hv_removed)


def score_points_pipeline(
    points: np.ndarray,
    front: np.ndarray,
    ref: Tuple[float, float, float],
    index: Optional[DominanceIndex3D] = None,
    assume_points_sorted_by_x_desc: bool = False,
) -> np.ndarray:
    """
    Full scoring pipeline for a batch of candidate points (N,3) vs a fixed front.

    This is a convenience function that:
      - computes hv_front once
      - builds dominance index if not provided
      - classifies dominated/nondominated via Fenwick offline sweep
      - scores each point using your rules

    For 2.9M points you will likely want:
      - call classify_dominated_offline() once (pass1)
      - then score in chunks / parallel (pass2)
    but this function is helpful for correctness tests or smaller batches.
    """
    P = np.asarray(points, dtype=np.float64)
    F = np.asarray(front, dtype=np.float64)

    hv_front = hv_3d_max(F, ref)

    if index is None:
        index = build_dominance_index_3d(F)

    dominated = classify_dominated_offline(P, index, assume_sorted_by_x_desc=assume_points_sorted_by_x_desc)

    scores = np.empty((P.shape[0],), dtype=np.float64)
    for i in range(P.shape[0]):
        scores[i] = score_point(P[i], F, hv_front, ref, bool(dominated[i]))
    return scores

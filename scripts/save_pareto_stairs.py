#!/usr/bin/env python3
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


_ITER_RE = re.compile(r"iter[_-]?(\d+)", re.IGNORECASE)


def extract_iter_num(path_str):
    stem = Path(path_str).stem
    m = _ITER_RE.search(stem)
    if not m:
        return None
    return int(m.group(1))


def pareto_front_2d(points):
    """
    2D Pareto front for MAXIMIZATION of both objectives.
    Returns stairs: sorted by x desc, with y strictly increasing.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return pts.reshape(0, 2)

    # sort by x desc, then y desc
    pts = pts[np.lexsort((-pts[:, 1], -pts[:, 0]))]

    front = []
    best_y = -np.inf
    for x, y in pts:
        if y > best_y:
            front.append((x, y))
            best_y = y

    return np.asarray(front, dtype=float)


def build_stairs_for_folder(folder, target_cols, negate_targets):
    folder = Path(folder)
    files = sorted(folder.glob("iter_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No iter_*.parquet in: {folder}")

    rows = []
    for f in files:
        it = extract_iter_num(str(f))
        if it is None:
            continue

        df = pd.read_parquet(f, columns=list(target_cols))
        df = df.dropna(subset=list(target_cols))
        pts = df[list(target_cols)].values.astype(np.float32)

        # jeśli cele są do minimalizacji (np. docking score), to negujemy, żeby mieć MAX
        if negate_targets:
            pts = -pts

        front = pareto_front_2d(pts)  # (K,2)
        if len(front) == 0:
            continue

        for j, (x, y) in enumerate(front):
            rows.append((it, j, float(x), float(y)))

    out_df = pd.DataFrame(rows, columns=["iter", "point_idx", "x", "y"])
    out_df = out_df.sort_values(["iter", "point_idx"], ascending=[True, True]).reset_index(drop=True)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="data/labels",
                    help="Base folder containing {ellipsoid,rectangle}/k_*/iter_*.parquet")
    ap.add_argument("--algorithms", nargs="*", default=["ellipsoid", "rectangle"])
    ap.add_argument("--ks", nargs="*", default=["2.0", "3.0"], help="List of k folders, e.g. 2.0 3.0")
    ap.add_argument("--target_cols", nargs=2, default=["score_3GVB", "score_6D6P"])
    ap.add_argument("--negate_targets", action="store_true",
                    help="If set: negate targets so that Pareto is for maximization.")
    ap.add_argument("--out_name", default="pareto_stairs.parquet",
                    help="Output file name inside each k-folder (one file per folder).")
    args = ap.parse_args()

    base = Path(args.base_dir)

    for algo in args.algorithms:
        for k in args.ks:
            folder = base / algo / f"k_{k}"
            if not folder.exists():
                print(f"[skip] missing folder: {folder}")
                continue

            print(f"[build] {folder}")
            out_df = build_stairs_for_folder(folder, args.target_cols, args.negate_targets)

            out_path = folder / args.out_name
            out_df.to_parquet(out_path, index=False)
            print(f"[saved] {out_path} | rows={len(out_df)} | iters={out_df['iter'].nunique() if len(out_df) else 0}")


if __name__ == "__main__":
    main()


# python scripts/save_pareto_stairs.py --base_dir data/labels --algorithms ellipsoid rectangle --ks 2.0 3.0 --target_cols score_3GVB score_6D6P --negate_targets

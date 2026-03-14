#!/usr/bin/env python3

import argparse
import time
import heapq
import logging
from pathlib import Path
import glob
import re

import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from typing import Tuple

from mlp_model import MLP


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# PARETO (OLD: for penalty)
# ======================================================
def get_pareto_points(props: np.ndarray) -> np.ndarray:
    is_pareto = np.ones(props.shape[0], dtype=bool)
    for i, p in enumerate(props):
        if is_pareto[i]:
            is_pareto[is_pareto] = (
                np.any(props[is_pareto] > p, axis=1)
                | np.all(props[is_pareto] == p, axis=1)
            )
    return props[is_pareto]


# ======================================================
# PARETO (2D stair front for HV rectangle)
# ======================================================
def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """
    2D Pareto front for MAXIMIZATION of both objectives.
    Returns "stair" representation sorted by x desc, with y strictly increasing.
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


def stair_y_at_x(front: np.ndarray, x: float, ref: Tuple[float, float]) -> float:
    """
    Given stair front (x desc, y inc), returns baseline y (stair height) at coordinate x.
    If x <= rx -> baseline is ry.
    """
    rx, ry = float(ref[0]), float(ref[1])
    if x <= rx or len(front) == 0:
        return ry

    xs = front[:, 0]  # desc
    ys = front[:, 1]  # inc

    if x >= xs[0]:
        return float(ys[0])

    # find i such that xs[i] >= x > xs[i+1] (in desc ordering)
    # Use searchsorted on -xs (ascending).
    i = int(np.searchsorted(-xs, -x, side="right") - 1)
    i = max(0, min(i, len(xs) - 1))
    return float(ys[i])


def delta_hv_rectangle_single(front: np.ndarray, x: float, y: float, ref: Tuple[float, float]) -> float:
    """
    "Wystający prostokąt": (x-rx)*(y - stair_y_at_x(front, x)), clipped at >=0.
    Returns 0 if point does not stick out above current Pareto stairs at x.
    """
    rx, ry = float(ref[0]), float(ref[1])
    if x <= rx or y <= ry:
        return 0.0
    base = stair_y_at_x(front, x, ref)
    if y <= base:
        return 0.0
    return float((x - rx) * (y - base))


# ======================================================
# MC DROPOUT (existing: projected stats)
# ======================================================
def mc_stats_proj_cov(model, x_batch, w_tensor_dev, n_passes=200):
    """
    Computes:
      - mean_proj: (batch, n_w) = mean of projections w^T y
      - std_proj : (batch, n_w) = std of projections computed from full covariance of y (2D)
    """
    model.train()

    mean_y = None              # (batch, 2)
    m2 = None                  # (batch, 2, 2)
    t = 0

    with torch.no_grad():
        for _ in range(n_passes):
            t += 1
            y = model(x_batch)  # (batch, 2)

            if mean_y is None:
                mean_y = y
                m2 = torch.zeros((y.shape[0], y.shape[1], y.shape[1]), device=y.device, dtype=y.dtype)
            else:
                delta = y - mean_y
                mean_y = mean_y + delta / t
                delta2 = y - mean_y
                m2 = m2 + delta.unsqueeze(-1) * delta2.unsqueeze(-2)

    cov = m2 / max(t - 1, 1)  # (batch, 2, 2)

    mean_proj = mean_y @ w_tensor_dev.T  # (batch, n_w)

    cov_w = torch.matmul(cov, w_tensor_dev.T.unsqueeze(0).expand(cov.size(0), -1, -1))  # (batch,2,n_w)
    var_proj = (w_tensor_dev.T.unsqueeze(0) * cov_w).sum(dim=1)  # (batch, n_w)

    std_proj = torch.sqrt(torch.clamp(var_proj, min=1e-9))
    return mean_proj, std_proj


def mc_stats_2d_cov(model, x_batch, n_passes=200):
    """
    For rectangle(HV):
      - mean_2d: (batch, 2)
      - std_2d : (batch, 2)  (sqrt of diagonal of covariance)
    """
    model.train()

    mean_y = None              # (batch, 2)
    m2 = None                  # (batch, 2, 2)
    t = 0

    with torch.no_grad():
        for _ in range(n_passes):
            t += 1
            y = model(x_batch)  # (batch, 2)

            if mean_y is None:
                mean_y = y
                m2 = torch.zeros((y.shape[0], y.shape[1], y.shape[1]), device=y.device, dtype=y.dtype)
            else:
                delta = y - mean_y
                mean_y = mean_y + delta / t
                delta2 = y - mean_y
                m2 = m2 + delta.unsqueeze(-1) * delta2.unsqueeze(-2)

    cov = m2 / max(t - 1, 1)  # (batch, 2, 2)
    var = torch.diagonal(cov, dim1=-2, dim2=-1)  # (batch, 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9))
    return mean_y, std


# ======================================================
# DIVERSITY PENALTY (only for ellipsoid mode)
# ======================================================
def compute_penalties_from_diversity(div_df, target_cols, w_tensor_dev, negate_targets=True):
    all_props = div_df[target_cols].values.astype(np.float32)
    if negate_targets:
        all_props = -all_props
    pareto_np = get_pareto_points(all_props)
    pareto_dev = torch.from_numpy(pareto_np).float().to(device)
    penalties = (pareto_dev @ w_tensor_dev.T).max(dim=0).values
    return penalties


# ======================================================
# HELPERS: build div_df from iter*.parquet
# ======================================================
_ITER_RE = re.compile(r"iter[_-]?(\d+)", re.IGNORECASE)

def _extract_iter_num(path: str):
    m = _ITER_RE.search(Path(path).stem)
    if not m:
        return None
    return int(m.group(1))


def load_diversity_df_from_glob(iter_glob: str, target_cols, max_iter=None) -> pd.DataFrame:
    files = sorted(glob.glob(iter_glob))
    if not files:
        raise FileNotFoundError(f"No files match --diversity_iter_glob: {iter_glob}")

    if max_iter is not None:
        kept = []
        for f in files:
            it = _extract_iter_num(f)
            if it is None or it <= max_iter:
                kept.append(f)
        files = kept
        if not files:
            raise FileNotFoundError(
                f"After applying --diversity_max_iter={max_iter}, no files remain from: {iter_glob}"
            )

    logging.info("Building diversity set from %d files (glob=%s)", len(files), iter_glob)

    dfs = []
    needed_cols = ["ID"] + list(target_cols)

    for f in files:
        df = pd.read_parquet(f, columns=needed_cols)
        df["ID"] = df["ID"].astype(str)
        dfs.append(df)

    div_df = pd.concat(dfs, ignore_index=True)

    before = len(div_df)
    div_df = div_df.dropna(subset=needed_cols)
    after = len(div_df)
    if after < before:
        logging.info("Dropped %d rows with missing ID/targets from diversity set", before - after)

    logging.info(
        "Diversity rows=%d | unique IDs=%d | targets=%s",
        len(div_df),
        div_df["ID"].nunique(),
        list(target_cols),
    )
    return div_df


# ======================================================
# SCORING: two separate logics
# ======================================================
def score_ellipsoid(mean_proj, std_proj, penalties_dev, k_ucb):
    """
    Ellipsoid = scalarized UCB over directions (W=22), then max over W.
    Returns:
      alpha_max (N,), w_idx_max (N,)
    """
    alpha = (mean_proj + k_ucb * std_proj) - penalties_dev  # (N,W) - (W,) => (N,W)
    alpha_np = alpha.detach().cpu().numpy()
    w_idx_max = alpha_np.argmax(axis=1).astype(np.int32)
    alpha_max = alpha_np.max(axis=1).astype(np.float32)
    return alpha_max, w_idx_max


def score_rectangle_hv(mean_2d, std_2d, front_np, hv_ref, k_ucb):
    """
    Rectangle(HV) = optimistic point (mean + k*std) in 2D, then
    'wystający prostokąt' area above Pareto stairs at that x.
    Returns:
      alpha_max (N,) as delta_hv >= 0, w_idx_max zeros.
    Also returns optimistic points (x_opt, y_opt) for saving/debugging.
    """
    pts_opt = mean_2d + k_ucb * std_2d  # (N,2)
    pts_np = pts_opt.detach().cpu().numpy()

    out = np.zeros(len(pts_np), dtype=np.float32)
    for i, (x, y) in enumerate(pts_np):
        out[i] = delta_hv_rectangle_single(front_np, float(x), float(y), hv_ref)

    w_idx_max = np.zeros(len(out), dtype=np.int32)
    return out, w_idx_max, pts_np  # include optimistic points for optional save


# ======================================================
# SCREEN
# ======================================================
def screen_parquet(parquet_path, model,
                  w_tensor_dev=None, penalties_dev=None,
                  pareto_front_np=None, hv_ref=None,
                  exclude_ids=None, include_ids=None,
                  n_passes=20, k_ucb=2.0, top_n=1000, batch_size=50000,
                  batch_keep=5000, progress_every=50000, total_hint=3000000,
                  algorithm="ellipsoid"):

    pf = pq.ParquetFile(parquet_path)
    global_top = []
    processed = 0
    skipped = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=["ID", "X_ecfp_2"]):
        t0 = time.time()

        rb = batch
        ids = np.array(rb.column(0).to_pylist(), dtype=str)

        if include_ids is not None:
            mask_np = np.fromiter((i in include_ids for i in ids), dtype=bool, count=len(ids))
        elif exclude_ids is not None:
            mask_np = np.fromiter((i not in exclude_ids for i in ids), dtype=bool, count=len(ids))
        else:
            mask_np = np.ones(len(ids), dtype=bool)

        if not np.any(mask_np):
            skipped += len(ids)
            processed += len(ids)
            continue

        mask_pa = pa.array(mask_np)
        rb_f = pc.filter(rb, mask_pa)

        df = rb_f.to_pandas()
        ids_f = df["ID"].astype(str).values
        x_np = np.stack(df["X_ecfp_2"].values).astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).to(device)

        if algorithm == "ellipsoid":
            if w_tensor_dev is None or penalties_dev is None:
                raise ValueError("ellipsoid requires w_tensor_dev and penalties_dev")
            mean_proj, std_proj = mc_stats_proj_cov(model, x, w_tensor_dev, n_passes=n_passes)
            alpha_max, w_idx_max = score_ellipsoid(mean_proj, std_proj, penalties_dev, k_ucb)
            opt_pts = None  # not used

        elif algorithm == "rectangle":
            if pareto_front_np is None or hv_ref is None:
                raise ValueError("rectangle requires pareto_front_np and hv_ref")
            mean_2d, std_2d = mc_stats_2d_cov(model, x, n_passes=n_passes)
            alpha_max, w_idx_max, opt_pts = score_rectangle_hv(mean_2d, std_2d, pareto_front_np, hv_ref, k_ucb)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        k = min(batch_keep, len(alpha_max))
        if k > 0:
            idx = np.argpartition(alpha_max, -k)[-k:]
            for j in idx:
                val = float(alpha_max[j])
                if val <= 0.0 and algorithm == "rectangle":
                    # nie wnosi HV -> nie ma sensu trzymać w top
                    continue
                mol_id = ids_f[j]
                widx = int(w_idx_max[j])

                # For rectangle we want to keep also optimistic point for saving/debugging
                extra = None
                if algorithm == "rectangle" and opt_pts is not None:
                    extra = (float(opt_pts[j, 0]), float(opt_pts[j, 1]))  # (x_opt, y_opt)

                item = (val, mol_id, widx, extra)

                if len(global_top) < top_n:
                    heapq.heappush(global_top, item)
                elif val > global_top[0][0]:
                    heapq.heapreplace(global_top, item)

        processed += len(ids)
        if processed % progress_every == 0:
            logging.info(
                f"Progress {processed}/{total_hint} | skipped={skipped} | last_batch_time={time.time()-t0:.2f}s"
            )

    # sort by score desc
    return sorted(global_top, key=lambda x: x[0], reverse=True)


# ======================================================
# SAVE
# ======================================================
def save_top_list_ellipsoid(top_list, weights_np, out_csv):
    rows = []
    for score, mol_id, widx, _extra in top_list:
        w1, w2 = weights_np[widx]
        rows.append([mol_id, score, int(widx), float(w1), float(w2)])
    pd.DataFrame(rows, columns=["ID", "alpha_max", "w_idx", "w1", "w2"]).to_csv(out_csv, index=False)
    logging.info("Saved %d -> %s", len(rows), out_csv)


def save_top_list_rectangle(top_list, out_csv):
    rows = []
    for score, mol_id, _widx, extra in top_list:
        if extra is None:
            rows.append([mol_id, score, np.nan, np.nan])
        else:
            xopt, yopt = extra
            rows.append([mol_id, score, xopt, yopt])
    pd.DataFrame(rows, columns=["ID", "delta_hv", "x_opt", "y_opt"]).to_csv(out_csv, index=False)
    logging.info("Saved %d -> %s", len(rows), out_csv)


# ======================================================
# MAIN
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool_parquet", required=True)

    ap.add_argument("--diversity_labels", default=None,
                    help="Parquet z danymi referencyjnymi do Pareto/exclude (stary tryb).")

    ap.add_argument("--diversity_iter_glob", default=None,
                    help='Glob do iteracji, np. "data/ellipsoid/2.0/iter*.parquet". '
                         "Jeśli podasz, to --diversity_labels jest ignorowane.")
    ap.add_argument("--diversity_max_iter", type=int, default=None,
                    help="Jeśli ustawione, to do Pareto/exclude wejdą tylko iteracje <= max_iter.")

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--target_cols", nargs=2, default=["score_3GVB", "score_6D6P"])
    ap.add_argument("--negate_targets", action="store_true")

    # IMPORTANT: two modes only, per your spec
    ap.add_argument("--algorithm", default="ellipsoid", choices=["ellipsoid", "rectangle"])

    ap.add_argument("--k_ucb", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=50000)

    ap.add_argument("--stage1_passes", type=int, default=20)
    ap.add_argument("--stage1_top", type=int, default=200000)
    ap.add_argument("--stage1_keep", type=int, default=5000)

    ap.add_argument("--stage2_passes", type=int, default=200)
    ap.add_argument("--final_top", type=int, default=1000)
    ap.add_argument("--stage2_keep", type=int, default=5000)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = MLP(
        in_features=ckpt["in_features"],
        hidden_dim=ckpt["hparams"]["hidden_dim"],
        num_hidden_layers=ckpt["hparams"]["num_hidden_layers"],
        dropout_rate=ckpt["hparams"]["dropout_rate"],
        out_features=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ===== build diversity dataframe =====
    if args.diversity_iter_glob is not None:
        div_df = load_diversity_df_from_glob(
            iter_glob=args.diversity_iter_glob,
            target_cols=args.target_cols,
            max_iter=args.diversity_max_iter,
        )
    else:
        if args.diversity_labels is None:
            raise ValueError("Provide either --diversity_labels (old) or --diversity_iter_glob (recommended).")
        div_df = pd.read_parquet(args.diversity_labels)
        div_df["ID"] = div_df["ID"].astype(str)
        div_df = div_df.dropna(subset=["ID"] + list(args.target_cols))
        logging.info(
            "Diversity (single file) rows=%d | unique IDs=%d | file=%s",
            len(div_df),
            div_df["ID"].nunique(),
            args.diversity_labels,
        )

    exclude_ids = set(div_df["ID"].astype(str).values.tolist())

    # ===== prepare mode-specific objects =====
    weights_np = None
    w_tensor_dev = None
    penalties_dev = None

    pareto_front_np = None
    hv_ref = None

    if args.algorithm == "ellipsoid":
        # directions for scalarization
        weights_np = np.column_stack([np.linspace(1, 0, 22), np.linspace(0, 1, 22)]).astype(np.float32)
        w_tensor_dev = torch.tensor(weights_np, dtype=torch.float32, device=device)

        penalties_dev = compute_penalties_from_diversity(
            div_df, args.target_cols, w_tensor_dev, negate_targets=args.negate_targets
        )
        logging.info("Penalty vector ready: shape=%s", Tuple(penalties_dev.shape))

    elif args.algorithm == "rectangle":
        # build Pareto stair front for HV rectangles
        props = div_df[list(args.target_cols)].values.astype(np.float32)
        if args.negate_targets:
            props = -props

        pareto_front_np = pareto_front_2d(props)
        if len(pareto_front_np) == 0:
            raise ValueError("Pareto front is empty after filtering/negation; cannot run rectangle(HV).")

        # reference point: slightly worse than the worst observed values (for maximization)
        eps = 1e-6
        rx = float(np.min(props[:, 0]) - eps)
        ry = float(np.min(props[:, 1]) - eps)
        hv_ref = (rx, ry)

        logging.info(
            "Rectangle(HV) ready: pareto_front=%d | hv_ref=(%.6g, %.6g)",
            len(pareto_front_np), hv_ref[0], hv_ref[1]
        )

    # ==============================
    # STAGE 1
    # ==============================
    logging.info("STAGE1 | algorithm=%s | k=%.2f | passes=%d", args.algorithm, args.k_ucb, args.stage1_passes)

    top_stage1 = screen_parquet(
        parquet_path=args.pool_parquet,
        model=model,
        w_tensor_dev=w_tensor_dev,
        penalties_dev=penalties_dev,
        pareto_front_np=pareto_front_np,
        hv_ref=hv_ref,
        exclude_ids=exclude_ids,
        include_ids=None,
        n_passes=args.stage1_passes,
        k_ucb=args.k_ucb,
        top_n=args.stage1_top,
        batch_size=args.batch_size,
        batch_keep=args.stage1_keep,
        algorithm=args.algorithm,
    )

    shortlist_path = str(out_dir / "shortlist.csv")
    if args.algorithm == "ellipsoid":
        save_top_list_ellipsoid(top_stage1, weights_np, shortlist_path)
    else:
        save_top_list_rectangle(top_stage1, shortlist_path)

    # ==============================
    # STAGE 2
    # ==============================
    logging.info("STAGE2 | passes=%d", args.stage2_passes)

    short_df = pd.read_csv(shortlist_path)
    include_ids = set(short_df["ID"].astype(str).values.tolist())

    top_stage2 = screen_parquet(
        parquet_path=args.pool_parquet,
        model=model,
        w_tensor_dev=w_tensor_dev,
        penalties_dev=penalties_dev,
        pareto_front_np=pareto_front_np,
        hv_ref=hv_ref,
        exclude_ids=None,
        include_ids=include_ids,
        n_passes=args.stage2_passes,
        k_ucb=args.k_ucb,
        top_n=args.final_top,
        batch_size=args.batch_size,
        batch_keep=args.stage2_keep,
        algorithm=args.algorithm,
    )

    top_path = str(out_dir / "top1000.csv")
    if args.algorithm == "ellipsoid":
        save_top_list_ellipsoid(top_stage2, weights_np, top_path)
    else:
        save_top_list_rectangle(top_stage2, top_path)


if __name__ == "__main__":
    main()

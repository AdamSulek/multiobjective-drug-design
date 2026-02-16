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

from mlp_model import MLP


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# PARETO
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
# MC DROPOUT
# ======================================================

def mc_stats_proj_cov(model, x_batch, w_tensor_dev, n_passes=200):
    """
    Liczy:
      - mean_proj: (batch, n_w) = średnia z projekcji w^T y
      - std_proj : (batch, n_w) = std z projekcji, ale wyliczone z pełnej kowariancji y (2D)
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
                delta = y - mean_y                 # (batch, 2)
                mean_y = mean_y + delta / t        # (batch, 2)
                delta2 = y - mean_y                # (batch, 2)
                # outer product per sample: (batch,2,1)*(batch,1,2) -> (batch,2,2)
                m2 = m2 + delta.unsqueeze(-1) * delta2.unsqueeze(-2)

    cov = m2 / max(t - 1, 1)  # (batch, 2, 2)

    # mean projection: E[w^T y] = w^T E[y]
    mean_proj = mean_y @ w_tensor_dev.T  # (batch, n_w)

    # variance along each w: Var(w^T y) = w^T Cov(y) w
    # cov_w = cov @ w  -> (batch, 2, n_w)
    cov_w = torch.matmul(cov, w_tensor_dev.T.unsqueeze(0).expand(cov.size(0), -1, -1))
    # var = sum over dim=1: w * (Cov w)
    var_proj = (w_tensor_dev.T.unsqueeze(0) * cov_w).sum(dim=1)  # (batch, n_w)

    std_proj = torch.sqrt(torch.clamp(var_proj, min=1e-9))
    return mean_proj, std_proj


# ======================================================
# DIVERSITY PENALTY
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

    # opcjonalnie utnij do max_iter
    if max_iter is not None:
        kept = []
        for f in files:
            it = _extract_iter_num(f)
            # jeśli nie umiemy wyciągnąć iter — zostaw, ale lepiej logować
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

    # tylko rekordy, które faktycznie mają targety (żeby Pareto było sensowne)
    before = len(div_df)
    div_df = div_df.dropna(subset=needed_cols)
    after = len(div_df)
    if after < before:
        logging.info("Dropped %d rows with missing ID/targets from diversity set", before - after)

    # jeśli masz duplikaty ID z różnych iteracji, to ich NIE kasuję automatycznie,
    # bo czasem chcesz zachować wszystkie pomiary; ale exclude_ids i tak będzie set().
    logging.info(
        "Diversity rows=%d | unique IDs=%d | targets=%s",
        len(div_df),
        div_df["ID"].nunique(),
        list(target_cols),
    )
    return div_df


# ======================================================
# ACQUISITION
# ======================================================
def acquisition(mean_proj, std_proj, penalties_dev, k_ucb, algorithm):
    if algorithm == "ellipsoid":
        return (mean_proj + k_ucb * std_proj) - penalties_dev

    elif algorithm == "rectangle":
        # klasyczny wariant: brak bonusu niepewności
        return mean_proj - penalties_dev

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ======================================================
# SCREEN
# ======================================================
def screen_parquet(parquet_path, model, w_tensor_dev, penalties_dev,
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

        mean_proj, std_proj = mc_stats_proj_cov(
            model, x, w_tensor_dev, n_passes=n_passes
        )

        alpha = acquisition(mean_proj, std_proj, penalties_dev, k_ucb, algorithm)

        alpha_np = alpha.detach().cpu().numpy()
        w_idx_max = alpha_np.argmax(axis=1)
        alpha_max = alpha_np.max(axis=1)

        k = min(batch_keep, len(alpha_max))
        if k > 0:
            idx = np.argpartition(alpha_max, -k)[-k:]
            for j in idx:
                val = float(alpha_max[j])
                mol_id = ids_f[j]
                widx = int(w_idx_max[j])
                if len(global_top) < top_n:
                    heapq.heappush(global_top, (val, mol_id, widx))
                elif val > global_top[0][0]:
                    heapq.heapreplace(global_top, (val, mol_id, widx))

        processed += len(ids)
        if processed % progress_every == 0:
            logging.info(
                f"Progress {processed}/{total_hint} | skipped={skipped} | last_batch_time={time.time()-t0:.2f}s"
            )

    return sorted(global_top, key=lambda x: x[0], reverse=True)


# ======================================================
# SAVE
# ======================================================
def save_top_list(top_list, weights_np, out_csv):
    rows = []
    for score, mol_id, widx in top_list:
        w1, w2 = weights_np[widx]
        rows.append([mol_id, score, int(widx), float(w1), float(w2)])
    pd.DataFrame(rows, columns=["ID", "alpha_max", "w_idx", "w1", "w2"]).to_csv(out_csv, index=False)
    logging.info("Saved %d -> %s", len(rows), out_csv)


# ======================================================
# MAIN
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool_parquet", required=True)

    # stary tryb
    ap.add_argument("--diversity_labels", default=None,
                    help="Parquet z danymi referencyjnymi do Pareto/exclude (stary tryb).")

    # nowy tryb: z iter*.parquet
    ap.add_argument("--diversity_iter_glob", default=None,
                    help='Glob do iteracji, np. "data/ellipsoid/2.0/iter*.parquet". '
                         "Jeśli podasz, to --diversity_labels jest ignorowane.")
    ap.add_argument("--diversity_max_iter", type=int, default=None,
                    help="Jeśli ustawione, to do Pareto/exclude wejdą tylko iteracje <= max_iter.")

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--target_cols", nargs=2, default=["score_3GVB", "score_6D6P"])
    ap.add_argument("--negate_targets", action="store_true")

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

    weights_np = np.column_stack([np.linspace(1, 0, 22), np.linspace(0, 1, 22)]).astype(np.float32)
    w_tensor_dev = torch.tensor(weights_np, dtype=torch.float32, device=device)

    # ===== build diversity dataframe (Pareto reference) =====
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
    penalties_dev = compute_penalties_from_diversity(
        div_df, args.target_cols, w_tensor_dev, negate_targets=args.negate_targets
    )
    logging.info("Penalty vector ready: shape=%s", tuple(penalties_dev.shape))

    # STAGE 1
    logging.info("STAGE1 | algorithm=%s | k=%.2f", args.algorithm, args.k_ucb)
    top_stage1 = screen_parquet(
        parquet_path=args.pool_parquet,
        model=model,
        w_tensor_dev=w_tensor_dev,
        penalties_dev=penalties_dev,
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
    save_top_list(top_stage1, weights_np, shortlist_path)

    # STAGE 2
    logging.info("STAGE2")
    short_df = pd.read_csv(shortlist_path)
    include_ids = set(short_df["ID"].astype(str).values.tolist())

    top_stage2 = screen_parquet(
        parquet_path=args.pool_parquet,
        model=model,
        w_tensor_dev=w_tensor_dev,
        penalties_dev=penalties_dev,
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
    save_top_list(top_stage2, weights_np, top_path)


if __name__ == "__main__":
    main()

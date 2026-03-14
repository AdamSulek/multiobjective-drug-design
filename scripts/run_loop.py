import argparse
import json
import subprocess
import logging
from pathlib import Path
import shutil
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT = Path(".")

DATA_POOL = PROJECT / "data" / "pool" / "savi_merged_all.parquet"
SEED_FILE = PROJECT / "data" / "seeds" / "seed_0001.parquet"

MAX_ITERS = 10

SMILES_COL = "smiles"


# ======================
# utils
# ======================

def run(cmd, log_file):
    cmd = [str(x) for x in cmd]
    logging.info("CMD: %s", " ".join(cmd))
    with open(log_file, "a") as f:
        f.write("\n\n=== " + " ".join(cmd) + " ===\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def ensure_seed_exists(labels_dir: Path):
    labels_dir.mkdir(parents=True, exist_ok=True)
    first = labels_dir / "iter_000.parquet"
    if not first.exists():
        logging.info("Seed missing → copy %s", first)
        shutil.copy(SEED_FILE, first)


def detect_last_completed_iter(labels_dir: Path):
    iters = []
    for p in labels_dir.glob("iter_*.parquet"):
        try:
            it = int(p.stem.split("_")[1])
            iters.append(it)
        except:
            pass
    return max(iters) if iters else -1


def iter_label_paths_or_stop(labels_dir: Path, upto_iter: int):
    paths = []
    for i in range(upto_iter + 1):
        p = labels_dir / f"iter_{i:03d}.parquet"
        if not p.exists():
            logging.warning("Missing label file: %s", p)
            return None
        paths.append(p)
    return paths


# ======================
# MAIN
# ======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", required=True, choices=["ellipsoid", "rectangle"])
    ap.add_argument("--k_ucb", required=True, type=float)
    ap.add_argument("--max_iters", type=int, default=MAX_ITERS)
    args = ap.parse_args()

    # ======================
    # katalogi eksperymentu
    # ======================
    RUNS_BASE = PROJECT / "runs" / args.algorithm / f"k_{args.k_ucb}"
    LOGS_BASE = PROJECT / "logs" / args.algorithm / f"k_{args.k_ucb}"
    LABELS_BASE = PROJECT / "data" / "labels" / args.algorithm / f"k_{args.k_ucb}"

    RUNS_BASE.mkdir(parents=True, exist_ok=True)
    LOGS_BASE.mkdir(parents=True, exist_ok=True)
    LABELS_BASE.mkdir(parents=True, exist_ok=True)

    logging.info("RUNS: %s", RUNS_BASE)
    logging.info("LABELS: %s", LABELS_BASE)

    # ======================
    # seed start
    # ======================
    ensure_seed_exists(LABELS_BASE)

    # ======================
    # resume
    # ======================
    last_done = detect_last_completed_iter(LABELS_BASE)
    start_iter = last_done  # trenujemy na historii do tej iteracji

    logging.info("Last detected iter: %d", last_done)

    # ======================
    # LOOP
    # ======================
    for i in range(start_iter, args.max_iters):

        run_dir = RUNS_BASE / f"iter_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        log_file = LOGS_BASE / f"iter_{i:03d}.log"

        ckpt_path = run_dir / "ckpt.pt"
        top_csv = run_dir / "top1000.csv"
        diversity_hist = run_dir / "diversity_history.parquet"

        # ======================
        # labels
        # ======================
        label_paths = iter_label_paths_or_stop(LABELS_BASE, i)
        if label_paths is None:
            logging.warning("Waiting for labels.")
            return

        # ======================
        # diversity history
        # ======================
        dfs = []
        for p in label_paths:
            df = pd.read_parquet(p, columns=["ID"])
            dfs.append(df)

        hist = pd.concat(dfs, ignore_index=True)
        hist["ID"] = hist["ID"].astype(str).drop_duplicates()

        pool = pd.read_parquet(DATA_POOL, columns=["ID", "score_3GVB", "score_6D6P"])
        pool["ID"] = pool["ID"].astype(str)

        hist = hist.merge(pool, on="ID", how="left")

        diversity_hist.parent.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(diversity_hist, index=False)

        # ======================
        # TRAIN
        # ======================
        rc = run([
            "python", "scripts/train_mlp.py",
            "--label_paths", *[str(p) for p in label_paths],
            "--pool_parquet", str(DATA_POOL),
            "--out_ckpt", str(ckpt_path),
            "--meta_out", str(run_dir / "meta.json"),
            "--negate_targets",
            "--epochs", "10",
            "--batch_size", "4096"
        ], log_file)

        if rc != 0:
            logging.error("TRAIN failed.")
            return

        # ======================
        # SCREEN
        # ======================
        rc = run([
            "python", "scripts/screen_2stage.py",
            "--ckpt", str(ckpt_path),
            "--pool_parquet", str(DATA_POOL),

            "--diversity_iter_glob", f"data/labels/{args.algorithm}/k_{args.k_ucb}/iter*.parquet",
            "--diversity_max_iter", str(i),

            "--out_dir", str(run_dir),
            "--negate_targets",
            "--stage1_passes", "20",
            "--stage1_top", "200000",
            "--stage2_passes", "200",
            "--final_top", "1000",
            "--batch_size", "50000",
            "--k_ucb", str(args.k_ucb),
            "--algorithm", args.algorithm,
        ], log_file)

        if rc != 0:
            logging.error("SCREEN failed.")
            return

        if not top_csv.exists():
            logging.error("Missing top1000.csv")
            return

        # ======================
        # NEXT LABELS
        # ======================
        next_labels = LABELS_BASE / f"iter_{i+1:03d}.parquet"

        subprocess.check_call([
            "python", "scripts/make_next_labels.py",
            "--selected_csv", str(top_csv),
            "--pool_parquet", str(DATA_POOL),
            "--out_parquet", str(next_labels),
            "--smiles_col", SMILES_COL,
        ])

        logging.info("ITER %d DONE", i)


if __name__ == "__main__":
    main()


# nohup python scripts/run_loop.py --algorithm ellipsoid --k_ucb 3.0 > logs/run_loop_elipsoid_k3.out 2>&1 &
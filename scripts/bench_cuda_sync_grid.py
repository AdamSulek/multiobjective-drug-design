#!/usr/bin/env python3
"""Grid benchmark with PAL_CUDA_SYNC_DIAG=1 (CUDA batch sync for attribution)."""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pal.config import ModelConfig
from pal.model import build_model, mc_predict, predict_eval, train_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; aborting.")
        sys.exit(1)

    device = "cuda"
    cfg = ModelConfig()
    d, n_out = 512, 3
    cfg.in_features = d
    cfg.hidden_sizes = (128, 64)
    cfg.dropout = 0.2

    rng = np.random.default_rng(42)
    n_train = 12288
    X = rng.standard_normal((n_train, d), dtype=np.float32)
    Y = rng.standard_normal((n_train, n_out), dtype=np.float32)

    print("=== BENCHMARK_ROWS (wall_s = full train_model call) ===")
    for nw in (2, 4, 8):
        for bs in (128, 256, 512):
            model = build_model(cfg, device=device, out_features=n_out)
            t0 = time.perf_counter()
            train_model(
                model,
                X,
                Y,
                epochs=2,
                batch_size=bs,
                device=device,
                patience=0,
                min_epochs=1,
                num_workers=nw,
            )
            wall = time.perf_counter() - t0
            print(f"TRAIN\tnum_workers={nw}\ttrain_batch={bs}\twall_s={wall:.4f}")

    n_inf = 65536
    X_inf = rng.standard_normal((n_inf, d), dtype=np.float32)
    model = build_model(cfg, device=device, out_features=n_out)
    train_model(
        model,
        X[:4096],
        Y[:4096],
        epochs=2,
        batch_size=256,
        device=device,
        patience=0,
        min_epochs=1,
        num_workers=4,
    )

    print("=== BENCHMARK_ROWS inference predict_eval ===")
    for ibs in (512, 2048, 4096):
        try:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            predict_eval(model, X_inf, batch_size=ibs, device=device)
            wall = time.perf_counter() - t0
            print(f"PREDICT_EVAL\tinfer_batch={ibs}\twall_s={wall:.4f}")
        except RuntimeError as e:
            print(f"PREDICT_EVAL\tinfer_batch={ibs}\tFAILED\t{e}")

    print("=== BENCHMARK_ROWS inference mc_predict (n_passes=8) ===")
    for ibs in (512, 2048, 4096):
        try:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            mc_predict(model, X_inf, n_passes=8, batch_size=ibs, device=device)
            wall = time.perf_counter() - t0
            print(f"MC_PREDICT\tinfer_batch={ibs}\twall_s={wall:.4f}")
        except RuntimeError as e:
            print(f"MC_PREDICT\tinfer_batch={ibs}\tFAILED\t{e}")


if __name__ == "__main__":
    if os.environ.get("PAL_CUDA_SYNC_DIAG", "").strip() not in ("1", "true", "yes"):
        print("Set PAL_CUDA_SYNC_DIAG=1 before running.", file=sys.stderr)
        sys.exit(2)
    main()

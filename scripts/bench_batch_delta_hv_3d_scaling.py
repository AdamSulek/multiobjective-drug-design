#!/usr/bin/env python3
"""
Sweep BATCH_DELTA_HV_3D_WORKERS and BATCH_DELTA_HV_3D_MIN_PER_CHUNK; compare SHM vs pickled.

Usage:
  python scripts/bench_batch_delta_hv_3d_scaling.py

Optional env:
  BENCH_B=2000   candidate count
  BENCH_REPS=3  timed repetitions after warmup
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pal.pareto import batch_delta_hv_3d, shutdown_batch_delta_hv_executor


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    return ys[len(ys) // 2]


def run_config(
    *,
    workers: int,
    min_chunk: int,
    use_shm: bool,
    front: np.ndarray,
    cands: np.ndarray,
    ref: tuple[float, float, float],
    reps: int,
) -> float:
    os.environ["BATCH_DELTA_HV_3D_PARALLEL"] = "1"
    os.environ["BATCH_DELTA_HV_3D_WORKERS"] = str(workers)
    os.environ["BATCH_DELTA_HV_3D_MIN_PER_CHUNK"] = str(min_chunk)
    os.environ["BATCH_DELTA_HV_3D_USE_SHM"] = "1" if use_shm else "0"
    shutdown_batch_delta_hv_executor()
    # Warmup (creates pool + first SHM path)
    batch_delta_hv_3d(front, cands, ref, compute_negative=True)
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        batch_delta_hv_3d(front, cands, ref, compute_negative=True)
        times.append(time.perf_counter() - t0)
    return _median(times)


def main() -> None:
    rng = np.random.default_rng(0)
    B = int(os.environ.get("BENCH_B", "2000"))
    reps = int(os.environ.get("BENCH_REPS", "3"))

    front = rng.random((28, 3), dtype=np.float64) * 2.0 + 0.1
    cands = rng.random((B, 3), dtype=np.float64) * 2.0 + 0.05
    ref = (0.0, 0.0, 0.0)

    os.environ["BATCH_DELTA_HV_3D_PARALLEL"] = "0"
    shutdown_batch_delta_hv_executor()
    t0 = time.perf_counter()
    ref_out = batch_delta_hv_3d(front, cands, ref, compute_negative=True)
    t_serial = time.perf_counter() - t0

    worker_grid = [8, 12, 16, 20]
    chunk_grid = [32, 50, 100, 200]

    print(f"BENCH_B={B} BENCH_REPS={reps} serial_one_shot_s={t_serial:.4f}")
    print("workers\tmin_chunk\tshm_median_s\tpickled_median_s\tshm_ok\tpickled_ok")

    best_shm = (1e9, -1, -1)
    best_pk = (1e9, -1, -1)

    for nw in worker_grid:
        for mc in chunk_grid:
            shutdown_batch_delta_hv_executor()
            t_shm = run_config(
                workers=nw,
                min_chunk=mc,
                use_shm=True,
                front=front,
                cands=cands,
                ref=ref,
                reps=reps,
            )
            out_shm = batch_delta_hv_3d(front, cands, ref, compute_negative=True)
            ok_shm = np.array_equal(ref_out, out_shm)

            shutdown_batch_delta_hv_executor()
            t_pk = run_config(
                workers=nw,
                min_chunk=mc,
                use_shm=False,
                front=front,
                cands=cands,
                ref=ref,
                reps=reps,
            )
            out_pk = batch_delta_hv_3d(front, cands, ref, compute_negative=True)
            ok_pk = np.array_equal(ref_out, out_pk)

            print(f"{nw}\t{mc}\t{t_shm:.4f}\t{t_pk:.4f}\t{ok_shm}\t{ok_pk}")

            if t_shm < best_shm[0]:
                best_shm = (t_shm, nw, mc)
            if t_pk < best_pk[0]:
                best_pk = (t_pk, nw, mc)

    print(
        f"BEST_SHM wall_median_s={best_shm[0]:.4f} workers={best_shm[1]} min_chunk={best_shm[2]} "
        f"speedup_vs_serial={t_serial/best_shm[0]:.2f}x"
    )
    print(
        f"BEST_PICKLED wall_median_s={best_pk[0]:.4f} workers={best_pk[1]} min_chunk={best_pk[2]} "
        f"speedup_vs_serial={t_serial/best_pk[0]:.2f}x"
    )
    if best_pk[0] < 1e8 and best_shm[0] < 1e8:
        print(f"SHM_vs_pickled_speedup={best_pk[0]/best_shm[0]:.2f}x (pickled/shm)")

    shutdown_batch_delta_hv_executor()


if __name__ == "__main__":
    main()

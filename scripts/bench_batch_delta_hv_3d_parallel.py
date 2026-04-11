#!/usr/bin/env python3
"""Compare serial vs parallel batch_delta_hv_3d (bitwise + timing)."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pal.pareto import batch_delta_hv_3d


def main() -> None:
    rng = np.random.default_rng(0)
    front = rng.random((28, 3), dtype=np.float64) * 2.0 + 0.1
    ref = (0.0, 0.0, 0.0)
    B = 800
    cands = rng.random((B, 3), dtype=np.float64) * 2.0 + 0.05

    os.environ["BATCH_DELTA_HV_3D_PARALLEL"] = "0"
    t0 = time.perf_counter()
    out_s = batch_delta_hv_3d(front, cands, ref, compute_negative=True)
    t_serial = time.perf_counter() - t0

    os.environ["BATCH_DELTA_HV_3D_PARALLEL"] = "1"
    t0 = time.perf_counter()
    out_p = batch_delta_hv_3d(front, cands, ref, compute_negative=True)
    t_par = time.perf_counter() - t0

    same = np.array_equal(out_s, out_p)
    max_abs = float(np.max(np.abs(out_s - out_p))) if out_s.size else 0.0
    print(f"bitwise_equal={same} max_abs_diff={max_abs}")
    print(f"wall_serial_s={t_serial:.6f} wall_parallel_s={t_par:.6f} speedup={t_serial/t_par:.2f}x")


if __name__ == "__main__":
    main()

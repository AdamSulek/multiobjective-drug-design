#!/usr/bin/env python3
"""Benchmark Pareto front and hypervolume routines used in PAL."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pal.pareto import (
    batch_delta_hv_2d,
    batch_delta_hv_3d,
    hypervolume_2d,
    hypervolume_3d,
    hypervolume_nd,
    pareto_front,
    pareto_front_2D,
    pareto_front_2d,
    pareto_front_3d_fenwick,
    pareto_front_max_3d_fast,
    pareto_skyline,
)


@dataclass
class BenchCase:
    family: str
    name: str
    dim: int
    fn: Callable[[], float]


def _time_ms(fn: Callable[[], float], repeats: int) -> tuple[float, float, float]:
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min())


def _auto_ref(points: np.ndarray) -> tuple[float, ...]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    ref = mins - 0.01 * span
    return tuple(ref.tolist())


def _build_cases_for_dim(
    *,
    dim: int,
    points: np.ndarray,
    candidates: np.ndarray,
    ref: tuple[float, ...],
) -> tuple[list[BenchCase], dict[str, float]]:
    cases: list[BenchCase] = []
    meta: dict[str, float] = {
        "n_points": float(points.shape[0]),
        "n_candidates": float(candidates.shape[0]),
    }

    if dim == 2:
        cases.extend(
            [
                BenchCase("pareto", "pareto_front_2D", 2, lambda: float(len(pareto_front_2D(points)))),
                BenchCase("pareto", "pareto_front_2d", 2, lambda: float(len(pareto_front_2d(points)))),
                BenchCase("pareto", "pareto_front_generic", 2, lambda: float(len(pareto_front(points)))),
            ]
        )
        front_idx = pareto_front_2D(points)
        front = points[front_idx]
        meta["front_size"] = float(front.shape[0])
        hv_base = float(hypervolume_2d(front, ref))
        meta["hv_base"] = hv_base

        def hv_loop_naive_2d() -> float:
            out = 0.0
            for p in candidates:
                out += float(hypervolume_2d(np.vstack([front, p]), ref) - hv_base)
            return out

        cases.extend(
            [
                BenchCase("hv", "batch_delta_hv_2d", 2, lambda: float(np.sum(batch_delta_hv_2d(front, candidates, ref, compute_negative=False)))),
                BenchCase("hv", "delta_hv_2d_naive_loop", 2, hv_loop_naive_2d),
                BenchCase("hv", "hypervolume_2d", 2, lambda: float(hypervolume_2d(front, ref))),
            ]
        )
        return cases, meta

    if dim == 3:
        cases.extend(
            [
                BenchCase("pareto", "pareto_front_max_3d_fast", 3, lambda: float(len(pareto_front_max_3d_fast(points)))),
                BenchCase("pareto", "pareto_front_3d_fenwick", 3, lambda: float(len(pareto_front_3d_fenwick(points)))),
                BenchCase("pareto", "pareto_front_generic", 3, lambda: float(len(pareto_front(points)))),
            ]
        )
        front = pareto_front_max_3d_fast(points)
        meta["front_size"] = float(front.shape[0])
        hv_base = float(hypervolume_3d(front, ref))
        meta["hv_base"] = hv_base

        def hv_loop_naive_3d() -> float:
            out = 0.0
            for p in candidates:
                out += float(hypervolume_3d(np.vstack([front, p]), ref) - hv_base)
            return out

        cases.extend(
            [
                BenchCase("hv", "batch_delta_hv_3d", 3, lambda: float(np.sum(batch_delta_hv_3d(front, candidates, ref, compute_negative=False)))),
                BenchCase("hv", "delta_hv_3d_naive_loop", 3, hv_loop_naive_3d),
                BenchCase("hv", "hypervolume_3d", 3, lambda: float(hypervolume_3d(front, ref))),
            ]
        )
        return cases, meta

    if dim in (4, 5):
        cases.extend(
            [
                BenchCase("pareto", "pareto_skyline", dim, lambda: float(len(pareto_skyline(points)))),
                BenchCase("pareto", "pareto_front_generic", dim, lambda: float(len(pareto_front(points)))),
            ]
        )
        idx = pareto_skyline(points)
        front = points[idx]
        meta["front_size"] = float(front.shape[0])
        cases.append(BenchCase("hv", "hypervolume_nd", dim, lambda: float(hypervolume_nd(front, ref))))
        return cases, meta

    raise ValueError(f"Unsupported dim={dim}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Pareto/HV methods across dimensions.")
    p.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--n_points", type=int, nargs="+", default=[1000, 3000, 5000])
    p.add_argument("--n_candidates", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="results/benchmark/pareto_hv_benchmark.csv")
    p.add_argument("--out_summary_csv", type=str, default="results/benchmark/pareto_hv_summary.csv")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_project", type=str, default="multiobjective-drug-design")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default="pareto-hv-benchmark")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except Exception as exc:
            print(f"[warn] wandb disabled: {exc}")
            wandb_run = None

    rows: list[dict[str, float | int | str]] = []

    for dim in args.dims:
        for n in args.n_points:
            points = rng.normal(size=(n, dim)).astype(np.float64)
            candidates = rng.normal(size=(args.n_candidates, dim)).astype(np.float64)
            ref = _auto_ref(points)

            cases, meta = _build_cases_for_dim(
                dim=dim,
                points=points,
                candidates=candidates,
                ref=ref,
            )

            for case in cases:
                mean_ms, std_ms, min_ms = _time_ms(case.fn, args.repeats)
                row = {
                    "family": case.family,
                    "func": case.name,
                    "dim": case.dim,
                    "n_points": int(meta["n_points"]),
                    "n_candidates": int(meta["n_candidates"]),
                    "front_size": int(meta.get("front_size", 0.0)),
                    "repeats": int(args.repeats),
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "min_ms": min_ms,
                }
                rows.append(row)

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "family": row["family"],
                            "func": row["func"],
                            "dim": row["dim"],
                            "n_points": row["n_points"],
                            "n_candidates": row["n_candidates"],
                            "front_size": row["front_size"],
                            "mean_ms": row["mean_ms"],
                            "std_ms": row["std_ms"],
                            "min_ms": row["min_ms"],
                        }
                    )

                print(
                    f"[bench] dim={dim} n={n} family={case.family} func={case.name} "
                    f"mean={mean_ms:.3f}ms std={std_ms:.3f}ms"
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("")

    groups: dict[tuple[str, int, int], dict[str, float | int | str]] = {}
    for row in rows:
        key = (str(row["family"]), int(row["dim"]), int(row["n_points"]))
        prev = groups.get(key)
        if prev is None or float(row["mean_ms"]) < float(prev["mean_ms"]):
            groups[key] = row

    summary_rows: list[dict[str, float | int | str]] = []
    for (family, dim, n_points), r in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        summary_rows.append(
            {
                "family": family,
                "dim": dim,
                "n_points": n_points,
                "winner_func": str(r["func"]),
                "winner_mean_ms": float(r["mean_ms"]),
                "std_ms": float(r["std_ms"]),
                "min_ms": float(r["min_ms"]),
            }
        )

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with out_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        out_summary.write_text("")

    print(f"[done] saved benchmark rows: {out_csv}")
    print(f"[done] saved winners:        {out_summary}")

    if wandb_run is not None:
        wandb_run.summary["rows"] = int(len(rows))
        wandb_run.summary["winners"] = int(len(summary_rows))
        wandb_run.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build performance tables from PAL result folders."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIME_RE = re.compile(
    r"\[TIME\] \[(?P<strategy>.+?)\] it\s+(?P<it>\d+)\s+\| total=(?P<total>[0-9.]+)s"
)
ORACLE_RE = re.compile(r"Global Pareto HV =\s*([0-9.eE+-]+)")
NAME_3D_RE = re.compile(
    r"^3d_(?P<strategy>random|ucb|ellipse_fast|ellipse_directions)_(?P<neg>two|all3|none)_(?P<clip>clip0|clip1)$"
)
NAME_2D_RE = re.compile(
    r"^2d_(?P<strategy>random|ucb|ellipse_fast|ellipse_directions)_(?P<neg>both|none)_(?P<clip>clip0|clip1)$"
)
K_RE = re.compile(r"k=([0-9.]+)")



def parse_oracle_hv(compare_log: Path) -> float | None:
    if not compare_log.exists():
        return None
    for line in compare_log.read_text(errors="replace").splitlines():
        m = ORACLE_RE.search(line)
        if m:
            return float(m.group(1))
    return None



def is_run_done(run_dir: Path) -> bool:
    compare_log = run_dir / "compare.log"
    if not compare_log.exists():
        return False
    txt = compare_log.read_text(errors="replace")
    return "Done!" in txt



def parse_runtime_by_strategy(log_file: Path) -> pd.DataFrame:
    if not log_file.exists():
        return pd.DataFrame(columns=["strategy", "replicate", "runtime_iter_s"])

    seq: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for line in log_file.read_text(errors="replace").splitlines():
        m = TIME_RE.search(line)
        if not m:
            continue
        strategy = m.group("strategy")
        it = int(m.group("it"))
        total = float(m.group("total"))
        seq[strategy].append((it, total))

    rows = []
    for strategy, items in seq.items():
        rep = 0
        prev_it = -1
        rep_vals: dict[int, list[float]] = defaultdict(list)
        for it, total in items:
            if prev_it >= 0 and it <= prev_it:
                rep += 1
            rep_vals[rep].append(total)
            prev_it = it
        for r, vals in rep_vals.items():
            rows.append(
                {
                    "strategy": strategy,
                    "replicate": int(r),
                    "runtime_total_s": float(np.sum(vals)),
                    "runtime_iter_s": float(np.mean(vals)),
                    "runtime_n_iters": int(len(vals)),
                }
            )
    return pd.DataFrame(rows)



def filter_complete_strategy_replicates(hv: pd.DataFrame, target_iter: int) -> pd.DataFrame:
    if hv.empty:
        return hv
    max_it = hv.groupby(["strategy", "replicate"], as_index=False)["iteration"].max()
    done_keys = max_it[max_it["iteration"] >= int(target_iter)][["strategy", "replicate"]]
    if done_keys.empty:
        return hv.iloc[0:0].copy()
    return hv.merge(done_keys, on=["strategy", "replicate"], how="inner")



def strategy_to_method_k(strategy: str) -> tuple[str, str, int, float]:
    s = strategy.strip()
    if s.lower().startswith("random"):
        return "Random", "–", 0, -1.0
    if s.startswith("UCB("):
        m = K_RE.search(s)
        k = float(m.group(1)) if m else np.nan
        if np.isfinite(k) and abs(k) < 1e-12:
            return "UCB/Greedy", "0.0", 1, 0.0
        return "UCB", f"{k:.1f}", 2, k
    if "FastEllipse" in s:
        m = K_RE.search(s)
        k = float(m.group(1)) if m else np.nan
        return "FastEllipse", f"{k:.1f}", 3, k
    if "EllipseDirections" in s:
        m = K_RE.search(s)
        k = float(m.group(1)) if m else np.nan
        return "EllipseDirections", f"{k:.1f}", 4, k
    return s, "–", 99, np.nan



def collect_run_records(
    run_dir: Path,
    logs_root: Path,
    target_iter: int,
    *,
    require_done: bool,
) -> pd.DataFrame:
    if require_done and not is_run_done(run_dir):
        return pd.DataFrame()

    hv_csv = run_dir / "hv_convergence.csv"
    if not hv_csv.exists():
        return pd.DataFrame()
    hv = pd.read_csv(hv_csv)

    if require_done:
        hv = filter_complete_strategy_replicates(hv, target_iter)
        if hv.empty:
            return pd.DataFrame()

    oracle_hv = parse_oracle_hv(run_dir / "compare.log")

    runtime_log_candidates = [
        logs_root / f"{run_dir.name}.log",
        logs_root / f"{run_dir.name}.err",
        logs_root / f"{run_dir.name}.out",
        run_dir / "compare.log",
    ]
    runtime_log_candidates.extend(sorted(logs_root.glob(f"{run_dir.name}_*.log")))
    runtime_log_candidates.extend(sorted(logs_root.glob(f"{run_dir.name}_*.err")))
    runtime_log_candidates.extend(sorted(logs_root.glob(f"{run_dir.name}_*.out")))

    runtime_df = pd.DataFrame(columns=["strategy", "replicate", "runtime_total_s", "runtime_iter_s", "runtime_n_iters"])
    for p in runtime_log_candidates:
        parsed = parse_runtime_by_strategy(p)
        if len(parsed) > len(runtime_df):
            runtime_df = parsed

    rows = []
    for (strategy, rep), g in hv.groupby(["strategy", "replicate"], sort=False):
        g = g.sort_values("iteration")
        row_it = g[g["iteration"] == target_iter]
        if row_it.empty:
            row_it = g.iloc[[-1]]

        hv_at = float(row_it["hypervolume"].iloc[0])
        auc = float(np.trapz(g["hypervolume"].to_numpy(dtype=float), g["iteration"].to_numpy(dtype=float)))
        pct = float("nan") if not oracle_hv or oracle_hv <= 0 else 100.0 * hv_at / float(oracle_hv)

        rt = np.nan
        rt_n = 0
        rt_m = runtime_df[
            (runtime_df["strategy"] == strategy)
            & (runtime_df["replicate"].astype(int) == int(rep))
        ]
        if not rt_m.empty:
            rt = float(rt_m["runtime_total_s"].iloc[0])
            rt_n = int(rt_m["runtime_n_iters"].iloc[0])

        method, k_label, order, k_num = strategy_to_method_k(str(strategy))
        rows.append(
            {
                "run_dir": run_dir.name,
                "strategy": strategy,
                "replicate": int(rep),
                "method": method,
                "k": k_label,
                "order": order,
                "k_num": k_num,
                "hypervolume_at_iter": hv_at,
                "hv_auc": auc,
                "oracle_hv_pct": pct,
                "runtime_total_s": rt,
                "runtime_n_iters": rt_n,
            }
        )

    return pd.DataFrame(rows)



def summarize(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    out = (
        records.groupby(["method", "k", "order", "k_num"], as_index=False)
        .agg(
            hv_mean=("hypervolume_at_iter", "mean"),
            hv_std=("hypervolume_at_iter", "std"),
            auc_mean=("hv_auc", "mean"),
            auc_std=("hv_auc", "std"),
            pct_mean=("oracle_hv_pct", "mean"),
            pct_std=("oracle_hv_pct", "std"),
            rt_mean=("runtime_total_s", "mean"),
            rt_std=("runtime_total_s", "std"),
            rt_n_iters_mean=("runtime_n_iters", "mean"),
        )
        .sort_values(["order", "k_num"], na_position="last")
    )
    out["Hypervolume @20"] = out.apply(lambda r: f"{r.hv_mean:.1f} +/- {0.0 if np.isnan(r.hv_std) else r.hv_std:.1f}", axis=1)
    out["HV AUC"] = out.apply(lambda r: f"{r.auc_mean:.2f} +/- {0.0 if np.isnan(r.auc_std) else r.auc_std:.2f}", axis=1)
    out["% Oracle HV"] = out.apply(lambda r: f"{r.pct_mean:.2f}% +/- {0.0 if np.isnan(r.pct_std) else r.pct_std:.2f}%", axis=1)
    out["Runtime / replicate (s)"] = out.apply(
        lambda r: f"{r.rt_mean:.2f} +/- {0.0 if np.isnan(r.rt_std) else r.rt_std:.2f}",
        axis=1,
    )
    return out



def write_table(out_dir: Path, name: str, table: pd.DataFrame) -> None:
    if table.empty:
        print(f"Skip (empty): {name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    table.to_csv(csv_path, index=False)
    show = table[["method", "k", "Hypervolume @20", "HV AUC", "% Oracle HV", "Runtime / replicate (s)"]].rename(
        columns={"method": "Method", "k": "k"}
    )
    md_path.write_text(show.to_markdown(index=False) + "\n")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")



def expand_dirs(results_root: Path, names: Iterable[str]) -> list[Path]:
    out = []
    for n in names:
        p = Path(n)
        if not p.is_absolute():
            p = results_root / n
        out.append(p)
    return out



def auto_groups(results_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    if not results_root.exists():
        return {}

    for p in sorted(results_root.iterdir()):
        if not p.is_dir():
            continue

        m3 = NAME_3D_RE.match(p.name)
        if m3:
            key = f"3d_{m3.group('neg')}_{m3.group('clip')}"
            groups[key].append(p)
            continue

        m2 = NAME_2D_RE.match(p.name)
        if m2:
            key = f"2d_{m2.group('neg')}_{m2.group('clip')}"
            groups[key].append(p)
            continue

    return dict(groups)



def main() -> None:
    ap = argparse.ArgumentParser(description="Export PAL tables (mean +/- std over replicates).")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--logs-root", type=Path, default=Path("logs"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    ap.add_argument("--iter", type=int, default=20, help="Iteration used for Hypervolume @N.")
    ap.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include incomplete runs/methods (default: false, i.e. done-only).",
    )
    ap.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="Optional explicit run dirs (relative to results-root). If omitted, auto-build tables from 2D/3D matrix folders.",
    )
    args = ap.parse_args()

    require_done = not args.allow_incomplete

    if args.run_dirs:
        dirs = [p for p in expand_dirs(args.results_root, args.run_dirs) if p.exists()]
        recs = [collect_run_records(d, args.logs_root, args.iter, require_done=require_done) for d in dirs]
        non_empty = [r for r in recs if not r.empty]
        if not non_empty:
            print("No data found for --run-dirs, nothing to export.")
            return
        rec = pd.concat(non_empty, ignore_index=True)
        table = summarize(rec)
        write_table(args.out_dir, "custom_table", table)
        return

    groups = auto_groups(args.results_root)
    if not groups:
        print("No auto-detected 2D/3D run directories found, nothing to export.")
        return

    for name in sorted(groups.keys()):
        dirs = groups[name]
        recs = [collect_run_records(d, args.logs_root, args.iter, require_done=require_done) for d in dirs]
        non_empty = [r for r in recs if not r.empty]
        if not non_empty:
            print(f"Skip (no done data): {name}")
            continue
        rec = pd.concat(non_empty, ignore_index=True)
        table = summarize(rec)
        write_table(args.out_dir, name, table)


if __name__ == "__main__":
    main()

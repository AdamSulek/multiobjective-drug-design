#!/usr/bin/env python3
"""Generate PAL summary tables and HV plots in one command."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def filter_complete_strategy_replicates(hv: pd.DataFrame, target_iter: int) -> pd.DataFrame:
    if hv.empty:
        return hv
    max_it = hv.groupby(["strategy", "replicate"], as_index=False)["iteration"].max()
    done_keys = max_it[max_it["iteration"] >= int(target_iter)][["strategy", "replicate"]]
    if done_keys.empty:
        return hv.iloc[0:0].copy()
    return hv.merge(done_keys, on=["strategy", "replicate"], how="inner")


def parse_runtime_by_strategy(log_file: Path) -> pd.DataFrame:
    if not log_file.exists():
        return pd.DataFrame(columns=["strategy", "replicate", "runtime_total_s", "runtime_iter_s", "runtime_n_iters"])

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


def collect_run_records(run_dir: Path, logs_root: Path, target_iter: int, require_done: bool) -> pd.DataFrame:
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


def load_case_df(run_dirs: Iterable[Path], target_iter: int, require_done: bool) -> tuple[pd.DataFrame, float | None]:
    frames = []
    oracle = None
    for d in run_dirs:
        if require_done and not is_run_done(d):
            continue
        hv_path = d / "hv_convergence.csv"
        if not hv_path.exists():
            continue

        df = pd.read_csv(hv_path)
        if require_done:
            df = filter_complete_strategy_replicates(df, target_iter)
            if df.empty:
                continue
        df["run_dir"] = d.name
        frames.append(df)

        if oracle is None:
            oracle = parse_oracle_hv(d / "compare.log")

    if not frames:
        return pd.DataFrame(), None
    return pd.concat(frames, ignore_index=True), oracle


def plot_case(ax, df: pd.DataFrame, title: str, normalize: bool, oracle: float | None) -> None:
    if df.empty:
        ax.set_title(f"{title}\\n(no data)")
        ax.grid(True, alpha=0.3)
        return

    y_col = "hypervolume"
    if normalize and oracle and oracle > 0:
        df = df.copy()
        df[y_col] = df[y_col] / oracle
        y_label = "HV / oracle"
    else:
        y_label = "Hypervolume"

    plotted = 0
    for strategy, g in df.groupby("strategy", sort=False):
        pivot = g.pivot_table(index="replicate", columns="iteration", values=y_col, aggfunc="mean")
        if pivot.empty:
            continue
        xs = np.array(sorted(pivot.columns), dtype=int)
        vals = pivot[xs].to_numpy(dtype=float)
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)
        ax.plot(xs, mean, label=strategy)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
        plotted += 1

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    if plotted > 0:
        ax.legend(fontsize=7)


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


def export_tables(groups: dict[str, list[Path]], logs_root: Path, out_dir: Path, target_iter: int, require_done: bool) -> int:
    saved = 0
    for name in sorted(groups.keys()):
        dirs = groups[name]
        recs = [collect_run_records(d, logs_root, target_iter, require_done=require_done) for d in dirs]
        non_empty = [r for r in recs if not r.empty]
        if not non_empty:
            print(f"Skip table (no done data): {name}")
            continue
        rec = pd.concat(non_empty, ignore_index=True)
        table = summarize(rec)
        write_table(out_dir, name, table)
        if not table.empty:
            saved += 1
    return saved


def export_plots(groups: dict[str, list[Path]], out_dir: Path, target_iter: int, require_done: bool, normalize_oracle: bool, dpi: int) -> int:
    saved = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(groups.keys()):
        df, oracle = load_case_df(groups[key], target_iter=target_iter, require_done=require_done)
        if df.empty:
            print(f"Skip plot (no done data): {key}")
            continue

        parts = key.split("_")
        dim = parts[0]
        mode = parts[1]
        clip = parts[2]
        title = f"{dim.upper()} ({mode}) - {clip}"
        out_path = out_dir / f"hv_{key}.png"

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_case(ax, df, title, normalize_oracle, oracle)
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

        print(f"Saved: {out_path}")
        saved += 1

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PAL tables and plots for 2D/3D cases.")
    ap.add_argument("--project", type=str, default=None, help="Project name under results/<project> and logs/<project>.")
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--logs-root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write tables/plots. Default: <results-root>/tables")
    ap.add_argument("--iter", type=int, default=20)
    ap.add_argument(
        "--allow-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include incomplete runs/methods (default: false, i.e. done-only).",
    )
    ap.add_argument(
        "--normalize-oracle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize HV by oracle when available.",
    )
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument(
        "--mode",
        choices=["all", "tables", "plots"],
        default="all",
        help="What to generate.",
    )
    args = ap.parse_args()

    if args.project:
        results_root = args.results_root or Path("results") / args.project
        logs_root = args.logs_root or Path("logs") / args.project
    else:
        results_root = args.results_root or Path("results")
        logs_root = args.logs_root or Path("logs")

    out_dir = args.out_dir or (results_root / "tables")
    require_done = not args.allow_incomplete

    groups = auto_groups(results_root)
    if not groups:
        print(f"No auto-detected 2D/3D case folders in: {results_root}")
        return

    n_tables = 0
    n_plots = 0

    if args.mode in ("all", "tables"):
        n_tables = export_tables(groups, logs_root, out_dir, args.iter, require_done)
    if args.mode in ("all", "plots"):
        n_plots = export_plots(groups, out_dir, args.iter, require_done, args.normalize_oracle, args.dpi)

    print(f"Done. tables={n_tables}, plots={n_plots}, out_dir={out_dir}")


if __name__ == "__main__":
    main()

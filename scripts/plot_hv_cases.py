#!/usr/bin/env python3
"""Plot HV convergence charts for detected 2D/3D clip cases."""

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


ORACLE_RE = re.compile(r"Global Pareto HV =\s*([0-9.eE+-]+)")
NAME_3D_RE = re.compile(
    r"^3d_(?P<strategy>random|ucb|ellipse_fast|ellipse_directions)_(?P<neg>two|all3|none)_(?P<clip>clip0|clip1)$"
)
NAME_2D_RE = re.compile(
    r"^2d_(?P<strategy>random|ucb|ellipse_fast|ellipse_directions)_(?P<neg>both|none)_(?P<clip>clip0|clip1)$"
)



def parse_oracle(compare_log: Path) -> float | None:
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
    return "Done!" in compare_log.read_text(errors="replace")



def filter_complete_strategy_replicates(hv: pd.DataFrame, target_iter: int) -> pd.DataFrame:
    if hv.empty:
        return hv
    max_it = hv.groupby(["strategy", "replicate"], as_index=False)["iteration"].max()
    done_keys = max_it[max_it["iteration"] >= int(target_iter)][["strategy", "replicate"]]
    if done_keys.empty:
        return hv.iloc[0:0].copy()
    return hv.merge(done_keys, on=["strategy", "replicate"], how="inner")



def load_case_df(
    run_dirs: Iterable[Path],
    *,
    target_iter: int,
    require_done: bool,
) -> tuple[pd.DataFrame, float | None]:
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
            oracle = parse_oracle(d / "compare.log")

    if not frames:
        return pd.DataFrame(), None
    return pd.concat(frames, ignore_index=True), oracle



def plot_case(ax, df: pd.DataFrame, title: str, normalize: bool, oracle: float | None) -> None:
    if df.empty:
        ax.set_title(f"{title}\n(no data)")
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
        pivot = g.pivot_table(
            index="replicate",
            columns="iteration",
            values=y_col,
            aggfunc="mean",
        )
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



def auto_cases(results_root: Path) -> dict[str, list[Path]]:
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
    ap = argparse.ArgumentParser(description="Generate HV convergence plots for detected 2D/3D clip cases.")
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    ap.add_argument("--iter", type=int, default=20, help="Required completed iteration for done-only filtering.")
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
    args = ap.parse_args()

    require_done = not args.allow_incomplete

    groups = auto_cases(args.results_root)
    if not groups:
        print("No auto-detected 2D/3D cases found, nothing to plot.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for key in sorted(groups.keys()):
        df, oracle = load_case_df(
            groups[key],
            target_iter=args.iter,
            require_done=require_done,
        )
        if df.empty:
            print(f"Skip (no done data): {key}")
            continue

        parts = key.split("_")
        dim = parts[0]
        mode = parts[1]
        clip = parts[2]
        title = f"{dim.upper()} ({mode}) - {clip}"
        out_path = args.out_dir / f"hv_{key}.png"

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plot_case(ax, df, title, args.normalize_oracle, oracle)
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)

        print(f"Saved: {out_path}")
        saved += 1

    if saved == 0:
        print("No non-empty cases to plot.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figures for one compare_quantizers.py run.

Input: a CSV written by compare_quantizers.py (one row per dataset, method,
bits). One panel column per dataset. Columns read here:
  dataset             str    gauss<dim> or the .npy stem (optional in old CSVs)
  method              str    turboquant | rabitq | rabitq-algorithm1 | ...
  bits                int    bits per coordinate (1, 4, 8)
  dim                 int    input dimension before padding
  code_bytes          int    bytes per stored code
  recall_at_10        float  [0, 1], overlap with exact squared-L2 top-10
  rel_mae             float  > 0, mean relative error of the asym distance;
                             NaN on FastScan rows (top-k index, no per-pair output)
  encode_vps          float  > 0, vectors/s through encode_batch
  search_pairs_per_s  float  > 0, pairs/s through distance_1_to_n
  host, package_version, timestamp   str, used in the figure title only

Outputs, under --out-dir (PNG, 150 dpi):
  compare_accuracy.png    recall@10 and rel_mae against bytes per code
  compare_throughput.png  encode and search rates, log scale

Usage:
  uv run python python/benchmarks/plot_compare.py \\
      python/benchmarks/results/compare_<stamp>.csv --out-dir docs/img
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# column -> numpy dtype family it must belong to (None: any, e.g. strings)
_REQUIRED = {
    "method": None,
    "bits": np.integer,
    "dim": np.integer,
    "code_bytes": np.integer,
    "recall_at_10": np.floating,
    "rel_mae": np.floating,
    "encode_vps": np.floating,
    "search_pairs_per_s": np.floating,
}

# One colour and marker per method, so every panel reads the same way.
# Bare `rabitq` is the library default: windowed_scale at 4/8 bits, the sign
# code at 1 bit.
_STYLE = {
    "turboquant": ("#1f77b4", "o"),
    "rabitq": ("#d62728", "s"),
    "rabitq-fixed-scale": ("#8c564b", "v"),
    "rabitq-trained-scale": ("#7f7f7f", "x"),
    "rabitq-algorithm1": ("#ff7f0e", "^"),
    "rabitq-windowed-scale": ("#2ca02c", "D"),
    "turboquant-fastscan": ("#17becf", "P"),
    "rabitq-fastscan": ("#9467bd", "X"),
}
_FALLBACK_STYLE = ("#7f7f7f", "x")
_DPI = 150
_PANEL_WIDTH_IN = 4.6


def load_rows(path: Path) -> pd.DataFrame:
    """Read a compare CSV and fail with the offending column on schema drift."""
    if not path.is_file():
        raise SystemExit(f"no such CSV: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(_REQUIRED) - set(df.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if df.empty:
        raise SystemExit(f"{path}: no rows")
    for column, kind in _REQUIRED.items():
        if kind is not None and not np.issubdtype(df[column].dtype, kind):
            raise SystemExit(f"{path}: column {column} has dtype {df[column].dtype}")
    if not df["recall_at_10"].between(0.0, 1.0).all():
        raise SystemExit(f"{path}: recall_at_10 outside [0, 1]")
    positive = ["encode_vps", "search_pairs_per_s"]
    if not (df[positive] > 0).all().all():
        raise SystemExit(f"{path}: non-positive value in {positive} (log axes)")
    if not (df["rel_mae"].dropna() > 0).all():
        raise SystemExit(f"{path}: non-positive rel_mae (log axis)")
    if "dataset" not in df.columns:  # CSVs written before --data existed
        df = df.assign(dataset="gauss" + df["dim"].astype(str))
    if df.duplicated(["dataset", "method", "bits"]).any():
        raise SystemExit(f"{path}: duplicate (dataset, method, bits) rows")
    return df


def _panels(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(title, rows) per dataset: synthetic first by dim, then real data."""
    keys = df.drop_duplicates("dataset")[["dataset", "dim"]]
    keys = keys.assign(real=~keys["dataset"].str.fullmatch(r"gauss\d+")).sort_values(["real", "dim"])
    return [(f"{name} · dim {dim}", df[df["dataset"] == name])
            for name, dim in keys[["dataset", "dim"]].itertuples(index=False)]


def _title(df: pd.DataFrame) -> str:
    first = df.iloc[0]
    return (f"vsq {first.get('package_version', '?')} · {first.get('host', '?')} · "
            f"{first.get('timestamp', '?')}")


def plot_accuracy(df: pd.DataFrame, out: Path) -> None:
    """Rows: recall@10, rel_mae. Columns: one per dim. x: bytes per code."""
    panels = _panels(df)
    fig, axes = plt.subplots(2, len(panels), figsize=(_PANEL_WIDTH_IN * len(panels), 7.2),
                             squeeze=False, sharey="row")
    for col, (title, sub) in enumerate(panels):
        for method, group in sub.groupby("method", sort=False):
            color, marker = _STYLE.get(method, _FALLBACK_STYLE)
            group = group.sort_values("code_bytes")
            for row, metric in enumerate(("recall_at_10", "rel_mae")):
                points = group.dropna(subset=[metric])  # FastScan has no rel_mae
                if points.empty:
                    continue
                axes[row][col].plot(points["code_bytes"], points[metric], marker=marker,
                                    color=color, linewidth=1, markersize=6, label=method)
        axes[0][col].set_title(title, fontsize=10)
        axes[1][col].set_xlabel("bytes per code")
        axes[1][col].set_yscale("log")
        for ax in axes[:, col]:
            ax.grid(True, which="both", alpha=0.3)
    axes[0][0].set_ylabel("recall@10 (higher is better)")
    axes[1][0].set_ylabel("relative distance error (lower is better)")
    by_label = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            by_label.setdefault(label, handle)
    handles, labels = list(by_label.values()), list(by_label.keys())
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(f"Accuracy vs code size — {_title(df)}")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def plot_throughput(df: pd.DataFrame, out: Path) -> None:
    """Rows: encode, search. Columns: one per dim. One bar per method:bits."""
    panels = _panels(df)
    metrics = (("encode_vps", "encode, vectors/s"),
               ("search_pairs_per_s", "1-to-N search, codes/s"))
    order = df.drop_duplicates(["method", "bits"])[["method", "bits"]]
    labels = [f"{m}:{b}" for m, b in order.itertuples(index=False)]
    colors = [_STYLE.get(m, _FALLBACK_STYLE)[0] for m in order["method"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(len(metrics), len(panels),
                             figsize=(_PANEL_WIDTH_IN * len(panels), 8.0),
                             squeeze=False, sharey="row")
    for col, (title, sub) in enumerate(panels):
        sub = sub.set_index(sub["method"] + ":" + sub["bits"].astype(str)).reindex(labels)
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row][col]
            ax.bar(x, sub[metric].to_numpy(), color=colors)
            ax.set_yscale("log")
            ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
            ax.grid(True, axis="y", which="both", alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[0][col].set_title(title, fontsize=10)
    fig.suptitle(f"Single-thread throughput — {_title(df)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="compare_<stamp>.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    df = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for plot, name in ((plot_accuracy, "compare_accuracy.png"),
                       (plot_throughput, "compare_throughput.png")):
        path = args.out_dir / name
        plot(df, path)
        print(path)


if __name__ == "__main__":
    main()

"""TurboQuant benchmark: SIFT1M recall/throughput + synthetic throughput.

Downloads SIFT1M on first run (cached in ~/.cache/turboquant/sift/), runs the
full matrix of (dataset, bits, num_threads), writes a CSV, and produces
seaborn plots. Opens plots in a window if a display is available.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from turboquant import TurboQuantSpace


# ---------------------------------------------------------------------------
# SIFT1M download + .fvecs/.ivecs reader
# ---------------------------------------------------------------------------

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
CACHE_DIR = Path.home() / ".cache" / "turboquant" / "sift"


def _read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    n = raw.size // (d + 1)
    view = raw.reshape(n, d + 1)
    return view[:, 1:].copy().view(np.float32)


def _read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    n = raw.size // (d + 1)
    return raw.reshape(n, d + 1)[:, 1:].copy()


# sift.tar.gz is fetched via `wget`, which already prints its own progress.


# def load_sift1m() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Return (base[1e6,128], query[1e4,128], gt[1e4,100])."""
#     CACHE_DIR.mkdir(parents=True, exist_ok=True)
#     base_path = CACHE_DIR / "sift_base.fvecs"
#     query_path = CACHE_DIR / "sift_query.fvecs"
#     gt_path = CACHE_DIR / "sift_groundtruth.ivecs"
#
#     if not (base_path.exists() and query_path.exists() and gt_path.exists()):
#         tar_path = CACHE_DIR / "sift.tar.gz"
#
#         # 1. Check if the existing tar is corrupted/incomplete
#         if tar_path.exists():
#             print(f"[sift] checking existing {tar_path}...")
#             try:
#                 with tarfile.open(tar_path, "r:gz") as tf:
#                     tf.getmembers()
#             except (EOFError, tarfile.ReadError):
#                 print("[sift] file corrupted, deleting and re-downloading...")
#                 tar_path.unlink()
#
#         # 2. Download via wget with extra robustness
#         if not tar_path.exists():
#             print(f"[sift] downloading {SIFT_URL} via wget...")
#             # --continue: resume partial downloads
#             # --tries: retry on connection resets
#             subprocess.run([
#                 "wget", "--continue", "--tries=10",
#                 "-O", str(tar_path), SIFT_URL
#             ], check=True)
#
#         # 3. Extract
#         print(f"[sift] extracting to {CACHE_DIR}")
#         with tarfile.open(tar_path, "r:gz") as tf:
#             for member in tf.getmembers():
#                 name = os.path.basename(member.name)
#                 if name in ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"):
#                     # We create a new TarInfo to avoid modifying original while iterating
#                     member.name = name
#                     tf.extract(member, CACHE_DIR)
#     else:
#         print(f"[sift] cached at {CACHE_DIR}")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _compute_gt_l2(base: np.ndarray, query: np.ndarray, gt_k: int) -> np.ndarray:
    base_sq = np.einsum("ij,ij->i", base, base)
    gt = np.empty((query.shape[0], gt_k), dtype=np.int32)
    chunk = 32
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        d = base_sq[None, :] - 2.0 * (q @ base.T)
        idx = np.argpartition(d, kth=gt_k, axis=1)[:, :gt_k]
        rows = np.arange(d.shape[0])[:, None]
        order = np.argsort(d[rows, idx], axis=1)
        gt[i:i + chunk] = idx[rows, order].astype(np.int32)
    return gt


def load_sift1m(metric: str = "l2") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (base[1e6,128], query[1e4,128], gt[1e4,100])."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_path = CACHE_DIR / "sift_base.fvecs"
    query_path = CACHE_DIR / "sift_query.fvecs"
    gt_path = CACHE_DIR / "sift_groundtruth.ivecs"

    if not (base_path.exists() and query_path.exists() and gt_path.exists()):
        tar_path = CACHE_DIR / "sift.tar.gz"
        if not tar_path.exists():
            subprocess.run(["wget", "-O", str(tar_path), SIFT_URL], check=True)
        print(f"[sift] extracting to {CACHE_DIR}")
        with tarfile.open(tar_path, "r:gz") as tf:
            for member in tf.getmembers():
                name = os.path.basename(member.name)
                if name in ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"):
                    member.name = name
                    tf.extract(member, CACHE_DIR, filter='data')
    else:
        print(f"[sift] cached at {CACHE_DIR}")

    print("[sift] reading vectors")
    base = _read_fvecs(base_path)
    query = _read_fvecs(query_path)
    gt = _read_ivecs(gt_path)

    if metric == "cosine":
        base = _l2_normalize(base)
        query = _l2_normalize(query)
        # cached gt was built for raw L2 on un-normalized vectors; rebuild for cosine.
        gt = _compute_gt_l2(base, query, gt_k=gt.shape[1])

    print(f"[sift] base={base.shape} query={query.shape} gt={gt.shape} metric={metric}")
    return base, query, gt


def load_dbpedia_single_file(file_path: str = "dbpedia_openai_100K_vectors.npy",
                             n_query: int = 100,
                             gt_k: int = 100,
                             metric: str = "l2") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"[dbpedia] loading {file_path} metric={metric}")
    data = np.load(file_path)

    dim = 1536
    n_vectors = data.size // dim
    data = data[:n_vectors * dim].reshape(n_vectors, dim)

    query = data[:n_query].astype(np.float32)
    base = data[n_query:].astype(np.float32)

    if metric == "cosine":
        base = _l2_normalize(base)
        query = _l2_normalize(query)

    print(f"[dbpedia] computing brute-force GT (k={gt_k}) on base={base.shape}")
    gt = _compute_gt_l2(base, query, gt_k)

    print(f"[dbpedia] split complete: base={base.shape}, query={query.shape}, gt={gt.shape}")
    return base, query, gt


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _time_min(fn, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def _topk_from_distances(dists: np.ndarray, k: int) -> np.ndarray:
    # dists: (n_query, n_base), lower = closer. argpartition is O(n).
    idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    # sort within top-k so recall-by-intersection is stable for any k'<=k
    rows = np.arange(dists.shape[0])[:, None]
    order = np.argsort(dists[rows, idx], axis=1)
    return idx[rows, order]


def _mton_topk_chunked(
    space, queries: np.ndarray, codes: np.ndarray, k: int, chunk: int,
    progress: bool = False,
) -> np.ndarray:
    """Chunked M-to-N + top-k aggregation. Returns (m, k) int32 indices.

    Avoids materializing the full (m, n) distance matrix: for SIFT1M with
    m=10k and n=1M that would be 40 GB. We stream by query chunks of size
    `chunk`, take top-k per chunk, and keep only the k indices per query.
    """
    m = queries.shape[0]
    out = np.empty((m, k), dtype=np.int32)
    it = range(0, m, chunk)
    if progress:
        it = tqdm(it, desc="recall M-to-N", leave=False, unit="chunk")
    for i in it:
        block = space.distance_m_to_n(queries[i : i + chunk], codes)
        top = _topk_from_distances(block, k=k)
        out[i : i + top.shape[0]] = top
    return out


def _mton_timed_chunked(
    space, queries: np.ndarray, codes: np.ndarray, chunk: int
) -> None:
    """Run chunked M-to-N, discard results. Used under _time_min."""
    m = queries.shape[0]
    for i in range(0, m, chunk):
        space.distance_m_to_n(queries[i : i + chunk], codes)


def _recall(pred_topk: np.ndarray, gt_topk: np.ndarray, k: int) -> float:
    n = pred_topk.shape[0]
    hits = 0
    gt_k = gt_topk[:, :k]
    for i in range(n):
        hits += np.intersect1d(pred_topk[i, :k], gt_k[i], assume_unique=False).size
    return hits / (n * k)


# ---------------------------------------------------------------------------
# Single-configuration run
# ---------------------------------------------------------------------------

def run_config(
    dataset: str,
    base: np.ndarray,
    query: np.ndarray,
    gt: np.ndarray | None,
    bits: int,
    num_threads: int,
    n_query_mton: int,
    repeats: int,
    query_chunk: int,
    metric: str = "l2",
) -> dict:
    dim = base.shape[1]
    n_base = base.shape[0]
    n_query = query.shape[0]

    space = TurboQuantSpace(dim, bits_per_coord=bits, num_threads=num_threads)

    # warmup: small batch
    warm = space.encode_batch(base[:512])
    space.distance_1_to_n(query[0], warm)
    space.distance_m_to_n(query[:8], warm)

    # encode
    t_enc = _time_min(lambda: space.encode_batch(base), repeats=repeats)
    codes = space.encode_batch(base)

    # 1-to-N (single query vs all base)
    t_1ton = _time_min(lambda: space.distance_1_to_n(query[0], codes), repeats=repeats)

    # M-to-N (subset of queries vs all base), chunked to cap peak memory at
    # ~ chunk * n_base * 4 bytes.
    q_sub = query[:n_query_mton]
    t_mton = _time_min(
        lambda: _mton_timed_chunked(space, q_sub, codes, query_chunk),
        repeats=repeats,
    )

    compression_ratio = (dim * 4) / space.code_size_bytes()

    row = {
        "dataset": dataset,
        "metric": metric,
        "n_base": n_base,
        "n_query": n_query,
        "dim": dim,
        "bits": bits,
        "num_threads": space.num_threads(),
        "encode_ms": t_enc * 1000,
        "encode_vps": n_base / t_enc,
        "query_1ton_ms": t_1ton * 1000,
        "query_1ton_vps": n_base / t_1ton,
        "query_mton_ms": t_mton * 1000,
        "query_mton_vps": (n_query_mton * n_base) / t_mton,
        "compression_ratio": compression_ratio,
        "recall_at_1": float("nan"),
        "recall_at_10": float("nan"),
        "recall_at_100": float("nan"),
    }

    if gt is not None:
        # Full M-to-N over all queries for recall (separate, not timed),
        # chunked so we never materialize the full (m, n) matrix.
        pred100 = _mton_topk_chunked(space, query, codes, k=100,
                                     chunk=query_chunk, progress=True)
        row["recall_at_1"] = _recall(pred100, gt, k=1)
        row["recall_at_10"] = _recall(pred100, gt, k=10)
        row["recall_at_100"] = _recall(pred100, gt, k=100)

    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    # 1. Threading scaling on M-to-N throughput — all datasets on one figure.
    scaling = df.copy()
    scaling["throughput_Mvps"] = scaling["query_mton_vps"] / 1e6
    scaling["series"] = scaling["dataset"] + " dim=" + scaling["dim"].astype(str)
    g = sns.relplot(
        data=scaling,
        x="num_threads",
        y="throughput_Mvps",
        hue="series",
        style="bits",
        kind="line",
        markers=True,
        dashes=False,
        height=4.5,
        aspect=1.4,
    )
    g.set_axis_labels("threads", "M-to-N throughput (M dist/s)")
    g.fig.suptitle("Threading scaling — distance_m_to_n (all datasets)", y=1.02)
    p = plots_dir / "threading_scaling_mton.png"
    g.fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)

    # 1b. Threading scaling on 1-to-N throughput (where parallelism is visible).
    scaling1 = df.copy()
    scaling1["throughput_Mvps"] = scaling1["query_1ton_vps"] / 1e6
    scaling1["series"] = scaling1["dataset"] + " dim=" + scaling1["dim"].astype(str)
    g1 = sns.relplot(
        data=scaling1,
        x="num_threads",
        y="throughput_Mvps",
        hue="series",
        style="bits",
        kind="line",
        markers=True,
        dashes=False,
        height=4.5,
        aspect=1.4,
    )
    g1.set_axis_labels("threads", "1-to-N throughput (M dist/s)")
    g1.fig.suptitle("Threading scaling — distance_1_to_n (all datasets)", y=1.02)
    p = plots_dir / "threading_scaling_1ton.png"
    g1.fig.savefig(p, dpi=150, bbox_inches="tight")
    saved.append(p)

    # 2. Recall vs bits — all real datasets on one figure.
    real = df[df["recall_at_10"].notna()].copy()
    if not real.empty:
        recall_df = (
            real[real["num_threads"] == real["num_threads"].min()]
            .melt(
                id_vars=["dataset", "bits"],
                value_vars=["recall_at_1", "recall_at_10", "recall_at_100"],
                var_name="k",
                value_name="recall",
            )
        )
        g3 = sns.catplot(
            data=recall_df, x="bits", y="recall", hue="k",
            col="dataset", kind="bar", height=4, aspect=1.1,
        )
        g3.set(ylim=(0, 1))
        g3.fig.suptitle("Recall vs bits per coord", y=1.02)
        p = plots_dir / "recall.png"
        g3.fig.savefig(p, dpi=150, bbox_inches="tight")
        saved.append(p)

    # 3. Encode/query throughput by dim (synthetic, threads=max)
    synth = df[df["dataset"] == "synthetic"].copy()
    if not synth.empty:
        max_t = synth["num_threads"].max()
        throughput = synth[synth["num_threads"] == max_t].melt(
            id_vars=["dim", "bits"],
            value_vars=["encode_vps", "query_1ton_vps", "query_mton_vps"],
            var_name="task",
            value_name="vps",
        )
        throughput["Mvps"] = throughput["vps"] / 1e6
        g2 = sns.catplot(
            data=throughput,
            x="dim",
            y="Mvps",
            hue="task",
            col="bits",
            kind="bar",
            height=4,
            aspect=1.1,
        )
        g2.set_axis_labels("dim", "throughput (M ops/s)")
        g2.fig.suptitle(f"Synthetic throughput by dim (threads={max_t})", y=1.02)
        p = plots_dir / "synthetic_throughput.png"
        g2.fig.savefig(p, dpi=150, bbox_inches="tight")
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=str, default="4,8",
                        help="comma-separated bits_per_coord values")
    parser.add_argument("--threads", type=str, default="1,2,4,8",
                        help="comma-separated num_threads values")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-base", type=int, default=None,
                        help="limit base vectors (SIFT1M is 1M)")
    parser.add_argument("--n-query-mton", type=int, default=128,
                        help="queries for timed M-to-N")
    parser.add_argument("--query-chunk", type=int, default=64,
                        help="chunk size for M-to-N streaming (peak mem ~ chunk*n_base*4B)")
    parser.add_argument("--synthetic-dims", type=str, default="128,512,1024")
    parser.add_argument("--synthetic-n-base", type=int, default=100_000)
    parser.add_argument("--synthetic-n-query", type=int, default=1_000)
    parser.add_argument("--skip-sift", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2",
                        help="distance metric; cosine applies L2-normalization "
                             "to base/query so existing L2^2 path becomes 2(1-cos)")
    parser.add_argument("--datasets", type=str, default="sift1m,dbpedia",
                        help="comma-separated real datasets to benchmark "
                             "(subset of: sift1m, dbpedia)")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open plot windows")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]
    threads_list = [int(t) for t in args.threads.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"results_{ts}.csv"
    plots_dir = args.output_dir / "plots"

    fieldnames = [
        "dataset", "metric", "n_base", "n_query", "dim", "bits", "num_threads",
        "encode_ms", "encode_vps",
        "query_1ton_ms", "query_1ton_vps",
        "query_mton_ms", "query_mton_vps",
        "compression_ratio",
        "recall_at_1", "recall_at_10", "recall_at_100",
    ]
    rows: list[dict] = []

    synth_dims = [int(d) for d in args.synthetic_dims.split(",")]
    real_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    valid = {"sift1m", "dbpedia"}
    bad = [d for d in real_datasets if d not in valid]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}; valid: {sorted(valid)}")
    total_cfgs = 0
    if not args.skip_sift:
        total_cfgs += len(real_datasets) * len(bits_list) * len(threads_list)
    if not args.skip_synthetic:
        total_cfgs += len(synth_dims) * len(bits_list) * len(threads_list)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        bar = tqdm(total=total_cfgs, desc="benchmark", unit="cfg")

        if not args.skip_sift:
            for ds_name in real_datasets:
                if ds_name == "sift1m":
                    base, query, gt = load_sift1m(metric=args.metric)
                else:
                    base, query, gt = load_dbpedia_single_file(metric=args.metric)
                full_base_n = base.shape[0]
                if args.n_base is not None:
                    base = base[: args.n_base]
                    gt_for_run = gt if args.n_base >= full_base_n else None
                else:
                    gt_for_run = gt
                for bits in bits_list:
                    for nt in threads_list:
                        bar.set_postfix_str(f"{ds_name} bits={bits} t={nt}")
                        row = run_config(
                            ds_name, base, query, gt_for_run,
                            bits=bits, num_threads=nt,
                            n_query_mton=args.n_query_mton,
                            repeats=args.repeats,
                            query_chunk=args.query_chunk,
                            metric=args.metric,
                        )
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                        bar.write(
                            f"[{ds_name} b={bits} t={nt}] encode={row['encode_ms']:.1f}ms "
                            f"1-to-N={row['query_1ton_ms']:.1f}ms "
                            f"M-to-N={row['query_mton_ms']:.1f}ms "
                            f"recall@10={row['recall_at_10']}"
                        )
                        bar.update(1)

        if not args.skip_synthetic:
            rng = np.random.default_rng(0)
            for dim in synth_dims:
                base = rng.standard_normal(
                    (args.synthetic_n_base, dim)).astype(np.float32)
                query = rng.standard_normal(
                    (args.synthetic_n_query, dim)).astype(np.float32)
                for bits in bits_list:
                    for nt in threads_list:
                        bar.set_postfix_str(f"synth dim={dim} bits={bits} t={nt}")
                        row = run_config(
                            "synthetic", base, query, None,
                            bits=bits, num_threads=nt,
                            n_query_mton=min(args.n_query_mton, args.synthetic_n_query),
                            repeats=args.repeats,
                            query_chunk=args.query_chunk,
                            metric=args.metric,
                        )
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                        bar.write(
                            f"[synth d={dim} b={bits} t={nt}] "
                            f"encode={row['encode_ms']:.1f}ms "
                            f"1-to-N={row['query_1ton_ms']:.1f}ms "
                            f"M-to-N={row['query_mton_ms']:.1f}ms"
                        )
                        bar.update(1)

        bar.close()

    print(f"\n[csv] wrote {csv_path}")

    df = pd.DataFrame(rows)
    saved = make_plots(df, plots_dir)
    for p in saved:
        print(f"[plot] {p}")

    if not args.no_show:
        try:
            plt.show()
        except Exception as e:
            print(f"[plot] cannot show windows: {e}")


if __name__ == "__main__":
    main()

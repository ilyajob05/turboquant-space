"""TurboQuant benchmark — end-to-end: downloads all datasets, runs recall/throughput
matrix, writes CSV, produces seaborn plots.

Default data directory: <repo>/python/benchmarks/data/
Override with --data-dir.

Usage:
    uv run python python/benchmarks/run_benchmark.py
    uv run python python/benchmarks/run_benchmark.py --datasets sift1m,dbpedia --bits 4,8
    uv run python python/benchmarks/run_benchmark.py --data-dir /mnt/data
"""

from __future__ import annotations

import argparse
import csv
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
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
DEFAULT_DATA_DIR = _HERE / "data"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, desc: str) -> None:
    """Download url → dest with a tqdm progress bar. Resumes if partial."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={existing}-"})
    try:
        resp = urllib.request.urlopen(req)
        total = existing + int(resp.headers.get("Content-Length", 0))
        mode = "ab"
    except Exception:
        resp = urllib.request.urlopen(url)
        total = int(resp.headers.get("Content-Length", 0))
        mode = "wb"
        existing = 0

    with open(dest, mode) as f, tqdm(
        total=total, initial=existing, unit="B", unit_scale=True, desc=desc
    ) as bar:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


# ---------------------------------------------------------------------------
# SIFT1M
# ---------------------------------------------------------------------------

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


def _read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def _read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def load_sift1m(data_dir: Path, metric: str = "l2") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = data_dir / "sift"
    base_path  = cache / "sift_base.fvecs"
    query_path = cache / "sift_query.fvecs"
    gt_path    = cache / "sift_groundtruth.ivecs"

    if not (base_path.exists() and query_path.exists() and gt_path.exists()):
        tar = cache / "sift.tar.gz"
        if not tar.exists():
            _download(SIFT_URL, tar, "sift.tar.gz")
        print(f"[sift] extracting to {cache}")
        with tarfile.open(tar, "r:gz") as tf:
            for m in tf.getmembers():
                name = Path(m.name).name
                if name in ("sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs"):
                    m.name = name
                    tf.extract(m, cache, filter="data")
    else:
        print(f"[sift] cached at {cache}")

    base  = _read_fvecs(base_path)
    query = _read_fvecs(query_path)
    gt    = _read_ivecs(gt_path)

    if metric == "cosine":
        base, query = _l2_normalize(base), _l2_normalize(query)
        gt = _compute_gt_l2(base, query, gt.shape[1])

    print(f"[sift] base={base.shape} query={query.shape} gt={gt.shape}")
    return base, query, gt


# ---------------------------------------------------------------------------
# HuggingFace datasets (streaming → cached .npy)
# ---------------------------------------------------------------------------

HF_DATASETS: dict[str, dict] = {
    "dbpedia": {
        "repo": "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
        "config": None, "split": "train",
        "emb_field": "text-embedding-3-small-1536-embedding",
        "dim": 1536, "default_limit": 100_000, "source_key": "dbpedia",
    },
    "beir-msmarco": {
        "repo": "CohereLabs/beir-embed-english-v3",
        "config": "msmarco-corpus", "split": "train",
        "emb_field": "emb",
        "dim": 1024, "default_limit": 100_000, "source_key": "beir-msmarco",
    },
    "openai-v3-small": {
        "repo": "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
        "config": None, "split": "train",
        "emb_field": "text-embedding-3-small-1536-embedding",
        "dim": 1536, "default_limit": 100_000, "source_key": "openai-v3-small",
    },
    "openai-v3-large": {
        "repo": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-100K",
        "config": None, "split": "train",
        "emb_field": "text-embedding-3-large-3072-embedding",
        "dim": 3072, "default_limit": 10_000, "source_key": "openai-v3-large",
    },
    # Matryoshka truncations — reuse openai-v3-large cache
    "openai-v3-large-512":  {"repo": None, "config": None, "split": None, "emb_field": None,
                              "dim": 512,  "default_limit": 10_000, "source_key": "openai-v3-large"},
    "openai-v3-large-1024": {"repo": None, "config": None, "split": None, "emb_field": None,
                              "dim": 1024, "default_limit": 10_000, "source_key": "openai-v3-large"},
    "openai-v3-large-1536": {"repo": None, "config": None, "split": None, "emb_field": None,
                              "dim": 1536, "default_limit": 10_000, "source_key": "openai-v3-large"},
}


def _hf_cache_path(data_dir: Path, source_key: str, limit: int) -> Path:
    return data_dir / "hf" / source_key / f"embeddings_{limit}.npy"


def _download_hf(spec: dict, limit: int, cache_path: Path) -> np.ndarray:
    from datasets import load_dataset
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_str = f" [{spec['config']}]" if spec["config"] else ""
    print(f"[hf] streaming {spec['repo']}{cfg_str} limit={limit}")
    kwargs = {"split": spec["split"], "streaming": True}
    if spec["config"]:
        stream = load_dataset(spec["repo"], spec["config"], **kwargs)
    else:
        stream = load_dataset(spec["repo"], **kwargs)
    embs = np.empty((limit, spec["dim"]), dtype=np.float32)
    with tqdm(total=limit, desc=f"  {spec['repo'].split('/')[-1]}", unit="vec") as bar:
        for i, ex in enumerate(stream):
            if i >= limit:
                break
            embs[i] = np.asarray(ex[spec["emb_field"]], dtype=np.float32)
            bar.update(1)
    if i + 1 < limit:
        embs = embs[:i + 1]
    np.save(cache_path, embs)
    print(f"[hf] saved {cache_path} shape={embs.shape}")
    return embs


def load_hf_dataset(
    key: str, data_dir: Path,
    limit: int | None = None, n_query: int = 100,
    gt_k: int = 100, metric: str = "l2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec       = HF_DATASETS[key]
    source_key = spec["source_key"]
    src_spec   = HF_DATASETS[source_key]
    if limit is None:
        limit = src_spec["default_limit"]

    cache_path = _hf_cache_path(data_dir, source_key, limit)
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"[hf] {key}: loaded {data.shape} from cache")
    else:
        data = _download_hf(src_spec, limit, cache_path)

    # Matryoshka truncation
    target_dim = spec["dim"]
    if data.shape[1] != target_dim:
        data = data[:, :target_dim].copy()
        print(f"[hf] {key}: truncated to dim={target_dim}")

    n_actual = data.shape[0]
    if n_query >= n_actual:
        raise ValueError(f"{key}: n_query={n_query} >= dataset size {n_actual}")

    query = data[:n_query].astype(np.float32, copy=False)
    base  = data[n_query:].astype(np.float32, copy=False)

    if metric == "cosine":
        base, query = _l2_normalize(base), _l2_normalize(query)

    print(f"[hf] {key}: computing GT (k={gt_k}) on base={base.shape}")
    gt = _compute_gt_l2(base, query, gt_k)
    print(f"[hf] {key}: base={base.shape} query={query.shape} gt={gt.shape}")
    return base, query, gt


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def _compute_gt_l2(base: np.ndarray, query: np.ndarray, gt_k: int) -> np.ndarray:
    """Brute-force L2 ground truth, chunked to ~512 MB peak RAM."""
    base_sq = np.einsum("ij,ij->i", base, base)
    nq = query.shape[0]
    gt = np.empty((nq, gt_k), dtype=np.int32)
    chunk = 128
    for i in range(0, nq, chunk):
        q = query[i:i + chunk]
        d = base_sq - 2.0 * (q @ base.T)
        part = np.argpartition(d, kth=gt_k, axis=1)[:, :gt_k]
        d_part = d[np.arange(d.shape[0])[:, None], part]
        order = np.argsort(d_part, axis=1)
        gt[i:i + q.shape[0]] = part[np.arange(q.shape[0])[:, None], order].astype(np.int32)
    return gt


def _topk_from_distances(dists: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    rows = np.arange(dists.shape[0])[:, None]
    order = np.argsort(dists[rows, idx], axis=1)
    return idx[rows, order]


def _mton_topk_chunked(space, queries, codes, k, chunk, progress=False):
    m = queries.shape[0]
    out = np.empty((m, k), dtype=np.int32)
    it = range(0, m, chunk)
    if progress:
        it = tqdm(it, desc="recall M-to-N", leave=False, unit="chunk")
    for i in it:
        block = space.distance_m_to_n(queries[i:i + chunk], codes)
        out[i:i + block.shape[0]] = _topk_from_distances(block, k=k)
    return out


def _mton_timed_chunked(space, queries, codes, chunk):
    for i in range(0, queries.shape[0], chunk):
        space.distance_m_to_n(queries[i:i + chunk], codes)


def _recall(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    p = pred[:, :k, None].astype(np.int32)   # (n, k, 1)
    g = gt[:, None, :k].astype(np.int32)     # (n, 1, k)
    return int((p == g).any(axis=2).sum()) / (pred.shape[0] * k)


def _time_min(fn, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# Single-configuration benchmark
# ---------------------------------------------------------------------------

def run_config(
    dataset: str, base: np.ndarray, query: np.ndarray, gt: np.ndarray | None,
    bits: int, num_threads: int, n_query_mton: int,
    repeats: int, query_chunk: int, metric: str = "l2",
) -> dict:
    n_base, dim = base.shape
    space = TurboQuantSpace(dim, bits_per_coord=bits, num_threads=num_threads)

    # warmup
    warm = space.encode_batch(base[:512])
    space.distance_1_to_n(query[0], warm)
    space.distance_m_to_n(query[:8], warm)

    codes = None
    def _enc():
        nonlocal codes
        codes = space.encode_batch(base)
    t_enc = _time_min(_enc, repeats=repeats)

    t_1ton = _time_min(lambda: space.distance_1_to_n(query[0], codes), repeats=repeats)

    q_sub = query[:n_query_mton]
    t_mton = _time_min(
        lambda: _mton_timed_chunked(space, q_sub, codes, query_chunk),
        repeats=repeats,
    )

    row = {
        "dataset": dataset, "metric": metric,
        "n_base": n_base, "n_query": query.shape[0], "dim": dim, "bits": bits,
        "num_threads": space.num_threads(),
        "encode_ms": t_enc * 1000,        "encode_vps": n_base / t_enc,
        "query_1ton_ms": t_1ton * 1000,   "query_1ton_vps": n_base / t_1ton,
        "query_mton_ms": t_mton * 1000,   "query_mton_vps": (n_query_mton * n_base) / t_mton,
        "compression_ratio": (dim * 4) / space.code_size_bytes(),
        "recall_at_1": float("nan"), "recall_at_10": float("nan"), "recall_at_100": float("nan"),
    }

    if gt is not None:
        pred100 = _mton_topk_chunked(space, query, codes, k=100,
                                     chunk=query_chunk, progress=True)
        row["recall_at_1"]   = _recall(pred100, gt, k=1)
        row["recall_at_10"]  = _recall(pred100, gt, k=10)
        row["recall_at_100"] = _recall(pred100, gt, k=100)

    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved: list[Path] = []

    df["series"] = df["dataset"] + " dim=" + df["dim"].astype(str)

    # 1a. M-to-N threading scaling
    for col, ylabel, fname in [
        ("query_mton_vps", "M-to-N throughput (M dist/s)", "threading_scaling_mton.png"),
        ("query_1ton_vps", "1-to-N throughput (M dist/s)", "threading_scaling_1ton.png"),
    ]:
        data = df.copy()
        data["Mvps"] = data[col] / 1e6
        g = sns.relplot(data=data, x="num_threads", y="Mvps",
                        hue="series", style="bits", kind="line",
                        markers=True, dashes=False, height=4.5, aspect=1.4)
        g.set_axis_labels("threads", ylabel)
        p = plots_dir / fname
        g.fig.savefig(p, dpi=150, bbox_inches="tight")
        saved.append(p)
        plt.close(g.fig)

    # 2. Recall vs bits per dataset
    real = df[df["recall_at_10"].notna()].copy()
    if not real.empty:
        recall_df = (
            real[real["num_threads"] == real["num_threads"].min()]
            .melt(id_vars=["dataset", "bits"],
                  value_vars=["recall_at_1", "recall_at_10", "recall_at_100"],
                  var_name="k", value_name="recall")
        )
        g3 = sns.catplot(data=recall_df, x="bits", y="recall", hue="k",
                         col="dataset", kind="bar", height=4, aspect=1.1, col_wrap=4)
        g3.set(ylim=(0, 1))
        g3.fig.suptitle("Recall vs bits per coord", y=1.02)
        p = plots_dir / "recall.png"
        g3.fig.savefig(p, dpi=150, bbox_inches="tight")
        saved.append(p)
        plt.close(g3.fig)

    # 3. Synthetic throughput by dim
    synth = df[df["dataset"] == "synthetic"].copy()
    if not synth.empty:
        max_t = synth["num_threads"].max()
        tp = (synth[synth["num_threads"] == max_t]
              .melt(id_vars=["dim", "bits"],
                    value_vars=["encode_vps", "query_1ton_vps", "query_mton_vps"],
                    var_name="task", value_name="vps"))
        tp["Mvps"] = tp["vps"] / 1e6
        g2 = sns.catplot(data=tp, x="dim", y="Mvps", hue="task",
                         col="bits", kind="bar", height=4, aspect=1.1)
        g2.set_axis_labels("dim", "throughput (M ops/s)")
        g2.fig.suptitle(f"Synthetic throughput (threads={max_t})", y=1.02)
        p = plots_dir / "synthetic_throughput.png"
        g2.fig.savefig(p, dpi=150, bbox_inches="tight")
        saved.append(p)
        plt.close(g2.fig)

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="directory for cached datasets (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=_HERE / "results")
    parser.add_argument("--datasets", type=str,
                        default="sift1m,dbpedia,beir-msmarco,openai-v3-small,"
                                "openai-v3-large,openai-v3-large-512,"
                                "openai-v3-large-1024,openai-v3-large-1536",
                        help="comma-separated: sift1m + any key from HF_DATASETS")
    parser.add_argument("--bits",          type=str, default="4,8")
    parser.add_argument("--threads",       type=str, default="1,2,4,8")
    parser.add_argument("--repeats",       type=int, default=5)
    parser.add_argument("--n-base",        type=int, default=None,
                        help="cap base vectors per dataset")
    parser.add_argument("--n-query-mton",  type=int, default=128,
                        help="queries for timed M-to-N")
    parser.add_argument("--query-chunk",   type=int, default=64,
                        help="chunk size for M-to-N streaming")
    parser.add_argument("--hf-limit",      type=int, default=None,
                        help="override per-dataset HF download limit")
    parser.add_argument("--synthetic-dims",    type=str, default="128,512,1024")
    parser.add_argument("--synthetic-n-base",  type=int, default=100_000)
    parser.add_argument("--synthetic-n-query", type=int, default=1_000)
    parser.add_argument("--skip-real",      action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bits_list    = [int(b) for b in args.bits.split(",")]
    threads_list = [int(t) for t in args.threads.split(",")]
    synth_dims   = [int(d) for d in args.synthetic_dims.split(",")]
    real_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    valid = {"sift1m"} | set(HF_DATASETS.keys())
    bad = [d for d in real_datasets if d not in valid]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}; valid: {sorted(valid)}")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"results_{ts}.csv"
    plots_dir = args.output_dir / "plots"

    fieldnames = [
        "dataset", "metric", "n_base", "n_query", "dim", "bits", "num_threads",
        "encode_ms", "encode_vps", "query_1ton_ms", "query_1ton_vps",
        "query_mton_ms", "query_mton_vps", "compression_ratio",
        "recall_at_1", "recall_at_10", "recall_at_100",
    ]
    rows: list[dict] = []

    total_cfgs = 0
    if not args.skip_real:
        total_cfgs += len(real_datasets) * len(bits_list) * len(threads_list)
    if not args.skip_synthetic:
        total_cfgs += len(synth_dims) * len(bits_list) * len(threads_list)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        bar = tqdm(total=total_cfgs, desc="benchmark", unit="cfg")

        if not args.skip_real:
            for ds_name in real_datasets:
                if ds_name == "sift1m":
                    base, query, gt = load_sift1m(args.data_dir, metric=args.metric)
                else:
                    base, query, gt = load_hf_dataset(
                        ds_name, args.data_dir,
                        limit=args.hf_limit, metric=args.metric,
                    )
                full_n = base.shape[0]
                if args.n_base is not None:
                    base = base[:args.n_base]
                    gt_for_run = gt if args.n_base >= full_n else None
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
                            f"[{ds_name} b={bits} t={nt}]  "
                            f"encode={row['encode_ms']:.1f}ms  "
                            f"1toN={row['query_1ton_ms']:.1f}ms  "
                            f"MtoN={row['query_mton_ms']:.1f}ms  "
                            f"recall@10={row['recall_at_10']}"
                        )
                        bar.update(1)

        if not args.skip_synthetic:
            rng = np.random.default_rng(0)
            for dim in synth_dims:
                base  = rng.standard_normal((args.synthetic_n_base,  dim)).astype(np.float32)
                query = rng.standard_normal((args.synthetic_n_query, dim)).astype(np.float32)
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
                            f"[synth d={dim} b={bits} t={nt}]  "
                            f"encode={row['encode_ms']:.1f}ms  "
                            f"1toN={row['query_1ton_ms']:.1f}ms  "
                            f"MtoN={row['query_mton_ms']:.1f}ms"
                        )
                        bar.update(1)

        bar.close()

    print(f"\n[csv] {csv_path}")
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

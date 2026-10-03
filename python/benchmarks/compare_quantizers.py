#!/usr/bin/env python3
"""Comparative accuracy and throughput for TurboQuant and RaBitQ.

One process draws a shared float32 N(0, 1) matrix per dimension. Every
method is scored on that same matrix, so rows differ by quantizer and bit
width only.

Accuracy (n_base vectors, n_query queries):
  exact squared L2 is ||q - x||^2 in float64.
  recall@k is the overlap of the k nearest codes, ranked by the quantizer's
  asymmetric distance, with the k nearest under exact squared L2.
  rel_mae is the mean of |d_hat - ||q-x||^2| / ||q-x||^2 over those pairs.
  Pairs with exact distance <= 1e-8 are left out of rel_mae.

Throughput (a separate draw of n_speed vectors, one query, min of --repeats
after one warmup):
  both encodes       -> encode_batch
  both searches      -> distance_1_to_n
  TurboQuant is constructed with num_threads=1. RaBitQ encode_batch is
  one serial C++ pass: encode can throw, so it does not use the OpenMP
  pool TurboQuant uses for Lloyd-Max.
  Both spaces use rot_seed (default 42). TurboQuant keeps its default qjl_seed.

Bits. TurboQuant: 4 or 8. RaBitQ: 1, 4, or 8.
The default grid is dims {128, 1024} and every legal method/bits pair.
--quick is a fixed smoke preset and refuses an explicit size flag.

Outputs, under --out-dir:
  compare_<UTC timestamp>.csv
  compare_<UTC timestamp>.md
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from turboquant import RaBitQSpace, TurboQuantSpace

_ALLOWED_BITS = {
    "turboquant": (4, 8),
    "rabitq": (1, 4, 8),
}
_EXACT_FLOOR = 1e-8
_QUICK_CONFLICTS = (
    "--dims",
    "--methods",
    "--n-base",
    "--n-query",
    "--n-speed",
    "--repeats",
    "--k",
)

_CSV_FIELDS = (
    "timestamp",
    "host",
    "python",
    "package_version",
    "method",
    "bits",
    "dim",
    "padded_dim",
    "code_bytes",
    "float32_bytes",
    "compression_ratio",
    "n_base",
    "n_query",
    "n_speed",
    "k",
    "seed",
    "rot_seed",
    "num_threads",
    "recall_at_1",
    "recall_at_10",
    "rel_mae",
    "rel_mae_pairs",
    "encode_api",
    "encode_vps",
    "encode_us",
    "search_api",
    "search_pairs_per_s",
    "distance_kernel",
)


@dataclass(frozen=True)
class MethodSpec:
    """One row of the comparison grid."""

    name: str
    bits: int

    def encode_api(self) -> str:
        return "encode_batch"

    def search_api(self) -> str:
        return "distance_1_to_n"


def parse_methods(text: str) -> list[MethodSpec]:
    """Parse 'turboquant:4,rabitq:1'. Empty and unknown tokens raise ValueError."""
    specs: list[MethodSpec] = []
    if not text.strip():
        raise ValueError("methods is empty")
    for token in text.split(","):
        piece = token.strip().lower()
        if ":" not in piece:
            raise ValueError(
                f"method {token!r} must look like turboquant:4 or rabitq:1"
            )
        name, bits_text = piece.split(":", 1)
        if name not in _ALLOWED_BITS:
            raise ValueError(
                f"unknown method {name!r}; expected turboquant or rabitq"
            )
        try:
            bits = int(bits_text)
        except ValueError as exc:
            raise ValueError(f"bits in {token!r} are not an integer") from exc
        legal = _ALLOWED_BITS[name]
        if bits not in legal:
            raise ValueError(
                f"{name} bits must be one of {legal}, got {bits}"
            )
        specs.append(MethodSpec(name, bits))
    return specs


def parse_dims(text: str) -> list[int]:
    """Parse '128,1024'. Each dim must be a positive integer."""
    if not text.strip():
        raise ValueError("dims is empty")
    dims: list[int] = []
    for token in text.split(","):
        try:
            dim = int(token.strip())
        except ValueError as exc:
            raise ValueError(f"dim {token!r} is not an integer") from exc
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        dims.append(dim)
    return dims


def draw_matrix(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """float32 N(0, 1) with shape (n, dim), C-contiguous."""
    if n <= 0:
        raise ValueError(f"row count must be positive, got {n}")
    return rng.standard_normal((n, dim)).astype(np.float32, copy=False)


def squared_l2(base: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Exact squared L2, shape (n_query, n_base), float64.

    base is (n_base, dim) and query is (n_query, dim). Both are finite.
    """
    if base.ndim != 2 or query.ndim != 2:
        raise ValueError(
            f"squared_l2 expects matrices, got {base.shape} and {query.shape}"
        )
    if base.shape[1] != query.shape[1]:
        raise ValueError(
            f"dim mismatch: base {base.shape[1]} vs query {query.shape[1]}"
        )
    b = np.asarray(base, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    if not np.isfinite(b).all() or not np.isfinite(q).all():
        raise ValueError("squared_l2 input has a non-finite value")
    base_sq = np.einsum("nd,nd->n", b, b)
    query_sq = np.einsum("md,md->m", q, q)
    return query_sq[:, None] + base_sq[None, :] - 2.0 * (q @ b.T)


def topk_indices(dist: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances per row, nearest first. int32."""
    if dist.ndim != 2:
        raise ValueError(f"topk expects a matrix, got {dist.shape}")
    n = dist.shape[1]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in 1..{n - 1}, got {k}")
    part = np.argpartition(dist, kth=k, axis=1)[:, :k]
    rows = np.arange(dist.shape[0])[:, None]
    order = np.argsort(dist[rows, part], axis=1)
    return part[rows, order].astype(np.int32, copy=False)


def recall_at(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Fraction of true top-k ids present in the predicted top-k."""
    if pred.shape != gt.shape:
        raise ValueError(f"recall shape {pred.shape} vs {gt.shape}")
    if k > pred.shape[1]:
        raise ValueError(f"recall k={k} exceeds stored {pred.shape[1]}")
    hits = (pred[:, :k, None] == gt[:, None, :k]).any(axis=2).sum()
    return float(hits) / (pred.shape[0] * k)


def relative_mae(estimate: np.ndarray, exact: np.ndarray) -> tuple[float, int]:
    """Mean |estimate - exact| / exact where exact > 1e-8. Returns (mae, n)."""
    if estimate.shape != exact.shape:
        raise ValueError(
            f"rel_mae shape {estimate.shape} vs {exact.shape}"
        )
    err = np.abs(estimate.astype(np.float64) - exact)
    mask = exact > _EXACT_FLOOR
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    return float((err[mask] / exact[mask]).mean()), count


def time_min(fn, repeats: int) -> float:
    """Seconds for the fastest of `repeats` calls after one untimed warmup."""
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    fn()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    if not np.isfinite(best) or best <= 0.0:
        raise RuntimeError(f"non-positive timing {best}")
    return best


class TurboQuantRunner:
    """Batch encode and 1-to-N search. num_threads is fixed at 1."""

    encode_api = "encode_batch"
    search_api = "distance_1_to_n"

    def __init__(self, dim: int, bits: int, rot_seed: int) -> None:
        self.space = TurboQuantSpace(
            dim, bits_per_coord=bits, rot_seed=rot_seed, num_threads=1
        )
        self.bits = bits

    @property
    def padded_dim(self) -> int:
        return int(self.space.padded_dim())

    @property
    def code_bytes(self) -> int:
        return int(self.space.code_size_bytes())

    @property
    def num_threads(self) -> int:
        return int(self.space.num_threads())

    @property
    def distance_kernel(self) -> str:
        return "distance_1_to_n"

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        codes = self.space.encode_batch(vectors)
        expected = (vectors.shape[0], self.code_bytes)
        if codes.shape != expected or codes.dtype != np.uint8:
            raise RuntimeError(
                f"encode_batch returned {codes.shape} {codes.dtype}, "
                f"expected {expected} uint8"
            )
        return codes

    def distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        out = np.asarray(self.space.distance_1_to_n(query, codes), dtype=np.float64)
        if out.shape != (codes.shape[0],):
            raise RuntimeError(
                f"distance_1_to_n returned {out.shape}, expected {(codes.shape[0],)}"
            )
        return out


class RaBitQRunner:
    """Encode and search each take one C++ pass over the rows."""

    encode_api = "encode_batch"
    search_api = "distance_1_to_n"

    def __init__(self, dim: int, bits: int, rot_seed: int) -> None:
        self.space = RaBitQSpace(dim, rot_seed=rot_seed, bits=bits)
        self.bits = bits

    @property
    def padded_dim(self) -> int:
        return int(self.space.padded_dim())

    @property
    def code_bytes(self) -> int:
        return int(self.space.code_size_bytes())

    @property
    def num_threads(self) -> int:
        return 1

    @property
    def distance_kernel(self) -> str:
        return str(self.space.distance_kernel())

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        codes = np.asarray(self.space.encode_batch(vectors), dtype=np.uint8)
        expected = (vectors.shape[0], self.code_bytes)
        if codes.shape != expected:
            raise RuntimeError(f"encode_batch returned {codes.shape}, expected {expected}")
        return codes

    def distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        out = np.asarray(self.space.distance_1_to_n(query, codes), dtype=np.float64)
        if out.shape != (codes.shape[0],):
            raise RuntimeError(
                f"distance_1_to_n returned {out.shape}, expected {(codes.shape[0],)}"
            )
        return out


def make_runner(spec: MethodSpec, dim: int, rot_seed: int):
    if spec.name == "turboquant":
        return TurboQuantRunner(dim, spec.bits, rot_seed)
    return RaBitQRunner(dim, spec.bits, rot_seed)


def score_method(
    spec: MethodSpec,
    dim: int,
    base: np.ndarray,
    query: np.ndarray,
    speed_base: np.ndarray,
    speed_query: np.ndarray,
    k: int,
    repeats: int,
    rot_seed: int,
    seed: int,
) -> dict:
    """Accuracy on `base`/`query`, throughput on the speed draw."""
    runner = make_runner(spec, dim, rot_seed)
    if runner.padded_dim < dim:
        raise RuntimeError(
            f"padded_dim {runner.padded_dim} is below input dim {dim}"
        )

    codes = runner.encode(base)
    exact = squared_l2(base, query)
    estimate = np.empty_like(exact)
    for i in range(query.shape[0]):
        estimate[i] = runner.distances(query[i], codes)
    if not np.isfinite(estimate).all():
        raise RuntimeError(f"{spec.name}:{spec.bits} produced a non-finite distance")

    gt = topk_indices(exact, k)
    pred = topk_indices(estimate, k)
    rel, rel_n = relative_mae(estimate, exact)

    encode_s = time_min(lambda: runner.encode(speed_base), repeats)
    speed_codes = runner.encode(speed_base)
    search_s = time_min(
        lambda: runner.distances(speed_query, speed_codes), repeats
    )
    n_speed = speed_base.shape[0]
    float32_bytes = dim * 4
    code_bytes = runner.code_bytes

    return {
        "method": spec.name,
        "bits": spec.bits,
        "dim": dim,
        "padded_dim": runner.padded_dim,
        "code_bytes": code_bytes,
        "float32_bytes": float32_bytes,
        "compression_ratio": float32_bytes / code_bytes,
        "n_base": base.shape[0],
        "n_query": query.shape[0],
        "n_speed": n_speed,
        "k": k,
        "seed": seed,
        "rot_seed": rot_seed,
        "num_threads": runner.num_threads,
        "recall_at_1": recall_at(pred, gt, 1),
        "recall_at_10": recall_at(pred, gt, min(10, k)),
        "rel_mae": rel,
        "rel_mae_pairs": rel_n,
        "encode_api": runner.encode_api,
        "encode_vps": n_speed / encode_s,
        "encode_us": encode_s * 1e6 / n_speed,
        "search_api": runner.search_api,
        "search_pairs_per_s": n_speed / search_s,
        "distance_kernel": runner.distance_kernel,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in _CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict], stamp: str, host: str) -> None:
    lines = [
        f"# Quantizer comparison {stamp}",
        "",
        f"Host `{host}`, Python {rows[0]['python']}, "
        f"turboquant {rows[0]['package_version']}.",
        "",
        "Shared float32 N(0, 1) draw per dimension. "
        f"Accuracy uses n_base={rows[0]['n_base']}, n_query={rows[0]['n_query']}, "
        f"k={rows[0]['k']}. "
        f"Throughput uses n_speed={rows[0]['n_speed']} and the fastest of the timed repeats. "
        "recall is overlap with exact squared L2. "
        "rel_mae is the mean relative error of the asymmetric distance. "
        "Both search rates are `distance_1_to_n`: the query is prepared once, then every slot is scored in one C++ pass.",
        "",
        "| method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s | search pairs/s | search api |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {bits} | {dim} | {code_bytes} | {r1:.4f} | {r10:.4f} | "
            "{mae:.4f} | {enc:,.0f} | {search:,.0f} | {api} |".format(
                method=row["method"],
                bits=row["bits"],
                dim=row["dim"],
                code_bytes=row["code_bytes"],
                r1=row["recall_at_1"],
                r10=row["recall_at_10"],
                mae=row["rel_mae"],
                enc=row["encode_vps"],
                search=row["search_pairs_per_s"],
                api=row["search_api"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'method':<12} {'bits':>4} {'dim':>6} {'bytes':>6} "
        f"{'R@1':>7} {'R@10':>7} {'rel_mae':>8} "
        f"{'encode/s':>12} {'search/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<12} {row['bits']:4d} {row['dim']:6d} "
            f"{row['code_bytes']:6d} {row['recall_at_1']:7.4f} "
            f"{row['recall_at_10']:7.4f} {row['rel_mae']:8.4f} "
            f"{row['encode_vps']:12,.0f} {row['search_pairs_per_s']:12,.0f}"
        )


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="128,1024", help="comma-separated input dims")
    parser.add_argument(
        "--methods",
        default="turboquant:4,turboquant:8,rabitq:1,rabitq:4,rabitq:8",
        help="comma-separated method:bits",
    )
    parser.add_argument("--n-base", type=int, default=2000)
    parser.add_argument("--n-query", type=int, default=40)
    parser.add_argument("--n-speed", type=int, default=400)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1, help="data draw seed")
    parser.add_argument("--rot-seed", type=int, default=42, help="SRHT seed for both spaces")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=here / "results")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke preset: dim 128, turboquant:4, rabitq:1, rabitq:4, "
        "n_base=64, n_query=8, n_speed=32, repeats=1, k=10",
    )
    return parser


def apply_quick(args: argparse.Namespace) -> None:
    conflict = [flag for flag in _QUICK_CONFLICTS if flag in sys.argv]
    if conflict:
        raise SystemExit(
            "--quick is a fixed preset; drop " + ", ".join(conflict)
        )
    args.dims = "128"
    args.methods = "turboquant:4,rabitq:1,rabitq:4"
    args.n_base = 64
    args.n_query = 8
    args.n_speed = 32
    args.k = 10
    args.repeats = 1


def validate_sizes(n_base: int, n_query: int, n_speed: int, k: int) -> None:
    if n_base <= 0 or n_query <= 0 or n_speed <= 0:
        raise SystemExit("n-base, n-query, and n-speed must be positive")
    if k <= 0:
        raise SystemExit("k must be positive")
    if k >= n_base:
        raise SystemExit(f"k={k} must be smaller than n-base={n_base}")
    if k < 10:
        raise SystemExit("k must be at least 10 so recall@10 is defined")


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        apply_quick(args)
    try:
        dims = parse_dims(args.dims)
        methods = parse_methods(args.methods)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validate_sizes(args.n_base, args.n_query, args.n_speed, args.k)
    if args.rot_seed < 0:
        raise SystemExit("rot-seed must be non-negative")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = f"{platform.system()} {platform.machine()}"
    py = platform.python_version()
    try:
        import turboquant

        version = turboquant.__version__
    except Exception:
        version = "unknown"

    rows: list[dict] = []
    for dim in dims:
        rng = np.random.default_rng(args.seed + dim)
        base = draw_matrix(args.n_base, dim, rng)
        query = draw_matrix(args.n_query, dim, rng)
        speed_base = draw_matrix(args.n_speed, dim, rng)
        speed_query = draw_matrix(1, dim, rng)[0]
        for spec in methods:
            print(f"running {spec.name}:{spec.bits} dim={dim}", flush=True)
            row = score_method(
                spec, dim, base, query, speed_base, speed_query,
                args.k, args.repeats, args.rot_seed, args.seed,
            )
            row["timestamp"] = stamp
            row["host"] = host
            row["python"] = py
            row["package_version"] = version
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"compare_{stamp}.csv"
    md_path = args.out_dir / f"compare_{stamp}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, stamp, host)
    print()
    print_table(rows)
    print()
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()

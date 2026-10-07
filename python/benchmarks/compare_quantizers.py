#!/usr/bin/env python3
"""Comparative accuracy and throughput for TurboQuant and RaBitQ.

Every method is scored on the same datasets, so rows differ by quantizer,
bit width and search API only:
  gaussN   float32 N(0, 1) draw per --dims entry (isotropic, no structure)
  --data   real vectors from a (rows, dim) .npy: base = first n_base rows,
           queries = last n_query rows (held out from the base)

Configurations (each quantizer at its most accurate setting by default):
  TurboQuant  library defaults: IVF centering with 256 k-means centroids
              fitted on the base, corrected estimator, no QJL.
  RaBitQ      space.train(base) centers codes on the base mean
              (--rabitq-centroid mean), float queries for 1-bit codes
              (--rabitq-query-bits 0), and bare `rabitq:4/8` keeps the
              library default encode mode, windowed_scale, which matches the
              Algorithm 1 sweep's accuracy at a fraction of its encode cost.
  Both spaces use num_threads=1 and rot_seed (default 42); training and
  centroid fitting happen once, outside any timed region.

Methods (method:bits):
  turboquant:4|8             per-pair distance_1_to_n
  turboquant-fastscan:4      TurboQuantFastScan.search, k*--rerank re-scored
  rabitq:1|4|8               per-pair distance_1_to_n, library default mode
  rabitq-fixed-scale:4|8     static: one scale frozen from N(0,1); fastest
  rabitq-windowed-scale:4|8  one scale per vector inside the tight window
  rabitq-trained-scale:4|8   one scale calibrated by train() on the base
  rabitq-algorithm1:1|4|8    bit-exact Extended RaBitQ sweep (reference)
  rabitq-fastscan:1|4|8      RaBitQFastScan.search, 1-bit stage + --eps0
                             bound, refinement with the full code

Accuracy (n_base codes, n_query queries):
  exact squared L2 is ||q - x||^2 in float64.
  recall@k is the overlap of the predicted top-k with the exact top-k.
  Per-pair rows rank all n_base estimates; FastScan rows use search().
  rel_mae is the mean of |d_hat - ||q-x||^2| / ||q-x||^2 (per-pair rows
  only; NaN for FastScan). Pairs with exact distance <= 1e-8 are skipped.

Throughput (n_speed codes, one query, min of --repeats after a warmup):
  encode_vps          vectors/s through encode_batch (FastScan: index build)
  search_pairs_per_s  n_speed / time of one query against n_speed codes;
                      FastScan time includes top-k selection and re-ranking

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

from vsq import RaBitQFastScan, RaBitQSpace, TurboQuantFastScan, TurboQuantSpace
from vsq._autotune.metrics import recall_at, squared_l2, topk_indices

# family -> legal bits. The *-fastscan families are flat top-k indexes, not
# per-pair distance APIs; their rows report recall from index.search().
_ALLOWED_BITS = {
    "turboquant": (4, 8),
    "rabitq": (1, 4, 8),
    "turboquant-fastscan": (4,),
    "rabitq-fastscan": (1, 4, 8),
}
_FASTSCAN_FAMILIES = ("turboquant-fastscan", "rabitq-fastscan")
_MAX_TRAIN_ROWS = 65536  # TurboQuant k-means sample cap

# Default grid. Bare `rabitq` / `rabitq-fastscan` use the library default
# encode mode (windowed_scale at 4/8 bits, the most accurate per encode
# cost); the other RaBitQ tokens pin the static, trained and exact modes.
_DEFAULT_METHODS = (
    "turboquant:4,turboquant:8,turboquant-fastscan:4,"
    "rabitq:1,rabitq:4,rabitq:8,"
    "rabitq-fixed-scale:4,rabitq-fixed-scale:8,"
    "rabitq-trained-scale:4,rabitq-trained-scale:8,"
    "rabitq-algorithm1:4,rabitq-algorithm1:8,"
    "rabitq-fastscan:1,rabitq-fastscan:4,rabitq-fastscan:8"
)
_EXACT_FLOOR = 1e-8
_QUICK_CONFLICTS = (
    "--data",
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
    "dataset",
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
    "index_param",
    "refined_frac",
)


@dataclass(frozen=True)
class MethodSpec:
    """One row of the comparison grid.

    encode_mode is algorithm1, fixed_scale, windowed_scale, trained_scale,
    or None (the library default). Only RaBitQ reads it.
    """

    name: str
    bits: int
    encode_mode: str | None = None

    def encode_api(self) -> str:
        return "encode_batch"

    def search_api(self) -> str:
        return "distance_1_to_n"


_RABITQ_MODES = {
    "fixed-scale": "fixed_scale",
    "windowed-scale": "windowed_scale",
    "trained-scale": "trained_scale",
    "algorithm1": "algorithm1",
}


def parse_methods(text: str) -> list[MethodSpec]:
    """Parse 'turboquant:4,rabitq:4,rabitq-algorithm1:8'.

    Empty and unknown tokens raise ValueError. Bare rabitq:1 is algorithm1.
    Bare rabitq / rabitq-fastscan keep the library default mode (None).
    rabitq-{fixed,windowed,trained}-scale accept bits 4 and 8 only.
    rabitq-algorithm1 accepts 1, 4, and 8.
    """
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
        encode_mode: str | None = None
        report_name = name
        family = name
        if name.startswith("rabitq-") and name not in _FASTSCAN_FAMILIES:
            family, _, suffix = name.partition("-")
            if suffix not in _RABITQ_MODES:
                raise ValueError(
                    f"unknown RaBitQ mode {suffix!r}; expected fixed-scale, "
                    "windowed-scale, trained-scale, or algorithm1"
                )
            encode_mode = _RABITQ_MODES[suffix]
        if family not in _ALLOWED_BITS:
            raise ValueError(
                f"unknown method {name!r}; expected turboquant or rabitq"
            )
        try:
            bits = int(bits_text)
        except ValueError as exc:
            raise ValueError(f"bits in {token!r} are not an integer") from exc
        legal = _ALLOWED_BITS[family]
        if bits not in legal:
            raise ValueError(
                f"{family} bits must be one of {legal}, got {bits}"
            )
        # None = the library default (algorithm1 at 1 bit, windowed_scale at 4/8).
        if encode_mode not in (None, "algorithm1") and bits == 1:
            raise ValueError(f"{report_name} requires bits 4 or 8, got 1")
        specs.append(MethodSpec(report_name, bits, encode_mode))
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


@dataclass(frozen=True)
class Dataset:
    """One evaluation set. All arrays float32, C-contiguous, same dim.

    base (n_base, dim) accuracy base, also the TurboQuant / RaBitQ training set
    query (n_query, dim) accuracy queries, disjoint from base for real data
    speed_base (n_speed, dim) codes timed by encode and search
    speed_query (dim,) the one timed query
    """

    name: str
    base: np.ndarray
    query: np.ndarray
    speed_base: np.ndarray
    speed_query: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.base.shape[1])


def gaussian_dataset(dim: int, n_base: int, n_query: int, n_speed: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed + dim)
    return Dataset(
        name=f"gauss{dim}",
        base=draw_matrix(n_base, dim, rng),
        query=draw_matrix(n_query, dim, rng),
        speed_base=draw_matrix(n_speed, dim, rng),
        speed_query=draw_matrix(1, dim, rng)[0],
    )


def npy_dataset(path: Path, n_base: int, n_query: int, n_speed: int) -> Dataset:
    """Real vectors from a (rows, dim) .npy file.

    base = first n_base rows, query = last n_query rows (held out), speed
    codes = first n_speed rows, timed query = last row. Needs
    max(n_base, n_speed) + n_query <= rows.
    """
    if not path.is_file():
        raise ValueError(f"no such dataset file: {path}")
    data = np.load(path, mmap_mode="r")
    if data.ndim != 2:
        raise ValueError(f"{path}: expected a (rows, dim) matrix, got shape {data.shape}")
    rows = data.shape[0]
    need = max(n_base, n_speed) + n_query
    if need > rows:
        raise ValueError(f"{path}: needs {need} rows (max(n_base, n_speed) + n_query), has {rows}")

    def take(sl: slice) -> np.ndarray:
        out = np.ascontiguousarray(data[sl], dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError(f"{path}: non-finite value in rows {sl}")
        return out

    query = take(slice(rows - n_query, rows))
    return Dataset(
        name=path.stem,
        base=take(slice(0, n_base)),
        query=query,
        speed_base=take(slice(0, n_speed)),
        speed_query=query[-1].copy(),
    )


def draw_matrix(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """float32 N(0, 1) with shape (n, dim), C-contiguous."""
    if n <= 0:
        raise ValueError(f"row count must be positive, got {n}")
    return rng.standard_normal((n, dim)).astype(np.float32, copy=False)


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
    """Batch encode and 1-to-N search. num_threads is fixed at 1.

    Library defaults (IVF centering with 256 k-means centroids, corrected
    estimator, no QJL) are also its most accurate configuration. The
    centering is fitted once on `train` (n, dim) float32, outside any timing.
    """

    encode_api = "encode_batch"
    search_api = "distance_1_to_n"

    def __init__(self, dim: int, bits: int, rot_seed: int, train: np.ndarray) -> None:
        self.space = TurboQuantSpace(dim, bits, rot_seed=rot_seed, num_threads=1)
        self.space.train(np.ascontiguousarray(train[:_MAX_TRAIN_ROWS]))
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
    """Encode and search each take one C++ pass over the rows.

    centroid: "mean" calls space.train(train), which centers codes on the
    mean of `train` (and calibrates trained_scale); "none" keeps the zero
    centroid and cannot run trained_scale. query_bits=0 scores 1-bit codes
    against the float query instead of the paper's 4-bit quantized query.
    """

    encode_api = "encode_batch"
    search_api = "distance_1_to_n"

    def __init__(
        self,
        dim: int,
        bits: int,
        rot_seed: int,
        encode_mode: str | None,
        train: np.ndarray,
        centroid: str,
        query_bits: int | None,
    ) -> None:
        if centroid not in ("mean", "none"):
            raise ValueError(f"centroid must be mean or none, got {centroid!r}")
        if centroid == "none" and encode_mode == "trained_scale":
            raise ValueError("trained_scale needs train(); drop --rabitq-centroid none")
        self.space = RaBitQSpace(
            dim, rot_seed=rot_seed, bits=bits, encode_mode=encode_mode,
            query_bits=query_bits, num_threads=1,
        )
        if centroid == "mean":
            self.space.train(np.ascontiguousarray(train, dtype=np.float32))
        self.bits = bits
        self.encode_mode = self.space.encode_mode()

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


class TurboQuantFastScanRunner(TurboQuantRunner):
    """4-bit TurboQuant codes behind TurboQuantFastScan.

    build: encode_batch, then repack into 32-code FastScan blocks.
    search: LUT-quantized scan of every block, then the k * rerank best
    candidates are re-scored with the full-precision code distance.
    """

    encode_api = "encode_batch+TurboQuantFastScan"
    search_api = "TurboQuantFastScan.search"

    def __init__(self, dim: int, bits: int, rot_seed: int, train: np.ndarray,
                 rerank: int) -> None:
        if bits != 4:
            raise ValueError(f"TurboQuantFastScan needs 4-bit codes, got {bits}")
        if rerank < 1:
            raise ValueError(f"rerank must be >= 1, got {rerank}")
        super().__init__(dim, bits, rot_seed, train)
        self.rerank = rerank

    @property
    def distance_kernel(self) -> str:
        return "fastscan_lut4+rerank"

    @property
    def index_param(self) -> str:
        return f"rerank={self.rerank}"

    def build(self, vectors: np.ndarray) -> TurboQuantFastScan:
        codes = self.encode(vectors)
        # The index keeps a reference to `codes` (keep_alive in the binding).
        return TurboQuantFastScan(self.space, codes)

    def search(self, index: TurboQuantFastScan, query: np.ndarray, k: int) -> tuple[np.ndarray, float]:
        ids, _ = index.search(query, k, rerank=self.rerank)
        refined = min(k * self.rerank, len(index)) / len(index)
        return np.asarray(ids, dtype=np.int64), refined


class RaBitQFastScanRunner(RaBitQRunner):
    """RaBitQ codes behind RaBitQFastScan (the paper's two-stage search).

    build: encode every vector from float32 into the FastScan layout.
    search: 1-bit FastScan estimate and error bound for every code; codes
    whose lower bound beats the current k-th distance are refined with the
    full B-bit code. eps0 sets the bound width.
    """

    encode_api = "RaBitQFastScan(X)"
    search_api = "RaBitQFastScan.search"

    def __init__(self, dim: int, bits: int, rot_seed: int, encode_mode: str | None,
                 train: np.ndarray, centroid: str, query_bits: int | None,
                 eps0: float) -> None:
        if not eps0 > 0.0:
            raise ValueError(f"eps0 must be positive, got {eps0}")
        super().__init__(dim, bits, rot_seed, encode_mode, train, centroid, query_bits)
        self.eps0 = eps0

    @property
    def distance_kernel(self) -> str:
        return "fastscan_1bit+bound_refine"

    @property
    def index_param(self) -> str:
        return f"eps0={self.eps0:g}"

    def build(self, vectors: np.ndarray) -> RaBitQFastScan:
        index = RaBitQFastScan(self.space, vectors, eps0=self.eps0)
        if len(index) != vectors.shape[0]:
            raise RuntimeError(f"RaBitQFastScan holds {len(index)} codes, expected {vectors.shape[0]}")
        return index

    def search(self, index: RaBitQFastScan, query: np.ndarray, k: int) -> tuple[np.ndarray, float]:
        ids, _, n_refined = index.search(query, k)
        return np.asarray(ids, dtype=np.int64), n_refined / len(index)


@dataclass(frozen=True)
class RunConfig:
    """Grid-wide knobs shared by every row (CLI flags)."""

    k: int
    repeats: int
    rot_seed: int
    seed: int
    rerank: int  # turboquant-fastscan
    eps0: float  # rabitq-fastscan
    rabitq_centroid: str  # "mean" | "none"
    rabitq_query_bits_1bit: int | None  # 1-bit codes only; None = library default (4)
    cooldown: float  # seconds idle before each row's timed section


def make_runner(spec: MethodSpec, dim: int, train: np.ndarray, cfg: RunConfig):
    """train: (n, dim) float32, the accuracy base; fits TurboQuant IVF / RaBitQ centroid."""
    if spec.name == "turboquant":
        return TurboQuantRunner(dim, spec.bits, cfg.rot_seed, train)
    if spec.name == "turboquant-fastscan":
        return TurboQuantFastScanRunner(dim, spec.bits, cfg.rot_seed, train, cfg.rerank)
    query_bits = cfg.rabitq_query_bits_1bit if spec.bits == 1 else None
    rabitq_args = (dim, spec.bits, cfg.rot_seed, spec.encode_mode, train,
                   cfg.rabitq_centroid, query_bits)
    if spec.name == "rabitq-fastscan":
        return RaBitQFastScanRunner(*rabitq_args, cfg.eps0)
    if spec.name == "rabitq" or spec.name.startswith("rabitq-"):
        return RaBitQRunner(*rabitq_args)
    raise ValueError(f"unknown method {spec.name!r}")


def _flat_timing(runner, speed_base, speed_query, k, repeats) -> tuple[float, float]:
    """Seconds for encode_batch(speed_base) and one distance_1_to_n pass."""
    encode_s = time_min(lambda: runner.encode(speed_base), repeats)
    speed_codes = runner.encode(speed_base)
    search_s = time_min(lambda: runner.distances(speed_query, speed_codes), repeats)
    return encode_s, search_s


def _index_timing(runner, speed_base, speed_query, k, repeats) -> tuple[float, float]:
    """Seconds for building the index over speed_base and one top-k search."""
    encode_s = time_min(lambda: runner.build(speed_base), repeats)
    speed_index = runner.build(speed_base)
    search_s = time_min(lambda: runner.search(speed_index, speed_query, k), repeats)
    return encode_s, search_s


def _flat_accuracy(runner, base, query, exact, k):
    """Per-pair API: rank all n_base estimates per query. -> (pred, rel, n, refined)."""
    codes = runner.encode(base)
    estimate = np.empty_like(exact)
    for i in range(query.shape[0]):
        estimate[i] = runner.distances(query[i], codes)
    if not np.isfinite(estimate).all():
        raise RuntimeError("non-finite distance")
    rel, rel_n = relative_mae(estimate, exact)
    return topk_indices(estimate, k), rel, rel_n, 1.0


def _index_accuracy(runner, base, query, k):
    """Top-k index API: recall of index.search(). -> (pred, nan, 0, refined).

    rel_mae is undefined (the index returns k ids, not n distances).
    refined is the mean fraction of the base re-scored with the full code.
    """
    index = runner.build(base)
    pred = np.empty((query.shape[0], k), dtype=np.int64)
    refined = np.empty(query.shape[0], dtype=np.float64)
    for i in range(query.shape[0]):
        ids, refined[i] = runner.search(index, query[i], k)
        if ids.shape != (k,):
            raise RuntimeError(f"search returned {ids.shape} ids, expected ({k},)")
        pred[i] = ids
    return pred, float("nan"), 0, float(refined.mean())


def score_method(
    spec: MethodSpec,
    dim: int,
    base: np.ndarray,
    query: np.ndarray,
    speed_base: np.ndarray,
    speed_query: np.ndarray,
    cfg: RunConfig,
) -> dict:
    """Accuracy on `base`/`query`, throughput on the speed draw.

    Flat rows rank every distance_1_to_n estimate; FastScan rows use the
    index's own top-k search. search_pairs_per_s is n_speed / (time of one
    query against n_speed codes) in both cases, so the rates compare per
    code scanned; FastScan time includes its top-k selection and re-ranking.
    """
    k, repeats = cfg.k, cfg.repeats
    runner = make_runner(spec, dim, base, cfg)
    if runner.padded_dim < dim:
        raise RuntimeError(
            f"padded_dim {runner.padded_dim} is below input dim {dim}"
        )

    # Timing runs first, after an optional idle pause, so a thermally limited
    # (fanless) CPU is not still hot from the previous row's accuracy pass.
    is_index = hasattr(runner, "build")
    if cfg.cooldown > 0:
        time.sleep(cfg.cooldown)
    timing = _index_timing if is_index else _flat_timing
    encode_s, search_s = timing(runner, speed_base, speed_query, k, repeats)

    exact = squared_l2(base, query)
    gt = topk_indices(exact, k)
    if is_index:
        pred, rel, rel_n, refined_frac = _index_accuracy(runner, base, query, k)
    else:
        pred, rel, rel_n, refined_frac = _flat_accuracy(runner, base, query, exact, k)

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
        "seed": cfg.seed,
        "rot_seed": cfg.rot_seed,
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
        "index_param": getattr(runner, "index_param", ""),
        "refined_frac": refined_frac,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in _CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict], stamp: str, host: str, cfg: RunConfig) -> None:
    first = rows[0]
    datasets = ", ".join(dict.fromkeys(f"`{r['dataset']}` (dim {r['dim']})" for r in rows))
    query_bits = "float" if cfg.rabitq_query_bits_1bit == 0 else (
        "library default" if cfg.rabitq_query_bits_1bit is None else f"{cfg.rabitq_query_bits_1bit}-bit")
    lines = [
        f"# Quantizer comparison {stamp}",
        "",
        (
            f"Host `{host}`, Python {first['python']}, vsq {first['package_version']}, "
            f"num_threads={first['num_threads']}."
        ),
        "",
        (
            f"Datasets: {datasets}. `gaussN` is a float32 N(0, 1) draw; a file name is real "
            "vectors (base = first rows, queries = last rows, held out). "
            f"Accuracy uses n_base={first['n_base']}, n_query={first['n_query']}, k={first['k']}; "
            f"throughput uses n_speed={first['n_speed']} codes and the fastest of {cfg.repeats} repeats."
        ),
        "",
        (
            "Configurations: TurboQuant uses its library defaults (IVF centering, 256 k-means "
            "centroids fitted on the base, corrected estimator, no QJL). RaBitQ uses "
            f"centroid={cfg.rabitq_centroid} (`train()` on the base), {query_bits} queries for "
            "1-bit codes, and bare `rabitq` keeps the library default encode mode "
            "(`windowed_scale` at 4/8 bits). "
            f"`turboquant-fastscan` re-scores k·{cfg.rerank} candidates; `rabitq-fastscan` uses "
            f"eps0={cfg.eps0:g}."
        ),
        "",
        (
            "recall is overlap with exact squared L2. rel_mae is the mean relative error of the "
            "asymmetric distance. Flat rows rank every `distance_1_to_n` estimate; FastScan rows use "
            "the index's own top-k `search` (rel_mae is undefined there). search codes/s is n_speed "
            "divided by the time of one query against n_speed codes. `refined` is the mean share of "
            "the base re-scored with the full code."
        ),
        "",
        (
            "| dataset | method | bits | dim | bytes | recall@1 | recall@10 | rel_mae | encode /s "
            "| search codes/s | search api | refined |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        mae = "—" if np.isnan(row["rel_mae"]) else f"{row['rel_mae']:.4f}"
        api = row["search_api"] + (f" ({row['index_param']})" if row["index_param"] else "")
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['bits']} | {row['dim']} "
            f"| {row['code_bytes']} | {row['recall_at_1']:.4f} | {row['recall_at_10']:.4f} "
            f"| {mae} | {row['encode_vps']:,.0f} | {row['search_pairs_per_s']:,.0f} | {api} "
            f"| {row['refined_frac']:.1%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'dataset':<12} {'method':<22} {'bits':>4} {'dim':>6} {'bytes':>6} "
        f"{'R@1':>7} {'R@10':>7} {'rel_mae':>8} "
        f"{'encode/s':>12} {'search/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['dataset']:<12} {row['method']:<22} {row['bits']:4d} {row['dim']:6d} "
            f"{row['code_bytes']:6d} {row['recall_at_1']:7.4f} "
            f"{row['recall_at_10']:7.4f} {row['rel_mae']:8.4f} "
            f"{row['encode_vps']:12,.0f} {row['search_pairs_per_s']:12,.0f}"
        )


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="128,1024",
                        help="comma-separated N(0, 1) dims; 'none' runs only --data")
    parser.add_argument("--data", type=Path, action="append", default=[],
                        help="(rows, dim) float .npy of real vectors; repeatable")
    parser.add_argument("--methods", default=_DEFAULT_METHODS, help="comma-separated method:bits")
    parser.add_argument("--n-base", type=int, default=2000)
    parser.add_argument("--n-query", type=int, default=40)
    parser.add_argument("--n-speed", type=int, default=400)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1, help="data draw seed")
    parser.add_argument("--rot-seed", type=int, default=42, help="SRHT seed for both spaces")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="seconds idle before timing each row (fanless / thermally "
                        "limited CPUs); timing always runs before the accuracy pass")
    parser.add_argument("--rerank", type=int, default=4,
                        help="turboquant-fastscan: k * rerank candidates are re-scored")
    parser.add_argument("--eps0", type=float, default=1.9,
                        help="rabitq-fastscan: error-bound width (library default 1.9)")
    parser.add_argument("--rabitq-centroid", choices=("mean", "none"), default="mean",
                        help="RaBitQ residual centroid: mean of the base (default) or none")
    parser.add_argument("--rabitq-query-bits", type=int, default=0,
                        help="1-bit RaBitQ query: 0 = float (default, most accurate), "
                        "4 = paper's B_q, -1 = library default")
    parser.add_argument("--out-dir", type=Path, default=here / "results")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke preset: dim 128, turboquant:4, turboquant-fastscan:4, rabitq:1, rabitq:4, "
        "rabitq-fastscan:4, "
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
    args.methods = "turboquant:4,turboquant-fastscan:4,rabitq:1,rabitq:4,rabitq-fastscan:4"
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
        dims = [] if args.dims.strip().lower() == "none" else parse_dims(args.dims)
        methods = parse_methods(args.methods)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    validate_sizes(args.n_base, args.n_query, args.n_speed, args.k)
    if args.rot_seed < 0:
        raise SystemExit("rot-seed must be non-negative")
    if args.rerank < 1:
        raise SystemExit("rerank must be >= 1")
    if args.cooldown < 0:
        raise SystemExit("cooldown must be >= 0")
    if not args.eps0 > 0.0:
        raise SystemExit("eps0 must be positive")
    cfg = RunConfig(
        k=args.k, repeats=args.repeats, rot_seed=args.rot_seed, seed=args.seed,
        rerank=args.rerank, eps0=args.eps0, rabitq_centroid=args.rabitq_centroid,
        rabitq_query_bits_1bit=None if args.rabitq_query_bits < 0 else args.rabitq_query_bits,
        cooldown=args.cooldown,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = f"{platform.system()} {platform.machine()}"
    py = platform.python_version()
    try:
        import vsq

        version = vsq.__version__
    except Exception:
        version = "unknown"

    try:
        datasets = [gaussian_dataset(d, args.n_base, args.n_query, args.n_speed, args.seed)
                    for d in dims]
        datasets += [npy_dataset(path, args.n_base, args.n_query, args.n_speed)
                     for path in args.data]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not datasets:
        raise SystemExit("nothing to run: --dims none and no --data")

    rows: list[dict] = []
    for ds in datasets:
        for spec in methods:
            print(f"running {spec.name}:{spec.bits} {ds.name} dim={ds.dim}", flush=True)
            row = score_method(spec, ds.dim, ds.base, ds.query, ds.speed_base,
                               ds.speed_query, cfg)
            row["dataset"] = ds.name
            row["timestamp"] = stamp
            row["host"] = host
            row["python"] = py
            row["package_version"] = version
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"compare_{stamp}.csv"
    md_path = args.out_dir / f"compare_{stamp}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, stamp, host, cfg)
    print()
    print_table(rows)
    print()
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()

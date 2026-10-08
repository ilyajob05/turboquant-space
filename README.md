# vsq — vector search quantization

![License](https://img.shields.io/pypi/l/vsq)
![Build](https://img.shields.io/github/actions/workflow/status/ilyajob05/turboquant-space/publish.yml)
![Python](https://img.shields.io/pypi/pyversions/vsq)
![PyPI](https://img.shields.io/pypi/v/vsq)

Vector quantization for approximate nearest-neighbour search: **TurboQuant**
(ICLR 2026, arXiv:2504.19874) and **RaBitQ / Extended RaBitQ** (SIGMOD 2024,
arXiv:2409.09913). Header-only C++17 with Python bindings, SIMD kernels
(AVX2 chosen at runtime, NEON on arm64), designed to be embedded in
[hnswlib](https://github.com/nmslib/hnswlib) as a distance space.

```bash
pip install vsq
```

```python
import numpy as np
from vsq import TurboQuantSpace

X = np.random.randn(100_000, 768).astype(np.float32)
q = np.random.randn(768).astype(np.float32)

space = TurboQuantSpace(768, bits=4)      # format v2, IVF centering by default
space.train(X)                             # fit the centroids once
codes = space.encode_batch(X)              # (100_000, 398) uint8
dists = space.distance_1_to_n(q, codes)    # (100_000,) float32, squared L2
```

`encode` once, then `distance_*` against the codes. The space itself is the
only state besides the codes; it pickles (with a format version).

---

## What it does

A vector `x` is encoded relative to a centroid `c` (none, the dataset mean, or
the nearest of `k` k-means centroids). The unit residual `(x - c) / ||x - c||`
is rotated by a random orthogonal transform (signs + Walsh–Hadamard blocks)
so its coordinates are approximately i.i.d. Gaussian, then quantized per
coordinate with a Lloyd–Max quantizer (4 or 8 bits) or stored as fp16
(16 bits). A per-code scale makes the inner-product estimate unbiased
(RaBitQ-style correction). Distances are computed directly on the codes:

* **asymmetric** — raw query × code (search), one rotation per query;
* **symmetric** — code × code (graph construction, e.g. HNSW links).

### Design decisions (measured, see `.agent/planning/plan.md`)

| question | decision | evidence (recall@10, 4 bits/coord) |
|---|---|---|
| spend 1 bit on QJL or on MSE? | MSE + correction by default; QJL optional | openai-v3-small .973 vs .966; msmarco .962 vs .941; SIFT .857 vs .630 |
| centering | IVF (k = 256) by default | openai-v3-large .987 (ivf) vs .985 (mean) vs .975 (none); SIFT .925–.932 vs .857 |
| padding | multiple of 64, not a power of two | 1536 stays 1536 (was 2048: −25% code size and compute) |

---

## Install

Prebuilt wheels: CPython 3.11–3.13 on Linux (x86\_64, aarch64) and macOS
(x86\_64, arm64). Wheels are compiled for the **baseline** ISA (x86-64 /
armv8-a); the AVX2 + FMA kernels are compiled in with target attributes and
selected **at runtime by CPUID**, so one wheel is fast on AVX2 machines and
still runs (scalar) on older x86 CPUs. `vsq.detected_isa()` reports
the choice (`"avx2"`, `"neon"` or `"scalar"`).

```bash
pip install vsq --no-binary vsq   # build with -march=native
git clone https://github.com/ilyajob05/turboquant-space && cd turboquant-space && uv sync
```

Source builds need CMake ≥ 3.18 and a C++17 compiler (`brew install libomp`
on macOS for OpenMP). `-DVSQ_PORTABLE=ON` selects the wheel baseline.

---

## TurboQuantSpace (format v2)

```python
TurboQuantSpace(
    dim: int,                      # input dimension >= 1
    bits: int = 4,                 # 4 | 8 | 16 stored bits per coordinate (16 = fp16)
    *,
    qjl: bool = False,             # 4/8 only: one of `bits` becomes a QJL sign bit
    estimator: str = "corrected",  # "corrected" (unbiased scale) | "plain"
    qjl_cross: str = "linear",     # full-symmetric <e_a, e_b> term: "linear" | "arcsine"
    centering: str = "ivf",        # "none" | "mean" | "ivf"
    n_clusters: int = 256,         # ivf: requested k (<= n_train / 39, <= 65535)
    rotation_rounds: int = 3,      # rounds of the block Walsh–Hadamard/Kac rotation
    rot_seed: int = 42, qjl_seed: int = 137,
    num_threads: int = 0,          # batch helpers; 0 = OpenMP default
    isa: str = "auto",             # "auto" | "scalar" | "neon" | "avx2" (clamped to the CPU)
)
```

| method | input | output |
|---|---|---|
| `train(X, seed=1234, iters=10)` | `(n, dim)` float32 | fits mean / k-means centroids |
| `set_centroids(C)` / `centroids()` | `(k, dim)` float32 | |
| `encode(x)` / `encode_batch(X)` | `(dim,)` / `(n, dim)` | `(cs,)` / `(n, cs)` uint8 |
| `encode_into(x, out)` / `encode_batch_into(X, out)` | caller-owned buffers | |
| `decode(code)` | `(cs,)` | `(dim,)` approximate reconstruction |
| `distance(q, code)` | `(dim,)`, `(cs,)` | float |
| `distance_1_to_n(q, codes)` | `(dim,)`, `(n, cs)` | `(n,)` |
| `distance_m_to_n(Q, codes)` | `(m, dim)`, `(n, cs)` | `(m, n)` (tiled) |
| `distance_symmetric(a, b)` / `distance_m_to_n_symmetric(A, B)` | codes | float / `(m, n)` |
| `distance_symmetric_full(a, b)` / `distance_m_to_n_symmetric_full(A, B)` | codes (qjl spaces) | full 4-term QJL estimate |

Accessors: `dim()`, `padded_dim()`, `bits()`, `qjl()`, `centering()`,
`n_clusters()`, `trained()`, `code_size_bytes()`, `kernel_isa()`,
`num_threads()`, `rotation_rounds()`, `format_version()` (static, `2`).

All array arguments must be **C-contiguous** with the exact dtype (float32
vectors, uint8 codes); strided views raise `ValueError` instead of being read
with the wrong layout. Distances are squared L2, clamped at 0.

### Code layout (v2)

`D` = `dim` rounded up to a multiple of 64. Little-endian, no alignment.

| field | type | present | meaning |
|---|---|---|---|
| payload | `D/2` B (4-bit), `D` B (8-bit), `2D` B (fp16) | always | per-coordinate index, low nibble = even coordinate; with QJL the unit is `idx << 1 \| sign` |
| `f_sq` | float32 | always | `‖x − c‖²` |
| `f_mul` | float32 | always | `‖x − c‖ · s / √D` (`s` = estimator scale) |
| `f_ct` | float32 | ivf | estimate of `⟨c, x − c⟩` |
| `f_qjl` | float32 | qjl | `‖x − c‖ · √(π/2) / √D · γ` |
| `cid` | uint16 | ivf | cluster id |

`code_size_bytes() = D·bits/8 + 8` (+4 ivf, +4 qjl, +2 ivf). At dim 1536 /
4 bits: 782 B.

### Centering

`train(X)` is required for `mean` and `ivf` (encoding an untrained space
raises `RuntimeError`); `centering="none"` needs no training. IVF adds one
k-float table per query (`‖q − c_j‖²`), not per code, and symmetric distances
between codes of different clusters use the rotated centroids (one fused
pass). Bring your own centroids (e.g. from FAISS) with `set_centroids`.

### HNSW integration, threading, ISA

* Per-pair distance functions never allocate, never throw and never spawn
  threads; C++ adapters with hnswlib's `DISTFUNC` signature are
  `TurboQuantSpace::searchDistFunc` (prepared query × code) and
  `buildDistFunc` (code × code).
* Batch helpers (`encode_batch`, `distance_*_to_n`) use OpenMP; the Python
  bindings release the GIL. `distance_m_to_n` tiles queries × codes so the
  codes stay in cache.
* Kernels: AVX2 (runtime CPUID), NEON (aarch64), scalar reference. All macros
  are prefixed `VSQ_` and all code lives in namespace `vsq`, so the headers can
  sit next to hnswlib.

---

## RaBitQSpace

```python
RaBitQSpace(dim, rot_seed=42, centroid=None, bits=4, encode_mode=None, *,
            rotation="kac",        # "kac" (default) | "legacy" (0.1.x codes)
            rotation_rounds=3,
            query_bits=None,       # 1-bit: 4 (paper's B_q, popcount); 0 = float query
            num_threads=0, isa="auto")

space = RaBitQSpace(768)           # bits=4, encode_mode=None -> fixed_scale
space.train(X)                     # centroid = mean(X); recommended for real data
codes = space.encode_batch(X)
```

* `bits` ∈ {1, 4, 8}, default 4. 1-bit distances use the paper's quantized query by
  default: `(B_q + 1) · D/64` popcounts per code instead of `D` float FMAs;
  `query_bits=0` scores against the float query (recall@10 +0.01–0.03).
* `train(X)` — `(n, dim)` float32. Sets the centroid to the mean of `X` (in
  every mode) and calibrates `trained_scale`. Codes encoded before `train`
  used the old centroid and must be re-encoded. `trained()` and `centroid()`
  read the state.
* `distance_bound(q, code, eps0=1.9)` → `(estimate, lower, upper)` from the
  RaBitQ error bound (≈94% coverage at `eps0 = 1.9`).
* `x == centroid` encodes to an exact `‖q − c‖²`.
* RaBitQ is asymmetric only — do not use it as the HNSW link metric.

### Encode modes (4 and 8 bits)

A 4/8-bit code stores each rotated residual coordinate `o'ᵢ` as a grid index
`round(t · o'ᵢ)` clamped to `2^bits` levels. The **scale `t`** decides how
the unit vector is stretched over the grid; the modes differ only in how `t`
is chosen. They write the same slot and use the same distance kernel, so
search speed and code size do not depend on the mode.

| `encode_mode` | how `t` is chosen | needs `train` | encode cost | accuracy |
|---|---|---|---|---|
| `"windowed_scale"` (**8-bit default**) | per vector: the best `t` inside the RaBitQ tight interval, found with a min-heap of the next grid-step event per coordinate | no | medium | best, equal to `algorithm1` |
| `"fixed_scale"` (static, **4-bit default**) | one `t` for the whole space: the mean Algorithm 1 optimum over 100 N(0, 1) vectors, computed in the constructor | no | lowest, O(1) per coordinate | 4 bits: equal; 8 bits: lower (recall@10 −0.015 on N(0, 1), dim 128) |
| `"trained_scale"` (trained) | one `t` for the whole space: the mean Algorithm 1 optimum over up to 1024 residuals of *your* data, computed by `train(X)` | yes (`encode` raises before it) | lowest, as `fixed_scale` | as `fixed_scale` |
| `"algorithm1"` (reference) | per vector: bit-exact Extended RaBitQ Algorithm 1, sorting every threshold | no | highest | best (the definition) |

1-bit codes are signs and have no scale; `encode_mode` must be `None` or
`"algorithm1"` there.

**Which mode to use.** Keep the default (`encode_mode=None`): at 4 bits
`fixed_scale` matches `windowed_scale` within measurement noise and encodes
9–11× faster; at 8 bits `windowed_scale` is more accurate (recall@10
+0.001…+0.016), so it is the default there. Pin `fixed_scale` at 8 bits only
when encode throughput is the bottleneck. `trained_scale` pins the static scale to
your data, but it lands within ~1% of `fixed_scale`'s `t`: after the random
rotation the coordinates of any unit residual are close to N(0, 1/D), so the
best static scale hardly depends on the data. `algorithm1` is the reference
for tests and papers.

**What does depend on the data is the centroid.** Codes quantize `x − c`.
Real embeddings are far from zero-mean, and the default zero centroid spends
the grid on their common offset. Call `train(X)` (or pass `centroid=`) on
real data: on dbpedia OpenAI embeddings (dim 1536) recall@10 rises from 0.938
to 0.968 at 4 bits and from 0.66 to 0.82 at 1 bit. Measurements are in
[`docs/benchmarks.md`](docs/benchmarks.md).

---

## FastScan (flat 1-to-N)

HNSW needs per-pair distances; flat scans and re-ranking use the FastScan
block layout (32 codes, 4-bit sub-codes, PSHUFB/TBL lookups):

```python
from vsq import TurboQuantFastScan, RaBitQFastScan

index = TurboQuantFastScan(space, codes)        # 4-bit TurboQuant, no qjl
ids, dists = index.search(q, k=10, rerank=4)    # FastScan, then exact re-score

rq = RaBitQSpace(768, bits=4)
rq.train(X)                                     # centroid = mean(X)
rindex = RaBitQFastScan(rq, X)                  # two-stage search from the paper
ids, dists, n_refined = rindex.search(q, k=10)  # 1-bit estimate + bound, refine
```

---

## Autotune

`vsq.autotune` chooses the quantizer, bit width, search path (flat or
FastScan) and thread count for your data and an objective, measures every
candidate on a sample of your data and builds the winner:

```python
result = vsq.autotune(X, profile="speed")   # "accuracy" | "speed" | "energy"
ids, dists = result.index.search(q, k=10)
print(result.report())                      # candidates, measurements, why
```

| profile | objective | default constraint |
|---|---|---|
| `accuracy` | highest recall@k | — |
| `speed` | lowest single-query p50 latency at your n | recall@k ≥ 0.90 |
| `energy` | lowest energy per query (proxy) | recall@k ≥ 0.90 |

Optional limits: `min_recall`, `max_bytes_per_vector`, `max_latency_ms`,
`time_budget_s`; the recall floor is checked against the lower 95 % bound of
the measured recall. Data-independent rules narrow the catalog, cheap 4-bit
candidates are calibrated first and 8/16-bit ones only if they miss the
recall floor; `result.to_json()` stores the choice for reuse. Details:
[`docs/autotune.md`](docs/autotune.md).

### Presets (no calibration)

When you know the trade-off you want, or have fewer than ~1000 vectors,
build a named preset directly:

```python
index = vsq.build_index(X)                  # preset "balanced"
index = vsq.build_index(X, "compact")       # or "accurate", or a QuantizerConfig
ids, dists = index.search(q, k=10)
```

| preset | quantizer | size vs float32 | recall@10 on dbpedia-1536 | use when |
|---|---|---|---:|---|
| `compact` | RaBitQ 1 bit (FastScan above dim 256) | ~32× smaller | 0.83 | memory is the limit; re-rank a larger k |
| `balanced` (default) | TurboQuant 4 bit, FastScan | ~8× smaller | 0.97 | general use: fastest search at every measured dim |
| `accurate` | RaBitQ 8 bit, flat scan | ~4× smaller | 0.999 | recall matters more than speed |

`vsq.preset(name, dim)` returns the `QuantizerConfig`; presets are entries of
the autotune catalog, so `autotune(..., candidates=["balanced", "accurate"])`
compares just those and the report names the preset it chose. Evidence:
[Defaults and presets](docs/benchmarks.md#defaults-and-presets).

---

## Layout

C++ namespaces mirror the `include/` directories under the project prefix
`vsq`: `vsq::common`, `vsq::turboquant`, `vsq::rabitq`. The two algorithms
never see each other's names; shared code is always called with an explicit
`common::` qualifier. Macros use the same prefix (`VSQ_`).

```
include/
  common/            shared by both algorithms
    config.h         ISA/CPUID dispatch, OpenMP, VSQ_ macros
    fp16.h vecops.h  half floats, small vector helpers
    srht.h           SRHT primitives (RaBitQ "legacy" rotation)
    rotation.h       block Walsh–Hadamard + Kac rotation; LegacySrht
    fastscan.h       FastScan engine (32-code blocks, 4-bit LUT scan)
  turboquant/        TurboQuant
    space.h          TurboQuantSpace (code format v2)
    kernels.h        kernels_{scalar,neon,avx2}.h dispatch
    lloyd_max.h      Lloyd–Max tables, branch-free quantizer
    kmeans.h centering.h
    fastscan.h       TurboQuantFastScan (4-bit codes)
  rabitq/            RaBitQ / Extended RaBitQ
    space.h          RaBitQSpace
    kernels.h        float-query kernels
    bitwise.h        quantized query + popcount (1-bit)
    fastscan.h       RaBitQFastScan (two-stage search)
python/vsq/          bindings.cpp, __init__.py, _vsq.pyi
tests/cpp/           test_core.cpp (copy safety, SIMD == scalar)
docker/              amd64 test image + script (AVX2 via qemu-user)
```

## Tests

```bash
uv run pytest python/tests/ -v                                  # Python suite
cmake -S . -B build -DVSQ_BUILD_TESTS=ON && cmake --build build --target test_core && ctest --test-dir build
docker/test_amd64.sh                                             # x86-64: scalar fallback + AVX2 (qemu-user)
```

## Benchmarks

```bash
uv run python python/benchmarks/run_benchmark.py            # SIFT1M + synthetic sweep
uv run python python/benchmarks/compare_quantizers.py       # TurboQuant vs RaBitQ
uv run python python/benchmarks/plot_compare.py <compare_*.csv> --out-dir docs/img
```

`run_benchmark.py` downloads SIFT1M and HF datasets on first use (into
`python/benchmarks/data/`, override with `--data-dir`);
`compare_quantizers.py` draws N(0, 1) data and takes real vectors with
`--data X.npy`. The 0.2.0 numbers in [`docs/benchmarks.md`](docs/benchmarks.md)
come from `compare_quantizers.py` (Apple M3, single thread, each quantizer at
its most accurate default). On DBpedia OpenAI embeddings (dim 1536) recall@10
is 0.974 (TurboQuant) vs 0.968 (RaBitQ) at 4 bits and 0.999 for both at 8
bits. `TurboQuantFastScan` is the fastest flat scan (27–209 M codes/s, no
recall loss); RaBitQ's per-pair scan is ~2× TurboQuant's.

![recall@10 and relative distance error against bytes per code](docs/img/compare_accuracy.png)

## Citation

If you use this library in academic work, please cite the TurboQuant
(ICLR 2026) and RaBitQ (SIGMOD 2024) papers in addition to this repository.

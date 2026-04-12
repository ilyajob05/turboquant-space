# turboquant-space

![GitHub License](https://img.shields.io/github/license/ilyajob05/turboquant-space)
![Build](https://img.shields.io/github/actions/workflow/status/ilyajob05/turboquant-space/publish.yml)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/turboquant-space)
![Python](https://img.shields.io/pypi/pyversions/turboquant-space)
![PyPI](https://img.shields.io/pypi/v/turboquant-space)
![License](https://img.shields.io/pypi/l/turboquant-space)
![Downloads](https://img.shields.io/pypi/dm/turboquant-space)


This library was inspired by the article https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/. The library is optimized for efficient data allocation in memory for 3+1 and 7+1 bit quantization schemes.

SIMD-accelerated 4/8-bit vector quantization for approximate nearest neighbor
search, based on **TurboQuant** (ICLR 2026). Standalone C++17 library with
Python bindings.

```python
from turboquant import TurboQuantSpace
import numpy as np

space = TurboQuantSpace(dim=128, bits_per_coord=8, num_threads=4)
X = np.random.randn(100_000, 128).astype(np.float32)
q = np.random.randn(128).astype(np.float32)

codes = space.encode_batch(X)              # (100_000, code_size) uint8
dists = space.distance_1_to_n(q, codes)    # (100_000,) float32
```

That is the whole mental model: `encode` once, then `distance_*` against the
codes. No index to build, no state to persist beyond `codes`.

---

## What it does, briefly

TurboQuant encodes each float32 vector into a compact code of
`bits_per_coord` bits per coordinate using a randomized Walsh–Hadamard
rotation followed by Lloyd–Max scalar quantization, plus one QJL sign bit per
coordinate for an unbiased residual correction. Distances between a raw query
and a packed code (asymmetric) or between two packed codes (symmetric) are
computed directly on the quantized representation with hand-written NEON /
SSE / AVX kernels.

**Concretely you get:**

| bits_per_coord | layout               | bytes / vec (dim=128) | compression vs fp32 |
|----------------|----------------------|-----------------------|---------------------|
| 4              | nibble-packed        | 76                    | 6.7×                |
| 8              | one byte per coord   | 140                   | 3.7×                |

(Plus 12 bytes of metadata — norm, γ, σ — per code.)

**What it is not:** not a graph index, not an IVF, not a drop-in replacement
for FAISS. It is the *distance* layer. Plug it into your own index, or use
`distance_1_to_n` as brute-force search on batches up to a few million.

---

## Install

```bash
pip install turboquant-space
```

Prebuilt wheels are published for CPython 3.11–3.13 on Linux (x86\_64,
aarch64), macOS (x86\_64, arm64), and Windows (AMD64). They target a
conservative CPU baseline — **x86-64-v3** (AVX2 + FMA + BMI2) on x64 and
**armv8-a** (NEON) on arm64 — so a single wheel runs on anything produced in
the last ~8 years. A C++ compiler is **not** required for this path.

### Build from source for maximum performance

The prebuilt wheels trade a few percent for portability. If you have a C++
compiler and want the binary tuned to *your* CPU (AVX-512 on Zen4 / Ice Lake,
SVE on Graviton, etc.), force pip to skip the wheel and compile from sdist:

```bash
pip install turboquant-space --no-binary turboquant-space
```

This invokes CMake with `-march=native`, so every available instruction set
on the build machine is enabled. Requires CMake ≥ 3.18 and a C++17 compiler;
on macOS also `brew install libomp` for multi-threaded batch ops.

### From a git checkout

```bash
git clone https://github.com/ilyajob05/turboquant-space
cd turboquant-space
uv sync                       # or: pip install -e .
```

Same story: local builds use `-march=native` by default. Pass
`-DTURBOQUANT_PORTABLE=ON` to CMake if you need a portable baseline instead.

---

## API

Everything lives on a single class, `TurboQuantSpace`. All numpy arrays are
`float32`, C-contiguous; all codes are `uint8`.

```python
TurboQuantSpace(
    dim: int,                    # input dimensionality (any positive integer)
    bits_per_coord: int = 4,     # 2..9 — nibble-packed for bits<=4
    rot_seed: int = 42,          # Hadamard rotation seed
    qjl_seed: int = 137,         # QJL sign seed
    num_threads: int = 0,        # 0 = use OMP_NUM_THREADS / all cores
)
```

| method                                    | shape in                     | shape out                   |
|-------------------------------------------|------------------------------|-----------------------------|
| `encode(x)`                               | `(dim,)`                     | `(code_size_bytes,)` uint8  |
| `encode_batch(X)`                         | `(n, dim)`                   | `(n, code_size_bytes)` uint8|
| `encode_into(x, out)` / `encode_batch_into` | in-place into caller buffer | —                           |
| `distance(query, code)`                   | `(dim,)`, `(code_size,)`     | `float`                     |
| `distance_symmetric(code_a, code_b)`      | `(code_size,)` ×2            | `float`                     |
| `distance_1_to_n(q, codes)`               | `(dim,)`, `(n, code_size)`   | `(n,)` float32              |
| `distance_m_to_n(Q, codes)`               | `(m, dim)`, `(n, code_size)` | `(m, n)` float32            |
| `distance_m_to_n_symmetric(codes_a, b)`   | `(m, cs)`, `(n, cs)`         | `(m, n)` float32            |

Accessors: `dim()`, `padded_dim()`, `padded()`, `num_threads()`,
`code_size_bytes()`, `bits_per_coord()`.

### Dimensionality padding

Internally every operation works in a power-of-two dimension (a requirement
of the Walsh–Hadamard transform). If you pass `dim=100`, the space rounds up
to 128 and zero-pads on the fly; a one-time warning is printed, and
`space.padded_dim()` reports the internal size. Correctness is preserved —
zero-padding in ℝᵈ does not change L2 distances — but encode/query cost is
determined by `padded_dim()`, not `dim()`.

### Threading

All batch methods (`encode_batch`, `distance_1_to_n`, `distance_m_to_n`,
`distance_m_to_n_symmetric`) parallelize the outer loop with OpenMP,
`schedule(static)`, so each thread owns a contiguous range of codes —
prefetcher-friendly, no false sharing on output rows. Set `num_threads` in
the constructor, or leave it `0` to respect `OMP_NUM_THREADS`. For small
batches (≤ 64) execution stays single-threaded to avoid fork/join overhead.

Observed scaling on Apple M-series, dim=512, 50k codes × 128 queries, bits=8:
**1→2 = 1.94×, 1→4 = 3.49×, 1→8 = 4.50×** — see `python/benchmarks/` for the
full reproduction.

---

## Benchmarks

```bash
uv run python python/benchmarks/run_benchmark.py
```

On first run this downloads SIFT1M (~170 MB) to
`~/.cache/turboquant/sift/`; subsequent runs reuse the cache. The script
sweeps `bits_per_coord × num_threads` on SIFT1M (with recall@{1,10,100}
against the shipped ground truth) and on synthetic Gaussian data across
several dimensions, writes
`python/benchmarks/results/results_<timestamp>.csv`, and produces seaborn
plots under `results/plots/`:

- `threading_scaling.png` — M-to-N throughput vs `num_threads`, faceted by dim.
- `sift_recall.png` — recall@{1,10,100} vs bits on SIFT1M.
- `synthetic_throughput.png` — encode / 1-to-N / M-to-N vs dim.

Useful flags: `--skip-sift`, `--skip-synthetic`, `--threads 1,4,8`,
`--bits 4,8`, `--no-show` (for headless CI).

Measured numbers from real hardware (Apple M3 and more as they come in)
live in [`docs/benchmarks.md`](docs/benchmarks.md). Headline from M3,
`dim=128, batch=10000, bits=8`: **~88M symmetric M-to-N ops/sec** and
**~2.8M encode/sec** on a single laptop.

---

## Layout and build

```
include/turboquant/
  turbo_quant.h          # Hadamard, Lloyd–Max, TurboQuantCode
  space_turbo_quant.h    # TurboQuantSpace + SIMD distance kernels
python/turboquant/
  bindings.cpp           # pybind11 bindings
  __init__.py
python/tests/            # pytest suite
python/benchmarks/       # run_benchmark.py (CSV + seaborn plots)
CMakeLists.txt           # scikit-build-core entry point
pyproject.toml
```

The library is header-only in spirit — all algorithmic code is in
`include/turboquant/`. Only the Python module (`bindings.cpp`) is compiled as
a shared object. A C++ consumer can depend on the headers alone and call the
same API directly.

Build flags worth knowing:

- `-DTURBOQUANT_HAVE_OPENMP` — set by CMake when OpenMP is detected; enables
  all `#pragma omp` blocks. Absent → sequential fallback, same API.
- Release build uses `-O3 -ffast-math -fno-finite-math-only`. The
  `fno-finite-math-only` is intentional: it keeps `inf`/`nan` handling sane
  while preserving vectorization.

### Tests

```bash
uv run pytest python/tests/ -v
```

Covers asymmetric/symmetric distances across `bits ∈ {4, 8}` and
`dim ∈ {32..4096}`, batch variants, zero-copy torch interop, and padding
correctness.

---

## Roadmap

The immediate priorities, in order:

1. **Publish wheels to PyPI** (cibuildwheel workflow in place; awaiting first tagged release)

Contributions welcome. The codebase is small (two headers, one bindings
file, ~2k lines) and deliberately kept that way — if a change makes it
harder to read, that is a reason to push back on it.

---

## License

MIT. See `LICENSE`.

## Citation

If you use this library in academic work, please cite the original TurboQuant
paper (ICLR 2026) in addition to this repository.

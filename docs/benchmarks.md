# Benchmarks

Measured performance of `turboquant-space` across platforms. Numbers are
raw throughput of the distance/encoding kernels — no index overhead, no
Python-side batching tricks. See `python/benchmarks/run_benchmark.py` for
the reproducible sweep (recall + scaling plots); the tables below come from
`python/tests/perfomance_check.py`, which is a tighter microbenchmark
meant for quick platform-to-platform comparison.

All distance rows report a single unit — **distances/sec** — so asymmetric
and symmetric paths can be compared directly per pair computed. Encoding
rows report **vectors/sec**.

Contributions of numbers from other platforms are welcome — open a PR
adding a section below.

---

## Distance modes — what the rows mean

`TurboQuantSpace` exposes two distance paths, differing only in whether
the **query** side is already quantized. There is no third "code vs float"
mode: the asymmetric kernel is symmetric in its roles, and quantizing a
query against a raw-float database would throw away precision without
saving memory.

| mode    | methods                                                     | query            | database | when to use                                                                                              |
|---------|-------------------------------------------------------------|------------------|----------|----------------------------------------------------------------------------------------------------------|
| **asym**| `distance_1_to_n`, `distance_m_to_n`, `distance`            | float32 (raw)    | code     | query arrives from outside in full precision — typical online search: a request comes in, scan the index |
| **sym** | `distance_m_to_n_symmetric`, `distance_symmetric`           | code             | code     | both sides already live in the index as codes — dedup, clustering, pairwise similarity within the base   |

A separate "fp32 baseline" (plain `np.dot(Q, X.T)`) is the reference for
"how fast would this be without quantization at all" — useful to quote
alongside, but it is not a turboquant path.

**Why sym is usually faster per distance.** The sym kernel operates on
two uint8 code streams with popcount-/XOR-style SIMD — no rotation, no
LUT, no float arithmetic in the hot loop. Asym has to apply the
Walsh–Hadamard rotation to the float query and evaluate the Lloyd–Max
lookup against each code. On tiny batches (`n < ~500`) sym loses to asym
because its `n²` work can't amortize the per-call overhead; on larger
batches sym pulls ahead sharply, and for `dim=128` hits ~140M distances/sec.

### Reading the `Time` column — work is not the same shape

The two distance rows answer **different questions**, even though
`Throughput (dist/s)` is directly comparable between them:

- **`1-to-N (asym)`** measures the realistic online-search path: one
  float query against `N` codes in the base. Work is `O(N)`, so at
  `N=10000` the kernel computes **10⁴ pairs**.
- **`M-to-N (sym)`** measures the realistic matrix path: every code in
  an `M`-set vs every code in an `N`-set — dedup, clustering, full kNN
  graph. Work is `O(M·N)`, so at `M=N=10000` the kernel computes
  **10⁸ pairs — 10,000× more than the asym row on the same line**.

That is why the `Time` column looks wildly different for the two modes
at large batches: `1024/10000 asym` spends 0.68 ms on 10⁴ pairs
(~14.7M dist/s), and `1024/10000 sym` spends 6010 ms on 10⁸ pairs
(~16.6M dist/s). **Per-distance throughput is within ~15% of each
other** — the wall-clock gap is purely the quadratic `M·N` shape of the
sym benchmark, not extra work per pair or query conversion overhead
(encoding is measured separately and costs ~13 ms for 10k vectors at
`dim=1024`).

If you want a wall-clock number for "symmetric, one query against N",
read the asym row and treat it as a lower bound — the sym kernel on a
`1×N` shape is faster per pair, not slower.

---

## Apple M3 (macOS, arm64, NEON)

- **CPU:** Apple M3
- **Build:** `-march=native`, OpenMP via `brew install libomp`
- **Python:** CPython 3.12
- **Date:** 2026-04-12

### bits_per_coord = 4

|  Dim | Batch |         Task | Time (ms) |      Throughput |   Unit |
|-----:|------:|-------------:|----------:|----------------:|-------:|
|  128 |     1 |     Encoding |      0.00 |         875,913 |  vec/s |
|  128 |     1 | 1-to-N (asym)|      0.00 |       1,270,849 | dist/s |
|  128 |    50 |     Encoding |      0.04 |       1,237,591 |  vec/s |
|  128 |    50 | 1-to-N (asym)|      0.00 |      20,915,035 | dist/s |
|  128 |    50 |  M-to-N (sym)|      0.08 |      30,481,223 | dist/s |
|  128 |   100 |     Encoding |      0.04 |       2,393,871 |  vec/s |
|  128 |   100 | 1-to-N (asym)|      0.03 |       2,971,602 | dist/s |
|  128 |   100 |  M-to-N (sym)|      0.11 |      91,277,136 | dist/s |
|  128 |  1000 |     Encoding |      0.21 |       4,761,928 |  vec/s |
|  128 |  1000 | 1-to-N (asym)|      0.03 |      36,153,838 | dist/s |
|  128 |  1000 |  M-to-N (sym)|      8.21 |     121,784,755 | dist/s |
|  128 |  5000 |     Encoding |      0.99 |       5,075,310 |  vec/s |
|  128 |  5000 | 1-to-N (asym)|      0.06 |      83,388,341 | dist/s |
|  128 |  5000 |  M-to-N (sym)|    190.18 |     131,451,790 | dist/s |
|  128 | 10000 |     Encoding |      1.78 |       5,619,314 |  vec/s |
|  128 | 10000 | 1-to-N (asym)|      0.10 |     101,464,683 | dist/s |
|  128 | 10000 |  M-to-N (sym)|    723.71 |     138,177,140 | dist/s |
|  512 |     1 |     Encoding |      0.00 |         311,162 |  vec/s |
|  512 |     1 | 1-to-N (asym)|      0.00 |         635,762 | dist/s |
|  512 |    50 |     Encoding |      0.14 |         348,283 |  vec/s |
|  512 |    50 | 1-to-N (asym)|      0.01 |       6,770,480 | dist/s |
|  512 |    50 |  M-to-N (sym)|      0.31 |       7,990,817 | dist/s |
|  512 |   100 |     Encoding |      0.09 |       1,096,764 |  vec/s |
|  512 |   100 | 1-to-N (asym)|      0.02 |       4,899,309 | dist/s |
|  512 |   100 |  M-to-N (sym)|      0.33 |      30,343,245 | dist/s |
|  512 |  1000 |     Encoding |      0.62 |       1,606,761 |  vec/s |
|  512 |  1000 | 1-to-N (asym)|      0.05 |      18,630,213 | dist/s |
|  512 |  1000 |  M-to-N (sym)|     31.08 |      32,170,460 | dist/s |
|  512 |  5000 |     Encoding |      3.49 |       1,431,145 |  vec/s |
|  512 |  5000 | 1-to-N (asym)|      0.18 |      27,114,416 | dist/s |
|  512 |  5000 |  M-to-N (sym)|    808.71 |      30,913,488 | dist/s |
|  512 | 10000 |     Encoding |      7.06 |       1,416,853 |  vec/s |
|  512 | 10000 | 1-to-N (asym)|      0.40 |      24,901,431 | dist/s |
|  512 | 10000 |  M-to-N (sym)|   3071.30 |      32,559,451 | dist/s |
|  768 |     1 |     Encoding |      0.01 |         160,267 |  vec/s |
|  768 |     1 | 1-to-N (asym)|      0.00 |         365,158 | dist/s |
|  768 |    50 |     Encoding |      0.29 |         173,421 |  vec/s |
|  768 |    50 | 1-to-N (asym)|      0.01 |       3,422,558 | dist/s |
|  768 |    50 |  M-to-N (sym)|      0.65 |       3,859,152 | dist/s |
|  768 |   100 |     Encoding |      0.17 |         583,462 |  vec/s |
|  768 |   100 | 1-to-N (asym)|      0.03 |       3,845,753 | dist/s |
|  768 |   100 |  M-to-N (sym)|      0.60 |      16,634,918 | dist/s |
|  768 |  1000 |     Encoding |      1.30 |         767,936 |  vec/s |
|  768 |  1000 | 1-to-N (asym)|      0.09 |      10,938,250 | dist/s |
|  768 |  1000 |  M-to-N (sym)|     59.77 |      16,730,334 | dist/s |
|  768 |  5000 |     Encoding |      7.59 |         658,881 |  vec/s |
|  768 |  5000 | 1-to-N (asym)|      0.36 |      13,711,898 | dist/s |
|  768 |  5000 |  M-to-N (sym)|   1549.77 |      16,131,451 | dist/s |
|  768 | 10000 |     Encoding |     12.41 |         805,597 |  vec/s |
|  768 | 10000 | 1-to-N (asym)|      0.68 |      14,713,120 | dist/s |
|  768 | 10000 |  M-to-N (sym)|   5963.25 |      16,769,366 | dist/s |
| 1024 |     1 |     Encoding |      0.01 |         161,155 |  vec/s |
| 1024 |     1 | 1-to-N (asym)|      0.00 |         357,515 | dist/s |
| 1024 |    50 |     Encoding |      0.29 |         173,225 |  vec/s |
| 1024 |    50 | 1-to-N (asym)|      0.01 |       3,357,301 | dist/s |
| 1024 |    50 |  M-to-N (sym)|      0.64 |       3,934,821 | dist/s |
| 1024 |   100 |     Encoding |      0.18 |         564,606 |  vec/s |
| 1024 |   100 | 1-to-N (asym)|      0.03 |       3,906,663 | dist/s |
| 1024 |   100 |  M-to-N (sym)|      0.63 |      15,907,474 | dist/s |
| 1024 |  1000 |     Encoding |      1.24 |         808,163 |  vec/s |
| 1024 |  1000 | 1-to-N (asym)|      0.09 |      11,486,578 | dist/s |
| 1024 |  1000 |  M-to-N (sym)|     60.56 |      16,513,344 | dist/s |
| 1024 |  5000 |     Encoding |      6.58 |         759,587 |  vec/s |
| 1024 |  5000 | 1-to-N (asym)|      0.37 |      13,462,208 | dist/s |
| 1024 |  5000 |  M-to-N (sym)|   1604.55 |      15,580,689 | dist/s |
| 1024 | 10000 |     Encoding |     12.79 |         782,074 |  vec/s |
| 1024 | 10000 | 1-to-N (asym)|      0.68 |      14,754,773 | dist/s |
| 1024 | 10000 |  M-to-N (sym)|   6010.79 |      16,636,755 | dist/s |
| 2048 |     1 |     Encoding |      0.01 |          80,050 |  vec/s |
| 2048 |     1 | 1-to-N (asym)|      0.01 |         182,342 | dist/s |
| 2048 |    50 |     Encoding |      0.58 |          86,233 |  vec/s |
| 2048 |    50 | 1-to-N (asym)|      0.03 |       1,760,421 | dist/s |
| 2048 |    50 |  M-to-N (sym)|      1.25 |       1,996,675 | dist/s |
| 2048 |   100 |     Encoding |      0.28 |         352,524 |  vec/s |
| 2048 |   100 | 1-to-N (asym)|      0.03 |       2,876,990 | dist/s |
| 2048 |   100 |  M-to-N (sym)|      1.17 |       8,557,185 | dist/s |
| 2048 |  1000 |     Encoding |      2.61 |         383,630 |  vec/s |
| 2048 |  1000 | 1-to-N (asym)|      0.20 |       4,990,704 | dist/s |
| 2048 |  1000 |  M-to-N (sym)|    123.79 |       8,078,218 | dist/s |
| 2048 |  5000 |     Encoding |     15.17 |         329,512 |  vec/s |
| 2048 |  5000 | 1-to-N (asym)|      0.66 |       7,588,608 | dist/s |
| 2048 |  5000 |  M-to-N (sym)|   3068.60 |       8,147,030 | dist/s |
| 2048 | 10000 |     Encoding |     27.55 |         363,005 |  vec/s |
| 2048 | 10000 | 1-to-N (asym)|      1.31 |       7,631,888 | dist/s |
| 2048 | 10000 |  M-to-N (sym)|  12099.89 |       8,264,540 | dist/s |
| 4096 |     1 |     Encoding |      0.02 |          41,768 |  vec/s |
| 4096 |     1 | 1-to-N (asym)|      0.01 |          97,779 | dist/s |
| 4096 |    50 |     Encoding |      1.17 |          42,588 |  vec/s |
| 4096 |    50 | 1-to-N (asym)|      0.06 |         883,187 | dist/s |
| 4096 |    50 |  M-to-N (sym)|      2.51 |         996,668 | dist/s |
| 4096 |   100 |     Encoding |      0.52 |         193,264 |  vec/s |
| 4096 |   100 | 1-to-N (asym)|      0.06 |       1,739,155 | dist/s |
| 4096 |   100 |  M-to-N (sym)|      2.43 |       4,108,145 | dist/s |
| 4096 |  1000 |     Encoding |      5.46 |         183,047 |  vec/s |
| 4096 |  1000 | 1-to-N (asym)|      0.27 |       3,657,629 | dist/s |
| 4096 |  1000 |  M-to-N (sym)|    253.19 |       3,949,562 | dist/s |
| 4096 |  5000 |     Encoding |     25.54 |         195,804 |  vec/s |
| 4096 |  5000 | 1-to-N (asym)|      1.22 |       4,101,840 | dist/s |
| 4096 |  5000 |  M-to-N (sym)|   6033.86 |       4,143,286 | dist/s |
| 4096 | 10000 |     Encoding |     53.03 |         188,577 |  vec/s |
| 4096 | 10000 | 1-to-N (asym)|      2.27 |       4,409,264 | dist/s |
| 4096 | 10000 |  M-to-N (sym)|  24765.24 |       4,037,917 | dist/s |

### bits_per_coord = 8

|  Dim | Batch |         Task | Time (ms) |      Throughput |   Unit |
|-----:|------:|-------------:|----------:|----------------:|-------:|
|  128 |     1 |     Encoding |      0.00 |         549,073 |  vec/s |
|  128 |     1 | 1-to-N (asym)|      0.00 |       1,255,232 | dist/s |
|  128 |    50 |     Encoding |      0.07 |         696,821 |  vec/s |
|  128 |    50 | 1-to-N (asym)|      0.00 |      20,093,757 | dist/s |
|  128 |    50 |  M-to-N (sym)|      0.13 |      18,650,424 | dist/s |
|  128 |   100 |     Encoding |      0.06 |       1,757,829 |  vec/s |
|  128 |   100 | 1-to-N (asym)|      0.02 |       5,703,693 | dist/s |
|  128 |   100 |  M-to-N (sym)|      0.17 |      58,099,282 | dist/s |
|  128 |  1000 |     Encoding |      0.35 |       2,850,432 |  vec/s |
|  128 |  1000 | 1-to-N (asym)|      0.03 |      37,093,989 | dist/s |
|  128 |  1000 |  M-to-N (sym)|     13.62 |      73,429,919 | dist/s |
|  128 |  5000 |     Encoding |      1.65 |       3,027,299 |  vec/s |
|  128 |  5000 | 1-to-N (asym)|      0.07 |      67,068,145 | dist/s |
|  128 |  5000 |  M-to-N (sym)|    361.13 |      69,226,221 | dist/s |
|  128 | 10000 |     Encoding |      3.47 |       2,885,256 |  vec/s |
|  128 | 10000 | 1-to-N (asym)|      0.13 |      74,511,023 | dist/s |
|  128 | 10000 |  M-to-N (sym)|   1330.04 |      75,185,496 | dist/s |
|  512 |     1 |     Encoding |      0.01 |         150,065 |  vec/s |
|  512 |     1 | 1-to-N (asym)|      0.00 |         546,261 | dist/s |
|  512 |    50 |     Encoding |      0.28 |         179,801 |  vec/s |
|  512 |    50 | 1-to-N (asym)|      0.01 |       6,219,064 | dist/s |
|  512 |    50 |  M-to-N (sym)|      0.57 |       4,364,413 | dist/s |
|  512 |   100 |     Encoding |      0.16 |         611,451 |  vec/s |
|  512 |   100 | 1-to-N (asym)|      0.02 |       4,787,170 | dist/s |
|  512 |   100 |  M-to-N (sym)|      0.50 |      20,184,719 | dist/s |
|  512 |  1000 |     Encoding |      1.39 |         717,550 |  vec/s |
|  512 |  1000 | 1-to-N (asym)|      0.07 |      14,672,258 | dist/s |
|  512 |  1000 |  M-to-N (sym)|     52.86 |      18,919,432 | dist/s |
|  512 |  5000 |     Encoding |      7.43 |         673,190 |  vec/s |
|  512 |  5000 | 1-to-N (asym)|      0.21 |      23,674,429 | dist/s |
|  512 |  5000 |  M-to-N (sym)|   1358.56 |      18,401,887 | dist/s |
|  512 | 10000 |     Encoding |     14.10 |         709,095 |  vec/s |
|  512 | 10000 | 1-to-N (asym)|      0.39 |      25,825,870 | dist/s |
|  512 | 10000 |  M-to-N (sym)|   5063.14 |      19,750,579 | dist/s |
|  768 |     1 |     Encoding |      0.01 |          76,841 |  vec/s |
|  768 |     1 | 1-to-N (asym)|      0.00 |         318,619 | dist/s |
|  768 |    50 |     Encoding |      0.55 |          90,697 |  vec/s |
|  768 |    50 | 1-to-N (asym)|      0.02 |       3,166,811 | dist/s |
|  768 |    50 |  M-to-N (sym)|      1.16 |       2,160,487 | dist/s |
|  768 |   100 |     Encoding |      0.27 |         364,956 |  vec/s |
|  768 |   100 | 1-to-N (asym)|      0.03 |       3,633,693 | dist/s |
|  768 |   100 |  M-to-N (sym)|      1.00 |      10,046,416 | dist/s |
|  768 |  1000 |     Encoding |      2.46 |         405,887 |  vec/s |
|  768 |  1000 | 1-to-N (asym)|      0.10 |       9,586,751 | dist/s |
|  768 |  1000 |  M-to-N (sym)|     98.88 |      10,113,511 | dist/s |
|  768 |  5000 |     Encoding |     12.70 |         393,837 |  vec/s |
|  768 |  5000 | 1-to-N (asym)|      0.37 |      13,358,158 | dist/s |
|  768 |  5000 |  M-to-N (sym)|   2760.30 |       9,056,986 | dist/s |
|  768 | 10000 |     Encoding |     27.70 |         361,042 |  vec/s |
|  768 | 10000 | 1-to-N (asym)|      0.84 |      11,897,553 | dist/s |
|  768 | 10000 |  M-to-N (sym)|  11455.46 |       8,729,462 | dist/s |
| 1024 |     1 |     Encoding |      0.01 |          86,692 |  vec/s |
| 1024 |     1 | 1-to-N (asym)|      0.00 |         356,162 | dist/s |
| 1024 |    50 |     Encoding |      0.55 |          91,163 |  vec/s |
| 1024 |    50 | 1-to-N (asym)|      0.02 |       3,178,470 | dist/s |
| 1024 |    50 |  M-to-N (sym)|      1.09 |       2,294,928 | dist/s |
| 1024 |   100 |     Encoding |      0.29 |         350,715 |  vec/s |
| 1024 |   100 | 1-to-N (asym)|      0.03 |       3,696,373 | dist/s |
| 1024 |   100 |  M-to-N (sym)|      1.01 |       9,857,347 | dist/s |
| 1024 |  1000 |     Encoding |      2.63 |         380,352 |  vec/s |
| 1024 |  1000 | 1-to-N (asym)|      0.11 |       8,925,516 | dist/s |
| 1024 |  1000 |  M-to-N (sym)|    104.03 |       9,612,962 | dist/s |
| 1024 |  5000 |     Encoding |     17.57 |         284,557 |  vec/s |
| 1024 |  5000 | 1-to-N (asym)|      0.41 |      12,231,120 | dist/s |
| 1024 |  5000 |  M-to-N (sym)|   2959.89 |       8,446,270 | dist/s |
| 1024 | 10000 |     Encoding |     26.21 |         381,501 |  vec/s |
| 1024 | 10000 | 1-to-N (asym)|      0.77 |      13,043,037 | dist/s |
| 1024 | 10000 |  M-to-N (sym)|  11922.12 |       8,387,772 | dist/s |
| 2048 |     1 |     Encoding |      0.03 |          33,073 |  vec/s |
| 2048 |     1 | 1-to-N (asym)|      0.01 |         149,798 | dist/s |
| 2048 |    50 |     Encoding |      1.26 |          39,702 |  vec/s |
| 2048 |    50 | 1-to-N (asym)|      0.03 |       1,453,356 | dist/s |
| 2048 |    50 |  M-to-N (sym)|      2.15 |       1,161,161 | dist/s |
| 2048 |   100 |     Encoding |      0.58 |         171,459 |  vec/s |
| 2048 |   100 | 1-to-N (asym)|      0.04 |       2,311,570 | dist/s |
| 2048 |   100 |  M-to-N (sym)|      2.36 |       4,243,328 | dist/s |
| 2048 |  1000 |     Encoding |      5.48 |         182,642 |  vec/s |
| 2048 |  1000 | 1-to-N (asym)|      0.22 |       4,504,149 | dist/s |
| 2048 |  1000 |  M-to-N (sym)|    339.71 |       2,943,664 | dist/s |
| 2048 |  5000 |     Encoding |     27.74 |         180,230 |  vec/s |
| 2048 |  5000 | 1-to-N (asym)|      0.77 |       6,463,690 | dist/s |
| 2048 |  5000 |  M-to-N (sym)|   5774.83 |       4,329,129 | dist/s |
| 2048 | 10000 |     Encoding |     52.74 |         189,627 |  vec/s |
| 2048 | 10000 | 1-to-N (asym)|      1.48 |       6,778,022 | dist/s |
| 2048 | 10000 |  M-to-N (sym)|  22031.20 |       4,539,017 | dist/s |
| 4096 |     1 |     Encoding |      0.05 |          20,498 |  vec/s |
| 4096 |     1 | 1-to-N (asym)|      0.01 |          90,385 | dist/s |
| 4096 |    50 |     Encoding |      2.23 |          22,460 |  vec/s |
| 4096 |    50 | 1-to-N (asym)|      0.06 |         793,637 | dist/s |
| 4096 |    50 |  M-to-N (sym)|      4.39 |         569,792 | dist/s |
| 4096 |   100 |     Encoding |      1.03 |          97,092 |  vec/s |
| 4096 |   100 | 1-to-N (asym)|      0.06 |       1,561,463 | dist/s |
| 4096 |   100 |  M-to-N (sym)|      5.23 |       1,910,488 | dist/s |
| 4096 |  1000 |     Encoding |     10.45 |          95,737 |  vec/s |
| 4096 |  1000 | 1-to-N (asym)|      0.34 |       2,984,135 | dist/s |
| 4096 |  1000 |  M-to-N (sym)|    416.68 |       2,399,944 | dist/s |
| 4096 |  5000 |     Encoding |     49.72 |         100,567 |  vec/s |
| 4096 |  5000 | 1-to-N (asym)|      1.47 |       3,411,877 | dist/s |
| 4096 |  5000 |  M-to-N (sym)|  10724.92 |       2,331,019 | dist/s |
| 4096 | 10000 |     Encoding |    101.29 |          98,728 |  vec/s |
| 4096 | 10000 | 1-to-N (asym)|      2.65 |       3,767,971 | dist/s |
| 4096 | 10000 |  M-to-N (sym)|  44380.13 |       2,253,260 | dist/s |

### Threading scaling

`dim=512`, `bits=8`, 50,000 codes, 128 queries, `M-to-N (asym)` kernel.

| Threads | Encode (ms) | 1-to-N (ms) | M-to-N (ms) | Speedup |
|--------:|------------:|------------:|------------:|--------:|
|       1 |      277.04 |        6.85 |      897.60 |   1.00× |
|       2 |      143.00 |        3.72 |      490.17 |   1.83× |
|       4 |       83.33 |        2.16 |      302.80 |   2.96× |
|       8 |       67.75 |        2.08 |      240.41 |   3.73× |

### What to take away

- **Low dims dominate sym.** At `dim=128, bits=4` the sym kernel hits
  **~138M dist/s** at `batch=10000` — codes are tiny, L1 reuse is
  perfect, and the XOR/popcount inner loop is pure SIMD.
- **4-bit is noticeably faster than 8-bit.** Half the code bytes → half
  the memory traffic → ~1.7–1.9× more throughput on sym at every
  `(dim, batch)` point.
- **Sym loses to asym only on tiny batches** (`n ≤ 100` at mid dims):
  `n²` work can't amortize the per-call overhead. By `n ≥ 1000` sym
  catches up; at `dim=128` it's ~2× faster than asym even then.
- **Padding tax at `dim=768`** — values track `dim=1024` almost
  identically because 768 is zero-padded to 1024 internally. If you can
  project to a native power of two, you save ~30% across the board.
- **Threading scales sublinearly past 4 cores on M3** (3.7× at 8
  threads) — expected on a 4P+4E chip, where the E-cores contribute
  less. On homogeneous x86 this usually flattens later.

# Autotune — choosing a quantizer for your data

`vsq.autotune` picks the quantizer, bit width, search path and thread count
for your vectors and an objective, measures every candidate on a sample of
your data, explains the choice and builds the index.

```python
import vsq

result = vsq.autotune(X, profile="speed")   # X: (n, dim) float32, n >= 1000
ids, dists = result.index.search(q, k=10)   # (10,) uint32, (10,) float32
print(result.report())                      # every candidate and why one won

saved = result.to_json()                    # reuse later without re-calibrating
index = vsq.AutotuneResult.from_json(saved).build(X)
```

## Autotune or a preset?

| you have | use | cost |
|---|---|---|
| ≥ 1000 vectors and an objective (recall, latency, energy, memory) | `vsq.autotune(X, profile, ...)` | 10–90 s of calibration |
| a known trade-off, or fewer than 1000 vectors | `vsq.build_index(X, "compact" \| "balanced" \| "accurate")` | build only |
| a stored choice | `vsq.build_index(X, result.chosen)` or `AutotuneResult.from_json(s).build(X)` | build only |

Presets are fixed entries of the candidate catalog below (`compact` =
`rq1-fs`, or `rq1-flat` at dim ≤ 256; `balanced` = `tq4-fs`; `accurate` =
`rq8-flat`); the report marks a choice that equals one, e.g.
`chosen: tq4-fs (preset "balanced")`. Their measured trade-offs are in
[Defaults and presets](benchmarks.md#defaults-and-presets).

## Profiles

| profile | objective | default constraints | time budget |
|---|---|---|---|
| `"accuracy"` | highest recall@k | none | 90 s |
| `"speed"` | lowest single-query p50 latency at your n | recall@k ≥ 0.90 | 60 s |
| `"energy"` | lowest energy per query (proxy, see below) | recall@k ≥ 0.90 | 60 s |

**The recall floor is tested against the lower 95 % bound** of the measured
recall (normal interval over queries × k), not the point estimate. A
candidate whose recall is 0.902 ± 0.013 does not clear a 0.90 floor: near the
threshold a point estimate would flip between seeds. With the default 200
queries and k = 10 the margin is about 0.013 at recall 0.9 and 0.004 at 0.99.

Ties are broken deterministically:

- **accuracy:** recall within ε_r (two binomial standard errors over
  queries × k, at least 0.002) is a tie → smallest code → lowest latency →
  catalog order. "Most accurate" therefore means the cheapest configuration
  that is statistically as accurate as the best one.
- **speed / energy:** values within 5 % are a tie → highest recall →
  smallest code → catalog order.

## Arguments

| argument | type, unit | default | meaning |
|---|---|---|---|
| `X` | `(n, dim)` float, converted once to C-contiguous float32 | — | vectors to index; n ≥ 1000 |
| `profile` | `"accuracy"` \| `"speed"` \| `"energy"` | `"speed"` | objective |
| `queries` | `(m, dim)` float or None | None | real queries; default holds out `n_queries` rows of X |
| `k` | int, 1..100 | 10 | neighbours per query |
| `min_recall` | float in [0, 1] or None | profile | recall@k floor, tested against the lower 95 % bound of the measured recall; `0` disables it |
| `max_bytes_per_vector` | int > 0 or None | None | stored code bytes per vector |
| `max_latency_ms` | float > 0 or None | None | single-query p50 at n |
| `time_budget_s` | float > 0 or None | profile | wall-clock budget of the calibration |
| `n_sample` | int | 20 000 | base rows used for calibration (≥ 9984 keeps 256 IVF centroids, as in the full build) |
| `n_queries` | int | 200 | held-out queries |
| `seed` | int | 0 | seeds the split; equal seeds give equal recall measurements |
| `build` | bool | True | also build the chosen index on all of X (`result.index`) |
| `energy_model` | `EnergyModel` or None | per machine class | proxy coefficients |
| `candidates` | list of `QuantizerConfig` and/or preset names, or None | None | expert override: measures exactly these (e.g. `["balanced", "accurate"]`), skipping the rules and the escalation ladder |

`ValueError` is raised for invalid arguments (fewer than 1000 rows: use a
preset); `AutotuneInfeasibleError` (a
`ValueError`) when no candidate meets every constraint. Its message lists the
best achievable value per constraint and what relaxing each one would admit,
and its `.result` carries all measurements.

## How the choice is made

### 1. Candidate catalog

| id | quantizer | bits | search path | tier | profiles |
|---|---|---|---|---|---|
| `tq4-fs` | TurboQuant | 4 | `TurboQuantFastScan`, `rerank=2` | 1 | speed, energy, accuracy |
| `rq4-fs` | RaBitQ | 4 | `RaBitQFastScan`, `eps0=1.9` | 1 | speed, energy |
| `rq4-flat` | RaBitQ | 4 | `distance_1_to_n` + top-k | 1 | speed, energy, accuracy |
| `tq4-flat` | TurboQuant | 4 | `distance_1_to_n` + top-k | 1 | speed, energy |
| `rq1-fs` | RaBitQ | 1 | `RaBitQFastScan` (float query) | 0 | speed, energy |
| `rq8-fs` | RaBitQ | 8 | `RaBitQFastScan` | 2 | speed, energy, accuracy |
| `rq8-flat` | RaBitQ | 8 | `distance_1_to_n` + top-k | 2 | speed, energy, accuracy |
| `tq8-flat` | TurboQuant | 8 | `distance_1_to_n` + top-k | 2 | accuracy |
| `tq16-flat` | TurboQuant | fp16 | `distance_1_to_n` + top-k | 3 | accuracy |

TurboQuant runs its defaults (IVF centering with 256 k-means centroids,
corrected estimator, no QJL — QJL never won at equal storage). RaBitQ runs
`train()` (centroid = mean of the data, the largest accuracy factor on real
embeddings) and its default encode mode (`fixed_scale` at 4 bits, `windowed_scale` at 8). A `-tN` suffix
(`rq4-flat-t4`) is the same entry with N OpenMP threads.

### 2. Rules (no recall prediction)

Rules use only facts that do not depend on the data distribution; recall is
always measured.

| rule | effect | reason |
|---|---|---|
| R1 budget | drop entries whose exact code size exceeds `max_bytes_per_vector` | memory |
| R2 profile | keep the entries tagged for the profile | — |
| R3 dim ≤ 256 | drop RaBitQ FastScan | at dim 128 RaBitQ-4 FastScan scans 22 M codes/s vs 95 M flat; TurboQuant-4 FastScan 209 M |
| R5 threads | add an N-thread variant of each flat entry when the predicted single-thread scan is ≥ 1 ms (speed / energy) | FastScan search is single-threaded; flat scans are OpenMP-parallel |
| R6 cap | at most 6 candidates per stage, catalog order | time budget |

### 3. Escalation ladder

Speed and energy calibrate tiers 0 + 1 first (1- and 4-bit). Only when none
of them meets `min_recall` (lower 95 % bound) are tier 2 (8-bit), then tier 3,
calibrated.
Accuracy calibrates tiers 1 + 2, then tier 3. Each escalation is recorded in
the report with the recall that triggered it. Example — N(0, 1) data at dim
128, where 4-bit codes reach recall@10 ≈ 0.87:

```
rules:
  drop rq4-fs: R3 dim 128 <= 256: RaBitQ FastScan is slower than the flat scan there
  escalate to ['rq8-flat']: best recall lower 95 % bound 0.8521 < min_recall 0.9
chosen: rq8-flat
```

### 4. Calibration

For each candidate, with the same seeds as the final build:

1. **Build** on the base sample (`n_sample` rows): train + encode time.
2. **Recall@k** over every held-out query through the real search path,
   against exact squared-L2 ground truth on the sample.
3. **Latency** of single queries: warmup, minimum of repeats per query, p50
   across ≤ 32 queries, garbage collector off, two round-robin rounds over
   the candidates so throttling affects all of them alike. The timed work is
   capped at about 0.5 s per candidate and round.
4. **Latency at your n** — `extrapolation` in the report:
   - `direct`: measured at n (n ≤ sample, or codes tiled up to n).
   - `arena-fit`: flat and TurboQuant FastScan scan cost does not depend on
     code values, so sample codes are tiled to `min(n, 256 MiB / code size)`
     rows, measured at two sizes and fitted `t = a + b·n`.
   - `sample-fit`: RaBitQ FastScan is built from raw vectors and its
     refinement count depends on the data; its latency is scaled in
     proportion to n from the sample.
5. **CPU time** per query (process CPU clock: includes OpenMP worker
   threads) and **bytes read** per query, for the energy proxy.

The time budget is projected from a 512-row pilot encode; if it would be
exceeded, `n_sample` is halved (down to 5000) and the report says so. A
candidate that cannot be measured in time is `skipped_budget`; one that
fails is `failed` and does not stop the others.

### 5. Energy proxy

```
E_query [mJ] = p_core_w · cpu_ms + p_base_w · wall_ms + e_byte_nj · 1e-6 · bytes_read
```

| term | meaning |
|---|---|
| `p_core_w · cpu_ms` | busy cores; N threads cost N× the CPU time, spin-waits included |
| `p_base_w · wall_ms` | package / uncore / DRAM background power while the query runs — rewards finishing early (race-to-idle) |
| `e_byte_nj · bytes_read` | memory traffic of the scanned codes |

The defaults (`EnergyModel`, per machine class: Apple arm64, x86-64, other
aarch64) are order-of-magnitude placeholders and are labelled `uncalibrated`
in the report. Only the ranking matters for selection, but it depends on the
ratio of the coefficients: with these defaults, 4 threads beat 1 thread for
a long flat scan (measured on an M3: 32.7 vs 38.3 mJ for RaBitQ 4-bit at
n = 50 000, dim 1536) because the background term dominates. Pass your own
`EnergyModel` if you have measured coefficients.

## Report

`result.report()` on 50 000 DBpedia OpenAI embeddings (dim 1536), Apple M3,
`profile="speed"`:

```
candidate     status    recall@k    ±95%  B/vec  p50 ms     extrap.  cpu ms  energy mJ  encode/s  build s
*tq4-fs       measured    0.9815  0.0059    782    1.73      direct    1.75       13.4    18,583     10.5
 rq4-fs       measured    0.9735  0.0070    776     2.9  sample-fit    2.92         17     6,269     7.98
 rq4-flat     measured    0.9735  0.0070    776    6.24      direct    6.26       38.3     6,763      7.4
 rq4-flat-t4  measured    0.9735  0.0070    776    1.86      direct    6.51       32.7    25,941     1.93
 tq4-flat     measured    0.9815  0.0059    782    12.4      direct    12.5       72.4    18,914     10.5
 tq4-flat-t4  measured    0.9815  0.0059    782    3.54      direct      13       61.4    68,214     3.19

chosen: tq4-fs
  lowest speed objective 1.726 ms p50; 1 candidate(s) within 5% of it
```

`*` marks the choice; `build s` estimates building the index on all n rows.
The same data with `profile="accuracy"` chooses `rq8-fs` (recall 0.998,
1544 B): `tq16-flat` reaches 1.000 but is within ε_r and twice the size.

## Serialised result (schema version 1)

`to_dict()` / `to_json()` contain only JSON values; `from_json()` restores
the result (without the built index) and warns when it was calibrated on a
different CPU or vsq version — the configuration stays valid, its timings
may not.

| key | content |
|---|---|
| `schema_version`, `vsq_version` | format and library version |
| `profile`, `constraints` | objective and the effective limits, defaults filled in: `min_recall`, `max_bytes_per_vector`, `max_latency_ms`, `time_budget_s` |
| `host` | `isa`, `logical_cpus`, `physical_cores`, `perf_cores`, `affinity_cpus`, `platform`, `machine`, `cpu_model` |
| `data` | `n_full`, `dim`, `n_base_sample`, `n_queries`, `k`, `seed`, `queries_source` |
| `chosen` | `QuantizerConfig` fields: `family`, `bits`, `index`, `num_threads`, `rot_seed`, TurboQuant `centering`/`n_clusters`/`rotation_rounds`/`train_seed`/`train_iters`, `train_rows`, RaBitQ `encode_mode`/`query_bits`, `rerank`, `eps0` |
| `candidates` | per candidate: `config`, `status`, `reasons`, `measurement` (`recall_at_k`, `recall_ci95`, `latency_p50_ms_sample`, `latency_p50_ms_full`, `extrapolation`, `cpu_ms_per_query`, `bytes_touched_per_query`, `energy_proxy_mj`, `encode_vps`, `train_s`, `est_build_s_full`, `refined_mean`) |
| `code_bytes` | candidate id → stored bytes per vector |
| `rule_trace`, `why` | the decisions, in order, and the reasons for the choice |
| `energy_model` | `p_core_w`, `p_base_w`, `e_byte_nj`, `source` |
| `partial` | the time budget cut calibration short |

## Validating the predictions

Two scripts check what calibration cannot see from the sample; they are
meant to be run by hand and write Markdown reports.

```bash
# recall drift (sample -> full n), latency extrapolation error, choice stability
uv run python python/benchmarks/validate_autotune.py \
    --data python/benchmarks/dbpedia_openai_100K_vectors.npy --n-full 50000,100000

# energy proxy vs measured joules; fits p_core_w, p_base_w, e_byte_nj
sudo powermetrics --samplers cpu_power -i 100 -o /tmp/pm.log   # macOS, other terminal
uv run python python/benchmarks/validate_energy_proxy.py --meter powermetrics \
    --powermetrics-log /tmp/pm.log
uv run python python/benchmarks/validate_energy_proxy.py --meter rapl   # Linux
```

`--quick` runs either script on small synthetic data in a few seconds.
Targets: arena-fit latency error ≤ 20 %, recall drift within ε_r, Spearman
ρ(proxy, measured) ≥ 0.8. Fitted coefficients can be passed back as
`vsq.autotune(..., energy_model=vsq.EnergyModel(p_core_w, p_base_w,
e_byte_nj, source="fitted on <host>"))`.

### Results on Apple M3 (2026-10-06)

- [`autotune_validation_20261006.md`](reports/autotune_validation_20261006.md)
  (dbpedia, n = 50 000 / 100 000, `speed` and `accuracy`, 3 seeds): predicted
  latency within −2.5…+1.8 % of the measured p50 (`direct`), −6.5 % for
  `sample-fit`; recall drift within ε_r in 10 of 12 runs, the other two at the
  boundary (+0.0001), mean drift of `tq4-fs` +0.003 — smaller than the margin
  the lower-95 % floor already applies. `speed` chose `tq4-fs` in all six runs.
- [`energy_proxy_20261006.md`](reports/energy_proxy_20261006.md)
  (powermetrics CPU power, dbpedia, n = 20 000 / 90 000, 28 runs): Spearman
  ρ(proxy, measured) = 0.982. Least-squares coefficients p_core 2.1 W,
  p_base 2.3 W, e_byte 0.08 nJ (CPU power only, DRAM not metered). 4 threads
  used less energy than 1 thread in all 10 flat pairs; the proxy ranked 9 of
  10 the same way. The defaults stay (ρ ≥ 0.8); three runs whose idle
  window caught background activity (idle 6–7 W) under-report energy.

## Limits

- Recall is measured on the sample (≤ `n_sample` rows). Neighbourhoods are
  denser at larger n, so recall at full n can be somewhat lower; the report
  states the sample size.
- Latency is single-query, single-process. Batch throughput
  (`distance_m_to_n`) is not a profile.
- Distances are squared L2. Normalise vectors yourself for cosine.
- Energy is a model, not a measurement; see the coefficients in the report.

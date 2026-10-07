"""Calibration: measure each candidate on a sample of the user's data.

Protocol per candidate (same seeds as the final build):
  1. build on the base sample (n_s rows) -> train_s, encode_vps
  2. recall@k over every held-out query through QuantizedIndex.search
  3. latency at the user's n (timing rounds, round-robin over candidates):
       flat / TurboQuant FastScan   scan cost is data-independent: codes are
           tiled to n_t = min(n_full, arena_bytes / code_size); n_t >= n_full
           -> measured ("direct"), else measured at n_t/2 and n_t and fitted
           t = a + b*n ("arena-fit")
       RaBitQ FastScan   built from raw vectors, cannot be tiled: scaled
           in proportion to n from the sample ("sample-fit")
  4. CPU and bytes per query -> energy proxy

The escalation ladder calibrates later stages only when no candidate meets
the recall floor -- lower 95 % bound of recall >= min_recall -- (speed /
energy) or always (accuracy), within the budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import CandidatePlan
from .energy import EnergyModel
from .index import QuantizedIndex, build_index
from .metrics import as_matrix, exact_topk, recall_at, recall_ci95, split_sample
from .timing import Clock, fit_linear, time_queries
from .types import (
    CandidateResult,
    Constraints,
    DataProfile,
    Measurement,
    QuantizerConfig,
)

_PILOT_ROWS = 512
_MIN_SAMPLE = 5000
# Per candidate outside encoding: timing rounds (bounded by time_queries'
# max_time_s, 2 rounds x <= 2 sizes x 0.5 s) plus the recall pass.
_TIMING_OVERHEAD_S = 2.5


@dataclass(frozen=True)
class Options:
    """Calibration knobs."""

    arena_bytes: int = 256 << 20  # tiled-code timing arena
    warmup: int = 3
    repeats: int = 5
    rounds: int = 2               # round-robin timing rounds over candidates


@dataclass(frozen=True)
class CalibrationSet:
    """Sample used by every candidate: base (n_s, dim), queries (m, dim) float32,
    gt (m, k) int64 exact top-k ids into base."""

    base: np.ndarray
    queries: np.ndarray
    gt: np.ndarray
    data: DataProfile


def prepare(X: np.ndarray, queries: np.ndarray | None, *, k: int, n_sample: int,
            n_queries: int, seed: int) -> CalibrationSet:
    """Seeded split (or user queries), exact ground truth on the base sample."""
    X = as_matrix(X, "X")
    n, dim = X.shape
    if queries is None:
        base_idx, query_idx = split_sample(n, n_sample, n_queries, seed)
        Q, source = X[query_idx], "holdout"
    else:
        Q = as_matrix(queries, "queries", dim)
        base_idx = np.sort(np.random.default_rng(seed).permutation(n)[:n_sample])
        source = "user"
    base = np.ascontiguousarray(X[base_idx])
    if k > base.shape[0]:
        raise ValueError(f"k={k} exceeds the calibration base of {base.shape[0]} rows")
    data = DataProfile(n_full=n, dim=dim, n_base_sample=base.shape[0], n_queries=Q.shape[0],
                       k=k, seed=seed, queries_source=source)
    return CalibrationSet(base, np.ascontiguousarray(Q), exact_topk(base, Q, k), data)


def shrink(cset: CalibrationSet, n_rows: int, seed: int) -> CalibrationSet:
    """Seeded subset of the base (ground truth recomputed)."""
    if n_rows >= cset.base.shape[0]:
        return cset
    idx = np.sort(np.random.default_rng(seed + 1).choice(cset.base.shape[0], n_rows,
                                                         replace=False))
    base = np.ascontiguousarray(cset.base[idx])
    data = DataProfile(**{**cset.data.to_dict(), "n_base_sample": n_rows})
    return CalibrationSet(base, cset.queries, exact_topk(base, cset.queries, data.k), data)


# ---------------------------------------------------------------------------


@dataclass
class _Built:
    """Per-candidate state between the build/recall phase and timing."""

    config: QuantizerConfig
    index: QuantizedIndex
    recall: float
    refined_frac: float  # mean share of the sample re-scored with the full code
    train_s: float
    encode_vps: float
    best: dict = field(default_factory=dict)  # n -> (wall_ms, cpu_ms), min over rounds


def bytes_per_query(index: QuantizedIndex, n: int, refined_frac: float, k: int) -> int:
    """Code bytes one query reads at n codes."""
    cfg, size = index.config, index.code_size_bytes
    padded = int(index.space.padded_dim())
    if cfg.index == "flat":
        return n * size
    if cfg.family == "turboquant":  # 4-bit FastScan block codes + reranked full codes
        return n * padded // 2 + k * (cfg.rerank or 4) * size
    return n * padded // 8 + int(refined_frac * n) * size  # 1-bit stage + refinement


def _timing_sizes(item: _Built, n_full: int, opts: Options) -> list[int]:
    """Index sizes to time: the sample itself, n_full, or two arena points."""
    n_s = len(item.index)
    if n_full <= n_s or item.index.codes is None:
        return [n_s]
    n_t = min(n_full, max(n_s, opts.arena_bytes // item.index.code_size_bytes))
    return [n_full] if n_t >= n_full else [n_t // 2, n_t]


def _time_round(item: _Built, queries: np.ndarray, k: int, n_full: int, clock: Clock,
                opts: Options) -> None:
    for n in _timing_sizes(item, n_full, opts):
        index = item.index if n == len(item.index) else item.index.tiled(n)
        wall, cpu = time_queries(lambda q, ix=index: ix.search(q, k), queries, clock,
                                 opts.warmup, opts.repeats)
        prev = item.best.get(n)
        item.best[n] = (wall, cpu) if prev is None else (min(prev[0], wall), min(prev[1], cpu))


def _finish(item: _Built, data: DataProfile, energy: EnergyModel, m: int) -> Measurement:
    """Latency / CPU / energy at n_full from the timed sizes."""
    n_full, n_s = data.n_full, len(item.index)
    points = sorted(item.best.items())
    n_ref, (wall_ref, cpu_ref) = points[-1]
    if len(points) == 2:
        alpha, beta = fit_linear([(n, w) for n, (w, _) in points])
        wall_full, kind = max(alpha + beta * n_full, wall_ref), "arena-fit"
        wall_sample = max(alpha + beta * n_s, 1e-6)
    elif n_ref >= n_full:
        wall_full, kind = wall_ref, "direct"
        wall_sample = wall_ref * n_s / n_ref
    else:
        wall_full, kind = wall_ref * n_full / n_ref, "sample-fit"
        wall_sample = wall_ref
    cpu_full = cpu_ref * wall_full / max(wall_ref, 1e-12)
    nbytes = bytes_per_query(item.index, n_full, item.refined_frac, data.k)
    is_rq_fs = item.config.family == "rabitq" and item.config.index == "fastscan"
    return Measurement(
        recall_at_k=item.recall,
        recall_ci95=recall_ci95(item.recall, m, data.k),
        latency_p50_ms_sample=wall_sample,
        latency_p50_ms_full=wall_full,
        extrapolation=kind,
        cpu_ms_per_query=cpu_full,
        bytes_touched_per_query=nbytes,
        energy_proxy_mj=energy.query_mj(cpu_full, wall_full, nbytes),
        encode_vps=item.encode_vps,
        train_s=item.train_s,
        est_build_s_full=item.train_s + n_full / max(item.encode_vps, 1e-9),
        refined_mean=item.refined_frac * n_s if is_rq_fs else None,
    )


def _build_and_recall(config: QuantizerConfig, cset: CalibrationSet) -> _Built:
    index, stats = build_index(config, cset.base)
    k = cset.data.k
    pred = np.empty((cset.queries.shape[0], k), dtype=np.int64)
    refined = 0
    for i, q in enumerate(cset.queries):
        ids, _, r = index.search_stats(q, k)
        pred[i] = ids
        refined += r
    frac = refined / (cset.queries.shape[0] * len(index))
    return _Built(config, index, recall_at(pred, cset.gt, k), frac, stats.train_s,
                  stats.encode_vps)


def measure_stage(configs: tuple[QuantizerConfig, ...], cset: CalibrationSet, clock: Clock,
                  energy: EnergyModel, opts: Options, deadline_ns: int) -> list[CandidateResult]:
    """Build + recall each config, then round-robin timing; respects the deadline."""
    built: list[_Built] = []
    results: dict[str, CandidateResult] = {}
    for cfg in configs:
        if clock.perf_ns() > deadline_ns:
            results[cfg.name] = CandidateResult(cfg, None, "skipped_budget",
                                                ("time budget exhausted before build",))
            continue
        try:
            built.append(_build_and_recall(cfg, cset))
        except Exception as exc:  # noqa: BLE001 - one native failure must not abort the run
            results[cfg.name] = CandidateResult(cfg, None, "failed",
                                                (f"{type(exc).__name__}: {exc}",))
    for _ in range(opts.rounds):
        for item in built:
            if item.config.name in results or (item.best and clock.perf_ns() > deadline_ns):
                continue
            try:
                _time_round(item, cset.queries, cset.data.k, cset.data.n_full, clock, opts)
            except Exception as exc:  # noqa: BLE001
                results[item.config.name] = CandidateResult(
                    item.config, None, "failed", (f"timing: {type(exc).__name__}: {exc}",))
    m = cset.queries.shape[0]
    for item in built:
        if item.config.name in results:
            continue
        if not item.best:
            results[item.config.name] = CandidateResult(
                item.config, None, "skipped_budget", ("time budget exhausted before timing",))
            continue
        results[item.config.name] = CandidateResult(
            item.config, _finish(item, cset.data, energy, m), "measured")
    return [results[c.name] for c in configs]


def plan_sample_size(configs: tuple[QuantizerConfig, ...], cset: CalibrationSet, clock: Clock,
                     budget_s: float, trace: list[str]) -> int:
    """Largest n_s (halving, floor 5000) whose projected cost fits 80 % of the budget."""
    pilot = cset.base[:_PILOT_ROWS]
    per_row_s = 0.0
    for cfg in configs:
        start = clock.perf_ns()
        try:
            build_index(cfg, pilot)
        except Exception:  # noqa: BLE001, S112 - the failure is reported by measure_stage
            continue
        per_row_s += (clock.perf_ns() - start) / 1e9 / pilot.shape[0]
    n_s = cset.base.shape[0]
    overhead = _TIMING_OVERHEAD_S * len(configs)
    while n_s > _MIN_SAMPLE and n_s * per_row_s + overhead > 0.8 * budget_s:
        n_s = max(_MIN_SAMPLE, n_s // 2)
    if n_s < cset.base.shape[0]:
        trace.append(f"budget: n_sample {cset.base.shape[0]} -> {n_s} (projected "
                     f"{cset.base.shape[0] * per_row_s + overhead:.1f} s > 80 % of {budget_s:g} s)")
    return n_s


def calibrate(plan: CandidatePlan, cset: CalibrationSet, profile: str, constraints: Constraints,
              clock: Clock, energy: EnergyModel, opts: Options | None = None
              ) -> tuple[list[CandidateResult], list[str], bool, CalibrationSet]:
    """Run the escalation ladder. Returns (results, trace, partial, final set)."""
    opts = opts or Options()
    trace = list(plan.trace)
    started = clock.perf_ns()
    deadline = started + int(constraints.time_budget_s * 1e9)
    if plan.first:
        n_s = plan_sample_size(plan.first, cset, clock, constraints.time_budget_s, trace)
        cset = shrink(cset, n_s, cset.data.seed)
    results: list[CandidateResult] = []
    partial = False
    for i, stage in enumerate(plan.stages):
        if i > 0:
            best = max((r.measurement.recall_lower95 for r in results if r.measurement),
                       default=0.0)
            if profile != "accuracy" and best >= constraints.min_recall:
                break
            if clock.perf_ns() > deadline:
                trace.append(f"escalation to {[c.name for c in stage]} skipped: budget exhausted")
                partial = True
                break
            why = ("the accuracy profile measures every tier" if profile == "accuracy" else
                   f"best recall lower 95 % bound {best:.4f} < min_recall "
                   f"{constraints.min_recall:g}")
            trace.append(f"escalate to {[c.name for c in stage]}: {why}")
        results.extend(measure_stage(stage, cset, clock, energy, opts, deadline))
    partial = partial or any(r.status == "skipped_budget" for r in results)
    trace.append(f"calibration took {(clock.perf_ns() - started) / 1e9:.1f} s of the "
                 f"{constraints.time_budget_s:g} s budget")
    return results, trace, partial, cset

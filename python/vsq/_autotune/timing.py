"""Query timing: warmup, min of repeats per query, p50 across queries.

Wall time uses ``perf_counter_ns``; CPU time uses ``process_time_ns``, which
includes OpenMP worker threads (verified on macOS: 8 threads report ~6x the
wall time), so cpu_ms is the CPU spent by all threads of one query.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

MAX_TIMING_QUERIES = 32


class Clock(Protocol):
    def perf_ns(self) -> int: ...
    def cpu_ns(self) -> int: ...


class RealClock:
    """Wall and process-CPU nanosecond clocks."""

    def perf_ns(self) -> int:
        return time.perf_counter_ns()

    def cpu_ns(self) -> int:
        return time.process_time_ns()


def time_queries(search_fn: Callable[[np.ndarray], object], queries: np.ndarray,
                 clock: Clock, warmup: int = 3, repeats: int = 5,
                 max_time_s: float | None = 0.5) -> tuple[float, float]:
    """(p50 wall ms, mean CPU ms) per single query.

    queries: (m, dim) float32; at most MAX_TIMING_QUERIES are used. Each
    query's wall time is the minimum over the repeats; p50 is the median of
    those minima. CPU time is the total over all timed calls divided by their
    count. max_time_s bounds the timed loop: the warmup calls estimate the
    per-call cost and the repeats (then the query count) are reduced to fit,
    never below one call per query for >= 3 queries. The garbage collector is
    off while timing and always restored.
    """
    if repeats < 1 or warmup < 1:
        raise ValueError(f"repeats and warmup must be >= 1, got {repeats}, {warmup}")
    qs = queries[:MAX_TIMING_QUERIES]
    if len(qs) == 0:
        raise ValueError("time_queries needs at least one query")
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = clock.perf_ns()
        for i in range(warmup):
            search_fn(qs[i % len(qs)])
        if max_time_s is not None:
            per_call_s = max((clock.perf_ns() - start) / 1e9 / warmup, 1e-9)
            calls = max(1, int(max_time_s / per_call_s))
            repeats = max(1, min(repeats, calls // len(qs)))
            qs = qs[:max(3, min(len(qs), calls))]
        best = np.full(len(qs), np.inf)
        cpu_start = clock.cpu_ns()
        for _ in range(repeats):
            for i, q in enumerate(qs):
                start = clock.perf_ns()
                search_fn(q)
                best[i] = min(best[i], clock.perf_ns() - start)
        cpu_total = clock.cpu_ns() - cpu_start
    finally:
        if was_enabled:
            gc.enable()
    calls = repeats * len(qs)
    return float(np.median(best)) / 1e6, cpu_total / calls / 1e6


def fit_linear(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares (alpha, beta) of t = alpha + beta * n; needs >= 2 distinct n."""
    if len(points) < 2:
        raise ValueError(f"fit_linear needs >= 2 points, got {len(points)}")
    n = np.array([p[0] for p in points], dtype=np.float64)
    t = np.array([p[1] for p in points], dtype=np.float64)
    if np.ptp(n) == 0:
        raise ValueError("fit_linear needs at least two distinct n")
    beta, alpha = np.polyfit(n, t, 1)
    return float(alpha), float(beta)

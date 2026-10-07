"""``vsq.autotune``: choose and build the best quantizer for data + profile."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .calibrate import Options, calibrate, prepare
from .candidates import CandidatePlan, exact_code_size, generate_candidates
from .energy import EnergyModel, default_energy_model
from .host import probe_host
from .index import build_index
from .metrics import as_matrix
from .result import AutotuneResult
from .select import AutotuneInfeasibleError, select
from .timing import Clock, RealClock
from .types import PROFILES, Constraints, QuantizerConfig

_MIN_ROWS = 1000
_MAX_K = 100


def autotune(
    X,
    profile: str = "speed",
    *,
    queries=None,
    k: int = 10,
    min_recall: float | None = None,
    max_bytes_per_vector: int | None = None,
    max_latency_ms: float | None = None,
    time_budget_s: float | None = None,
    n_sample: int = 20_000,
    n_queries: int = 200,
    seed: int = 0,
    build: bool = True,
    energy_model: EnergyModel | None = None,
    candidates: Sequence[QuantizerConfig] | None = None,
    clock: Clock | None = None,
    options: Options | None = None,
) -> AutotuneResult:
    """Pick the quantizer configuration that best serves ``profile`` on X.

    Args:
      X: (n, dim) float vectors to index; converted once to C-contiguous
        float32. n >= 1000.
      profile: "accuracy" (max recall@k), "speed" (min single-query p50
        latency at n), or "energy" (min energy per query, proxy).
      queries: optional (m, dim) real queries; default holds out n_queries
        rows of X.
      k: neighbours per query, 1..100.
      min_recall, max_bytes_per_vector, max_latency_ms, time_budget_s:
        constraints; None keeps the profile default (speed/energy:
        min_recall 0.90, budget 60 s; accuracy: no floor, budget 90 s).
        min_recall=0 disables the floor.
      n_sample: base rows used for calibration (>= 9984 keeps 256 IVF
        centroids, as in the full build).
      seed: seeds the split; equal seeds give equal recall measurements.
      build: also build the chosen index on all of X into ``result.index``.
      energy_model: proxy coefficients (default: uncalibrated placeholders
        for this machine class).
      candidates: expert override; skips the rules and the escalation ladder.
      clock, options: timing clock and calibration knobs (tests / experts).

    Returns: AutotuneResult (``chosen``, every measurement, ``report()``,
      ``to_json()``, ``index``).
    Raises: ValueError for invalid arguments; AutotuneInfeasibleError (a
      ValueError) when no candidate meets every constraint -- its ``result``
      holds all measurements.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
    if not 1 <= k <= _MAX_K:
        raise ValueError(f"k must be in [1, {_MAX_K}], got {k}")
    if n_sample < 1 or n_queries < 1:
        raise ValueError(f"n_sample and n_queries must be >= 1, got {n_sample}, {n_queries}")
    X = as_matrix(X, "X")
    held_out = n_queries if queries is None else 0
    need = max(_MIN_ROWS, held_out + 10 * k)
    if X.shape[0] < need:
        raise ValueError(f"autotune needs n >= {need} rows (got {X.shape[0]}); for smaller "
                         "data build a QuantizerConfig directly")
    constraints = Constraints.for_profile(
        profile, min_recall=min_recall, max_bytes_per_vector=max_bytes_per_vector,
        max_latency_ms=max_latency_ms, time_budget_s=time_budget_s)
    clock = clock or RealClock()
    host = probe_host()
    energy = energy_model or default_energy_model(host)

    cset = prepare(X, queries, k=k, n_sample=n_sample, n_queries=n_queries, seed=seed)
    if candidates is not None:
        configs = tuple(candidates)
        if not configs:
            raise ValueError("candidates must not be empty")
        plan = CandidatePlan((configs,), ("expert candidate list: rules and escalation skipped",))
    else:
        plan = generate_candidates(profile, constraints, cset.data, host)
    results, trace, partial, cset = calibrate(plan, cset, profile, constraints, clock, energy,
                                              options or Options())
    code_bytes = {r.config.name: exact_code_size(r.config, X.shape[1]) for r in results}

    from .. import __version__

    def make(chosen, why) -> AutotuneResult:
        return AutotuneResult(
            profile=profile, constraints=constraints, host=host, data=cset.data,
            chosen=chosen, candidates=tuple(results), code_bytes=code_bytes,
            rule_trace=tuple(trace), why=tuple(why), energy_model=energy,
            vsq_version=__version__, partial=partial)

    try:
        chosen, why = select(results, profile, constraints, code_bytes,
                             cset.queries.shape[0], k)
    except AutotuneInfeasibleError as exc:
        exc.result = make(None, ())
        raise
    result = make(chosen, why)
    if build:
        result = replace(result, index=build_index(chosen, X)[0])
    return result

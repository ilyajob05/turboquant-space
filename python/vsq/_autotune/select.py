"""Choose the configuration: feasibility, profile objective, tie-breaks.

    accuracy  max recall (ties within eps_r) -> min bytes -> min latency -> order
    speed     min p50 at n_full (ties within 5 %) -> max recall -> min bytes -> order
    energy    min energy proxy (ties within 5 %) -> max recall -> min bytes -> order

"order" is the candidate order (catalog order), so the choice never depends
on the order measurements arrived in. eps_r is two binomial standard errors.
"""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import recall_tie_tolerance
from .types import CandidateResult, Constraints, QuantizerConfig

_REL_TIE = 0.05  # speed / energy: values within 5 % are a tie
_CONSTRAINTS = ("min_recall", "max_latency_ms", "max_bytes_per_vector")


class AutotuneInfeasibleError(ValueError):
    """No calibrated candidate meets every constraint. ``result`` holds the
    AutotuneResult with all measurements (its ``chosen`` is None)."""

    def __init__(self, message: str, result=None) -> None:
        super().__init__(message)
        self.result = result


def violations(r: CandidateResult, c: Constraints, code_bytes: dict[str, int]) -> list[str]:
    """Constraint names a measured candidate violates (empty = feasible).

    code_bytes maps candidate name -> stored bytes per vector.
    """
    m = r.measurement
    out = []
    if m.recall_lower95 < c.min_recall:
        out.append("min_recall")
    if c.max_latency_ms is not None and m.latency_p50_ms_full > c.max_latency_ms:
        out.append("max_latency_ms")
    if (c.max_bytes_per_vector is not None
            and code_bytes[r.config.name] > c.max_bytes_per_vector):
        out.append("max_bytes_per_vector")
    return out


def select(results: Sequence[CandidateResult], profile: str, constraints: Constraints,
           code_bytes: dict[str, int], m: int, k: int) -> tuple[QuantizerConfig, list[str]]:
    """(chosen config, explanation lines); raises AutotuneInfeasibleError.

    results in catalog order; m held-out queries and k neighbours size the
    recall tie tolerance.
    """
    order = {r.config.name: i for i, r in enumerate(results)}
    measured = [r for r in results if r.measurement is not None]
    feasible = [r for r in measured if not violations(r, constraints, code_bytes)]
    if not feasible:
        raise AutotuneInfeasibleError(_infeasible_message(measured, results, constraints,
                                                          code_bytes))

    def nbytes(r: CandidateResult) -> int:
        return code_bytes[r.config.name]

    if profile == "accuracy":
        top = max(r.measurement.recall_at_k for r in feasible)
        eps = recall_tie_tolerance(top, m, k)
        tied = [r for r in feasible if r.measurement.recall_at_k >= top - eps]
        win = min(tied, key=lambda r: (nbytes(r), r.measurement.latency_p50_ms_full,
                                       order[r.config.name]))
        why = [(f"highest recall@{k} {top:.4f}; {len(tied)} candidate(s) within "
                f"eps_r={eps:.4f} of it")]
        if len(tied) > 1:
            why.append(f"tie broken by the smallest code ({nbytes(win)} B/vector), then latency")
        return win.config, why

    if profile == "speed":
        metric, unit = (lambda r: r.measurement.latency_p50_ms_full), "ms p50"
    else:
        metric, unit = (lambda r: r.measurement.energy_proxy_mj), "mJ/query (proxy)"
    best = min(metric(r) for r in feasible)
    tied = [r for r in feasible if metric(r) <= best * (1.0 + _REL_TIE)]
    win = min(tied, key=lambda r: (-r.measurement.recall_at_k, nbytes(r), order[r.config.name]))
    why = [(f"lowest {profile} objective {best:.4g} {unit}; {len(tied)} candidate(s) within "
            f"{_REL_TIE:.0%} of it")]
    if len(tied) > 1:
        why.append(f"tie broken by the highest recall ({win.measurement.recall_at_k:.4f}), "
                   "then the smallest code")
    return win.config, why


def _infeasible_message(measured: list[CandidateResult], results: Sequence[CandidateResult],
                        c: Constraints, code_bytes: dict[str, int]) -> str:
    """Best achievable value per active constraint, and what relaxing one admits."""
    if not measured:
        states = ", ".join(f"{r.config.name}: {r.status}" for r in results) or "no candidates"
        return f"no candidate could be measured ({states})"
    lines = ["no candidate meets every constraint:"]
    if c.min_recall > 0:
        r = max(measured, key=lambda r: r.measurement.recall_lower95)
        lines.append(f"  min_recall={c.min_recall:g} (lower 95 % bound): best "
                     f"{r.measurement.recall_lower95:.4f}, recall {r.measurement.recall_at_k:.4f} "
                     f"({r.config.name})")
    if c.max_latency_ms is not None:
        r = min(measured, key=lambda r: r.measurement.latency_p50_ms_full)
        lines.append(f"  max_latency_ms={c.max_latency_ms:g}: best "
                     f"{r.measurement.latency_p50_ms_full:.3g} ms ({r.config.name})")
    if c.max_bytes_per_vector is not None:
        r = min(measured, key=lambda r: code_bytes[r.config.name])
        lines.append(f"  max_bytes_per_vector={c.max_bytes_per_vector}: smallest "
                     f"{code_bytes[r.config.name]} B ({r.config.name})")
    for name in _CONSTRAINTS:
        admitted = [r.config.name for r in measured
                    if violations(r, c, code_bytes) == [name]]
        if admitted:
            lines.append(f"  relaxing {name} alone admits: {', '.join(admitted)}")
    return "\n".join(lines)

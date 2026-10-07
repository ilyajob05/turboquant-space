"""Plain-text report of an AutotuneResult."""

from __future__ import annotations

_HEADER = ("candidate", "status", "recall@k", "±95%", "B/vec", "p50 ms", "extrap.",
           "cpu ms", "energy mJ", "encode/s", "build s")


def _row(result, cand) -> tuple[str, ...]:
    name = cand.config.name
    mark = "*" if result.chosen is not None and cand.config == result.chosen else " "
    nbytes = str(result.code_bytes.get(name, "?"))
    m = cand.measurement
    if m is None:
        return (mark + name, cand.status, "—", "—", nbytes) + ("—",) * 6
    lo, hi = m.recall_ci95
    return (mark + name, cand.status, f"{m.recall_at_k:.4f}", f"{(hi - lo) / 2:.4f}", nbytes,
            f"{m.latency_p50_ms_full:.3g}", m.extrapolation, f"{m.cpu_ms_per_query:.3g}",
            f"{m.energy_proxy_mj:.3g}", f"{m.encode_vps:,.0f}", f"{m.est_build_s_full:.3g}")


def _table(rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt(r: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) if i < 2 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(r, widths)))

    return [fmt(rows[0]), "  ".join("-" * w for w in widths)] + [fmt(r) for r in rows[1:]]


def format_report(result) -> str:
    """Header, candidate table (* = chosen), rule trace, reasons, caveats."""
    c, d, h = result.constraints, result.data, result.host
    em = result.energy_model
    lines = [
        f"vsq.autotune  profile={result.profile}  vsq {result.vsq_version}",
        (f"constraints   min_recall={c.min_recall:g}  max_bytes_per_vector={c.max_bytes_per_vector}"
         f"  max_latency_ms={c.max_latency_ms}  time_budget_s={c.time_budget_s:g}"),
        (f"host          {h.cpu_model} ({h.machine}, {h.isa}), {h.physical_cores} cores"
         + (f", {h.perf_cores} performance" if h.perf_cores else "")),
        (f"data          n={d.n_full:,}  dim={d.dim}  calibrated on {d.n_base_sample:,} base "
         f"rows, {d.n_queries} {d.queries_source} queries, k={d.k}, seed={d.seed}"),
        (f"energy model  {em.source} (p_core={em.p_core_w} W, p_base={em.p_base_w} W, "
         f"e_byte={em.e_byte_nj} nJ)"),
        "",
    ]
    lines += _table([_HEADER] + [_row(result, cand) for cand in result.candidates])
    not_measured = [cand for cand in result.candidates if cand.reasons]
    if not_measured:
        lines += ["", "not measured:"] + [f"  {cand.config.name}: {'; '.join(cand.reasons)}"
                                          for cand in not_measured]
    if result.rule_trace:
        lines += ["", "rules:"] + [f"  {t}" for t in result.rule_trace]
    lines += ["", f"chosen: {result.chosen.name if result.chosen else 'none'}"]
    lines += [f"  {w}" for w in result.why]
    sampled = d.n_base_sample < d.n_full
    caveats = [
        f"recall measured at n={d.n_base_sample:,}; at n={d.n_full:,} it can be lower"
        if sampled else None,
        "latency is the p50 of single queries (min of repeats); 'arena-fit' and "
        "'sample-fit' rows are extrapolated to the full n" if sampled else None,
        "energy is a proxy (CPU time, latency, bytes read), not a measurement",
        "calibration was cut short by the time budget" if result.partial else None,
    ]
    lines += ["", "caveats:"] + [f"  - {cv}" for cv in caveats if cv]
    return "\n".join(lines)

#!/usr/bin/env python3
"""Validate vsq.autotune's sample-based predictions at the full n.

For every (n_full, profile, seed) the script runs vsq.autotune on the first
n_full rows, then rebuilds the chosen configuration on all base rows (the
same held-out queries excluded) and measures what calibration predicted:

  recall drift    recall@k at n_sample (autotune) minus recall@k at n_full
                  (exact ground truth over the full base, chunked); drift
                  beyond eps_r means autotune overstates recall at full n
  latency error   predicted latency_p50_ms_full / measured p50 at n_full - 1,
                  per extrapolation kind (direct / arena-fit / sample-fit)
  stability       the chosen configuration across seeds

Output: Markdown (one table per check) at --out-dir/autotune_validation_<date>.md
(default docs/reports; --quick writes to python/benchmarks/results).

Usage:
  uv run python python/benchmarks/validate_autotune.py --quick
  uv run python python/benchmarks/validate_autotune.py \\
      --data python/benchmarks/dbpedia_openai_100K_vectors.npy --n-full 50000,100000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import vsq
from vsq._autotune.index import build_index
from vsq._autotune.metrics import (
    as_matrix,
    exact_topk,
    recall_at,
    recall_tie_tolerance,
    split_sample,
)
from vsq._autotune.timing import RealClock, time_queries

_GT_BYTES = 256 << 20  # float64 distance block budget of the exact ground truth
_REPO = Path(__file__).resolve().parents[2]


def _today():
    return datetime.now(timezone.utc).date()


@dataclass
class Check:
    n_full: int
    profile: str
    seed: int
    chosen: str
    extrapolation: str
    recall_sample: float
    recall_full: float
    eps_r: float
    predicted_ms: float
    measured_ms: float

    @property
    def drift(self) -> float:
        return self.recall_sample - self.recall_full

    @property
    def latency_error(self) -> float:
        return self.predicted_ms / self.measured_ms - 1.0


def check_one(X: np.ndarray, profile: str, seed: int, args) -> Check:
    n = X.shape[0]
    result = vsq.autotune(X, profile, k=args.k, n_sample=args.n_sample, n_queries=args.n_queries,
                          seed=seed, time_budget_s=args.time_budget, build=False)
    cfg = result.chosen
    m = next(c.measurement for c in result.candidates if c.config == cfg)
    # Same held-out queries as autotune (prepare() uses split_sample with this seed).
    _, query_idx = split_sample(n, args.n_sample, args.n_queries, seed)
    mask = np.ones(n, dtype=bool)
    mask[query_idx] = False
    base, queries = np.ascontiguousarray(X[mask]), np.ascontiguousarray(X[query_idx])
    gt = exact_topk(base, queries, args.k, chunk=max(1, _GT_BYTES // (8 * base.shape[0])))
    index, _ = build_index(cfg, base)
    pred = np.stack([index.search(q, args.k)[0] for q in queries]).astype(np.int64)
    recall_full = recall_at(pred, gt, args.k)
    measured_ms, _ = time_queries(lambda q: index.search(q, args.k), queries, RealClock(),
                                  warmup=3, repeats=5, max_time_s=2.0)
    return Check(n, profile, seed, cfg.name, m.extrapolation, m.recall_at_k, recall_full,
                 recall_tie_tolerance(m.recall_at_k, args.n_queries, args.k),
                 m.latency_p50_ms_full, measured_ms)


def report(checks: list[Check], args, source: str) -> str:
    lines = [f"# Autotune validation {_today().isoformat()}", "",
             (f"Data: {source}. n_sample={args.n_sample}, n_queries={args.n_queries}, "
              f"k={args.k}, time_budget_s={args.time_budget}, vsq {vsq.__version__}."), "",
             "## Recall drift (sample -> full n)", "",
             ("| n_full | profile | seed | chosen | recall@n_sample | recall@n_full | drift | "
              "eps_r | within eps_r |"), "|---:|---|---:|---|---:|---:|---:|---:|---|"]
    for c in checks:
        lines.append(f"| {c.n_full} | {c.profile} | {c.seed} | {c.chosen} | "
                     f"{c.recall_sample:.4f} | {c.recall_full:.4f} | {c.drift:+.4f} | "
                     f"{c.eps_r:.4f} | {'yes' if c.drift <= c.eps_r else '**no**'} |")
    lines += ["", "## Latency extrapolation", "",
              "| n_full | profile | seed | chosen | kind | predicted ms | measured ms | error |",
              "|---:|---|---:|---|---|---:|---:|---:|"]
    for c in checks:
        lines.append(f"| {c.n_full} | {c.profile} | {c.seed} | {c.chosen} | {c.extrapolation} | "
                     f"{c.predicted_ms:.3g} | {c.measured_ms:.3g} | {c.latency_error:+.1%} |")
    lines += ["", "## Choice stability across seeds", "", "| n_full | profile | choices |",
              "|---:|---|---|"]
    groups: dict[tuple[int, str], list[str]] = {}
    for c in checks:
        groups.setdefault((c.n_full, c.profile), []).append(c.chosen)
    for (n, profile), names in sorted(groups.items()):
        verdict = "stable" if len(set(names)) == 1 else "**varies**"
        lines.append(f"| {n} | {profile} | {', '.join(names)} ({verdict}) |")
    worst_drift = max(c.drift - c.eps_r for c in checks)
    fits = [abs(c.latency_error) for c in checks if c.extrapolation == "arena-fit"]
    drift_text = "none" if worst_drift <= 0 else f"up to {worst_drift:+.4f}"
    fit_text = (f"max {max(fits):.1%} (target <= 20 %)" if fits
                else "no arena-fit choice in this run")
    lines += ["", "## Verdict", "", f"- recall drift beyond eps_r: {drift_text}",
              f"- arena-fit latency error: {fit_text}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="(rows, dim) .npy; default N(0,1)")
    parser.add_argument("--dim", type=int, default=256, help="dim of the N(0,1) data")
    parser.add_argument("--n-full", default="100000", help="comma-separated full sizes")
    parser.add_argument("--profiles", default="speed,accuracy")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-sample", type=int, default=20_000)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=_REPO / "docs" / "reports")
    parser.add_argument("--quick", action="store_true",
                        help="smoke preset: N(0,1) dim 64, n 6000, sample 3000, speed, seeds 0,1")
    args = parser.parse_args()
    if args.quick:
        args.data, args.dim, args.n_full, args.profiles, args.seeds = None, 64, "6000", "speed", "0,1"
        args.n_sample, args.n_queries, args.time_budget = 3000, 50, 10.0
        args.out_dir = Path(__file__).parent / "results"

    sizes = sorted(int(s) for s in args.n_full.split(",") if s)
    profiles = [p for p in args.profiles.split(",") if p]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if not sizes or not profiles or not seeds:
        raise SystemExit("--n-full, --profiles and --seeds must be non-empty")
    if min(sizes) <= args.n_sample:
        raise SystemExit(f"every --n-full must exceed --n-sample={args.n_sample} "
                         "(otherwise there is nothing to extrapolate)")
    if args.data is not None:
        raw = np.load(args.data, mmap_mode="r")
        if raw.ndim != 2 or raw.shape[0] < max(sizes):
            raise SystemExit(f"{args.data}: need >= {max(sizes)} rows, got shape {raw.shape}")
        X_all, source = as_matrix(raw[:max(sizes)], "data"), f"`{args.data.name}`"
    else:
        X_all = np.random.default_rng(0).standard_normal((max(sizes), args.dim)).astype(np.float32)
        source = f"N(0, 1), dim {args.dim}"

    checks = []
    for n in sizes:
        for profile in profiles:
            for seed in seeds:
                c = check_one(X_all[:n], profile, seed, args)
                print(f"n={n} {profile} seed={seed}: {c.chosen} drift={c.drift:+.4f} "
                      f"latency {c.extrapolation} {c.latency_error:+.1%}", flush=True)
                checks.append(c)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"autotune_validation_{_today():%Y%m%d}.md"
    out.write_text(report(checks, args, source), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    sys.exit(main())

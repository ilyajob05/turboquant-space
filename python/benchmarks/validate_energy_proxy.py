#!/usr/bin/env python3
"""Validate the autotune energy proxy against measured joules and fit it.

For every (catalog config, thread count, n) the script runs single queries
for --seconds, reads CPU and wall time like the autotuner does, computes the
proxy with the default EnergyModel, and -- if a meter is given -- the energy
actually drawn above idle:

  --meter rapl          Linux: /sys/class/powercap/intel-rapl:<i>/energy_uj
                        (package domains; wraparound via max_energy_range_uj)
  --meter powermetrics  macOS: a log written by a powermetrics process YOU
                        start in another terminal (this script never calls
                        sudo):
                          sudo powermetrics --samplers cpu_power -i 100 \\
                              -o /tmp/pm.log
                        then pass --powermetrics-log /tmp/pm.log. Samples
                        ("*** Sampled system activity (<date>) (<ms> elapsed)"
                        + "CPU Power: <mW> mW") are matched to each run window.
  --meter none          proxy only (no ranking check)

Before each run the machine idles for --cooldown seconds; that window gives
the idle power, which is subtracted (measured_mj is dynamic energy per query,
the quantity the proxy models).

Outputs, under --out-dir:
  energy_proxy_<stamp>.csv  one row per run:
    config        candidate id (tq4-fs, rq4-flat-t4, ...)
    threads       OpenMP threads of the space
    n             codes in the index
    queries       single queries executed in the window
    wall_ms       mean wall time per query
    cpu_ms        mean process CPU per query (all threads)
    bytes         code bytes read per query
    proxy_mj      default EnergyModel estimate per query
    measured_mj   metered energy above idle per query (NaN with --meter none)
    idle_w        idle power before the run (NaN with --meter none)
  energy_proxy_<stamp>.md   Spearman rho(proxy, measured), least-squares
                            fit of (p_core_w, p_base_w, e_byte_nj) and the
                            1- vs N-thread energy ranking per config.

Usage:
  uv run python python/benchmarks/validate_energy_proxy.py --meter none --quick
  uv run python python/benchmarks/validate_energy_proxy.py --meter rapl \\
      --data python/benchmarks/dbpedia_openai_100K_vectors.npy
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from vsq._autotune.calibrate import bytes_per_query
from vsq._autotune.candidates import CATALOG
from vsq._autotune.energy import default_energy_model
from vsq._autotune.host import probe_host
from vsq._autotune.index import build_index
from vsq._autotune.metrics import as_matrix

_K = 10


@dataclass
class Run:
    config: str
    threads: int
    n: int
    queries: int
    wall_ms: float
    cpu_ms: float
    bytes: int
    proxy_mj: float
    measured_mj: float = math.nan
    idle_w: float = math.nan
    t_idle: tuple[float, float] = (0.0, 0.0)  # epoch seconds, not written to CSV
    t_run: tuple[float, float] = (0.0, 0.0)


# --- meters --------------------------------------------------------------------

class RaplMeter:
    """Sum of package-level RAPL counters, microjoules -> joules."""

    def __init__(self) -> None:
        self.zones = sorted(z for z in glob.glob("/sys/class/powercap/intel-rapl:*")
                            if re.fullmatch(r".*/intel-rapl:\d+", z))
        if not self.zones:
            raise SystemExit("--meter rapl: no /sys/class/powercap/intel-rapl:<i> zones")
        try:
            self.ranges = [int(Path(z, "max_energy_range_uj").read_text()) for z in self.zones]
            self.read()
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--meter rapl: cannot read the counters ({exc}); "
                             "they are often root-only") from exc

    def read(self) -> list[int]:
        return [int(Path(z, "energy_uj").read_text()) for z in self.zones]

    def joules(self, start: list[int], stop: list[int]) -> float:
        total = 0
        for a, b, rng in zip(start, stop, self.ranges):
            total += b - a if b >= a else b + rng - a
        return total / 1e6


_PM_HEADER = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
_PM_CPU = re.compile(r"^CPU Power:\s+([\d.]+)\s*mW", re.MULTILINE)


def parse_powermetrics(text: str) -> list[tuple[float, float, float]]:
    """(window start, window end, CPU watts) per sample; epoch seconds."""
    samples = []
    blocks = _PM_HEADER.split(text)
    # split -> [prefix, date, elapsed, body, date, elapsed, body, ...]
    for i in range(1, len(blocks) - 2, 3):
        date_text, elapsed_ms, body = blocks[i], float(blocks[i + 1]), blocks[i + 2]
        power = _PM_CPU.search(body)
        if power is None:
            continue
        end = datetime.strptime(" ".join(date_text.split()), "%a %b %d %H:%M:%S %Y %z")
        stop = end.timestamp()
        samples.append((stop - elapsed_ms / 1e3, stop, float(power.group(1)) / 1e3))
    return samples


def window_joules(samples: list[tuple[float, float, float]], start: float, stop: float) -> float:
    """Energy of the overlap of every sample with [start, stop]."""
    total = 0.0
    for a, b, watts in samples:
        overlap = min(b, stop) - max(a, start)
        if overlap > 0:
            total += watts * overlap
    return total


# --- runs ----------------------------------------------------------------------

def _configs(names: list[str] | None, cores: int, threads: list[int]):
    for entry in CATALOG:
        if names and entry.id not in names:
            continue
        variants = threads if entry.config.index == "flat" else [1]  # FastScan is 1-thread
        for t in variants:
            if t <= cores:
                yield entry.config.with_threads(t)


def run_one(cfg, X: np.ndarray, n: int, queries: np.ndarray, seconds: float, energy) -> Run:
    index, _ = build_index(cfg, X[:n])
    for q in queries[:3]:
        index.search(q, _K)  # warmup
    count, refined = 0, 0
    wall0, cpu0 = time.perf_counter(), time.process_time()
    t0 = time.time()
    while time.perf_counter() - wall0 < seconds:
        _, _, r = index.search_stats(queries[count % len(queries)], _K)
        refined += r
        count += 1
    wall = (time.perf_counter() - wall0) / count * 1e3
    cpu = (time.process_time() - cpu0) / count * 1e3
    nbytes = bytes_per_query(index, n, refined / count / n, _K)
    return Run(cfg.name, cfg.num_threads, n, count, wall, cpu, nbytes,
               energy.query_mj(cpu, wall, nbytes), t_run=(t0, time.time()))


# --- analysis ------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_coefficients(runs: list[Run]) -> tuple[float, float, float] | None:
    """Least squares measured_mj = p_core*cpu_ms + p_base*wall_ms + e_byte*1e-6*bytes."""
    rows = [r for r in runs if math.isfinite(r.measured_mj)]
    if len(rows) < 3:
        return None
    A = np.array([[r.cpu_ms, r.wall_ms, r.bytes * 1e-6] for r in rows])
    y = np.array([r.measured_mj for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def summary(runs: list[Run], meter: str, energy) -> str:
    lines = [f"# Energy proxy validation ({meter})", "",
             (f"Default model: {energy.source} (p_core={energy.p_core_w} W, "
              f"p_base={energy.p_base_w} W, e_byte={energy.e_byte_nj} nJ)."), ""]
    measured = [r for r in runs if math.isfinite(r.measured_mj)]
    if len(measured) >= 3:
        rho = spearman(np.array([r.proxy_mj for r in measured]),
                       np.array([r.measured_mj for r in measured]))
        lines.append(f"Spearman rho(proxy, measured) over {len(measured)} runs: **{rho:.3f}**")
        fit = fit_coefficients(measured)
        if fit:
            lines.append(f"Fitted: p_core_w={fit[0]:.3g}, p_base_w={fit[1]:.3g}, "
                         f"e_byte_nj={fit[2]:.3g}")
    else:
        lines.append("No meter readings: proxy values only.")
    lines += ["", ("| config | n | fewest threads mJ (proxy / measured) | most threads mJ "
                   "(proxy / measured) |"), "|---|---:|---|---|"]
    by_key: dict[tuple[str, int], dict[int, Run]] = {}
    for r in runs:
        base = r.config.rsplit("-t", 1)[0]
        by_key.setdefault((base, r.n), {})[r.threads] = r
    for (base, n), variants in sorted(by_key.items()):
        if len(variants) < 2:
            continue
        few, many = variants[min(variants)], variants[max(variants)]
        lines.append(f"| {base} | {n} | {few.proxy_mj:.3g} / {few.measured_mj:.3g} "
                     f"({few.threads} t) | {many.proxy_mj:.3g} / {many.measured_mj:.3g} "
                     f"({many.threads} t) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meter", choices=("none", "rapl", "powermetrics"), required=True)
    parser.add_argument("--powermetrics-log", type=Path)
    parser.add_argument("--data", type=Path, help="(rows, dim) .npy; default N(0,1)")
    parser.add_argument("--dim", type=int, default=256, help="dim of the N(0,1) data")
    parser.add_argument("--n", default="20000,200000", help="comma-separated index sizes")
    parser.add_argument("--threads", default="", help="comma-separated; default 1 and all cores")
    parser.add_argument("--configs", default="", help="comma-separated catalog ids; default all")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--quick", action="store_true",
                        help="smoke preset: dim 64, n 2000, 0.2 s runs, 3 configs, threads 1,2")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    host = probe_host()
    energy = default_energy_model(host)
    if args.quick:
        args.dim, args.n, args.seconds, args.cooldown = 64, "2000", 0.2, 0.0
        args.configs, args.threads = "tq4-fs,rq4-flat,tq4-flat", "1,2"
    sizes = [int(s) for s in args.n.split(",") if s]
    threads = ([int(t) for t in args.threads.split(",") if t] if args.threads
               else sorted({1, host.parallel_cores}))
    if not sizes or min(sizes) < 100 or min(threads) < 1 or args.seconds <= 0:
        raise SystemExit("--n sizes must be >= 100, threads >= 1, seconds > 0")
    if args.meter == "powermetrics" and (args.powermetrics_log is None
                                         or not args.powermetrics_log.is_file()):
        raise SystemExit("--meter powermetrics needs --powermetrics-log pointing to the log "
                         "of a running `sudo powermetrics --samplers cpu_power -i 100 -o LOG`")
    rapl = RaplMeter() if args.meter == "rapl" else None

    n_max = max(sizes) + 256
    if args.data is not None:
        raw = np.load(args.data, mmap_mode="r")
        if raw.ndim != 2 or raw.shape[0] < n_max:
            raise SystemExit(f"{args.data}: need a 2-D array with >= {n_max} rows, got {raw.shape}")
        X = as_matrix(raw[:n_max], "data")
    else:
        X = np.random.default_rng(0).standard_normal((n_max, args.dim)).astype(np.float32)
    queries = X[-256:]
    names = [c for c in args.configs.split(",") if c] or None

    runs: list[Run] = []
    for cfg in _configs(names, host.logical_cpus, threads):
        for n in sizes:
            t_idle0 = time.time()
            idle_start = rapl.read() if rapl else None
            time.sleep(args.cooldown)
            idle_j = rapl.joules(idle_start, rapl.read()) if rapl else math.nan
            t_idle = (t_idle0, time.time())
            run_start = rapl.read() if rapl else None
            run = run_one(cfg, X, n, queries, args.seconds, energy)
            run.t_idle = t_idle
            if rapl:
                idle_w = idle_j / max(t_idle[1] - t_idle[0], 1e-9)
                total = rapl.joules(run_start, rapl.read())
                span = run.t_run[1] - run.t_run[0]
                run.idle_w = idle_w
                run.measured_mj = (total - idle_w * span) / run.queries * 1e3
            print(f"{run.config:<12} n={n:<8} q={run.queries:<7} wall={run.wall_ms:.3f} ms "
                  f"cpu={run.cpu_ms:.3f} ms proxy={run.proxy_mj:.3g} mJ", flush=True)
            runs.append(run)
    if not runs:
        raise SystemExit("no configuration matched --configs / --threads")

    if args.meter == "powermetrics":
        samples = parse_powermetrics(args.powermetrics_log.read_text(errors="replace"))
        if not samples:
            raise SystemExit(f"{args.powermetrics_log}: no 'CPU Power' samples found")
        for run in runs:
            idle_s = run.t_idle[1] - run.t_idle[0]
            run.idle_w = window_joules(samples, *run.t_idle) / max(idle_s, 1e-9)
            span = run.t_run[1] - run.t_run[0]
            dynamic = window_joules(samples, *run.t_run) - run.idle_w * span
            run.measured_mj = dynamic / run.queries * 1e3

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"energy_proxy_{stamp}.csv"
    fields = [f for f in asdict(runs[0]) if f not in ("t_idle", "t_run")]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({f: getattr(run, f) for f in fields})
    md_path = args.out_dir / f"energy_proxy_{stamp}.md"
    md_path.write_text(summary(runs, args.meter, energy), encoding="utf-8")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    sys.exit(main())

"""Candidate catalog, data-independent rules and the escalation ladder.

Rules use only facts that do not depend on the data distribution: code
size (memory budget), dimension (which kernel is fastest) and core count.
Recall is never predicted by a rule; calibration measures it, and the
ladder adds wider codes only when the cheaper tiers miss the recall floor.

Measured facts behind the rules (Apple M3, 1 thread, docs/benchmarks.md):
  dim 128:  RaBitQ-4 flat 95 M codes/s, RaBitQ-4 FastScan 22 M, TurboQuant-4
            FastScan 209 M  -> RaBitQ FastScan never wins at dim <= 256
  dim 1536: RaBitQ-4 FastScan 19 M vs flat 8.3 M; TurboQuant-4 FastScan 27 M
  FastScan search is single-threaded; flat distance_1_to_n is OpenMP-parallel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import Constraints, DataProfile, HostInfo, QuantizerConfig

# Scan bandwidth used only to decide whether a multi-threaded variant is
# worth measuring (bytes/s, conservative single-core DRAM stream rate).
_SCAN_BYTES_PER_S = 5e9
_THREAD_MIN_SCAN_S = 1e-3   # add a threaded variant above 1 ms predicted scan
_SMALL_DIM_RABITQ_FS = 256  # RaBitQ FastScan dropped at dim <= this
MAX_PER_STAGE = 6


@dataclass(frozen=True)
class CatalogEntry:
    """A candidate template: config, escalation tier, profiles it serves."""

    config: QuantizerConfig
    tier: int
    profiles: frozenset[str]

    @property
    def id(self) -> str:
        return self.config.name


def _entry(family: str, bits: int, index: str, tier: int, profiles: str, **kw) -> CatalogEntry:
    return CatalogEntry(QuantizerConfig(family, bits, index, **kw), tier,
                        frozenset(profiles.split()))


# Order is the last tie-break of selection. Tier 0: 1-bit; 1: 4-bit;
# 2: 8-bit; 3: fp16. RaBitQ 1-bit scores the float query (query_bits=0).
CATALOG: tuple[CatalogEntry, ...] = (
    _entry("turboquant", 4, "fastscan", 1, "speed energy accuracy", rerank=2),
    _entry("rabitq", 4, "fastscan", 1, "speed energy", eps0=1.9),
    _entry("rabitq", 4, "flat", 1, "speed energy accuracy"),
    _entry("turboquant", 4, "flat", 1, "speed energy"),
    _entry("rabitq", 1, "fastscan", 0, "speed energy", eps0=1.9, query_bits=0),
    _entry("rabitq", 8, "fastscan", 2, "speed energy accuracy", eps0=1.9),
    _entry("rabitq", 8, "flat", 2, "speed energy accuracy"),
    _entry("turboquant", 8, "flat", 2, "accuracy"),
    _entry("turboquant", 16, "flat", 3, "accuracy"),
)

# Tiers calibrated together, in escalation order, per profile.
STAGES: dict[str, tuple[tuple[int, ...], ...]] = {
    "speed": ((0, 1), (2,), (3,)),
    "energy": ((0, 1), (2,), (3,)),
    "accuracy": ((1, 2), (3,)),
}


@dataclass(frozen=True)
class CandidatePlan:
    """Candidate batches in escalation order plus the rule trace."""

    stages: tuple[tuple[QuantizerConfig, ...], ...]
    trace: tuple[str, ...]

    @property
    def first(self) -> tuple[QuantizerConfig, ...]:
        return self.stages[0] if self.stages else ()


def exact_code_size(config: QuantizerConfig, dim: int) -> int:
    """Stored bytes per vector, read from a constructed (untrained) space."""
    from .index import make_space  # native import kept out of the rule tests

    return int(make_space(config.with_threads(1), dim).code_size_bytes())


def generate_candidates(
    profile: str,
    constraints: Constraints,
    data: DataProfile,
    host: HostInfo,
    code_size: Callable[[QuantizerConfig, int], int] = exact_code_size,
) -> CandidatePlan:
    """Rules R1-R6 over CATALOG; returns batches per escalation stage."""
    if profile not in STAGES:
        raise ValueError(f"profile must be one of {sorted(STAGES)}, got {profile!r}")
    trace: list[str] = []
    sizes = {e.id: code_size(e.config, data.dim) for e in CATALOG}

    kept: list[CatalogEntry] = []
    for entry in CATALOG:
        reason = _drop_reason(entry, profile, constraints, data, sizes[entry.id])
        if reason:
            if reason != "profile":
                trace.append(f"drop {entry.id}: {reason}")
            continue
        kept.append(entry)

    if not kept and constraints.max_bytes_per_vector is not None:
        smallest = min(CATALOG, key=lambda e: sizes[e.id])
        trace.append(f"no candidate fits max_bytes_per_vector={constraints.max_bytes_per_vector}, "
                     f"smallest is {sizes[smallest.id]} B ({smallest.id})")

    stages: list[tuple[QuantizerConfig, ...]] = []
    for tiers in STAGES[profile]:
        batch = _with_threads([e for e in kept if e.tier in tiers], profile, data, host,
                              sizes, trace)
        if len(batch) > MAX_PER_STAGE:
            trace.append(f"cap tiers {tiers}: keep {[c.name for c in batch[:MAX_PER_STAGE]]}, "
                         f"drop {[c.name for c in batch[MAX_PER_STAGE:]]}")
            batch = batch[:MAX_PER_STAGE]
        if batch:
            stages.append(tuple(batch))
    return CandidatePlan(tuple(stages), tuple(trace))


def _drop_reason(entry: CatalogEntry, profile: str, constraints: Constraints,
                 data: DataProfile, size: int) -> str:
    """Why ``entry`` is excluded ('' keeps it, 'profile' is not traced)."""
    if profile not in entry.profiles:
        return "profile"
    budget = constraints.max_bytes_per_vector
    if budget is not None and size > budget:
        return f"R1 code {size} B > max_bytes_per_vector={budget}"
    cfg = entry.config
    if (cfg.family == "rabitq" and cfg.index == "fastscan"
            and data.dim <= _SMALL_DIM_RABITQ_FS):
        return (f"R3 dim {data.dim} <= {_SMALL_DIM_RABITQ_FS}: RaBitQ FastScan is slower "
                "than the flat scan there")
    return ""


def _with_threads(entries: list[CatalogEntry], profile: str, data: DataProfile,
                  host: HostInfo, sizes: dict[str, int], trace: list[str]) -> list[QuantizerConfig]:
    """R5: a threaded variant right after each flat entry whose predicted
    single-thread scan is >= 1 ms (FastScan search is single-threaded)."""
    out: list[QuantizerConfig] = []
    cores = host.parallel_cores
    for entry in entries:
        out.append(entry.config)
        if profile == "accuracy" or entry.config.index != "flat" or cores < 2:
            continue
        scan_s = data.n_full * sizes[entry.id] / _SCAN_BYTES_PER_S
        if scan_s >= _THREAD_MIN_SCAN_S:
            out.append(entry.config.with_threads(cores))
            trace.append(f"R5 add {entry.config.with_threads(cores).name}: predicted 1-thread "
                         f"scan {scan_s * 1e3:.1f} ms >= {_THREAD_MIN_SCAN_S * 1e3:.0f} ms")
    return out

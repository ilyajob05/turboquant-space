"""Value types of the autotuner: constraints, host / data fingerprints, a
quantizer configuration, measurements and per-candidate results.

Every type is a frozen dataclass with ``to_dict()`` / ``from_dict()``; the
dictionaries hold only JSON scalars, lists and nested dictionaries, so an
``AutotuneResult`` can be stored and a configuration rebuilt later without
re-calibrating. Units are part of the field names (``_ms``, ``_mj``, ``_s``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

SCHEMA_VERSION = 1

Profile = Literal["accuracy", "speed", "energy"]
PROFILES: tuple[str, ...] = ("accuracy", "speed", "energy")
FAMILIES: tuple[str, ...] = ("turboquant", "rabitq")
INDEXES: tuple[str, ...] = ("flat", "fastscan")
_BITS = {"turboquant": (4, 8, 16), "rabitq": (1, 4, 8)}
_FASTSCAN_BITS = {"turboquant": (4,), "rabitq": (1, 4, 8)}
_RABITQ_MODES = (None, "windowed_scale", "fixed_scale", "trained_scale", "algorithm1")
_TQ_CENTERING = ("ivf", "mean", "none")
EXTRAPOLATIONS: tuple[str, ...] = ("direct", "arena-fit", "sample-fit")
STATUSES: tuple[str, ...] = ("measured", "skipped_budget", "failed")


def _check_choice(name: str, value: Any, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _known_fields(cls, data: dict) -> dict:
    """Keyword arguments for ``cls`` from ``data``; unknown keys raise."""
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}")
    return dict(data)


# ---------------------------------------------------------------------------
# Constraints


_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "accuracy": {"min_recall": 0.0, "max_bytes_per_vector": None,
                 "max_latency_ms": None, "time_budget_s": 90.0},
    "speed": {"min_recall": 0.90, "max_bytes_per_vector": None,
              "max_latency_ms": None, "time_budget_s": 60.0},
    "energy": {"min_recall": 0.90, "max_bytes_per_vector": None,
               "max_latency_ms": None, "time_budget_s": 60.0},
}


@dataclass(frozen=True)
class Constraints:
    """Hard limits a chosen configuration must meet.

    min_recall            recall@k floor in [0, 1], tested against the lower 95 %
                          bound of the measured recall (robust to sampling noise);
                          0 disables it
    max_bytes_per_vector  stored code bytes per vector, > 0, or None
    max_latency_ms        single-query p50 latency at the user's n, > 0, or None
    time_budget_s         wall-clock budget of the calibration, > 0
    """

    min_recall: float = 0.0
    max_bytes_per_vector: int | None = None
    max_latency_ms: float | None = None
    time_budget_s: float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_recall <= 1.0:
            raise ValueError(f"min_recall must be in [0, 1], got {self.min_recall}")
        if self.max_bytes_per_vector is not None and self.max_bytes_per_vector <= 0:
            raise ValueError(
                f"max_bytes_per_vector must be > 0 or None, got {self.max_bytes_per_vector}")
        if self.max_latency_ms is not None and not self.max_latency_ms > 0:
            raise ValueError(f"max_latency_ms must be > 0 or None, got {self.max_latency_ms}")
        if not self.time_budget_s > 0:
            raise ValueError(f"time_budget_s must be > 0, got {self.time_budget_s}")

    @classmethod
    def for_profile(cls, profile: str, **overrides: Any) -> Constraints:
        """Profile defaults; an override of None keeps the default."""
        _check_choice("profile", profile, PROFILES)
        values = dict(_PROFILE_DEFAULTS[profile])
        for key, value in overrides.items():
            if key not in values:
                raise ValueError(f"unknown constraint {key!r}; expected {sorted(values)}")
            if value is not None:
                values[key] = value
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Constraints:
        return cls(**_known_fields(cls, data))


# ---------------------------------------------------------------------------
# Fingerprints


@dataclass(frozen=True)
class HostInfo:
    """Machine fingerprint read by the rules and printed in the report."""

    isa: str
    logical_cpus: int
    physical_cores: int
    perf_cores: int | None
    affinity_cpus: int
    platform: str
    machine: str
    cpu_model: str

    @property
    def parallel_cores(self) -> int:
        """Cores a multi-threaded query should use: performance cores if known."""
        cores = self.perf_cores or self.physical_cores
        return max(1, min(cores, self.affinity_cpus))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> HostInfo:
        return cls(**_known_fields(cls, data))


@dataclass(frozen=True)
class DataProfile:
    """Shapes of the calibration run (no data statistics)."""

    n_full: int
    dim: int
    n_base_sample: int
    n_queries: int
    k: int
    seed: int
    queries_source: Literal["holdout", "user"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DataProfile:
        return cls(**_known_fields(cls, data))


# ---------------------------------------------------------------------------
# Quantizer configuration


@dataclass(frozen=True)
class QuantizerConfig:
    """Everything needed to rebuild one index from raw vectors.

    family/bits/index pick the quantizer and search path; num_threads >= 1 is
    always explicit. TurboQuant reads centering, n_clusters, rotation_rounds,
    train_seed, train_iters; RaBitQ reads encode_mode and query_bits. rerank
    is TurboQuant FastScan only, eps0 RaBitQ FastScan only. train_rows caps
    the seeded subsample used by train().
    """

    family: Literal["turboquant", "rabitq"]
    bits: int
    index: Literal["flat", "fastscan"]
    num_threads: int = 1
    rot_seed: int = 42
    centering: str = "ivf"
    n_clusters: int = 256
    rotation_rounds: int = 3
    train_seed: int = 1234
    train_iters: int = 10
    train_rows: int = 65_536
    encode_mode: str | None = None
    query_bits: int | None = None
    rerank: int | None = None
    eps0: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def name(self) -> str:
        """Stable id: tq4-fs, rq8-flat, rq4-flat-t8 (thread count when > 1)."""
        short = "tq" if self.family == "turboquant" else "rq"
        kind = "fs" if self.index == "fastscan" else "flat"
        suffix = f"-t{self.num_threads}" if self.num_threads > 1 else ""
        return f"{short}{self.bits}-{kind}{suffix}"

    def validate(self) -> None:
        _check_choice("family", self.family, FAMILIES)
        _check_choice("index", self.index, INDEXES)
        _check_choice(f"{self.family} bits", self.bits, _BITS[self.family])
        if self.index == "fastscan":
            _check_choice(f"{self.family} fastscan bits", self.bits, _FASTSCAN_BITS[self.family])
        if self.num_threads < 1:
            raise ValueError(f"num_threads must be >= 1, got {self.num_threads}")
        if self.train_rows < 1:
            raise ValueError(f"train_rows must be >= 1, got {self.train_rows}")
        is_tq = self.family == "turboquant"
        if is_tq:
            _check_choice("centering", self.centering, _TQ_CENTERING)
            if self.encode_mode is not None or self.query_bits is not None:
                raise ValueError("encode_mode / query_bits are RaBitQ options")
        else:
            _check_choice("encode_mode", self.encode_mode, _RABITQ_MODES)
            if self.bits == 1 and self.encode_mode not in (None, "algorithm1"):
                raise ValueError("1-bit RaBitQ has no encode_mode other than algorithm1")
        if self.rerank is not None and not (is_tq and self.index == "fastscan" and self.rerank >= 1):
            raise ValueError("rerank (>= 1) applies to TurboQuant FastScan only")
        if self.eps0 is not None and not (not is_tq and self.index == "fastscan" and self.eps0 > 0):
            raise ValueError("eps0 (> 0) applies to RaBitQ FastScan only")

    def with_threads(self, num_threads: int) -> QuantizerConfig:
        return replace(self, num_threads=num_threads)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QuantizerConfig:
        return cls(**_known_fields(cls, data))


# ---------------------------------------------------------------------------
# Measurements


@dataclass(frozen=True)
class Measurement:
    """Calibration result of one candidate. Timings are min-of-repeats.

    recall_at_k             recall@k over the held-out queries at n_base_sample
    recall_ci95             95 % normal interval of recall_at_k
    latency_p50_ms_sample   single-query p50 at n_base_sample codes
    latency_p50_ms_full     p50 at n_full (measured or extrapolated, see extrapolation)
    cpu_ms_per_query        process CPU per query at n_full (all threads)
    bytes_touched_per_query code bytes read per query at n_full
    energy_proxy_mj         EnergyModel estimate per query, millijoules
    encode_vps              vectors/s through the build path (encode / index build)
    train_s                 seconds of train() on the calibration sample
    est_build_s_full        train_s + n_full / encode_vps
    refined_mean            RaBitQ FastScan: mean codes refined per query, else None
    """

    recall_at_k: float
    recall_ci95: tuple[float, float]
    latency_p50_ms_sample: float
    latency_p50_ms_full: float
    extrapolation: str
    cpu_ms_per_query: float
    bytes_touched_per_query: int
    energy_proxy_mj: float
    encode_vps: float
    train_s: float
    est_build_s_full: float
    refined_mean: float | None = None

    def __post_init__(self) -> None:
        _check_choice("extrapolation", self.extrapolation, EXTRAPOLATIONS)
        if not 0.0 <= self.recall_at_k <= 1.0:
            raise ValueError(f"recall_at_k must be in [0, 1], got {self.recall_at_k}")
        object.__setattr__(self, "recall_ci95", tuple(self.recall_ci95))

    @property
    def recall_lower95(self) -> float:
        """Lower 95 % bound of recall_at_k; the min_recall floor is tested against it."""
        return self.recall_ci95[0]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["recall_ci95"] = list(self.recall_ci95)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Measurement:
        return cls(**_known_fields(cls, data))


@dataclass(frozen=True)
class CandidateResult:
    """One candidate after calibration (measurement is None unless measured)."""

    config: QuantizerConfig
    measurement: Measurement | None
    status: str
    reasons: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        _check_choice("status", self.status, STATUSES)
        if (self.status == "measured") != (self.measurement is not None):
            raise ValueError("measurement must be present exactly when status == 'measured'")
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "measurement": None if self.measurement is None else self.measurement.to_dict(),
            "status": self.status,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CandidateResult:
        data = _known_fields(cls, data)
        m = data["measurement"]
        return cls(
            config=QuantizerConfig.from_dict(data["config"]),
            measurement=None if m is None else Measurement.from_dict(m),
            status=data["status"],
            reasons=tuple(data.get("reasons", ())),
        )

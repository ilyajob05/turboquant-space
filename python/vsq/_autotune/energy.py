"""Portable per-query energy proxy.

    E_q [J] = p_core_w * cpu_s + p_base_w * wall_s + e_byte_nj * 1e-9 * bytes

cpu_s    process CPU seconds of one query, all threads (spin-waits included:
         they are real energy)
wall_s   single-query p50 latency: package / uncore / DRAM background power
         while the query is in flight -- this term rewards race-to-idle
bytes    code bytes read by the query (DRAM traffic)

The default coefficients are order-of-magnitude placeholders per machine
class and are labelled "uncalibrated"; python/benchmarks/
validate_energy_proxy.py fits them against RAPL / powermetrics joules.
Only the ranking of candidates matters for selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .types import HostInfo


@dataclass(frozen=True)
class EnergyModel:
    """Coefficients of the proxy: watts per busy core, background watts,
    nanojoules per byte read. source says where they come from."""

    p_core_w: float
    p_base_w: float
    e_byte_nj: float
    source: str

    def __post_init__(self) -> None:
        for name in ("p_core_w", "p_base_w", "e_byte_nj"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")

    def query_mj(self, cpu_ms: float, wall_ms: float, bytes_touched: float) -> float:
        """Energy proxy of one query in millijoules (inputs in ms and bytes)."""
        if cpu_ms < 0 or wall_ms < 0 or bytes_touched < 0:
            raise ValueError(f"negative input: cpu_ms={cpu_ms}, wall_ms={wall_ms}, "
                             f"bytes={bytes_touched}")
        joules = (self.p_core_w * cpu_ms * 1e-3 + self.p_base_w * wall_ms * 1e-3
                  + self.e_byte_nj * 1e-9 * bytes_touched)
        return joules * 1e3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EnergyModel:
        return cls(**data)


# Placeholders: Apple P-core ~4 W at full clock, SoC background ~1.5 W,
# LPDDR ~0.1 nJ/byte; x86 server-class core ~6 W, package ~15 W, DDR ~0.3 nJ/B.
_DEFAULTS = {
    "arm64-apple": EnergyModel(4.0, 1.5, 0.10, "uncalibrated: arm64-apple placeholder"),
    "x86_64": EnergyModel(6.0, 15.0, 0.30, "uncalibrated: x86_64 placeholder"),
    "aarch64-other": EnergyModel(3.0, 3.0, 0.20, "uncalibrated: aarch64 placeholder"),
}


def default_energy_model(host: HostInfo) -> EnergyModel:
    """Placeholder coefficients for the host's machine class."""
    machine = host.machine.lower()
    if host.platform == "Darwin" and machine in ("arm64", "aarch64"):
        return _DEFAULTS["arm64-apple"]
    if machine in ("arm64", "aarch64"):
        return _DEFAULTS["aarch64-other"]
    return _DEFAULTS["x86_64"]

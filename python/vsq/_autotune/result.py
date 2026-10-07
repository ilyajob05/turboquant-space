"""AutotuneResult: the choice, every measurement and how to rebuild it."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

from .energy import EnergyModel
from .types import (
    SCHEMA_VERSION,
    CandidateResult,
    Constraints,
    DataProfile,
    HostInfo,
    QuantizerConfig,
)


@dataclass(frozen=True)
class AutotuneResult:
    """Outcome of :func:`vsq.autotune`.

    chosen      the selected configuration (None only inside an
                AutotuneInfeasibleError)
    candidates  every candidate in catalog order with its measurement/status
    code_bytes  candidate name -> stored bytes per vector
    rule_trace  rule, budget and escalation decisions, in order
    why         explanation of the final choice
    partial     True when the time budget cut calibration short
    index       the built QuantizedIndex (build=True); never serialised
    """

    profile: str
    constraints: Constraints
    host: HostInfo
    data: DataProfile
    chosen: QuantizerConfig | None
    candidates: tuple[CandidateResult, ...]
    code_bytes: dict
    rule_trace: tuple[str, ...]
    why: tuple[str, ...]
    energy_model: EnergyModel
    vsq_version: str
    partial: bool = False
    schema_version: int = SCHEMA_VERSION
    index: object | None = field(default=None, compare=False, repr=False)

    def build(self, X):
        """Build the chosen configuration on X (n, dim) float32 -> QuantizedIndex."""
        if self.chosen is None:
            raise ValueError("this result has no chosen configuration")
        from .index import build_index

        return build_index(self.chosen, X)[0]

    def report(self) -> str:
        """Human-readable table of candidates, the rule trace and the reasons."""
        from .report import format_report

        return format_report(self)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "vsq_version": self.vsq_version,
            "profile": self.profile,
            "constraints": self.constraints.to_dict(),
            "host": self.host.to_dict(),
            "data": self.data.to_dict(),
            "chosen": None if self.chosen is None else self.chosen.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "code_bytes": dict(self.code_bytes),
            "rule_trace": list(self.rule_trace),
            "why": list(self.why),
            "energy_model": self.energy_model.to_dict(),
            "partial": self.partial,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **{"indent": 2, **kwargs})

    @classmethod
    def from_dict(cls, data: dict) -> AutotuneResult:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"AutotuneResult schema_version {version} is not supported "
                             f"(this vsq reads {SCHEMA_VERSION})")
        host = HostInfo.from_dict(data["host"])
        _warn_if_different_host(host, data.get("vsq_version"))
        chosen = data["chosen"]
        return cls(
            profile=data["profile"],
            constraints=Constraints.from_dict(data["constraints"]),
            host=host,
            data=DataProfile.from_dict(data["data"]),
            chosen=None if chosen is None else QuantizerConfig.from_dict(chosen),
            candidates=tuple(CandidateResult.from_dict(c) for c in data["candidates"]),
            code_bytes=dict(data["code_bytes"]),
            rule_trace=tuple(data["rule_trace"]),
            why=tuple(data["why"]),
            energy_model=EnergyModel.from_dict(data["energy_model"]),
            vsq_version=data["vsq_version"],
            partial=bool(data["partial"]),
        )

    @classmethod
    def from_json(cls, text: str) -> AutotuneResult:
        return cls.from_dict(json.loads(text))


def _warn_if_different_host(host: HostInfo, version: str | None) -> None:
    """Timings are host-specific: warn when a stored result is reused elsewhere."""
    from .host import probe_host

    try:
        from .. import __version__ as current
    except ImportError:  # pragma: no cover
        current = None
    here = probe_host()
    diffs = []
    if (host.isa, host.cpu_model) != (here.isa, here.cpu_model):
        diffs.append(f"host {host.cpu_model}/{host.isa} vs {here.cpu_model}/{here.isa}")
    if version is not None and current is not None and version != current:
        diffs.append(f"vsq {version} vs {current}")
    if diffs:
        warnings.warn("autotune result was calibrated elsewhere (" + "; ".join(diffs)
                      + "): the configuration is valid, its timings may not be",
                      UserWarning, stacklevel=3)

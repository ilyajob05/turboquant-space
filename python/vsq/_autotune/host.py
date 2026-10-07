"""Host fingerprint: ISA, core counts and CPU model. Never raises."""

from __future__ import annotations

import os
import platform
import subprocess

from .types import HostInfo


def _sysctl(name: str) -> str | None:
    """macOS sysctl value, or None when unavailable."""
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                             timeout=2, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _linux_cpuinfo() -> tuple[str | None, int | None]:
    """(model name, physical cores) from /proc/cpuinfo, None where unknown."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None, None
    model = None
    cores: set[tuple[str, str]] = set()
    phys = core = None
    for line in text.splitlines() + [""]:
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "physical id":
            phys = value
        elif key == "core id":
            core = value
        elif not line.strip():
            if core is not None:
                cores.add((phys or "0", core))
            phys = core = None
    return model, (len(cores) or None)


def _affinity(logical: int) -> int:
    try:
        return len(os.sched_getaffinity(0))  # Linux only
    except (AttributeError, OSError):
        return logical


def _isa() -> str:
    try:
        from .._vsq import detected_isa

        return str(detected_isa())
    except Exception:  # noqa: BLE001 - extension may be absent in rule-only tests
        return "unknown"


def probe_host() -> HostInfo:
    """Fingerprint of this machine; unknown fields fall back to safe values."""
    logical = os.cpu_count() or 1
    system = platform.system()
    model: str | None = None
    physical: int | None = None
    perf: int | None = None
    if system == "Darwin":
        model = _sysctl("machdep.cpu.brand_string")
        physical = _int_or_none(_sysctl("hw.physicalcpu"))
        perf = _int_or_none(_sysctl("hw.perflevel0.physicalcpu"))
    elif system == "Linux":
        model, physical = _linux_cpuinfo()
    physical = max(1, physical or logical)
    return HostInfo(
        isa=_isa(),
        logical_cpus=logical,
        physical_cores=physical,
        perf_cores=perf,
        affinity_cpus=max(1, _affinity(logical)),
        platform=system or "unknown",
        machine=platform.machine() or "unknown",
        cpu_model=model or platform.processor() or "unknown",
    )

"""Docs stay true to the code; validation-script helpers are correct."""

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import vsq
from vsq._autotune.calibrate import Options

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "python" / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def _small_result():
    X = np.random.default_rng(0).standard_normal((2000, 32)).astype(np.float32) + 1.0
    return vsq.autotune(X, "speed", n_sample=1500, n_queries=40, time_budget_s=10,
                        options=Options(warmup=1, repeats=1, rounds=1), min_recall=0)


def test_readme_autotune_example_runs():
    text = (ROOT / "README.md").read_text()
    section = text.split("## Autotune", 1)[1]
    code = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)
    X = np.random.default_rng(1).standard_normal((2000, 32)).astype(np.float32)
    namespace = {"vsq": vsq, "X": X, "q": X[0], "print": lambda *a: None}
    exec(compile(code, "README.md#autotune", "exec"), namespace)  # noqa: S102
    assert namespace["ids"].shape == (10,)


def test_docs_list_every_serialised_key():
    """Every key of AutotuneResult.to_dict() (and nested) appears in the schema table."""
    doc = (ROOT / "docs" / "autotune.md").read_text()
    section = doc.split("## Serialised result", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"`([a-z_0-9]+)`", section))
    d = _small_result().to_dict()
    keys = set(d)
    keys |= set(d["constraints"]) | set(d["host"]) | set(d["data"]) | set(d["chosen"])
    keys |= set(d["energy_model"])
    cand = d["candidates"][0]
    keys |= set(cand) | set(cand["measurement"])
    missing = sorted(keys - documented)
    assert not missing, f"undocumented keys in docs/autotune.md: {missing}"


# --- validate_energy_proxy helpers ----------------------------------------------

POWERMETRICS_LOG = """\
Machine model: Mac15,13
*** Sampled system activity (Tue Oct  6 14:03:21 2026 +0300) (100.00ms elapsed) ***

**** Processor usage ****
CPU Power: 2000 mW
GPU Power: 0 mW

*** Sampled system activity (Tue Oct  6 14:03:22 2026 +0300) (1000.00ms elapsed) ***

CPU Power: 4000 mW
"""


def test_parse_powermetrics_and_windows():
    ep = _load_script("validate_energy_proxy")
    samples = ep.parse_powermetrics(POWERMETRICS_LOG)
    assert len(samples) == 2
    (a0, b0, w0), (a1, b1, w1) = samples
    assert w0 == pytest.approx(2.0) and w1 == pytest.approx(4.0)
    assert b0 - a0 == pytest.approx(0.1) and b1 - a1 == pytest.approx(1.0)
    # the second sample covers [b1 - 1, b1]; half of it inside the window -> 2 J
    assert ep.window_joules(samples, b1 - 0.5, b1) == pytest.approx(2.0)
    assert ep.window_joules(samples, b1 + 1, b1 + 2) == 0.0


def test_spearman_and_coefficient_fit():
    ep = _load_script("validate_energy_proxy")
    assert ep.spearman(np.array([1.0, 2, 3]), np.array([10.0, 20, 30])) == pytest.approx(1.0)
    assert ep.spearman(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == pytest.approx(-1.0)
    runs = []
    for cpu, wall, nbytes in [(1, 1, 1e6), (4, 1, 2e6), (2, 3, 5e5), (6, 2, 8e6)]:
        measured = 3.0 * cpu + 2.0 * wall + 0.5 * nbytes * 1e-6
        runs.append(ep.Run("x", 1, 1, 1, wall, cpu, int(nbytes), 0.0, measured))
    p_core, p_base, e_byte = ep.fit_coefficients(runs)
    assert (p_core, p_base, e_byte) == pytest.approx((3.0, 2.0, 0.5), rel=1e-6)
    assert ep.fit_coefficients(runs[:2]) is None
    assert math.isnan(ep.Run("x", 1, 1, 1, 1.0, 1.0, 1, 1.0).measured_mj)

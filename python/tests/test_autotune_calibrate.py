"""Metrics, timing, serialisation and calibration mechanics."""

import gc

import numpy as np
import pytest
from vsq._autotune import calibrate as cal
from vsq._autotune.candidates import CandidatePlan
from vsq._autotune.energy import EnergyModel
from vsq._autotune.index import build_index
from vsq._autotune.metrics import as_matrix, exact_topk, recall_at, split_sample
from vsq._autotune.timing import fit_linear, time_queries
from vsq._autotune.types import (
    CandidateResult,
    Constraints,
    DataProfile,
    HostInfo,
    Measurement,
    QuantizerConfig,
)

ENERGY = EnergyModel(4.0, 1.5, 0.1, "test")


class FakeClock:
    """perf_ns advances `step` ns per read; cpu_ns advances 2*step per read."""

    def __init__(self, step=1_000_000):
        self.t, self.c, self.step = 0, 0, step

    def perf_ns(self):
        self.t += self.step
        return self.t

    def cpu_ns(self):
        self.c += 2 * self.step
        return self.c


# --- metrics ---------------------------------------------------------------

@pytest.mark.parametrize("dim", [7, 33, 128])
def test_exact_topk_matches_argsort(dim):
    rng = np.random.default_rng(dim)
    base = rng.standard_normal((500, dim)).astype(np.float32)
    q = rng.standard_normal((70, dim)).astype(np.float32)
    got = exact_topk(base, q, 10, chunk=16)
    d = ((q[:, None, :].astype(np.float64) - base[None]) ** 2).sum(-1)
    np.testing.assert_array_equal(got, np.argsort(d, axis=1, kind="stable")[:, :10])


@pytest.mark.parametrize("bad,match", [
    (np.ones(5, np.float32), "2-D"),
    (np.ones((3, 4), np.int32), "float"),
    (np.array([[1.0, np.nan]], np.float32), "NaN"),
    (np.ones((0, 4), np.float32), "non-empty"),
])
def test_as_matrix_rejects_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        as_matrix(bad, "X")


def test_as_matrix_dim_mismatch_and_conversion():
    with pytest.raises(ValueError, match="dim 4, expected 5"):
        as_matrix(np.ones((2, 4)), "Q", 5)
    out = as_matrix(np.ones((3, 8))[:, ::2], "X")
    assert out.dtype == np.float32 and out.flags.c_contiguous and out.shape == (3, 4)


def test_split_sample_is_seeded_and_disjoint():
    a = split_sample(1000, 300, 50, seed=3)
    b = split_sample(1000, 300, 50, seed=3)
    c = split_sample(1000, 300, 50, seed=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])
    assert not set(a[0]) & set(a[1]) and len(a[1]) == 50 and len(a[0]) == 300
    with pytest.raises(ValueError):
        split_sample(10, 5, 10, seed=0)


def test_recall_and_topk_k_bounds():
    gt = np.array([[0, 1, 2], [3, 4, 5]])
    assert recall_at(np.array([[2, 1, 9], [9, 9, 9]]), gt, 3) == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        exact_topk(np.ones((3, 2), np.float32), np.ones((1, 2), np.float32), 4)


# --- timing ----------------------------------------------------------------

def test_time_queries_with_fake_clock_counts_only_timed_calls():
    clock = FakeClock(step=1_000_000)  # every perf read +1 ms, cpu read +2 ms
    calls = []
    p50, cpu = time_queries(calls.append, np.zeros((4, 2), np.float32), clock,
                            warmup=2, repeats=3, max_time_s=None)
    assert len(calls) == 2 + 3 * 4
    assert p50 == pytest.approx(1.0)       # one perf step between start and stop
    assert cpu == pytest.approx(2.0 / 12)  # one 2 ms cpu step over 12 timed calls


def test_time_queries_restores_gc_after_exception():
    def boom(_):
        raise RuntimeError("x")

    assert gc.isenabled()
    with pytest.raises(RuntimeError):
        time_queries(boom, np.zeros((2, 2), np.float32), FakeClock())
    assert gc.isenabled()


def test_time_queries_budget_reduces_work():
    calls = []
    time_queries(calls.append, np.zeros((32, 2), np.float32), FakeClock(step=100_000_000),
                 warmup=1, repeats=5, max_time_s=0.5)
    assert len(calls) <= 1 + 5  # 0.1 s per call -> <= 5 timed calls, one repeat


def test_fit_linear_recovers_coefficients():
    alpha, beta = fit_linear([(1000, 0.5 + 2e-3 * 1000), (5000, 0.5 + 2e-3 * 5000)])
    assert alpha == pytest.approx(0.5, rel=1e-6) and beta == pytest.approx(2e-3, rel=1e-6)
    with pytest.raises(ValueError):
        fit_linear([(1, 1.0), (1, 2.0)])


# --- types -------------------------------------------------------------------

def _measurement():
    return Measurement(0.9, (0.88, 0.92), 0.1, 0.5, "arena-fit", 0.6, 1000, 2.5, 1e5, 0.2, 3.0,
                       None)


@pytest.mark.parametrize("obj", [
    Constraints(0.9, 400, 2.0, 30.0),
    HostInfo("neon", 8, 8, 4, 8, "Darwin", "arm64", "M3"),
    DataProfile(10, 4, 5, 2, 1, 0, "holdout"),
    QuantizerConfig("rabitq", 8, "fastscan", eps0=1.9, num_threads=2),
    _measurement(),
    CandidateResult(QuantizerConfig("turboquant", 4, "flat"), _measurement(), "measured"),
    CandidateResult(QuantizerConfig("turboquant", 4, "flat"), None, "failed", ("x",)),
])
def test_types_roundtrip(obj):
    assert type(obj).from_dict(obj.to_dict()) == obj


@pytest.mark.parametrize("kwargs,match", [
    ({"family": "faiss", "bits": 4, "index": "flat"}, "family"),
    ({"family": "turboquant", "bits": 1, "index": "flat"}, "bits"),
    ({"family": "turboquant", "bits": 8, "index": "fastscan"}, "fastscan bits"),
    ({"family": "rabitq", "bits": 4, "index": "flat", "rerank": 2}, "rerank"),
    ({"family": "turboquant", "bits": 4, "index": "fastscan", "eps0": 1.9}, "eps0"),
    ({"family": "rabitq", "bits": 1, "index": "flat", "encode_mode": "fixed_scale"}, "1-bit"),
    ({"family": "rabitq", "bits": 4, "index": "flat", "num_threads": 0}, "num_threads"),
])
def test_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        QuantizerConfig(**kwargs)


def test_constraints_validation_and_defaults():
    assert Constraints.for_profile("speed").min_recall == 0.90
    assert Constraints.for_profile("speed", min_recall=0).min_recall == 0
    with pytest.raises(ValueError, match="min_recall"):
        Constraints(min_recall=1.5)
    with pytest.raises(ValueError, match="unknown constraint"):
        Constraints.for_profile("speed", recall=0.9)


# --- calibration ---------------------------------------------------------------

def _data(n=3000, dim=48, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((20, dim)) * 3
    return (centers[rng.integers(0, 20, n)] + rng.standard_normal((n, dim))).astype(np.float32)


def test_calibration_recall_matches_independent_index():
    X = _data()
    cset = cal.prepare(X, None, k=10, n_sample=2000, n_queries=50, seed=1)
    cfg = QuantizerConfig("rabitq", 4, "flat")
    res = cal.measure_stage((cfg,), cset, FakeClock(), ENERGY, cal.Options(rounds=1),
                            deadline_ns=10**15)[0]
    index, _ = build_index(cfg, cset.base)
    ids, _ = index.search_batch(cset.queries, 10)
    assert res.measurement.recall_at_k == pytest.approx(recall_at(ids.astype(np.int64),
                                                                  cset.gt, 10))
    again_set = cal.prepare(X, None, k=10, n_sample=2000, n_queries=50, seed=1)
    again = cal.measure_stage((cfg,), again_set, FakeClock(), ENERGY, cal.Options(rounds=1),
                              deadline_ns=10**15)[0]
    assert again.measurement.recall_at_k == res.measurement.recall_at_k


def test_failed_candidate_does_not_abort(monkeypatch):
    X = _data()
    cset = cal.prepare(X, None, k=10, n_sample=1500, n_queries=30, seed=0)
    real = cal._build_and_recall

    def flaky(cfg, cs):
        if cfg.family == "turboquant":
            raise RuntimeError("native failure")
        return real(cfg, cs)

    monkeypatch.setattr(cal, "_build_and_recall", flaky)
    cfgs = (QuantizerConfig("turboquant", 4, "flat"), QuantizerConfig("rabitq", 4, "flat"))
    out = cal.measure_stage(cfgs, cset, FakeClock(), ENERGY, cal.Options(rounds=1), 10**15)
    assert [r.status for r in out] == ["failed", "measured"]
    assert "native failure" in out[0].reasons[0]


def test_budget_projection_shrinks_sample():
    X = _data(n=12000)
    cset = cal.prepare(X, None, k=10, n_sample=11000, n_queries=50, seed=0)
    trace: list[str] = []
    n = cal.plan_sample_size((QuantizerConfig("rabitq", 4, "flat"),), cset,
                             FakeClock(step=10**9), budget_s=10.0, trace=trace)
    assert n == 5000 and trace and trace[0].startswith("budget: n_sample 11000 -> 5000")


def test_exhausted_budget_skips_candidates():
    X = _data()
    cset = cal.prepare(X, None, k=10, n_sample=1500, n_queries=30, seed=0)
    out = cal.measure_stage((QuantizerConfig("rabitq", 4, "flat"),), cset, FakeClock(), ENERGY,
                            cal.Options(rounds=1), deadline_ns=0)
    assert out[0].status == "skipped_budget"


def test_escalation_runs_next_stage_only_when_floor_missed():
    X = _data()
    cset = cal.prepare(X, None, k=10, n_sample=1500, n_queries=30, seed=0)
    plan = CandidatePlan(((QuantizerConfig("rabitq", 4, "flat"),),
                          (QuantizerConfig("rabitq", 8, "flat"),)), ())
    opts = cal.Options(rounds=1)
    low = Constraints.for_profile("speed", min_recall=0.01, time_budget_s=600)
    res, trace, partial, _ = cal.calibrate(plan, cset, "speed", low, FakeClock(), ENERGY, opts)
    assert [r.config.name for r in res] == ["rq4-flat"] and not partial
    high = Constraints.for_profile("speed", min_recall=1.0, time_budget_s=600)
    res, trace, _, _ = cal.calibrate(plan, cset, "speed", high, FakeClock(), ENERGY, opts)
    assert [r.config.name for r in res] == ["rq4-flat", "rq8-flat"]
    assert any(t.startswith("escalate to ['rq8-flat']: best recall") for t in trace)


def test_extrapolation_kinds():
    X = _data(n=3000)
    cset = cal.prepare(X, None, k=10, n_sample=1000, n_queries=20, seed=0)
    small_arena = cal.Options(rounds=1, arena_bytes=1500 * 80)  # n_t < n_full -> fit
    cfgs = (QuantizerConfig("turboquant", 4, "flat"),
            QuantizerConfig("rabitq", 4, "fastscan", eps0=1.9))
    out = cal.measure_stage(cfgs, cset, FakeClock(), ENERGY, small_arena, 10**15)
    assert out[0].measurement.extrapolation == "arena-fit"
    assert out[1].measurement.extrapolation == "sample-fit"
    out = cal.measure_stage(cfgs[:1], cset, FakeClock(), ENERGY, cal.Options(rounds=1), 10**15)
    assert out[0].measurement.extrapolation == "direct"


# --- host --------------------------------------------------------------------

def test_probe_host_survives_failing_sources(monkeypatch):
    from vsq._autotune import host

    def fail(*_a, **_k):
        raise OSError("no sysctl")

    monkeypatch.setattr(host.subprocess, "run", fail)
    monkeypatch.setattr("builtins.open", fail)
    for system in ("Darwin", "Linux", "Windows"):
        monkeypatch.setattr(host.platform, "system", lambda s=system: s)
        info = host.probe_host()
        assert info.physical_cores >= 1 and info.affinity_cpus >= 1 and info.parallel_cores >= 1


def test_linux_cpuinfo_counts_physical_cores(monkeypatch, tmp_path):
    from vsq._autotune import host

    text = "".join(f"processor\t: {i}\nmodel name\t: Test CPU\nphysical id\t: {i // 4}\n"
                   f"core id\t\t: {i % 2}\n\n" for i in range(8))  # 2 sockets x 2 cores x HT
    path = tmp_path / "cpuinfo"
    path.write_text(text)
    real_open = open
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: real_open(path if p == "/proc/cpuinfo" else p, *a, **k))
    assert host._linux_cpuinfo() == ("Test CPU", 4)

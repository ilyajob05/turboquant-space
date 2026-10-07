"""Selection, tie-breaks, infeasibility and the energy proxy (fake measurements)."""

import random

import pytest
from vsq._autotune.energy import EnergyModel
from vsq._autotune.select import AutotuneInfeasibleError, select
from vsq._autotune.types import (
    CandidateResult,
    Constraints,
    Measurement,
    QuantizerConfig,
)

CFG = {
    "tq4-fs": QuantizerConfig("turboquant", 4, "fastscan", rerank=2),
    "rq4-flat": QuantizerConfig("rabitq", 4, "flat"),
    "rq4-flat-t4": QuantizerConfig("rabitq", 4, "flat", num_threads=4),
    "rq8-flat": QuantizerConfig("rabitq", 8, "flat"),
    "tq16-flat": QuantizerConfig("turboquant", 16, "flat"),
}
BYTES = {"tq4-fs": 398, "rq4-flat": 392, "rq4-flat-t4": 392, "rq8-flat": 776, "tq16-flat": 1544}


def _m(recall, latency, energy=None):
    return Measurement(recall_at_k=recall, recall_ci95=(recall, recall),
                       latency_p50_ms_sample=latency, latency_p50_ms_full=latency,
                       extrapolation="direct", cpu_ms_per_query=latency,
                       bytes_touched_per_query=1, energy_proxy_mj=energy or latency,
                       encode_vps=1.0, train_s=0.0, est_build_s_full=1.0)


def _res(name, recall, latency, energy=None):
    return CandidateResult(CFG[name], _m(recall, latency, energy), "measured")


def _pick(results, profile, **c):
    cfg, _ = select(results, profile, Constraints.for_profile(profile, **c), BYTES, m=200, k=10)
    return cfg.name


@pytest.mark.parametrize("results,expected", [
    ([_res("tq4-fs", 0.97, 1.0), _res("rq4-flat", 0.96, 3.0)], "tq4-fs"),
    ([_res("tq4-fs", 0.85, 1.0), _res("rq8-flat", 0.99, 3.0)], "rq8-flat"),  # floor 0.90
    ([_res("rq4-flat", 0.95, 2.0), _res("rq4-flat-t4", 0.95, 0.7)], "rq4-flat-t4"),
])
def test_speed_scenarios(results, expected):
    assert _pick(results, "speed") == expected


def test_speed_tie_within_5_percent_prefers_recall_then_bytes():
    results = [_res("tq4-fs", 0.95, 1.00), _res("rq4-flat", 0.97, 1.04)]
    assert _pick(results, "speed") == "rq4-flat"
    results = [_res("tq4-fs", 0.95, 1.00), _res("rq4-flat", 0.95, 1.04)]
    assert _pick(results, "speed") == "rq4-flat"  # equal recall -> 392 B < 398 B


@pytest.mark.parametrize("results,expected", [
    ([_res("rq8-flat", 0.999, 5.0), _res("tq16-flat", 1.0, 4.0)], "rq8-flat"),   # tie -> bytes
    ([_res("rq8-flat", 0.95, 5.0), _res("tq16-flat", 1.0, 4.0)], "tq16-flat"),
    ([_res("tq4-fs", 0.97, 1.0), _res("rq4-flat", 0.97, 2.0)], "rq4-flat"),     # bytes
])
def test_accuracy_scenarios(results, expected):
    assert _pick(results, "accuracy") == expected


def test_accuracy_equal_bytes_breaks_on_latency():
    results = [_res("rq4-flat", 0.97, 2.0), _res("rq4-flat-t4", 0.97, 1.0)]
    assert _pick(results, "accuracy") == "rq4-flat-t4"


@pytest.mark.parametrize("results,expected", [
    ([_res("rq4-flat", 0.95, 2.0, 10.0), _res("rq4-flat-t4", 0.95, 0.7, 8.0)], "rq4-flat-t4"),
    ([_res("rq4-flat", 0.95, 2.0, 6.0), _res("rq4-flat-t4", 0.95, 0.7, 8.0)], "rq4-flat"),
    ([_res("tq4-fs", 0.80, 1.0, 1.0), _res("rq8-flat", 0.99, 3.0, 9.0)], "rq8-flat"),
])
def test_energy_scenarios(results, expected):
    assert _pick(results, "energy") == expected


def test_choice_is_stable_under_reordering_when_unambiguous():
    """Only the final tie-break reads candidate order; here recall decides."""
    results = [_res("tq4-fs", 0.95, 1.0), _res("rq4-flat", 0.95, 1.0), _res("rq8-flat", 0.99, 1.02)]
    rng = random.Random(0)
    for _ in range(10):
        shuffled = results[:]
        rng.shuffle(shuffled)
        assert _pick(shuffled, "speed") == "rq8-flat"


def test_infeasible_message_names_every_violated_constraint():
    results = [_res("tq4-fs", 0.97, 1.0), _res("rq8-flat", 0.99, 3.0)]
    with pytest.raises(AutotuneInfeasibleError) as err:
        _pick(results, "speed", min_recall=0.995, max_latency_ms=0.5, max_bytes_per_vector=100)
    msg = str(err.value)
    assert "min_recall=0.995 (lower 95 % bound): best 0.9900, recall 0.9900 (rq8-flat)" in msg
    assert "max_latency_ms=0.5: best 1 ms (tq4-fs)" in msg
    assert "max_bytes_per_vector=100: smallest 398 B (tq4-fs)" in msg


def test_infeasible_reports_what_relaxing_admits():
    results = [_res("tq4-fs", 0.97, 1.0)]
    with pytest.raises(AutotuneInfeasibleError, match="relaxing min_recall alone admits: tq4-fs"):
        _pick(results, "speed", min_recall=0.99)


def test_no_measured_candidate():
    failed = [CandidateResult(CFG["tq4-fs"], None, "failed", ("boom",))]
    with pytest.raises(AutotuneInfeasibleError, match="tq4-fs: failed"):
        _pick(failed, "speed")


def test_energy_proxy_formula():
    model = EnergyModel(p_core_w=4.0, p_base_w=2.0, e_byte_nj=0.5, source="test")
    # 4 W * 1 ms + 2 W * 0.5 ms + 0.5 nJ * 1e6 B = 4 + 1 + 0.5 mJ
    assert model.query_mj(cpu_ms=1.0, wall_ms=0.5, bytes_touched=1e6) == pytest.approx(5.5)
    with pytest.raises(ValueError):
        model.query_mj(-1.0, 1.0, 0)
    with pytest.raises(ValueError):
        EnergyModel(-1.0, 0.0, 0.0, "bad")


def test_background_power_flips_thread_choice():
    """Race-to-idle: 4 threads burn more CPU but finish sooner."""
    one = {"cpu_ms": 4.0, "wall_ms": 4.0, "bytes_touched": 0}
    four = {"cpu_ms": 6.0, "wall_ms": 1.5, "bytes_touched": 0}
    low = EnergyModel(4.0, 0.0, 0.0, "no background power")
    high = EnergyModel(4.0, 10.0, 0.0, "high background power")
    assert low.query_mj(**one) < low.query_mj(**four)
    assert high.query_mj(**one) > high.query_mj(**four)


def _res_ci(name, recall, lower, latency):
    m = Measurement(recall_at_k=recall, recall_ci95=(lower, min(1.0, 2 * recall - lower)),
                    latency_p50_ms_sample=latency, latency_p50_ms_full=latency,
                    extrapolation="direct", cpu_ms_per_query=latency, bytes_touched_per_query=1,
                    energy_proxy_mj=latency, encode_vps=1.0, train_s=0.0, est_build_s_full=1.0)
    return CandidateResult(CFG[name], m, "measured")


def test_recall_floor_uses_the_lower_95_bound():
    """0.902 +- 0.013 does not clear a 0.90 floor; the 8-bit candidate does."""
    results = [_res_ci("tq4-fs", 0.902, 0.889, 1.0), _res_ci("rq8-flat", 0.99, 0.986, 3.0)]
    assert _pick(results, "speed") == "rq8-flat"
    results = [_res_ci("tq4-fs", 0.93, 0.918, 1.0), _res_ci("rq8-flat", 0.99, 0.986, 3.0)]
    assert _pick(results, "speed") == "tq4-fs"


def test_infeasible_message_reports_the_lower_bound():
    results = [_res_ci("tq4-fs", 0.902, 0.889, 1.0)]
    with pytest.raises(AutotuneInfeasibleError,
                       match=r"min_recall=0.9 \(lower 95 % bound\): best 0.8890, recall 0.9020"):
        _pick(results, "speed")

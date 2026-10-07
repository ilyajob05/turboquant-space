"""Candidate catalog, rules R1-R6 and the escalation stages (no calibration)."""

import pytest
from vsq._autotune.candidates import CATALOG, MAX_PER_STAGE, generate_candidates
from vsq._autotune.types import Constraints, DataProfile, HostInfo, QuantizerConfig


def _host(cores=8, perf=4):
    return HostInfo(isa="neon", logical_cpus=cores, physical_cores=cores, perf_cores=perf,
                    affinity_cpus=cores, platform="Darwin", machine="arm64", cpu_model="test")


def _data(n=100_000, dim=768):
    return DataProfile(n_full=n, dim=dim, n_base_sample=20_000, n_queries=200, k=10, seed=0,
                       queries_source="holdout")


def _size(cfg: QuantizerConfig, dim: int) -> int:
    """Deterministic fake code size: bits/8 per coordinate + 8 bytes."""
    return dim * cfg.bits // 8 + 8


def _names(plan):
    return [[c.name for c in stage] for stage in plan.stages]


def test_catalog_entries_are_valid_and_unique():
    ids = [e.id for e in CATALOG]
    assert len(ids) == len(set(ids))
    for entry in CATALOG:
        entry.config.validate()
        assert entry.id == entry.config.name
        assert entry.profiles <= {"accuracy", "speed", "energy"}


def test_rules_are_deterministic():
    args = ("speed", Constraints.for_profile("speed"), _data(), _host())
    first = generate_candidates(*args, code_size=_size)
    for _ in range(10):
        assert generate_candidates(*args, code_size=_size) == first


@pytest.mark.parametrize("profile", ["speed", "energy", "accuracy"])
def test_every_stage_has_one_to_six_entries(profile):
    plan = generate_candidates(profile, Constraints.for_profile(profile), _data(), _host(),
                               code_size=_size)
    assert plan.stages
    assert all(1 <= len(stage) <= MAX_PER_STAGE for stage in plan.stages)


def test_r1_budget_drops_large_codes_and_traces_them():
    c = Constraints.for_profile("speed", max_bytes_per_vector=400)
    plan = generate_candidates("speed", c, _data(dim=768), _host(), code_size=_size)
    kept = [n for stage in _names(plan) for n in stage]
    assert kept and not any(n.startswith(("rq8", "tq8")) for n in kept)
    assert any("drop rq8-flat: R1" in t for t in plan.trace)


def test_r1_budget_below_smallest_code_gives_empty_plan():
    c = Constraints.for_profile("speed", max_bytes_per_vector=8)
    plan = generate_candidates("speed", c, _data(dim=768), _host(), code_size=_size)
    assert plan.stages == ()
    assert any("no candidate fits max_bytes_per_vector=8, smallest is 104 B (rq1-fs)" in t
               for t in plan.trace)


def test_r2_profile_tags():
    acc = generate_candidates("accuracy", Constraints.for_profile("accuracy"), _data(), _host(),
                              code_size=_size)
    names = {n for stage in _names(acc) for n in stage}
    assert "rq1-fs" not in names and "tq16-flat" in names
    speed = generate_candidates("speed", Constraints.for_profile("speed"), _data(), _host(),
                                code_size=_size)
    assert "tq16-flat" not in {n for stage in _names(speed) for n in stage}


@pytest.mark.parametrize("dim,dropped", [(128, True), (256, True), (512, False), (1536, False)])
def test_r3_small_dim_drops_rabitq_fastscan(dim, dropped):
    plan = generate_candidates("speed", Constraints.for_profile("speed"), _data(dim=dim),
                               _host(), code_size=_size)
    names = {n for stage in _names(plan) for n in stage}
    assert ("rq4-fs" not in names) == dropped
    assert "tq4-fs" in names


def test_r5_threads_only_for_long_flat_scans():
    long = generate_candidates("speed", Constraints.for_profile("speed"), _data(n=1_000_000),
                               _host(), code_size=_size)
    first = [c.name for c in long.first]
    assert "rq4-flat-t4" in first and first.index("rq4-flat-t4") == first.index("rq4-flat") + 1
    assert not any(n.endswith("-t4") and "fs" in n for n in first)
    short = generate_candidates("speed", Constraints.for_profile("speed"), _data(n=1000),
                                _host(), code_size=_size)
    assert not any(c.num_threads > 1 for stage in short.stages for c in stage)
    single = generate_candidates("speed", Constraints.for_profile("speed"), _data(n=1_000_000),
                                 _host(cores=1, perf=None), code_size=_size)
    assert not any(c.num_threads > 1 for stage in single.stages for c in stage)


def test_r5_never_for_accuracy():
    plan = generate_candidates("accuracy", Constraints.for_profile("accuracy"),
                               _data(n=1_000_000), _host(), code_size=_size)
    assert all(c.num_threads == 1 for stage in plan.stages for c in stage)


def test_r6_cap_is_traced():
    plan = generate_candidates("speed", Constraints.for_profile("speed"), _data(n=1_000_000),
                               _host(), code_size=_size)
    assert len(plan.first) == MAX_PER_STAGE
    assert any(t.startswith("cap tiers") and "rq1-fs" in t for t in plan.trace)


def test_escalation_stages_follow_tiers():
    plan = generate_candidates("speed", Constraints.for_profile("speed"), _data(n=1000), _host(),
                               code_size=_size)
    bits = [{c.bits for c in stage} for stage in plan.stages]
    assert bits[0] <= {1, 4} and bits[1] == {8}


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="profile"):
        generate_candidates("fast", Constraints(), _data(), _host(), code_size=_size)

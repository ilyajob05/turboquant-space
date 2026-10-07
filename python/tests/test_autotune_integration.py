"""End-to-end vsq.autotune on small clustered data."""

import numpy as np
import pytest
import vsq
from vsq._autotune.calibrate import Options

FAST = Options(warmup=1, repeats=1, rounds=1)


def _clustered(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((32, dim)) * 2
    X = centers[rng.integers(0, 32, n)] + rng.standard_normal((n, dim))
    return (X + 3.0).astype(np.float32)  # non-zero mean: the centroid matters


@pytest.mark.parametrize("dim", [64, 300])
@pytest.mark.parametrize("profile", ["speed", "accuracy", "energy"])
def test_every_profile_returns_a_built_index(profile, dim):
    X = _clustered(4000, dim)
    r = vsq.autotune(X, profile, n_sample=3000, n_queries=50, time_budget_s=20, options=FAST)
    assert r.chosen is not None and r.profile == profile
    chosen = next(c for c in r.candidates if c.config == r.chosen)
    assert chosen.measurement.recall_at_k >= r.constraints.min_recall
    ids, dists = r.index.search(X[0], 10)
    assert ids.dtype == np.uint32 and len(set(ids.tolist())) == 10 and ids.max() < len(X)
    assert np.all(np.diff(dists) >= -1e-6)
    text = r.report()
    for cand in r.candidates:
        assert cand.config.name in text


def _recalls(result):
    return [(c.config.name, c.measurement.recall_at_k) for c in result.candidates
            if c.measurement]


def test_recall_measurements_are_deterministic():
    X = _clustered(3000, 64)
    kw = {"n_sample": 2000, "n_queries": 40, "time_budget_s": 20, "options": FAST, "build": False, "seed": 5}
    assert _recalls(vsq.autotune(X, "speed", **kw)) == _recalls(vsq.autotune(X, "speed", **kw))


def test_json_roundtrip_rebuilds_identical_codes():
    X = _clustered(3000, 64)
    r = vsq.autotune(X, "speed", n_sample=2000, n_queries=40, time_budget_s=20, options=FAST,
                     candidates=[vsq.QuantizerConfig("turboquant", 4, "flat"),
                                 vsq.QuantizerConfig("rabitq", 8, "flat")],
                     min_recall=0)
    again = vsq.AutotuneResult.from_json(r.to_json())
    assert again.chosen == r.chosen and again.to_dict() == r.to_dict()
    np.testing.assert_array_equal(again.build(X).codes, r.index.codes)


def test_infeasible_raises_with_measurements():
    X = _clustered(3000, 64)
    with pytest.raises(vsq.AutotuneInfeasibleError) as err:
        vsq.autotune(X, "speed", min_recall=0.9999, max_bytes_per_vector=40, n_sample=2000,
                     n_queries=40, time_budget_s=20, options=FAST)
    assert err.value.result is not None and err.value.result.chosen is None
    assert "max_bytes_per_vector=40" in str(err.value)


@pytest.mark.parametrize("kwargs,match", [
    ({"profile": "fast"}, "profile"),
    ({"k": 0}, "k must be"),
    ({"k": 101}, "k must be"),
    ({"min_recall": 1.2}, "min_recall"),
    ({"time_budget_s": 0}, "time_budget_s"),
    ({"n_queries": 0}, "n_sample and n_queries"),
])
def test_invalid_arguments(kwargs, match):
    X = _clustered(1500, 16)
    with pytest.raises(ValueError, match=match):
        vsq.autotune(X, **{"build": False, **kwargs})


def test_too_few_rows():
    with pytest.raises(ValueError, match="needs n >="):
        vsq.autotune(_clustered(500, 16))

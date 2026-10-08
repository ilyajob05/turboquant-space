"""Named presets, vsq.build_index and preset names inside vsq.autotune."""

import numpy as np
import pytest
import vsq
from vsq._autotune.calibrate import Options
from vsq._autotune.candidates import CATALOG


def _data(n, dim, seed=0):
    return (np.random.default_rng(seed).standard_normal((n, dim)) + 1.0).astype(np.float32)


@pytest.mark.parametrize("name,dim,expected_id", [
    ("compact", 128, "rq1-flat"),   # R3: RaBitQ FastScan loses to the flat scan at dim <= 256
    ("compact", 768, "rq1-fs"),
    ("balanced", 128, "tq4-fs"),
    ("balanced", 1536, "tq4-fs"),
    ("accurate", 768, "rq8-flat"),
])
def test_preset_resolves_to_catalog_config(name, dim, expected_id):
    config = vsq.preset(name, dim)
    assert config.name == expected_id
    if expected_id != "rq1-flat":
        assert config in {e.config for e in CATALOG}


def test_preset_threads_and_validation():
    assert vsq.preset("accurate", 64, num_threads=4).name == "rq8-flat-t4"
    with pytest.raises(ValueError, match="unknown preset"):
        vsq.preset("fast", 64)
    with pytest.raises(ValueError):
        vsq.preset("balanced", 0)
    with pytest.raises(ValueError):
        vsq.preset("balanced", 64, num_threads=0)


def test_presets_are_ordered_by_code_size():
    X = _data(300, 768)
    sizes = [vsq.build_index(X, name).code_size_bytes
             for name in ("compact", "balanced", "accurate")]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


@pytest.mark.parametrize("name", sorted(vsq.PRESETS))
def test_build_index_from_preset_searches(name):
    X = _data(2000, 64)
    index = vsq.build_index(X, name)
    ids, dists = index.search(X[3], 10)
    assert ids.shape == (10,) and ids.dtype == np.uint32
    assert np.all(np.diff(dists) >= -1e-6)
    assert len(index) == len(X)


def test_build_index_defaults_to_balanced_and_accepts_config():
    X = _data(500, 32)
    assert vsq.build_index(X).config == vsq.preset("balanced", 32)
    config = vsq.QuantizerConfig("turboquant", 8, "flat")
    assert vsq.build_index(X, config, num_threads=2).config == config.with_threads(2)
    with pytest.raises(TypeError):
        vsq.build_index(X, 4)


def test_autotune_accepts_preset_names_and_labels_the_choice():
    X = _data(3000, 64)
    r = vsq.autotune(X, "accuracy", candidates=["balanced", "accurate"], n_sample=2000,
                     n_queries=40, time_budget_s=20, options=Options(warmup=1, repeats=1, rounds=1))
    assert [c.config.name for c in r.candidates] == ["tq4-fs", "rq8-flat"]
    chosen_line = r.report().split("chosen: ", 1)[1].splitlines()[0]
    assert '(preset "' in chosen_line


def test_small_data_error_points_to_presets():
    with pytest.raises(ValueError, match="build_index"):
        vsq.autotune(_data(200, 16))

"""Named presets: fixed quantizer configurations for building without calibration.

A preset is a size / accuracy tier, chosen from the vsq 0.2.0 measurements
(docs/benchmarks.md, Apple M3, 1 thread, N(0, 1) at dim 128/768/1024 and
DBpedia OpenAI embeddings at dim 1536). Every preset is an entry of the
autotune catalog, so its name in an autotune report is the same id.

    preset    id                 code size        recall@10 (dbpedia / N(0,1))
    compact   rq1-fs | rq1-flat  dim/8 + 8 B      0.83 / 0.26-0.30
    balanced  tq4-fs             dim/2 + 14 B     0.97 / 0.87
    accurate  rq8-flat           dim + 8 B        0.999 / 0.985-0.990

``compact`` uses the flat scan at dim <= 256, where RaBitQ FastScan is
slower than it (rule R3 of the catalog).
"""

from __future__ import annotations

from dataclasses import replace

from .candidates import _SMALL_DIM_RABITQ_FS, CATALOG
from .types import QuantizerConfig

DEFAULT_PRESET = "balanced"

PRESETS: dict[str, str] = {
    "compact": ("RaBitQ 1-bit (~32x smaller than float32): smallest codes; recall@10 "
                "~0.83 on real embeddings, so re-rank a larger k"),
    "balanced": ("TurboQuant 4-bit FastScan (~8x smaller): fastest search at every "
                 "measured dim, recall@10 ~0.97 on real embeddings"),
    "accurate": ("RaBitQ 8-bit flat scan (~4x smaller): recall@10 >= 0.98, "
                 "slower encode and search"),
}

_CATALOG_ID = {"compact": "rq1-fs", "balanced": "tq4-fs", "accurate": "rq8-flat"}


def _catalog_config(entry_id: str) -> QuantizerConfig:
    return next(e.config for e in CATALOG if e.id == entry_id)


def preset(name: str, dim: int, *, num_threads: int = 1) -> QuantizerConfig:
    """QuantizerConfig of preset ``name`` for vectors of dimension ``dim``.

    Args:
      name: one of PRESETS ("compact", "balanced", "accurate").
      dim: vector dimension, >= 1 (``compact`` picks its search path from it).
      num_threads: OpenMP threads of the flat scan and of encoding, >= 1.
    Raises: ValueError for an unknown name or invalid dim / num_threads.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose one of {sorted(PRESETS)}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    config = _catalog_config(_CATALOG_ID[name])
    if config.index == "fastscan" and config.family == "rabitq" and dim <= _SMALL_DIM_RABITQ_FS:
        config = replace(config, index="flat", eps0=None)
    return config.with_threads(num_threads)


def preset_name(config: QuantizerConfig, dim: int) -> str | None:
    """Name of the preset equal to ``config`` (thread count ignored), else None."""
    single = config.with_threads(1)
    return next((name for name in PRESETS if preset(name, dim) == single), None)

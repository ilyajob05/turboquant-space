"""TurboQuant / RaBitQ vector quantization for ANN search.

TurboQuantSpace writes code format v2. See README.md for the formats and options.
"""
from ._vsq import (
    RaBitQFastScan,
    RaBitQSpace,
    TurboQuantFastScan,
    TurboQuantSpace,
    detected_isa,
)

__version__ = "0.2.0"

from ._autotune.api import autotune
from ._autotune.energy import EnergyModel
from ._autotune.index import QuantizedIndex
from ._autotune.result import AutotuneResult
from ._autotune.select import AutotuneInfeasibleError
from ._autotune.types import Constraints, QuantizerConfig

__all__ = [
    "AutotuneInfeasibleError",
    "AutotuneResult",
    "Constraints",
    "EnergyModel",
    "QuantizedIndex",
    "QuantizerConfig",
    "RaBitQFastScan",
    "RaBitQSpace",
    "TurboQuantFastScan",
    "TurboQuantSpace",
    "autotune",
    "detected_isa",
]

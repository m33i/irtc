from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from irtc.domain.solar import SolarEstimate


@runtime_checkable
class SolarEstimatorPort(Protocol):
    def estimate(
        self,
        image_path: Path,
        sky_crop: np.ndarray | None = None,
    ) -> SolarEstimate: ...

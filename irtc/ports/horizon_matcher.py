from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from irtc.domain.horizon import HorizonMatchResult, HorizonProfile


@runtime_checkable
class HorizonMatcherPort(Protocol):
    def extract_profile(
        self, image_path: Path, sky_mask: np.ndarray,
    ) -> HorizonProfile: ...

    def match(
        self, photo_profile: HorizonProfile,
        lat_range: tuple[float, float] | None,
        lon_hint: float | None,
        target_lats: list[float] | None = None,
        target_lons: list[float] | None = None,
    ) -> HorizonMatchResult: ...

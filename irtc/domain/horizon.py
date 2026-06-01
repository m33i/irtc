from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class HorizonProfile:
    elevation_angles: np.ndarray
    azimuth_deg: np.ndarray
    confidence: float


@dataclass
class HorizonMatchResult:
    matched: bool
    estimated_lat: float | None = None
    estimated_lon: float | None = None
    lat_range: tuple[float, float] | None = None
    lon_range: tuple[float, float] | None = None
    camera_azimuth_deg: float | None = None
    correlation_score: float | None = None
    north: float | None = None
    south: float | None = None
    east: float | None = None
    west: float | None = None
    confidence: float = 0.0

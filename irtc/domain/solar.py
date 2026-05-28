"""Domain — Solar position estimate."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SolarEstimate:
    sun_visible: bool
    elevation_deg: float | None
    time_of_day: str             # "night" | "dawn_dusk" | "golden_hour" | "daytime" | "midday" | "overcast"
    hour_range: tuple[int, int]  # approximate UTC range
    hemisphere: str | None       # "north" | "south" | "equatorial" | "unknown"
    lat_range: tuple[float, float] | None
    season_hint: str | None      # "summer" | "winter" | "equinox" | "unknown"
    confidence: float

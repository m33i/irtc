"""Domain — Satellite search entities."""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BBox:
    """Bounding box geográfico (WGS84)."""
    west: float
    south: float
    east: float
    north: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


@dataclass
class SatelliteCandidate:
    id: str
    collection: str
    bbox: BBox
    captured_at: datetime       # UTC
    cloud_cover_pct: float
    thumbnail_url: str | None
    assets: dict[str, str]      # asset name → URL
    stac_url: str

    @property
    def lat(self) -> float:
        return self.bbox.center[1]

    @property
    def lon(self) -> float:
        return self.bbox.center[0]


@dataclass
class SatelliteSearchResult:
    candidates: list[SatelliteCandidate]
    total_searched: int
    collections_used: list[str]
    query_bbox: BBox | None
    query_date_range: tuple[str, str]

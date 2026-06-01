"""Domain — SearchConstraints and CloudAnalysis (full pipeline result)."""

from __future__ import annotations
from dataclasses import dataclass
import json

from irtc.domain.cloud import ClassificationResult, SegmentationResult, CloudFeatures
from irtc.domain.horizon import HorizonMatchResult
from irtc.domain.solar import SolarEstimate


@dataclass
class SearchConstraints:
    """Constraints derived from visual analysis to narrow the satellite search."""
    lat_range: tuple[float, float] | None
    hour_range: tuple[int, int] | None
    season_hint: str | None
    hemisphere: str | None
    cloud_coverage_min: float
    cloud_coverage_max: float
    cloud_level: str
    lon_range: tuple[float, float] | None = None


@dataclass
class CloudAnalysis:
    """Complete result of analysing one cloud photo."""
    image_path: str
    segmentation: SegmentationResult
    classification: ClassificationResult
    solar: SolarEstimate
    features: CloudFeatures
    search_constraints: SearchConstraints
    horizon_match: HorizonMatchResult | None = None

    def summary(self) -> str:
        seg_icon = "neural" if self.segmentation.method == "neural" else "heuristic"
        sc = self.search_constraints
        lines = [
            "CLOUD ANALYSIS — I Remember That Cloud",
            "=" * 50,
            f"SEGMENTATION ({seg_icon}):  {self.segmentation.sky_ratio:.0%} sky",
            "",
            "CLOUD TYPE:",
            f"{self.classification.primary.name} ({self.classification.primary.abbr})"
            f"  [{self.classification.primary.level} · {self.classification.primary.altitude_range}]",
            f"     {self.classification.primary.description}",
            f"     Confidence: {self.classification.confidence:.0%}",
            "",
            "  Top 3:",
        ]
        for name, prob in self.classification.top3:
            lines.append(f"    {name:<16} {'█' * int(prob * 20)} {prob:.0%}")

        if self.horizon_match and self.horizon_match.matched:
            hm = self.horizon_match
            lines += [
                "",
                "HORIZON MATCH:",
                f"Location:   {hm.estimated_lat:.3f}°, {hm.estimated_lon:.3f}°",
                f"Camera:     facing {hm.camera_azimuth_deg:.0f}°",
                f"Confidence: {hm.confidence:.0%}",
            ]
            if hm.west is not None and hm.east is not None:
                lines.append(f"FOV:        {hm.west:.0f}° – {hm.east:.0f}°")

        lines += [
            "",
            "SEARCH CONSTRAINTS:",
            f"Latitude:  {sc.lat_range or 'global'}",
            (f"Longitude: {sc.lon_range[0]:.1f}° to {sc.lon_range[1]:.1f}°"
             if sc.lon_range else ""),
            (f"UTC hour:  {sc.hour_range[0]}h – {sc.hour_range[1]}h"
             if sc.hour_range else "  UTC hour:  any"),
            f"Coverage:  {sc.cloud_coverage_min:.0f}% – {sc.cloud_coverage_max:.0f}%",
            f"Level:     {sc.cloud_level}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        sc = self.search_constraints
        return {
            "image_path": self.image_path,
            "sky_ratio": self.segmentation.sky_ratio,
            "cloud_type": {
                "name": self.classification.primary.name,
                "abbr": self.classification.primary.abbr,
                "level": self.classification.primary.level,
                "altitude_range": self.classification.primary.altitude_range,
                "confidence": self.classification.confidence,
                "top3": self.classification.top3,
            },
            "solar": {
                "sun_visible": self.solar.sun_visible,
                "elevation_deg": self.solar.elevation_deg,
                "time_of_day": self.solar.time_of_day,
                "hour_range": self.solar.hour_range,
                "hemisphere": self.solar.hemisphere,
                "lat_range": self.solar.lat_range,
                "season_hint": self.solar.season_hint,
                "confidence": self.solar.confidence,
                "azimuth_deg": self.solar.azimuth_deg,
                "estimated_lat": self.solar.estimated_lat,
                "lat_confidence": self.solar.lat_confidence,
            },
            "features": {
                "cloud_coverage_pct": self.features.cloud_coverage_pct,
                "dominant_brightness": self.features.dominant_brightness,
                "embedding_dim": len(self.features.embedding),
            },
            "horizon_match": {
                "matched": self.horizon_match.matched if self.horizon_match else False,
                "estimated_lat": self.horizon_match.estimated_lat if self.horizon_match else None,
                "estimated_lon": self.horizon_match.estimated_lon if self.horizon_match else None,
                "camera_azimuth_deg": self.horizon_match.camera_azimuth_deg if self.horizon_match else None,
                "correlation_score": self.horizon_match.correlation_score if self.horizon_match else None,
                "confidence": self.horizon_match.confidence if self.horizon_match else None,
            } if self.horizon_match else None,
            "search_constraints": {
                "lat_range": sc.lat_range,
                "lon_range": sc.lon_range,
                "hour_range": sc.hour_range,
                "season_hint": sc.season_hint,
                "hemisphere": sc.hemisphere,
                "cloud_coverage": [sc.cloud_coverage_min, sc.cloud_coverage_max],
                "cloud_level": sc.cloud_level,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

"""AnalyzeCloudUseCase — orchestrates the full photo analysis pipeline."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from irtc.domain.analysis import CloudAnalysis, SearchConstraints
from irtc.domain.cloud import ClassificationResult, CloudFeatures
from irtc.domain.horizon import HorizonMatchResult
from irtc.domain.solar import SolarEstimate
from irtc.ports.segmenter import SkySegmenterPort
from irtc.ports.classifier import CloudClassifierPort
from irtc.ports.solar_estimator import SolarEstimatorPort
from irtc.ports.feature_extractor import FeatureExtractorPort
from irtc.ports.horizon_matcher import HorizonMatcherPort


class AnalyzeCloudUseCase:

    def __init__(
        self,
        segmenter: SkySegmenterPort,
        classifier: CloudClassifierPort,
        solar_estimator: SolarEstimatorPort,
        feature_extractor: FeatureExtractorPort,
        horizon_matcher: HorizonMatcherPort | None = None,
    ) -> None:
        self._segmenter         = segmenter
        self._classifier        = classifier
        self._solar_estimator   = solar_estimator
        self._feature_extractor = feature_extractor
        self._horizon_matcher   = horizon_matcher

    def execute(self, image_path: Path) -> CloudAnalysis:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nAnalysing: {image_path.name}")

        print("  [0/4] Segmenting sky...")
        seg = self._segmenter.segment(image_path)
        print(f"        {seg.sky_ratio:.0%} sky ({seg.method})")

        sky_bbox = self._crop_sky_bbox(image_path, seg.mask)

        print("  [1/4] Classifying cloud type...")
        classification = self._classifier.classify(sky_bbox)
        print(f"        {classification.primary.name} ({classification.confidence:.0%})")

        print("  [2/4] Estimating solar/temporal position...")
        solar = self._solar_estimator.estimate(image_path, sky_crop=seg.sky_crop)
        if solar.time_of_day == "night":
            stars = "stars detected" if solar.confidence > 0.8 else "overcast"
            print(f"        night ({stars})")
        else:
            print(f"        {solar.time_of_day}, elevation {solar.elevation_deg}°")

        print("  [3/4] Extracting visual fingerprint...")
        features = self._feature_extractor.extract(sky_bbox, sky_full_rgb=seg.sky_crop)
        print(f"        {features.cloud_coverage_pct:.0f}% coverage, "
              f"brightness {features.dominant_brightness:.0f}, "
              f"{len(features.embedding)}-dim embedding")

        horizon_match = None
        if self._horizon_matcher is not None and solar.time_of_day != "night":
            print("  [4/4] Matching horizon against DEM...")
            try:
                photo_horizon = self._horizon_matcher.extract_profile(image_path, seg.mask)
                if photo_horizon.confidence > 0.3:
                    print(f"        profile: {len(photo_horizon.elevation_angles)} pts, confidence {photo_horizon.confidence:.0%}")
                    horizon_match = self._horizon_matcher.match(
                        photo_horizon,
                        lat_range=solar.estimated_lat if solar.lat_confidence else solar.lat_range,
                    )
                    if horizon_match and horizon_match.matched:
                        print(f"        matched at {horizon_match.estimated_lat:.3f}°, {horizon_match.estimated_lon:.3f}° (corr={horizon_match.correlation_score:.3f})")
                    else:
                        print("        no match")
                else:
                    print("        no terrain visible")
            except Exception as e:
                print(f"        horizon match failed: {e}")

        sc = self._build_constraints(classification, solar, features, horizon_match)

        return CloudAnalysis(
            image_path     = str(image_path),
            segmentation   = seg,
            classification = classification,
            solar          = solar,
            features       = features,
            search_constraints = sc,
            horizon_match  = horizon_match,
        )

    def _crop_sky_bbox(self, image_path: Path, mask: np.ndarray) -> Image.Image:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
            cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
            return Image.open(image_path).convert("RGB").crop((cmin, rmin, cmax, rmax))
        return Image.open(image_path).convert("RGB")

    def _build_constraints(
        self,
        classification: ClassificationResult,
        solar: SolarEstimate,
        features: CloudFeatures,
        horizon_match: HorizonMatchResult | None = None,
    ) -> SearchConstraints:
        cov = features.cloud_coverage_pct

        if solar.time_of_day == "night":
            stars_clear = solar.confidence > 0.8
            return SearchConstraints(
                lat_range          = None,
                hour_range         = (20, 6),
                season_hint        = None,
                hemisphere         = None,
                cloud_coverage_min = 0.0  if stars_clear else max(0.0, cov - 20),
                cloud_coverage_max = 25.0 if stars_clear else min(100.0, cov + 20),
                cloud_level        = classification.primary.level,
            )

        lat_range = solar.lat_range
        hour_range = solar.hour_range

        lon_range = None
        if horizon_match and horizon_match.matched and horizon_match.confidence > 0.3:
            margin = 0.5
            if horizon_match.estimated_lat is not None:
                lat_range = (
                    max(-90.0, round(horizon_match.estimated_lat - margin, 1)),
                    min( 90.0, round(horizon_match.estimated_lat + margin, 1)),
                )
            if horizon_match.west is not None and horizon_match.east is not None:
                lon_center = (horizon_match.west + horizon_match.east) / 2.0
                lon_margin = abs(horizon_match.east - horizon_match.west) / 2.0 + 2.0
                lon_range = (lon_center - lon_margin, lon_center + lon_margin)
        elif solar.estimated_lat is not None:
            margin = 3.0 if solar.time_of_day == "midday" else 5.0
            lat_range = (
                max(-90.0, round(solar.estimated_lat - margin, 1)),
                min( 90.0, round(solar.estimated_lat + margin, 1)),
            )

        if hour_range and hour_range != (0, 24):
            hour_range = (max(0, hour_range[0] - 1), min(24, hour_range[1] + 1))

        return SearchConstraints(
            lat_range          = lat_range,
            hour_range         = hour_range,
            season_hint        = solar.season_hint,
            hemisphere         = solar.hemisphere,
            cloud_coverage_min = max(0.0, cov - 20),
            cloud_coverage_max = min(100.0, cov + 20),
            cloud_level        = classification.primary.level,
            lon_range          = lon_range,
        )

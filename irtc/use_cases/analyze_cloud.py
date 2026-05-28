"""AnalyzeCloudUseCase — orchestrates the full photo analysis pipeline."""

from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from irtc.domain.analysis import CloudAnalysis, SearchConstraints
from irtc.domain.cloud import ClassificationResult, CloudFeatures
from irtc.domain.solar import SolarEstimate
from irtc.ports.segmenter import SkySegmenterPort
from irtc.ports.classifier import CloudClassifierPort
from irtc.ports.solar_estimator import SolarEstimatorPort
from irtc.ports.feature_extractor import FeatureExtractorPort


class AnalyzeCloudUseCase:

    def __init__(
        self,
        segmenter: SkySegmenterPort,
        classifier: CloudClassifierPort,
        solar_estimator: SolarEstimatorPort,
        feature_extractor: FeatureExtractorPort,
    ) -> None:
        self._segmenter         = segmenter
        self._classifier        = classifier
        self._solar_estimator   = solar_estimator
        self._feature_extractor = feature_extractor

    def execute(self, image_path: Path) -> CloudAnalysis:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nAnalysing: {image_path.name}")

        print("  [0/3] Segmenting sky...")
        seg = self._segmenter.segment(image_path)
        print(f"        {seg.sky_ratio:.0%} sky ({seg.method})")

        sky_bbox = self._crop_sky_bbox(image_path, seg.mask)

        print("  [1/3] Classifying cloud type...")
        classification = self._classifier.classify(sky_bbox)
        print(f"        {classification.primary.name} ({classification.confidence:.0%})")

        print("  [2/3] Estimating solar/temporal position...")
        solar = self._solar_estimator.estimate(image_path, sky_crop=seg.sky_crop)
        if solar.time_of_day == "night":
            stars = "stars detected" if solar.confidence > 0.8 else "overcast"
            print(f"        night ({stars})")
        else:
            print(f"        {solar.time_of_day}, elevation {solar.elevation_deg}°")

        print("  [3/3] Extracting visual fingerprint...")
        features = self._feature_extractor.extract(sky_bbox, sky_full_rgb=seg.sky_crop)
        print(f"        {features.cloud_coverage_pct:.0f}% coverage, "
              f"brightness {features.dominant_brightness:.0f}, "
              f"{len(features.embedding)}-dim embedding")

        return CloudAnalysis(
            image_path     = str(image_path),
            segmentation   = seg,
            classification = classification,
            solar          = solar,
            features       = features,
            search_constraints = self._build_constraints(classification, solar, features),
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
    ) -> SearchConstraints:
        cov = features.cloud_coverage_pct

        # Night: if stars visible, sky was clear → look for low-cloud satellite tiles
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
        if lat_range:
            lat_range = (max(-90.0, lat_range[0] - 5), min(90.0, lat_range[1] + 5))

        # Solar geometry tightening at midday in known season
        elev = solar.elevation_deg
        if (
            elev is not None
            and solar.time_of_day == "midday"
            and solar.season_hint in ("summer", "equinox")
        ):
            margin = 6.0
            if solar.hemisphere in ("north", "unknown"):
                geo_min = round(max(  0.0, 90.0 - elev - margin), 1)
                geo_max = round(min( 90.0, 90.0 - elev + 23.5 + margin), 1)
                lat_range = (
                    max(lat_range[0], geo_min) if lat_range else geo_min,
                    min(lat_range[1], geo_max) if lat_range else geo_max,
                )

        hour_range = solar.hour_range
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
        )

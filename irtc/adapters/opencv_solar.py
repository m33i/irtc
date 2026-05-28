"""OpenCV solar / night estimator. Implements SolarEstimatorPort."""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from irtc.domain.solar import SolarEstimate


class OpenCVSolarEstimator:

    def estimate(
        self,
        image_path: Path,
        sky_crop: np.ndarray | None = None,
    ) -> SolarEstimate:
        img_bgr = (
            cv2.cvtColor(sky_crop, cv2.COLOR_RGB2BGR)
            if sky_crop is not None
            else self._load(image_path)
        )

        if self._is_night(img_bgr):
            stars = self._has_stars(img_bgr)
            return SolarEstimate(
                sun_visible  = False,
                elevation_deg= None,
                time_of_day  = "night",
                hour_range   = (20, 6),
                hemisphere   = None,
                lat_range    = None,
                season_hint  = None,
                confidence   = 0.9 if stars else 0.5,
            )

        h, w = img_bgr.shape[:2]
        use_crop = sky_crop is not None

        sun_visible, sun_x, sun_y = self._detect_sun(img_bgr, use_crop)
        color_temp, time_of_day, hour_range = self._sky_color_analysis(img_bgr, use_crop)
        elevation_deg = self._estimate_elevation(sun_x, sun_y, h, time_of_day)
        hemisphere, lat_range = self._estimate_lat(elevation_deg, time_of_day)
        season_hint = self._estimate_season(color_temp, elevation_deg)

        return SolarEstimate(
            sun_visible  = sun_visible,
            elevation_deg= elevation_deg,
            time_of_day  = time_of_day,
            hour_range   = hour_range,
            hemisphere   = hemisphere,
            lat_range    = lat_range,
            season_hint  = season_hint,
            confidence   = self._confidence(sun_visible, elevation_deg, color_temp),
        )

    # Night / star detection

    @staticmethod
    def _is_night(img_bgr: np.ndarray) -> bool:
        return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].mean()) < 45

    @staticmethod
    def _has_stars(img_bgr: np.ndarray) -> bool:
        """Stars: many tiny bright point sources in a dark field."""
        h = img_bgr.shape[0]
        upper = img_bgr[:int(h * 0.8), :]       # ignore ground lights at bottom
        gray  = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in contours if cv2.contourArea(c) < 20) > 15

    # Sun detection

    def _detect_sun(
        self, img_bgr: np.ndarray, use_sky_crop: bool
    ) -> tuple[bool, int | None, int | None]:
        h, w = img_bgr.shape[:2]

        if use_sky_crop:
            sky_mask  = np.any(img_bgr > 15, axis=-1).astype(np.uint8) * 255
            analysis  = img_bgr
        else:
            sky_mask  = None
            analysis  = img_bgr[:int(h * 0.70), :]

        v = cv2.cvtColor(analysis, cv2.COLOR_BGR2HSV)[:, :, 2]
        if sky_mask is not None:
            v = cv2.bitwise_and(v, v, mask=sky_mask)

        _, bright = cv2.threshold(v, 248, 255, cv2.THRESH_BINARY)
        bright = cv2.erode( bright, np.ones((3, 3)), iterations=2)
        bright = cv2.dilate(bright, np.ones((5, 5)), iterations=3)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 30:
                continue
            circ = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-6)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            score  = circ * (1.5 if cy < h * 0.5 else 1.0)
            if score > best_score:
                best_score, best = score, (cx, cy)

        if best and best_score > 0.25:
            return True, best[0], best[1]

        # Fallback: brightest region as a sun-glare indicator
        if use_sky_crop and sky_mask is not None:
            blurred = cv2.GaussianBlur(cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY), (51, 51), 0)
            blurred = cv2.bitwise_and(blurred, blurred, mask=sky_mask)
        else:
            blurred = cv2.GaussianBlur(
                cv2.cvtColor(img_bgr[:int(h * 0.6), :], cv2.COLOR_BGR2GRAY), (51, 51), 0
            )
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        if max_val > 190:
            return False, max_loc[0], max_loc[1]
        return False, None, None

    # Sky analysis helpers

    def _sky_color_analysis(
        self, img_bgr: np.ndarray, use_sky_crop: bool
    ) -> tuple[float, str, tuple[int, int]]:
        if use_sky_crop:
            sky_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            sky_mask = np.any(sky_rgb > 15, axis=-1)
            rb = (float(sky_rgb[sky_mask, 0].mean()) / (float(sky_rgb[sky_mask, 2].mean()) + 1e-6)
                  if sky_mask.sum() > 200 else 0.6)
        else:
            h       = img_bgr.shape[0]
            sky_rgb = cv2.cvtColor(img_bgr[:int(h * 0.5), :], cv2.COLOR_BGR2RGB)
            rb      = float(sky_rgb[:, :, 0].mean()) / (float(sky_rgb[:, :, 2].mean()) + 1e-6)

        if   rb < 0.6: ct = 7000.0
        elif rb < 0.9: ct = 5500.0
        elif rb < 1.5: ct = 4000.0
        elif rb < 2.5: ct = 3000.0
        else:          ct = 2000.0

        if   ct > 6000: return ct, "midday",       (10, 15)
        if   ct > 5000: return ct, "daytime",      ( 8, 18)
        if   ct > 3500: return ct, "golden_hour",  ( 6,  9)
        if   ct > 2500: return ct, "dawn_dusk",    ( 5,  8)
        return ct, "overcast", (0, 24)

    # Estimation helpers

    @staticmethod
    def _estimate_elevation(
        sun_x: int | None, sun_y: int | None, h: int, tod: str
    ) -> float | None:
        if sun_x is not None and sun_y is not None:
            horizon_y = h * 0.85
            rel = max(0.0, min(1.0, (horizon_y - sun_y) / horizon_y))
            return round(rel * 90.0, 1)
        return {"midday": 55.0, "golden_hour": 15.0, "dawn_dusk": 5.0}.get(tod)

    @staticmethod
    def _estimate_lat(
        elev: float | None, tod: str
    ) -> tuple[str | None, tuple[float, float] | None]:
        if elev is None:
            return "unknown", None
        if tod != "midday":
            if elev > 70: return "equatorial", (-30.0, 30.0)
            if elev > 50: return "unknown",    (-50.0, 50.0)
            return "unknown", (-70.0, 70.0)
        # midday: elevation ≈ 90° − |lat − declination|; ±23.5° seasonal variance
        c = 90.0 - elev
        lat_min = max(-90.0, round(c - 33.5, 1))
        lat_max = min( 90.0, round(c + 33.5, 1))
        lat_mid = (lat_min + lat_max) / 2
        hem = "north" if lat_mid > 10 else ("south" if lat_mid < -10 else "unknown")
        return hem, (lat_min, lat_max)

    @staticmethod
    def _estimate_season(ct: float, elev: float | None) -> str:
        if elev is None:            return "unknown"
        if ct < 3500 and elev < 30: return "winter"
        if elev > 55 and ct > 5500: return "summer"
        if 35 < elev < 65:          return "equinox"
        return "unknown"

    @staticmethod
    def _confidence(sun_visible: bool, elev: float | None, ct: float) -> float:
        c = 0.2 + (0.5 if sun_visible else 0.0) + (0.2 if elev is not None else 0.0) + (0.1 if ct != 4000.0 else 0.0)
        return round(min(c, 1.0), 2)

    @staticmethod
    def _load(image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            img = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        return img

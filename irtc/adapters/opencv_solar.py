"""OpenCV solar / night estimator. Implements SolarEstimatorPort."""

from __future__ import annotations
import math
import datetime
from collections import Counter
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
        elevation_deg = self._estimate_elevation(sun_x, sun_y, h, time_of_day, use_crop, img_bgr)
        azimuth_deg = self._estimate_azimuth(sun_x, w) if sun_x is not None else None

        bt = self._backtrack_location(elevation_deg, hour_range, time_of_day)
        if bt is not None:
            hemisphere, lat_range, season_hint, estimated_lat, lat_conf = bt
        else:
            hemisphere = self._fallback_hemisphere(elevation_deg, time_of_day)
            lat_range = self._fallback_lat_range(elevation_deg, time_of_day)
            season_hint = self._fallback_season(color_temp, elevation_deg)
            estimated_lat = None
            lat_conf = None

        return SolarEstimate(
            sun_visible   = sun_visible,
            elevation_deg = elevation_deg,
            time_of_day   = time_of_day,
            hour_range    = hour_range,
            hemisphere    = hemisphere,
            lat_range     = lat_range,
            season_hint   = season_hint,
            confidence    = self._confidence(sun_visible, elevation_deg, color_temp, lat_conf),
            azimuth_deg   = azimuth_deg,
            estimated_lat = estimated_lat,
            lat_confidence= lat_conf,
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

    def _estimate_elevation(
        self, sun_x: int | None, sun_y: int | None, h: int, tod: str,
        use_crop: bool, img_bgr: np.ndarray,
    ) -> float | None:
        if sun_x is not None and sun_y is not None:
            horizon_y = self._detect_horizon(img_bgr, h, use_crop)
            rel = max(0.0, min(1.0, (horizon_y - sun_y) / horizon_y))
            return round(rel * 90.0, 1)
        return {"midday": 55.0, "golden_hour": 15.0, "dawn_dusk": 5.0}.get(tod)

    @staticmethod
    def _detect_horizon(img_bgr: np.ndarray, h: int, use_crop: bool) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        row_profile = gray.mean(axis=1).astype(np.float32)
        grad = np.abs(np.diff(row_profile))
        if len(grad) > 5:
            kernel = np.hanning(7)
            kernel /= kernel.sum()
            smooth = np.convolve(grad, kernel, mode="same") if len(grad) > 7 else grad
            peak = int(np.argmax(smooth))
            if peak > h * 0.2:
                return float(peak)
        return float(h * 0.85)

    @staticmethod
    def _estimate_azimuth(sun_x: int, img_width: int) -> float:
        fov_deg = 60.0
        rel_pos = (sun_x / img_width - 0.5) * 2
        return round(rel_pos * (fov_deg / 2), 1)

    def _backtrack_location(
        self, elev_deg: float | None, hour_range: tuple[int, int], tod: str,
    ) -> tuple[str, tuple[float, float], str, float, float] | None:
        if elev_deg is None:
            return None
        try:
            from suncalc import get_position
        except ImportError:
            return None

        season_dates = [
            ("summer", 6, 21), ("winter", 12, 21),
            ("equinox", 3, 20), ("equinox", 9, 22),
        ]

        if tod == "midday":
            search_hours = [12, 11, 13, 10, 14]
        else:
            lo, hi = hour_range
            if lo <= hi:
                search_hours = list(range(max(0, lo), min(24, hi + 1)))
            else:
                search_hours = list(range(lo, 24)) + list(range(0, hi + 1))
            if not search_hours:
                search_hours = list(range(6, 19))

        candidates: list[tuple[float, float, str]] = []
        for season_name, month, day in season_dates:
            for lat_deg in range(-90, 91, 1):
                for utc_hour in search_hours:
                    try:
                        dt = datetime.datetime(2024, month, day, utc_hour, 0, 0)
                        pos = get_position(dt, 0.0, float(lat_deg))
                        calc_elev = math.degrees(pos["altitude"])
                        if calc_elev < 0:
                            continue
                        diff = abs(calc_elev - elev_deg)
                        score = 1.0 - min(diff / 30.0, 1.0)
                        if score > 0.3:
                            candidates.append((score, float(lat_deg), season_name))
                    except Exception:
                        continue

        if not candidates:
            return None

        scores_by_lat: dict[float, float] = {}
        for score, lat, _ in candidates:
            scores_by_lat[lat] = scores_by_lat.get(lat, 0.0) + score

        sorted_lats = sorted(scores_by_lat.items(), key=lambda x: -x[1])
        top_lats = sorted_lats[:10]

        total_weight = sum(w for _, w in top_lats)
        if total_weight == 0:
            return None

        estimated_lat = sum(lat * w for lat, w in top_lats) / total_weight

        variance = sum(w * (lat - estimated_lat) ** 2 for lat, w in top_lats) / total_weight
        lat_std = math.sqrt(variance) if variance > 0 else 5.0
        margin = max(3.0, lat_std * 2)
        lat_range = (
            max(-90.0, round(estimated_lat - margin, 1)),
            min(90.0, round(estimated_lat + margin, 1)),
        )

        hem = "north" if estimated_lat > 15 else ("south" if estimated_lat < -15 else "equatorial")

        season_counts = Counter(s for _, _, s in candidates)
        best_season = season_counts.most_common(1)[0][0] if season_counts else "unknown"

        best_score = max(s for s, _, _ in candidates)
        lat_confidence = round(min(1.0, best_score * 1.5), 2)

        return hem, lat_range, best_season, round(estimated_lat, 2), lat_confidence

    @staticmethod
    def _fallback_hemisphere(elev: float | None, tod: str) -> str | None:
        if elev is None or tod != "midday":
            return None
        c = 90.0 - elev
        return "north" if c > 10 else ("south" if c < -10 else "equatorial")

    @staticmethod
    def _fallback_lat_range(elev: float | None, tod: str) -> tuple[float, float] | None:
        if elev is None:
            return None
        c = 90.0 - elev
        lat_min = max(-90.0, round(c - 23.5, 1))
        lat_max = min( 90.0, round(c + 23.5, 1))
        return (lat_min, lat_max)

    @staticmethod
    def _fallback_season(ct: float, elev: float | None) -> str:
        if elev is None:            return "unknown"
        if ct < 3500 and elev < 30: return "winter"
        if elev > 55 and ct > 5500: return "summer"
        if 35 < elev < 65:          return "equinox"
        return "unknown"

    @staticmethod
    def _confidence(sun_visible: bool, elev: float | None, ct: float, lat_conf: float | None = None) -> float:
        c = 0.2
        if sun_visible:   c += 0.3
        if elev is not None: c += 0.2
        if ct != 4000.0:  c += 0.1
        if lat_conf is not None: c += 0.2 * lat_conf
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

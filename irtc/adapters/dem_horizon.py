"""Horizon matcher using OpenTopoData DEM + photo sky mask."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from irtc.domain.horizon import HorizonMatchResult, HorizonProfile


class DemHorizonMatcher:

    def __init__(
        self,
        search_radius_km: float = 25.0,
        dem_step_deg: float = 0.05,
        dem_api_url: str = "https://api.opentopodata.org/v1/aster30m",
        cache_ttl: float = 600.0,
    ) -> None:
        self.search_radius_km = search_radius_km
        self.dem_step_deg = dem_step_deg
        self.dem_api_url = dem_api_url
        self._dem_cache: dict[str, tuple[float, np.ndarray]] = {}
        self._cache_ttl = cache_ttl
        self._horizon_cache: dict[str, tuple[float, np.ndarray]] = {}

    def extract_profile(
        self, image_path: Path, sky_mask: np.ndarray,
    ) -> HorizonProfile:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if sky_mask.shape[:2] != (h, w):
            sky_mask = np.array(
                Image.fromarray(sky_mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST),
                dtype=bool,
            )

        horizon_row = np.full(w, -1, dtype=np.float32)
        for col in range(w):
            sky_rows = np.where(sky_mask[:, col])[0]
            if len(sky_rows) > 0:
                horizon_row[col] = float(sky_rows[-1])

        valid = horizon_row >= 0
        valid_cols = np.where(valid)[0]
        if len(valid_cols) < 10:
            return HorizonProfile(
                elevation_angles=np.array([]),
                azimuth_deg=np.array([]),
                confidence=0.0,
            )

        vfov_deg = 50.0
        cy = h / 2.0
        rel = (horizon_row[valid] - cy) / cy
        elevation_angles = np.clip(-rel * (vfov_deg / 2.0), -45.0, 45.0)

        hfov_deg = 60.0
        azimuth_deg = ((valid_cols.astype(np.float32) / w) - 0.5) * hfov_deg

        smooth_win = min(21, len(elevation_angles) // 2 * 2 + 1)
        if smooth_win >= 3:
            kernel = np.hanning(smooth_win)
            kernel /= kernel.sum()
            padded = np.pad(elevation_angles, smooth_win // 2, mode="edge")
            elevation_angles = np.convolve(padded, kernel, mode="valid")

        confidence = min(1.0, len(valid_cols) / w)

        return HorizonProfile(
            elevation_angles=elevation_angles,
            azimuth_deg=azimuth_deg,
            confidence=confidence,
        )

    def match(
        self,
        photo_profile: HorizonProfile,
        lat_range: tuple[float, float] | None = None,
        lon_hint: float | None = None,
        target_lats: list[float] | None = None,
        target_lons: list[float] | None = None,
    ) -> HorizonMatchResult:
        if photo_profile.confidence < 0.2 or len(photo_profile.elevation_angles) < 10:
            return HorizonMatchResult(matched=False, confidence=0.0)

        if target_lats and target_lons:
            candidates = list(zip(target_lats, target_lons))[:5]
        elif lat_range:
            lo, hi = lat_range
            mid = (lo + hi) / 2.0
            candidates = [(float(mid), lon_hint or 0.0)]
            if hi - lo > 5:
                candidates.append(((lo + mid) / 2.0, lon_hint or 0.0))
                candidates.append(((mid + hi) / 2.0, lon_hint or 0.0))
            candidates = candidates[:3]
        else:
            candidates = [(0.0, 0.0)]

        best = HorizonMatchResult(matched=False, confidence=0.0)
        best_score = -1.0

        for lat, lon in candidates:
            dem = self._fetch_dem(lat, lon)
            if dem is None:
                continue

            dem_horizon = self._compute_dem_horizon(dem, lat, lon)
            if dem_horizon is None:
                continue

            dem_az, dem_el = dem_horizon
            result = self._cross_correlate(
                photo_profile.elevation_angles,
                photo_profile.azimuth_deg,
                dem_el, dem_az,
            )

            if result and result[0] > best_score:
                best_score = result[0]
                corr, orientation, matched_n, matched_s, matched_e, matched_w = result
                best = HorizonMatchResult(
                    matched=True,
                    estimated_lat=lat,
                    estimated_lon=lon,
                    camera_azimuth_deg=orientation,
                    correlation_score=corr,
                    confidence=corr,
                    north=matched_n,
                    south=matched_s,
                    east=matched_e,
                    west=matched_w,
                )

        return best

    def _fetch_dem(self, center_lat: float, center_lon: float) -> np.ndarray | None:
        cache_key = f"{center_lat:.2f},{center_lon:.2f}"
        now = time.time()
        if cache_key in self._dem_cache:
            ts, grid = self._dem_cache[cache_key]
            if now - ts < self._cache_ttl:
                return grid

        half_deg = self.search_radius_km / 111.0
        step = self.dem_step_deg
        lats = np.arange(center_lat - half_deg, center_lat + half_deg + step / 2, step)
        lons = np.arange(center_lon - half_deg, center_lon + half_deg + step / 2, step)

        locations = [
            f"{lat:.4f},{lon:.4f}"
            for lat in lats
            for lon in lons
        ]

        elevation_map: dict[str, float] = {}
        for i in range(0, len(locations), 100):
            batch = locations[i:i + 100]
            try:
                url = f"{self.dem_api_url}?locations={'|'.join(batch)}"
                resp = requests.get(url, timeout=10)
                data = resp.json()
                if data.get("status") != "OK":
                    continue
                for r in data.get("results", []):
                    if r.get("elevation") is not None:
                        key = f"{r['location']['lat']:.4f},{r['location']['lng']:.4f}"
                        elevation_map[key] = r["elevation"]
            except Exception:
                continue

        if len(elevation_map) < 10:
            return None

        grid = np.full((len(lats), len(lons)), np.nan)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                key = f"{lat:.4f},{lon:.4f}"
                if key in elevation_map:
                    grid[i, j] = elevation_map[key]

        if np.isnan(grid).all():
            return None

        grid = self._fill_nan(grid)

        self._dem_cache[cache_key] = (now, grid)
        return grid

    @staticmethod
    def _fill_nan(grid: np.ndarray) -> np.ndarray:
        mask = np.isnan(grid)
        if mask.all():
            return grid
        y, x = np.where(~mask)
        if len(y) == 0:
            return grid
        from scipy.interpolate import griddata
        points = np.column_stack((y, x))
        values = grid[~mask]
        xi, xj = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
        filled = griddata(points, values, (xi, xj), method="nearest")
        return filled.reshape(grid.shape)

    def _compute_dem_horizon(
        self, dem: np.ndarray, center_lat: float, center_lon: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        half_deg = self.search_radius_km / 111.0
        step = self.dem_step_deg
        lats = np.arange(center_lat - half_deg, center_lat + half_deg + step / 2, step)
        lons = np.arange(center_lon - half_deg, center_lon + half_deg + step / 2, step)

        if dem.shape != (len(lats), len(lons)):
            return None

        center_idx = np.array([len(lats) // 2, len(lons) // 2])
        obs_elev = float(dem[center_idx[0], center_idx[1]])
        if np.isnan(obs_elev) or obs_elev < -500:
            return None

        n_az = 180
        azimuths = np.arange(n_az, dtype=np.float32) * 2
        horizon_el = np.full(n_az, -np.inf, dtype=np.float32)

        max_dist_m = self.search_radius_km * 1000.0
        earth_r = 6371000.0

        for az_idx in range(n_az):
            az_rad = math.radians(float(az_idx))
            dx = math.sin(az_rad)
            dy = math.cos(az_rad)
            step_m = 100.0
            n_steps = int(max_dist_m / step_m)

            best_angle = -np.inf
            for s in range(1, n_steps + 1, 10):
                dist_m = s * step_m
                dlat = math.degrees(dist_m * dy / earth_r)
                dlon = math.degrees(dist_m * dx / (earth_r * math.cos(math.radians(center_lat + dlat / 2))))

                target_lat = center_lat + dlat
                target_lon = center_lon + dlon

                if not (-90 <= target_lat <= 90):
                    continue
                if not (-180 <= target_lon <= 180):
                    continue

                ti = (target_lat - (center_lat - half_deg)) / step
                tj = (target_lon - (center_lon - half_deg)) / step
                ti = np.clip(ti, 0, len(lats) - 1.001)
                tj = np.clip(tj, 0, len(lons) - 1.001)

                i0, i1 = int(ti), min(int(ti) + 1, len(lats) - 1)
                j0, j1 = int(tj), min(int(tj) + 1, len(lons) - 1)
                fi, fj = ti - i0, tj - j0

                elev = (
                    dem[i0, j0] * (1 - fi) * (1 - fj)
                    + dem[i1, j0] * fi * (1 - fj)
                    + dem[i0, j1] * (1 - fi) * fj
                    + dem[i1, j1] * fi * fj
                )

                if np.isnan(elev):
                    continue

                dz = elev - obs_elev
                ang = math.degrees(math.atan2(dz, dist_m))
                if ang > best_angle:
                    best_angle = ang

            horizon_el[az_idx] = best_angle if best_angle > -np.inf else -90.0

        kernel = np.hanning(7)
        kernel /= kernel.sum()
        padded = np.pad(horizon_el, 3, mode="edge")
        horizon_el = np.convolve(padded, kernel, mode="valid")

        return azimuths, horizon_el

    def _cross_correlate(
        self,
        photo_el: np.ndarray,
        photo_az: np.ndarray,
        dem_el: np.ndarray,
        dem_az: np.ndarray,
    ) -> tuple[float, float, float, float, float, float] | None:
        n_photo = len(photo_el)
        if n_photo < 5:
            return None

        fov_deg = float(photo_az[-1] - photo_az[0])
        if fov_deg <= 0:
            return None

        photo = photo_el - np.mean(photo_el)
        photo_norm = np.linalg.norm(photo)
        if photo_norm < 1e-6:
            return None
        photo = photo / photo_norm

        n_dem = len(dem_el)
        n_window = n_photo

        best_corr = -1.0
        best_offset = 0.0

        for offset in range(n_dem - n_window + 1):
            window = dem_el[offset:offset + n_window]
            window = window - np.mean(window)
            wnorm = np.linalg.norm(window)
            if wnorm < 1e-6:
                continue
            window = window / wnorm
            corr = float(np.dot(photo, window))
            if corr > best_corr:
                best_corr = corr
                best_offset = float(offset)

        if best_corr < 0.1:
            return None

        orientation = dem_az[int(best_offset)]
        half_fov = fov_deg / 2.0
        north = dem_az[int(max(0, best_offset))]
        south = dem_az[int(min(n_dem - 1, best_offset + n_window / 2))]
        east = dem_az[int(min(n_dem - 1, best_offset + n_window))]
        west = orientation

        return best_corr, orientation, north, south, east, west

    def clear_cache(self) -> None:
        self._dem_cache.clear()
        self._horizon_cache.clear()

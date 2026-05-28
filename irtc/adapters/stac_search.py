"""
STAC satellite search. Implements SatelliteSearchPort.

Geographic diversity strategy:
  - Splits the globe into 9 longitude bands to guarantee continental coverage
  - Each season is split into 2 sub-windows per year to sample different orbital passes
  - Post-filters by tile centroid (STAC returns intersecting tiles, not just contained ones)
"""

from __future__ import annotations
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pystac_client
import pystac

from irtc.domain.analysis import SearchConstraints
from irtc.domain.satellite import BBox, SatelliteCandidate, SatelliteSearchResult


_COLLECTIONS = [
    "sentinel-2-l2a",
    "landsat-c2-l2",
]
_STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Longitude bands — narrower around densely populated regions
_LON_BANDS: list[tuple[float, float]] = [
    (-180.0, -120.0),
    (-120.0,  -70.0),
    ( -70.0,  -25.0),
    ( -25.0,   -5.0),
    (  -5.0,   20.0),   # Western Europe (Spain, France, Italy, UK)
    (  20.0,   55.0),
    (  55.0,  110.0),
    ( 110.0,  155.0),
    ( 155.0,  180.0),
]

_SEASON_MONTHS_NORTH: dict[str, list[int]] = {
    "summer":  [5, 6, 7, 8, 9],
    "winter":  [11, 12, 1, 2, 3],
    "equinox": [3, 4, 5, 9, 10, 11],
}


def _season_months_for(season: str | None, hemisphere: str | None) -> list[int] | None:
    if not season or season not in _SEASON_MONTHS_NORTH:
        return None
    months = list(_SEASON_MONTHS_NORTH[season])
    if hemisphere == "south":
        months = [((m + 5) % 12) + 1 for m in months]
    return months


def _season_window_for_year(year: int, months: list[int]) -> tuple[str, str]:
    sorted_m = sorted(set(months))
    m_first, m_last = sorted_m[0], sorted_m[-1]
    last_day = calendar.monthrange(year, m_last)[1]
    return (
        f"{year}-{m_first:02d}-01T00:00:00Z",
        f"{year}-{m_last:02d}-{last_day:02d}T23:59:59Z",
    )


class StacSatelliteSearch:

    def __init__(
        self,
        endpoint: str = _STAC_ENDPOINT,
        collections: list[str] = _COLLECTIONS,
        years_back: int = 10,   # Sentinel-2 data starts 2015; cover 10 years
        max_items: int = 100,
        workers: int = 8,
    ) -> None:
        self._endpoint    = endpoint
        self._collections = collections
        self._years_back  = years_back
        self._max_items   = max_items
        self._workers     = workers
        self._client: pystac_client.Client | None = None

    def search(self, constraints: SearchConstraints) -> SatelliteSearchResult:
        client = self._get_client()

        cloud_min     = constraints.cloud_coverage_min
        cloud_max     = constraints.cloud_coverage_max
        season_months = _season_months_for(constraints.season_hint, constraints.hemisphere)
        lat_min, lat_max = self._lat_bounds(constraints)

        if constraints.hemisphere == "north":
            lat_min = max(lat_min, 0.0)
        elif constraints.hemisphere == "south":
            lat_max = min(lat_max, 0.0)

        print(f"  STAC: lat=[{lat_min:.1f}, {lat_max:.1f}] | "
              f"cloud={cloud_min:.0f}–{cloud_max:.0f}% | "
              f"season={constraints.season_hint or 'unknown'} "
              f"(months={season_months or 'all'})")

        now = datetime.now(tz=timezone.utc)
        items_per_task = max(8, self._max_items // len(_LON_BANDS))
        tasks: list[tuple[BBox, str, str]] = []

        for offset in range(self._years_back):
            year = now.year - offset
            if year < 2016:
                continue
            all_months = sorted(set(season_months)) if season_months else list(range(1, 13))
            mid = len(all_months) // 2
            month_windows = [all_months[:mid] or all_months, all_months[mid:] or all_months]

            for m_group in month_windows:
                win_start, win_end = _season_window_for_year(year, m_group)
                for lon_west, lon_east in _LON_BANDS:
                    tasks.append((
                        BBox(west=lon_west, south=lat_min, east=lon_east, north=lat_max),
                        win_start,
                        win_end,
                    ))

        raw_items: list[pystac.Item] = []

        def _fetch(task: tuple[BBox, str, str]) -> list[pystac.Item]:
            bbox, t0, t1 = task
            return self._query(client, bbox, t0, t1, cloud_min, cloud_max,
                               max_items=items_per_task)

        with ThreadPoolExecutor(max_workers=self._workers) as ex:
            for fut in as_completed({ex.submit(_fetch, t): t for t in tasks}):
                try:
                    raw_items.extend(fut.result())
                except Exception:
                    pass

        print(f"  {len(raw_items)} raw items from {len(tasks)} tasks "
              f"({len(_LON_BANDS)} bands × {self._years_back} years)")

        seen: set[str] = set()
        items: list[pystac.Item] = []
        for it in raw_items:
            if it.id not in seen:
                seen.add(it.id)
                items.append(it)

        print(f"  {len(items)} unique items")

        if constraints.lat_range:
            lat_lo = max(constraints.lat_range[0],
                         0.0 if constraints.hemisphere == "north" else -90.0)
            lat_hi = min(constraints.lat_range[1],
                         0.0 if constraints.hemisphere == "south" else 90.0)
            before = len(items)
            items = [
                it for it in items
                if it.bbox is not None and
                lat_lo <= (it.bbox[1] + it.bbox[3]) / 2 <= lat_hi
            ]
            if len(items) < before:
                print(f"  {len(items)} after centroid filter [{lat_lo:.1f}°, {lat_hi:.1f}°]")

        # Round-robin across years so recent tasks completing first via as_completed
        # don't monopolise the candidate list — guarantees temporal diversity.
        by_year: dict[int, list[pystac.Item]] = {}
        for it in items:
            yr = self._item_datetime(it).year
            by_year.setdefault(yr, []).append(it)

        diverse: list[pystac.Item] = []
        years = sorted(by_year)
        while len(diverse) < self._max_items:
            added = False
            for yr in years:
                if by_year[yr] and len(diverse) < self._max_items:
                    diverse.append(by_year[yr].pop(0))
                    added = True
            if not added:
                break

        candidates = [self._to_candidate(it) for it in diverse]
        print(f"  Years in candidates: {sorted(set(c.captured_at.year for c in candidates))}")

        target_cov = (cloud_min + cloud_max) / 2
        candidates.sort(key=lambda c: abs(c.cloud_cover_pct - target_cov))

        return SatelliteSearchResult(
            candidates=candidates,
            total_searched=len(candidates),
            collections_used=self._collections,
            query_bbox=BBox(west=-180.0, south=lat_min, east=180.0, north=lat_max),
            query_date_range=(
                f"{now.year - self._years_back}-01-01T00:00:00Z",
                now.strftime("%Y-%m-%dT23:59:59Z"),
            ),
        )

    def _get_client(self) -> pystac_client.Client:
        if self._client is None:
            self._client = pystac_client.Client.open(self._endpoint)
        return self._client

    def _lat_bounds(self, constraints: SearchConstraints) -> tuple[float, float]:
        if constraints.lat_range:
            return float(constraints.lat_range[0]), float(constraints.lat_range[1])
        return -90.0, 90.0

    def _query(
        self,
        client: pystac_client.Client,
        bbox: BBox,
        date_start: str,
        date_end: str,
        cloud_min: float,
        cloud_max: float,
        max_items: int = 5,
    ) -> list[pystac.Item]:
        try:
            search = client.search(
                collections=self._collections,
                bbox=bbox.as_list(),
                datetime=f"{date_start}/{date_end}",
                query={"eo:cloud_cover": {"gte": cloud_min, "lte": cloud_max}},
                max_items=max_items,
            )
            return list(search.items())
        except Exception:
            return []

    @staticmethod
    def _item_datetime(item: pystac.Item) -> datetime:
        """
        Return the best available datetime for a STAC item.
        STAC allows item.datetime to be None when the item covers a date range
        (start_datetime / end_datetime in properties).  Fall back through
        several sources before giving up and using now().
        """
        if item.datetime is not None:
            return item.datetime
        # Try common_metadata (pystac parses start/end into here)
        cm = item.common_metadata
        if cm.start_datetime is not None:
            return cm.start_datetime
        if cm.end_datetime is not None:
            return cm.end_datetime
        # Try raw properties strings
        for key in ("datetime", "start_datetime", "end_datetime"):
            raw = item.properties.get(key)
            if raw:
                try:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    pass
        return datetime.now(tz=timezone.utc)

    def _to_candidate(self, item: pystac.Item) -> SatelliteCandidate:
        bbox = BBox(
            west=item.bbox[0], south=item.bbox[1],
            east=item.bbox[2], north=item.bbox[3],
        )
        thumb = None
        for key in ("rendered_preview", "thumbnail", "overview"):
            if key in item.assets:
                thumb = item.assets[key].href
                break

        assets = {
            k: v.href
            for k, v in item.assets.items()
            if k in ("B02", "B03", "B04", "visual", "red", "green", "blue",
                     "SR_B2", "SR_B3", "SR_B4")
        }

        return SatelliteCandidate(
            id=item.id,
            collection=item.collection_id or "",
            bbox=bbox,
            captured_at=self._item_datetime(item),
            cloud_cover_pct=float(item.properties.get("eo:cloud_cover", 0)),
            thumbnail_url=thumb,
            assets=assets,
            stac_url=item.get_self_href() or "",
        )

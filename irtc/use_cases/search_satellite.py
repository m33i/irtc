"""SearchSatelliteUseCase — finds satellite tiles matching the cloud analysis."""

from __future__ import annotations

from irtc.domain.analysis import CloudAnalysis
from irtc.domain.satellite import SatelliteSearchResult
from irtc.ports.satellite_search import SatelliteSearchPort


class SearchSatelliteUseCase:

    def __init__(self, search: SatelliteSearchPort) -> None:
        self._search = search

    def execute(self, analysis: CloudAnalysis) -> SatelliteSearchResult:
        sc = analysis.search_constraints
        print(f"\nSearching satellite archive...")
        print(f"  Constraints: lat={sc.lat_range} | "
              f"cloud {sc.cloud_coverage_min:.0f}-{sc.cloud_coverage_max:.0f}%")
        result = self._search.search(sc)
        print(f"  {len(result.candidates)} candidates")
        return result

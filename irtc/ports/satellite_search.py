from typing import Protocol, runtime_checkable
from irtc.domain.analysis import SearchConstraints
from irtc.domain.satellite import SatelliteSearchResult


@runtime_checkable
class SatelliteSearchPort(Protocol):
    def search(self, constraints: SearchConstraints) -> SatelliteSearchResult: ...

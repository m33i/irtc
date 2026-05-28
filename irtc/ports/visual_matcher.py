from typing import Protocol, runtime_checkable
from irtc.domain.cloud import CloudFeatures
from irtc.domain.satellite import SatelliteCandidate
from irtc.domain.match import MatchingResult


@runtime_checkable
class VisualMatcherPort(Protocol):
    def match(
        self,
        features: CloudFeatures,
        candidates: list[SatelliteCandidate],
        cloud_type_name: str = "",
    ) -> MatchingResult: ...

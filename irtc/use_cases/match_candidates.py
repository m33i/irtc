"""MatchCandidatesUseCase — visual matching of satellite candidates against the cloud photo."""

from __future__ import annotations

from irtc.domain.analysis import CloudAnalysis
from irtc.domain.match import MatchingResult
from irtc.domain.satellite import SatelliteSearchResult
from irtc.ports.visual_matcher import VisualMatcherPort


class MatchCandidatesUseCase:

    def __init__(self, matcher: VisualMatcherPort) -> None:
        self._matcher = matcher

    def execute(
        self,
        analysis: CloudAnalysis,
        search_result: SatelliteSearchResult,
    ) -> MatchingResult:
        candidates = [c for c in search_result.candidates if c.thumbnail_url]
        print(f"\nVisual matching: {len(candidates)} candidates with thumbnail")

        result = self._matcher.match(
            analysis.features,
            candidates,
            cloud_type_name=analysis.classification.primary.name,
        )

        if result.best:
            best = result.best
            print(f"  Best: {best.candidate.id}")
            print(f"    score={best.combined_score:.3f}  "
                  f"clip={best.similarity:.3f}  coverage={best.coverage_score:.3f}")
            print(f"    loc={best.candidate.lat:+.4f},{best.candidate.lon:+.4f}  "
                  f"time={best.candidate.captured_at.strftime('%Y-%m-%d %H:%M UTC')}")

        return result

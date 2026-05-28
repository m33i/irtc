"""Domain — Visual matching results."""

from __future__ import annotations
from dataclasses import dataclass

from irtc.domain.satellite import SatelliteCandidate


@dataclass
class MatchResult:
    candidate: SatelliteCandidate
    similarity: float       # CLIP cosine similarity (0–1)
    coverage_score: float   # cloud coverage proximity score (0–1)
    combined_score: float   # weighted final score (0–1)


@dataclass
class MatchingResult:
    matches: list[MatchResult]  # sorted by combined_score desc
    query_embedding_dim: int

    @property
    def best(self) -> MatchResult | None:
        return self.matches[0] if self.matches else None

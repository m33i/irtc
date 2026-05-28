from typing import Protocol, runtime_checkable
from pathlib import Path
from irtc.domain.cloud import SegmentationResult


@runtime_checkable
class SkySegmenterPort(Protocol):
    def segment(self, image_path: Path) -> SegmentationResult: ...

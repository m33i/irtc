from typing import Protocol, runtime_checkable
from PIL import Image
from irtc.domain.cloud import ClassificationResult


@runtime_checkable
class CloudClassifierPort(Protocol):
    def classify(self, image: Image.Image) -> ClassificationResult: ...

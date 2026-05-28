from typing import Protocol, runtime_checkable
import numpy as np
from PIL import Image
from irtc.domain.cloud import CloudFeatures


@runtime_checkable
class FeatureExtractorPort(Protocol):
    def extract(self, sky_bbox: Image.Image, sky_full_rgb: np.ndarray) -> CloudFeatures: ...

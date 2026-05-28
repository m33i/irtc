"""Domain — Cloud entities. No external library imports."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CloudType:
    name: str
    abbr: str
    level: str
    altitude_range: str
    description: str


CLOUD_TYPES: list[CloudType] = [
    CloudType("Cirrus",        "Ci", "high",        "6000-12000m", "Thin ice-crystal wisps, clear sky"),
    CloudType("Cirrostratus",  "Cs", "high",        "6000-12000m", "Thin white veil, produces sun halos"),
    CloudType("Cirrocumulus",  "Cc", "high",        "6000-12000m", "Small white ripples, mackerel sky"),
    CloudType("Altostratus",   "As", "middle",      "2000-7000m",  "Uniform grey layer, diffuse sun"),
    CloudType("Altocumulus",   "Ac", "middle",      "2000-7000m",  "Grey-white rounded masses in rows"),
    CloudType("Stratus",       "St", "low",         "0-2000m",     "Featureless grey layer, like lifted fog"),
    CloudType("Stratocumulus", "Sc", "low",         "0-2000m",     "Patchy grey-white rolls, most common type"),
    CloudType("Nimbostratus",  "Ns", "low/middle",  "0-4000m",     "Dark dense layer, continuous rain"),
    CloudType("Cumulus",       "Cu", "vertical",    "500-3000m",   "Fluffy white clouds with flat base"),
    CloudType("Cumulonimbus",  "Cb", "vertical",    "500-15000m",  "Anvil-top storm tower, lightning"),
]


@dataclass
class ClassificationResult:
    primary: CloudType
    confidence: float
    top3: list[tuple[str, float]]


@dataclass
class SegmentationResult:
    """Sky pixels separated from terrain/buildings.
    sky_crop: RGB array (H, W, 3) — sky pixels at original position, rest is black.
    """
    mask: np.ndarray       # bool (H, W)
    sky_ratio: float
    method: str            # "neural" | "heuristic"
    sky_crop: np.ndarray   # RGB (H, W, 3)


@dataclass
class CloudFeatures:
    """Visual fingerprint of the cloud formation."""
    embedding: np.ndarray    # 512-dim CLIP semantic embedding
    texture_lbp: np.ndarray  # 256-dim LBP descriptor (cloud-masked)
    cloud_coverage_pct: float
    dominant_brightness: float

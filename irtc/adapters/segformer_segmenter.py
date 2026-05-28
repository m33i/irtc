"""SegFormer sky segmenter. Implements SkySegmenterPort."""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from irtc.domain.cloud import SegmentationResult


class SegformerSkySegmenter:

    MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
    SKY_CLASS  = 2   # ADE20K sky index

    def __init__(self, device: str | None = None, force_heuristic: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.force_heuristic = force_heuristic
        self._model = None
        self._processor = None

    def segment(self, image_path: Path) -> SegmentationResult:
        img_pil = Image.open(image_path).convert("RGB")
        img_rgb = np.array(img_pil)
        h, w = img_rgb.shape[:2]

        if self._try_load():
            mask   = self._neural_mask(img_pil, h, w)
            method = "neural"
        else:
            mask   = self._heuristic_mask(img_rgb)
            method = "heuristic"

        mask     = self._clean_mask(mask)
        sky_crop = img_rgb.copy()
        sky_crop[~mask] = 0

        return SegmentationResult(
            mask=mask,
            sky_ratio=round(float(mask.mean()), 4),
            method=method,
            sky_crop=sky_crop,
        )

    def _try_load(self) -> bool:
        if self._model is not None:
            return True
        if self.force_heuristic:
            return False
        try:
            import warnings
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            print("  Loading SegFormer sky segmenter...")
            with warnings.catch_warnings():
                # HuggingFace model config ships legacy 'feature_extractor_type'
                # that is no longer a valid constructor argument — safe to ignore.
                warnings.filterwarnings(
                    "ignore",
                    message=".*feature_extractor_type.*",
                    category=UserWarning,
                )
                self._processor = SegformerImageProcessor.from_pretrained(self.MODEL_NAME)
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self.MODEL_NAME
            ).to(self.device)
            self._model.eval()
            print("  SegFormer ready")
            return True
        except Exception as e:
            print(f"  SegFormer unavailable ({e}), falling back to heuristic")
            return False

    def _neural_mask(self, img_pil: Image.Image, h: int, w: int) -> np.ndarray:
        inputs = self._processor(images=img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        return (upsampled.argmax(dim=1).squeeze(0).cpu().numpy() == self.SKY_CLASS)

    def _heuristic_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        blue_sky  = cv2.inRange(hsv, np.array([ 90,  20, 100]), np.array([140, 200, 255]))
        white_sky = cv2.inRange(hsv, np.array([  0,   0, 180]), np.array([180,  40, 255]))
        gray_sky  = cv2.inRange(hsv, np.array([  0,   0, 140]), np.array([180,  30, 220]))
        raw = cv2.bitwise_or(cv2.bitwise_or(blue_sky, white_sky), gray_sky)
        return raw > 127

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones(( 5,  5)))
        m = cv2.GaussianBlur(m, (21, 21), 0)
        return m > 127

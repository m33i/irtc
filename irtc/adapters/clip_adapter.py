"""
Unified CLIP adapter.

Implements ClassifierPort, FeatureExtractorPort, and VisualMatcherPort in a
single class so the model is loaded once and shared across all three roles.

Scoring strategy (cross-view ground↑ vs satellite↓):
  CLIP image–image similarity is nearly flat for all candidates, so it gets a
  small weight.  The bulk of the score comes from features that survive the
  viewpoint flip:

    • Cloud coverage      40 % — fraction of bright/low-sat pixels
    • LBP texture         25 % — cloud-masked local binary patterns
    • Satellite text      20 % — CLIP image–text: thumbnail vs "X cloud from above"
    • Brightness          10 % — mean luminance of cloud regions
    • CLIP image–image    05 % — ground embedding vs satellite embedding
"""

from __future__ import annotations
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from irtc.domain.cloud import CLOUD_TYPES, ClassificationResult, CloudFeatures
from irtc.domain.match import MatchResult, MatchingResult
from irtc.domain.satellite import SatelliteCandidate

# Ground-level prompts for zero-shot classification
_GROUND_PROMPTS: dict[str, list[str]] = {
    "Cirrus": [
        "thin wispy white cloud streaks high in the sky made of ice crystals",
        "feathery white cirrus clouds at high altitude against blue sky",
    ],
    "Cirrostratus": [
        "thin white veil of ice cloud covering the entire sky creating a sun halo",
        "translucent whitish sheet of cloud high in the sky with sun halo",
    ],
    "Cirrocumulus": [
        "small white cloudlets arranged in rows at high altitude mackerel sky rippled",
        "tiny white rippled cloud pattern high altitude like fish scales",
    ],
    "Altostratus": [
        "uniform gray sheet of cloud covering the sky at medium altitude sun barely visible",
        "flat gray cloud layer blocking the sun mid altitude striated",
    ],
    "Altocumulus": [
        "gray and white rounded cloud patches in groups or waves at medium altitude",
        "rows of rounded cloud masses mid level sky gray and white",
    ],
    "Stratus": [
        "low featureless gray cloud layer like lifted fog covering the whole sky",
        "uniform gray overcast low cloud ceiling like fog",
    ],
    "Stratocumulus": [
        "low lumpy gray and white cloud layer patches covering most of the sky",
        "low rounded cloud rolls gray and white patchy overcast",
    ],
    "Nimbostratus": [
        "dark thick rain cloud layer covering entire sky producing heavy continuous rain",
        "dense dark gray cloud sheet low and thick with rain falling",
    ],
    "Cumulus": [
        "bright white fluffy cumulus clouds with flat dark base and rounded tops in blue sky",
        "puffy white fair weather clouds isolated in blue sky",
    ],
    "Cumulonimbus": [
        "massive towering storm cloud with anvil shaped top cumulonimbus thunderstorm dark base",
        "enormous dark storm cloud reaching very high altitude with anvil top and lightning",
    ],
}

# Satellite-view prompts for cross-view text matching (one per type)
_SAT_PROMPTS: dict[str, str] = {
    "Cirrus":        "thin white ice cloud streaks satellite view from above",
    "Cirrostratus":  "thin translucent white cloud veil large area satellite view from above",
    "Cirrocumulus":  "small white rippled cloud cells regular pattern satellite view from above",
    "Altostratus":   "uniform grey cloud sheet mid level satellite view from above",
    "Altocumulus":   "grey white rounded cloud cells in rows satellite view from above",
    "Stratus":       "flat grey low cloud layer featureless satellite view from above",
    "Stratocumulus": "patchy grey white cloud rolls broken layer satellite view from above",
    "Nimbostratus":  "thick dark grey rain cloud layer satellite view from above",
    "Cumulus":       "puffy white cloud tops rounded convective satellite view from above",
    "Cumulonimbus":  "large white anvil towering cumulonimbus storm satellite view from above",
}


class ClipAdapter:

    MODEL_NAME       = "openai/clip-vit-base-patch32"
    DOWNLOAD_WORKERS = 8
    TIMEOUT_S        = 10

    W_COVERAGE  = 0.40
    W_TEXTURE   = 0.25
    W_SAT_TEXT  = 0.20
    W_BRIGHT    = 0.10
    W_CLIP      = 0.05

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model     = None
        self._processor = None
        # Text features pre-computed at load time
        self._ground_text_feats: torch.Tensor | None = None
        self._ground_prompt_types: list[str] = []
        self._sat_text_feats: dict[str, np.ndarray] = {}

    # ClassifierPort 

    def classify(self, image: Image.Image) -> ClassificationResult:
        self._load()
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_f = self._model.get_image_features(**inputs)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        sims = (img_f @ self._ground_text_feats.T).squeeze(0).cpu().numpy()

        type_sims: dict[str, list[float]] = {ct.name: [] for ct in CLOUD_TYPES}
        for score, cloud_name in zip(sims, self._ground_prompt_types):
            type_sims[cloud_name].append(float(score))

        raw   = {k: float(np.mean(v)) for k, v in type_sims.items() if v}
        vals  = np.array(list(raw.values()))
        probs = np.exp((vals - vals.max()) * 10)
        probs /= probs.sum()
        prob_dict = dict(zip(raw.keys(), probs.tolist()))

        sorted_t        = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        best_name, best_prob = sorted_t[0]
        best_type       = next(ct for ct in CLOUD_TYPES if ct.name == best_name)
        return ClassificationResult(primary=best_type, confidence=best_prob, top3=sorted_t[:3])

    # FeatureExtractorPort 

    def extract(self, sky_bbox: Image.Image, sky_full_rgb: np.ndarray) -> CloudFeatures:
        self._load()
        sky_bgr = cv2.cvtColor(np.array(sky_bbox), cv2.COLOR_RGB2BGR)
        return CloudFeatures(
            embedding          = self._embed(sky_bbox),
            texture_lbp        = self._cloud_lbp_norm(sky_bbox),
            cloud_coverage_pct = self._coverage(sky_full_rgb),
            dominant_brightness= float(cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2GRAY).mean()),
        )

    # VisualMatcherPort 

    def match(
        self,
        features: CloudFeatures,
        candidates: list[SatelliteCandidate],
        cloud_type_name: str = "",
    ) -> MatchingResult:
        self._load()

        sat_text_feat = self._sat_text_feats.get(cloud_type_name) if cloud_type_name else None

        print(f"Downloading {len(candidates)} thumbnails...")
        thumbnails = self._download_thumbnails(candidates)
        print(f"  {sum(v is not None for v in thumbnails.values())} downloaded OK")

        print("Extracting satellite embeddings (batch)...")
        sat_embeddings = self._batch_embed(thumbnails, candidates)

        results = []
        for cand, sat_emb in zip(candidates, sat_embeddings):
            img = thumbnails.get(cand.id)
            if img is None or sat_emb is None:
                continue

            clip_score = (float(np.dot(features.embedding, sat_emb)) + 1.0) / 2.0
            cov_score  = max(0.0, 1.0 - abs(features.cloud_coverage_pct - cand.cloud_cover_pct) / 30.0)
            lbp_score  = (float(np.dot(features.texture_lbp, self._cloud_lbp_norm(img))) + 1.0) / 2.0
            bright     = max(0.0, 1.0 - abs(features.dominant_brightness - self._cloud_brightness(img)) / 128.0)
            type_score = (float(np.dot(sat_emb, sat_text_feat)) + 1.0) / 2.0 if sat_text_feat is not None else 0.5

            combined = (
                self.W_COVERAGE * cov_score
                + self.W_TEXTURE  * lbp_score
                + self.W_SAT_TEXT * type_score
                + self.W_BRIGHT   * bright
                + self.W_CLIP     * clip_score
            )
            results.append(MatchResult(
                candidate      = cand,
                similarity     = float(np.dot(features.embedding, sat_emb)),
                coverage_score = cov_score,
                combined_score = combined,
            ))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return MatchingResult(matches=results, query_embedding_dim=len(features.embedding))

    # Shared feature helpers

    def _embed(self, img: Image.Image) -> np.ndarray:
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    def _batch_embed(
        self,
        thumbnails: dict[str, Image.Image | None],
        candidates: list[SatelliteCandidate],
    ) -> list[np.ndarray | None]:
        BATCH = 16
        valid = [(cid, img) for cid, img in thumbnails.items() if img is not None]
        id_to_emb: dict[str, np.ndarray] = {}
        for i in range(0, len(valid), BATCH):
            ids   = [cid for cid, _ in valid[i:i+BATCH]]
            imgs  = [img for _, img  in valid[i:i+BATCH]]
            inputs = self._processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            for cid, emb in zip(ids, feats.cpu().numpy()):
                id_to_emb[cid] = emb.astype(np.float32)
        return [id_to_emb.get(c.id) for c in candidates]

    @staticmethod
    def _cloud_mask(img: Image.Image) -> np.ndarray:
        hsv = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        return (hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 80)

    def _cloud_lbp_norm(self, img: Image.Image) -> np.ndarray:
        gray = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        mask = self._cloud_mask(img)
        lbp  = np.zeros_like(gray, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (shifted >= gray).astype(np.uint8)
        pixels = lbp[mask].ravel() if mask.any() else lbp.ravel()
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
        v = hist.astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def _cloud_brightness(self, img: Image.Image) -> float:
        bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = self._cloud_mask(img)
        return float(gray[mask].mean() if mask.any() else gray.mean())

    @staticmethod
    def _coverage(img_rgb: np.ndarray) -> float:
        hsv  = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 80)
        return round(float(mask.mean() * 100), 1)

    def _download_thumbnails(
        self, candidates: list[SatelliteCandidate]
    ) -> dict[str, Image.Image | None]:
        def fetch(cand: SatelliteCandidate) -> tuple[str, Image.Image | None]:
            try:
                r = requests.get(cand.thumbnail_url, timeout=self.TIMEOUT_S)
                r.raise_for_status()
                return cand.id, Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                return cand.id, None

        results: dict[str, Image.Image | None] = {}
        with ThreadPoolExecutor(max_workers=self.DOWNLOAD_WORKERS) as ex:
            for cid, img in (f.result() for f in as_completed(
                {ex.submit(fetch, c): c for c in candidates if c.thumbnail_url}
            )):
                results[cid] = img
        return results

    # Model loading

    def _load(self) -> None:
        if self._model is not None:
            return
        import warnings
        from transformers import CLIPModel, CLIPProcessor

        print(f"  Loading CLIP on {self.device}...")
        self._model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*use_fast.*")
            self._processor = CLIPProcessor.from_pretrained(self.MODEL_NAME, use_fast=False)
        self._model.eval()

        # Pre-compute all text features in one batch
        ground_texts = [p for prompts in _GROUND_PROMPTS.values() for p in prompts]
        sat_texts    = list(_SAT_PROMPTS.values())
        all_texts    = ground_texts + sat_texts

        self._ground_prompt_types = [
            name for name, prompts in _GROUND_PROMPTS.items() for _ in prompts
        ]

        inputs = self._processor(text=all_texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        n_ground = len(ground_texts)
        self._ground_text_feats = feats[:n_ground]

        sat_names = list(_SAT_PROMPTS.keys())
        for i, name in enumerate(sat_names):
            self._sat_text_feats[name] = feats[n_ground + i].cpu().numpy().astype(np.float32)

        print("CLIP ready")

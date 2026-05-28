"""FastAPI backend for I Remember That Cloud."""

from __future__ import annotations
import asyncio
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from irtc.adapters.segformer_segmenter import SegformerSkySegmenter
from irtc.adapters.clip_adapter import ClipAdapter
from irtc.adapters.opencv_solar import OpenCVSolarEstimator
from irtc.adapters.stac_search import StacSatelliteSearch
from irtc.use_cases.analyze_cloud import AnalyzeCloudUseCase
from irtc.use_cases.search_satellite import SearchSatelliteUseCase
from irtc.use_cases.match_candidates import MatchCandidatesUseCase

app = FastAPI(title="I Remember That Cloud")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load once at startup — ClipAdapter is shared across all three roles
_clip    = ClipAdapter()
_analyze_uc = AnalyzeCloudUseCase(
    segmenter         = SegformerSkySegmenter(),
    classifier        = _clip,
    solar_estimator   = OpenCVSolarEstimator(),
    feature_extractor = _clip,
)
_search_uc = SearchSatelliteUseCase(search=StacSatelliteSearch(max_items=150))
_match_uc  = MatchCandidatesUseCase(matcher=_clip)


def _serialize_analysis(analysis) -> dict:
    return analysis.to_dict()


def _serialize_matches(match_result) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "id": m.candidate.id,
            "collection": m.candidate.collection,
            "lat": m.candidate.lat,
            "lon": m.candidate.lon,
            "captured_at": m.candidate.captured_at.isoformat(),
            "cloud_cover_pct": m.candidate.cloud_cover_pct,
            "thumbnail_url": m.candidate.thumbnail_url,
            "bbox": {
                "west":  m.candidate.bbox.west,
                "south": m.candidate.bbox.south,
                "east":  m.candidate.bbox.east,
                "north": m.candidate.bbox.north,
            },
            "scores": {
                "similarity": round(m.similarity, 4),
                "coverage": round(m.coverage_score, 4),
                "combined": round(m.combined_score, 4),
            },
        }
        for i, m in enumerate(match_result.matches[:50])
    ]


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accept a cloud photo and stream pipeline progress via Server-Sent Events."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()

    async def event_stream():
        def send(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "img.jpg").suffix, delete=False
        ) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            loop = asyncio.get_event_loop()

            yield send("progress", {"step": 1, "total": 5, "message": "Segmenting sky & analysing clouds..."})
            await asyncio.sleep(0)

            analysis = await loop.run_in_executor(None, _analyze_uc.execute, tmp_path)
            yield send("progress", {"step": 2, "total": 5, "message": "Analysis complete"})
            yield send("analysis", _serialize_analysis(analysis))
            await asyncio.sleep(0)

            yield send("progress", {"step": 3, "total": 5, "message": "Searching satellite archive..."})
            await asyncio.sleep(0)

            search_result = await loop.run_in_executor(None, _search_uc.execute, analysis)
            yield send("progress", {"step": 4, "total": 5, "message": f"{len(search_result.candidates)} candidates — visual matching..."})
            await asyncio.sleep(0)

            match_result = await loop.run_in_executor(None, _match_uc.execute, analysis, search_result)
            yield send("progress", {"step": 5, "total": 5, "message": "Done"})
            yield send("matches", {"matches": _serialize_matches(match_result)})
            yield send("done", {})

        except Exception as exc:
            yield send("error", {"message": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


_ui_dist = Path(__file__).parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="ui")

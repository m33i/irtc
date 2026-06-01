#!/usr/bin/env python3

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from irtc.adapters.dem_horizon import DemHorizonMatcher
from irtc.adapters.segformer_segmenter import SegformerSkySegmenter
from irtc.adapters.clip_adapter import ClipAdapter
from irtc.adapters.opencv_solar import OpenCVSolarEstimator
from irtc.adapters.stac_search import StacSatelliteSearch
from irtc.use_cases.analyze_cloud import AnalyzeCloudUseCase
from irtc.use_cases.search_satellite import SearchSatelliteUseCase
from irtc.use_cases.match_candidates import MatchCandidatesUseCase

console = Console()


@click.group()
def cli():
    """I Remember That Cloud — Forensic Cloud Geolocator"""


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--search", is_flag=True, help="Search satellite archive after analysis")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--device", default=None, help="Torch device (cpu / cuda)")
def analyze(image_path: Path, search: bool, output_json: bool, device: str | None):
    """Analyse a cloud photo and optionally search the satellite archive."""
    console.print(Panel.fit("[bold]I Remember That Cloud[/bold]"))

    clip = ClipAdapter(device=device)
    analyze_uc = AnalyzeCloudUseCase(
        segmenter         = SegformerSkySegmenter(device=device),
        classifier        = clip,
        solar_estimator   = OpenCVSolarEstimator(),
        feature_extractor = clip,
        horizon_matcher   = DemHorizonMatcher(),
    )
    analysis = analyze_uc.execute(image_path)

    if output_json and not search:
        print(analysis.to_json())
        return

    console.print(analysis.summary())

    if not search:
        return

    search_uc = SearchSatelliteUseCase(search=StacSatelliteSearch(max_items=150))
    search_result = search_uc.execute(analysis)

    match_uc   = MatchCandidatesUseCase(matcher=clip)
    match_result = match_uc.execute(analysis, search_result)

    if output_json:
        import json as _json
        print(_json.dumps({
            "analysis": analysis.to_dict(),
            "matches": [
                {
                    "rank": i + 1,
                    "id": m.candidate.id,
                    "collection": m.candidate.collection,
                    "lat": m.candidate.lat,
                    "lon": m.candidate.lon,
                    "captured_at": m.candidate.captured_at.isoformat(),
                    "cloud_cover_pct": m.candidate.cloud_cover_pct,
                    "scores": {
                        "similarity": round(m.similarity, 4),
                        "coverage": round(m.coverage_score, 4),
                        "combined": round(m.combined_score, 4),
                    },
                    "thumbnail_url": m.candidate.thumbnail_url,
                }
                for i, m in enumerate(match_result.matches[:50])
            ],
        }, indent=2, ensure_ascii=False))
        return

    table = Table(title=f"Top matches ({len(match_result.matches)} ranked)", border_style="white")
    table.add_column("Rank",     width=4)
    table.add_column("Date UTC", style="cyan")
    table.add_column("Lat / Lon")
    table.add_column("Cloud %",  style="yellow")
    table.add_column("Visual")
    table.add_column("Coverage")
    table.add_column("Score",    style="bold")

    for i, m in enumerate(match_result.matches[:10], 1):
        c   = m.candidate
        bar = "█" * int(m.combined_score * 10)
        table.add_row(
            f"#{i}",
            c.captured_at.strftime("%Y-%m-%d %H:%M"),
            f"{c.lat:+.3f} / {c.lon:+.3f}",
            f"{c.cloud_cover_pct:.1f}%",
            f"{m.similarity:.3f}",
            f"{m.coverage_score:.3f}",
            f"{bar} {m.combined_score:.3f}",
        )

    console.print(table)

    if match_result.best:
        best = match_result.best
        console.print(
            f"\n[bold]Best:[/bold] "
            f"[cyan]{best.candidate.captured_at.strftime('%Y-%m-%d %H:%M UTC')}[/cyan]  "
            f"lat={best.candidate.lat:+.4f}  lon={best.candidate.lon:+.4f}  "
            f"score={best.combined_score:.3f}"
        )


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--no-reload", is_flag=True, default=False, help="Disable auto-reload")
def serve(host: str, port: int, no_reload: bool):
    """Start the web server (builds UI first if dist/ is missing)."""
    import subprocess, sys
    from pathlib import Path as _Path
    if not _Path("ui/dist").exists():
        console.print("[dim]Building UI...[/dim]")
        subprocess.run(["npm", "run", "build"], cwd="ui", check=True)
    import uvicorn
    uvicorn.run("api:app", host=host, port=port, reload=not no_reload)


if __name__ == "__main__":
    cli()

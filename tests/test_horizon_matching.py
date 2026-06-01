#!/usr/bin/env python3
"""Test horizon matching: extracts profile from a cloud photo and searches across the globe for the best DEM match."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from irtc.adapters.segformer_segmenter import SegformerSkySegmenter
from irtc.adapters.dem_horizon import DemHorizonMatcher


def test_horizon(image_path: str, show_profile: bool = True):
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"File not found: {image_path}")
        return

    print(f"\n{'='*60}")
    print(f"  HORIZON MATCH TEST — {img_path.name}")
    print(f"{'='*60}\n")

    # 1. Segment sky
    print("[1/4] Segmenting sky...")
    t0 = time.time()
    seg = SegformerSkySegmenter()
    result = seg.segment(img_path)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    print(f"      {w}x{h}, {result.sky_ratio:.0%} sky ({result.method}), {time.time()-t0:.1f}s")

    # 2. Extract smooth SegFormer horizon
    print("[2/4] Extracting horizon profile...")
    t0 = time.time()

    sky_mask = result.mask
    horizon = np.full(w, -1, dtype=np.float32)
    for col in range(w):
        sky_rows = np.where(sky_mask[:, col])[0]
        if len(sky_rows) > 0:
            horizon[col] = float(sky_rows[-1])

    valid = horizon >= 0
    vc = np.where(valid)[0]

    vfov = 50.0
    cy = h / 2.0
    angles = -(horizon[valid] - cy) / cy * (vfov / 2.0)
    hfov = 60.0
    az = (vc.astype(np.float32) / w - 0.5) * hfov

    range_deg = float(angles.max() - angles.min())
    std_deg = float(angles.std())
    print(f"      {len(angles)} pts, range={range_deg:.2f}°, std={std_deg:.3f}° ({time.time()-t0:.1f}s)")

    if show_profile:
        print(f"\n  Horizon profile (every 30 cols):")
        for i in range(0, len(angles), 30):
            bar = "█" * max(1, int((angles[i] - angles.min() + 0.001) * 30))
            print(f"  {bar} {angles[i]:+.3f}°")

    # 3. Downsample for DEM matching
    n_target = min(180, len(angles))
    indices = np.linspace(0, len(angles) - 1, n_target, dtype=int)
    angles_ds = angles[indices]
    az_ds = az[indices]
    print(f"\n      downsampled to {n_target} pts for DEM matching")

    # 4. Broad search
    print(f"\n[3/4] Searching DEM horizon across globe...")
    print(f"      {'Lat':>6} {'Lon':>8} {'Corr':>7} {'Cam':>5} {'Relief':>7}")
    print(f"      {'-'*35}")

    hm = DemHorizonMatcher(search_radius_km=30.0, dem_step_deg=0.05)
    results = []
    search_lats = list(range(-60, 81, 10))
    search_lons = [-120, -100, -80, -60, 0, 60, 120]

    for lat in search_lats:
        for lon in search_lons:
            t0 = time.time()
            dem = hm._fetch_dem(float(lat), float(lon))
            if dem is None:
                continue
            hz = hm._compute_dem_horizon(dem, float(lat), float(lon))
            if hz is None:
                continue
            dem_az, dem_el = hz
            r = hm._cross_correlate(angles_ds, az_ds, dem_el, dem_az)
            dt = time.time() - t0
            if r:
                results.append((r[0], lat, lon, r[1], float(dem_el.max() - dem_el.min())))
                print(f"      {lat:>+4d}° {lon:>+4d}°  {r[0]:>.3f}  {r[1]:>3.0f}°  {dem_el.max()-dem_el.min():>5.1f}°  ({dt:.1f}s)")
            else:
                print(f"      {lat:>+4d}° {lon:>+4d}°    —       —    {dem_el.max()-dem_el.min():>5.1f}°  ({dt:.1f}s)")

    # 5. Results
    print(f"\n[4/4] Results")
    if results:
        results.sort(key=lambda x: -x[0])
        print(f"\n  Top 5 matches:")
        print(f"  {'Rank':>4} {'Corr':>6} {'Lat':>6} {'Lon':>6} {'Cam':>6}")
        print(f"  {'-'*30}")
        for rank, (corr, lat, lon, cam_az, relief) in enumerate(results[:5], 1):
            print(f"  {rank:>4} {corr:>.3f} {lat:>+4d}° {lon:>+4d}° {cam_az:>3.0f}°  (relief={relief:.1f}°)")
            if rank == 1:
                print(f"         └─ {'within range!' if abs(lat - 47) <= 10 else 'far from Montana'}")
    else:
        print("  No matches found anywhere on the globe.")
        print("  The photo likely has no distinctive horizon profile.")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test horizon matching on a cloud photo")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--no-profile", action="store_true", help="Skip ASCII profile plot")
    args = parser.parse_args()

    test_horizon(args.image, show_profile=not args.no_profile)

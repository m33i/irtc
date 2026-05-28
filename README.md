# IRTC - I Remember That Cloud

Trying to build a forensic cloud geolocation tool. Given only a photo of clouds ~~no GPS, no EXIF, no manual input~~ (though i'll probably end up adding some of that to make tracking work better... tbh) it determines where and when that cloud formation existed by matching against satellite imagery.

## How it (more or less) works

1. **Sky segmentation** — SegFormer B2 isolates sky pixels from terrain
2. **Cloud classification** — CLIP zero-shot identifies the cloud type (Cirrus, Cumulus, etc.)
3. **Solar estimation** — OpenCV estimates time of day, hemisphere and latitude from light and sun position
4. **Feature extraction** — CLIP embedding + LBP texture + cloud coverage fingerprint
5. **Satellite search** — queries Microsoft Planetary Computer STAC (Sentinel-2 + Landsat, 10 years back, global)
6. **Visual matching** — ranks candidates by cloud coverage, texture, brightness and CLIP similarity

ps: everything here is temporary and will probably change the architecture if I dont drop this project

## setup

```bash
make install
```

requires Python 3.10+ and Node 18+.

## how to run

```bash
make run          # web UI at localhost:8000
```

```bash
python3 cli.py analyze photo.jpg --search   # CLI
```

## stack

- **ML**: PyTorch, HuggingFace Transformers (SegFormer, CLIP)
- **Satellite data**: Microsoft Planetary Computer via pystac-client
- **Backend**: FastAPI + Server-Sent Events
- **Frontend**: React + CesiumJS + Tailwind CSS



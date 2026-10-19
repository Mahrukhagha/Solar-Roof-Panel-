#!/usr/bin/env python3
"""CLI:  python main.py --rgb rgb.tif --dsm dsm_full.tif --clipped dsm_clipped.tif --out outputs"""
import argparse
from dataclasses import fields
from roofpv.config import Config
from roofpv.pipeline import run


def main():
    ap = argparse.ArgumentParser(description="3D roof reconstruction and solar panel placement from DSM + RGB")
    ap.add_argument("--rgb", required=True); ap.add_argument("--dsm", required=True)
    ap.add_argument("--clipped", default=None, help="parcel-clipped DSM (nodata outside the parcel); optional")
    ap.add_argument("--out", default="outputs"); ap.add_argument("--no-orbit", action="store_true", help="skip the orbit GIF")
    for f in fields(Config):                                  # every config value can be overridden: --panel-long 1.65 ...
        t = f.type if f.type in (int, float) else None
        if f.type is bool:
            ap.add_argument("--" + f.name.replace("_", "-"), type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif t:
            ap.add_argument("--" + f.name.replace("_", "-"), type=t, default=None)
    a = ap.parse_args()
    cfg = Config()
    for f in fields(Config):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    run(a.rgb, a.dsm, a.clipped, a.out, cfg, orbit=not a.no_orbit)


if __name__ == "__main__":
    main()

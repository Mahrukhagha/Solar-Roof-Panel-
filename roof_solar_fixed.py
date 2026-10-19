"""DSM/RGB roof reconstruction + solar-panel placement  (fixed driver).

Keeps the same inputs/outputs as the original baseline script (rasters, CSV, shapefile, overlay PNG, metrics.json)
but uses the `roofpv` package (next to this file) for the actual work:
  building mask (ground = low percentile, height band, vegetation excluded, parcel from dsm_clipped)
  -> roof faces by sequential local RANSAC -> obstructions (DSM + RGB) -> panels packed in each face's own plane.

Run:   python roof_solar_fixed.py                      (uses the paths below)
       python roof_solar_fixed.py DSM.tif RGB.tif OUTDIR [DSM_CLIPPED.tif]
"""
import os, sys, csv, json, datetime
from pathlib import Path

# ---------------------------------------------------------------- PROJ setup (Windows / conda friendly)
python_dir = Path(sys.executable).parent
for p in [python_dir / "Library" / "share" / "proj", python_dir / "share" / "proj",
          python_dir / "Lib" / "site-packages" / "rasterio" / "proj_data",
          python_dir / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj"]:
    if (p / "proj.db").exists():
        os.environ["PROJ_DATA"] = str(p); os.environ["PROJ_LIB"] = str(p); break

sys.path.insert(0, str(Path(__file__).resolve().parent))      # find the roofpv/ folder next to this file
import numpy as np
import rasterio
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon

from roofpv.config import Config
from roofpv.io_utils import load_scene
from roofpv.preprocess import fill_nodata, denoise, building_mask
from roofpv.terrain import slope_aspect
from roofpv.segmentation import ransac_faces, merge_coplanar, snap_azimuths, assign_pixels
from roofpv.obstructions import detect_obstructions
from roofpv.reconstruction import build_roof_model, face_polygon
from roofpv.panels import place_panels
from roofpv import validation as val, viz

# ----------------------- EDIT THESE PATHS -----------------------
INPUT_DSM = r"C:\Users\Administrator\Downloads\roof_solar_project\roof_solar_project\src\dsm_full.tif"
INPUT_RGB = r"C:\Users\Administrator\Downloads\roof_solar_project\roof_solar_project\src\rgb.tif"
INPUT_CLIPPED = r"C:\Users\Administrator\Downloads\roof_solar_project\roof_solar_project\src\dsm_clipped.tif"  # optional: set to None
OUTPUT_DIR = r"C:\Users\Administrator\Downloads\roof_solar_project\outputs"
# ---------------------------------------------------------------

# ----------------------- PANEL PARAMETERS -----------------------
PANEL_WIDTH_M = 1.00        # short side
PANEL_LENGTH_M = 1.70       # long side
PANEL_GAP_M = 0.05
PANEL_WATTS = 400
EDGE_SETBACK_M = 0.30       # from ridge / hip / eave
OBSTACLE_CLEARANCE_M = 0.30 # around chimneys / windows / skylights
SEED_ROW_COL = None         # optional (row, col) pixel on the target roof, e.g. (400, 380); None = automatic
# ---------------------------------------------------------------


def make_config():
    cfg = Config()
    cfg.panel_long, cfg.panel_short, cfg.panel_gap = PANEL_LENGTH_M, PANEL_WIDTH_M, PANEL_GAP_M
    cfg.edge_setback = EDGE_SETBACK_M
    cfg.glass_buffer = OBSTACLE_CLEARANCE_M
    cfg.obstruction_buffer = max(OBSTACLE_CLEARANCE_M, 0.40)
    return cfg


def write_raster(path, arr, profile, dtype, nodata):
    p = profile.copy(); p.update(count=1, compress="deflate", dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.asarray(arr).astype(dtype), 1)


def main(dsm=INPUT_DSM, rgb_path=INPUT_RGB, outdir=OUTPUT_DIR, clipped=INPUT_CLIPPED):
    out = Path(outdir); (out / "rasters").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    cfg = make_config()

    print("1/8 Loading DSM and RGB...")
    sc = load_scene(rgb_path, dsm, clipped if clipped and os.path.exists(clipped) else None)
    with rasterio.open(dsm) as ds:
        profile = ds.profile.copy()

    print("2/8 Cleaning DSM (nodata fill + edge-preserving denoise)...")
    valid = np.isfinite(sc.dsm)
    z = denoise(fill_nodata(sc.dsm, valid), cfg.median_ksize, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)

    print("3/8 Slope / aspect (Horn) and building mask...")
    slope, aspect = slope_aspect(z, sc.px)
    mask, ndsm, ground = building_mask(z, sc.rgb, sc.parcel, sc.px, cfg, seed_rc=SEED_ROW_COL)
    print(f"    ground = {ground:.2f} m (2nd percentile), roof footprint = {mask.sum() * sc.px ** 2:.0f} m2")
    if mask.sum() * sc.px ** 2 < 25:
        raise RuntimeError("Roof mask too small - set SEED_ROW_COL to a pixel on the roof or relax thresholds in Config")

    print("4/8 Segmenting roof planes (sequential local RANSAC)...")
    rng = np.random.default_rng(cfg.random_seed)
    faces = merge_coplanar(ransac_faces(z, mask, sc.px, cfg, rng), z, sc.px, cfg)
    faces, theta = snap_azimuths(faces, z, sc.px, cfg)
    lab = assign_pixels(faces, z, mask, sc.px, cfg)
    if not faces:
        raise RuntimeError("No roof planes found")
    print(f"    {len(faces)} faces, building axis {theta:.1f} deg")

    print("5/8 Detecting obstacles (chimneys, dormers, skylights, roof windows, lantern)...")
    obs, _ = detect_obstructions(z, sc.rgb, mask, lab, faces, sc.px, cfg)
    model = build_roof_model(faces, sc.px, sc.height, cfg)
    support = {f.id: face_polygon(f.support, sc.px, sc.height, 0.10) for f in faces if f.support.sum() > 20}

    print("6/8 Placing panels in each face's own plane...")
    placements = place_panels(faces, model, obs, support, sc.px, sc.height, cfg, axis_deg=theta)
    n_pan = sum(len(p.panels_3d) for p in placements)
    total_w = n_pan * PANEL_WATTS

    print("7/8 Writing rasters / CSV / shapefiles...")
    face_id = np.zeros(z.shape, np.int16)
    for f in faces: face_id[f.mask] = f.id + 1
    obst = np.zeros(z.shape, np.uint8)
    for o in obs: obst[o.mask] = 255
    rec = val.reconstructed_dsm(faces, z.shape, sc.px)
    write_raster(out / "rasters" / "cleaned_dsm.tif", z, profile, "float32", -9999)
    write_raster(out / "rasters" / "slope_degrees.tif", slope, profile, "float32", -9999)
    write_raster(out / "rasters" / "aspect_degrees.tif", np.nan_to_num(aspect, nan=-9999), profile, "float32", -9999)
    write_raster(out / "rasters" / "roof_face_id.tif", face_id, profile, "int16", 0)
    write_raster(out / "rasters" / "fitted_roof.tif", np.nan_to_num(rec, nan=-9999), profile, "float32", -9999)
    write_raster(out / "rasters" / "roof_mask.tif", mask.astype(np.uint8) * 255, profile, "uint8", 0)
    write_raster(out / "rasters" / "obstacles.tif", obst, profile, "uint8", 0)

    e0, n0 = sc.origin_e, sc.origin_n - sc.height * sc.px          # local (x,y) -> world (E,N)
    rows, geoms = [], []
    fm = {f.id: f for f in faces}
    for pl in placements:
        f = fm[pl.face_id]
        for q in pl.panels_3d:
            geoms.append(Polygon([(e0 + x, n0 + y) for x, y, _ in q]))
            rows.append(dict(face_id=int(f.id), width_m=PANEL_WIDTH_M, length_m=PANEL_LENGTH_M, slope_deg=round(f.slope, 1),
                             azimuth=round(f.aspect, 1), layout=pl.orientation, capacity_w=PANEL_WATTS,
                             z_m=round(float(q[:, 2].mean()), 2)))
    with open(out / "panel_layout.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["panel_id"] + list(rows[0].keys()) if rows else ["panel_id"]); w.writeheader()
        for i, r in enumerate(rows, 1): w.writerow({"panel_id": i, **r})
    try:
        import geopandas as gpd
        if rows:
            gpd.GeoDataFrame([{"panel_id": i, **r} for i, r in enumerate(rows, 1)], geometry=geoms, crs=sc.crs).to_file(out / "panels.shp")
        gpd.GeoDataFrame([{"face_id": f.id, "kind": f.kind, "slope": round(f.slope, 1), "azimuth": round(f.aspect, 1)} for f in faces if f.id in model.polys],
                         geometry=[Polygon([(e0 + x, n0 + y) for x, y in model.polys[f.id].exterior.coords]) for f in faces if f.id in model.polys],
                         crs=sc.crs).to_file(out / "roof_faces.shp")
        gpd.GeoDataFrame([{"kind": o.kind, "area_m2": round(o.area, 2)} for o in obs],
                         geometry=[Polygon([(e0 + x, n0 + y) for x, y in o.polygon.exterior.coords]) for o in obs],
                         crs=sc.crs).to_file(out / "obstacles.shp")
    except Exception as ex:
        print("    (shapefile export skipped:", ex, ")")

    print("8/8 Figures + 3D model...")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(14, 14)); ax.imshow(sc.rgb)
    for o in obs:
        c = "#e63946" if o.kind == "protrusion" else "#00d1ff"
        xy = np.array(o.polygon.exterior.coords)
        ax.add_patch(patches.Polygon(np.c_[xy[:, 0] / sc.px, sc.height - xy[:, 1] / sc.px], closed=True, fc=c, ec=c, alpha=.4))
    for pl in placements:
        for q in pl.panels_3d:
            ax.add_patch(patches.Polygon(np.c_[q[:, 0] / sc.px, sc.height - q[:, 1] / sc.px], closed=True, fc="#1f3b8a", ec="w", lw=.6, alpha=.9))
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        c = c.squeeze()
        if c.ndim == 2: ax.plot(c[:, 0], c[:, 1], color="#39FF14", lw=2.5)
    rr, cc = np.nonzero(mask)
    ax.set_xlim(max(cc.min() - 40, 0), min(cc.max() + 40, sc.width)); ax.set_ylim(min(rr.max() + 40, sc.height), max(rr.min() - 40, 0)); ax.axis("off")
    save_path = out / "figures" / f"final_overlay_{ts}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0); plt.close(fig)
    viz.fig_faces  # (kept importable)
    items = viz.build_scene_meshes(faces, model, obs, placements, ground, cfg)
    viz.export_meshes(items, str(out / "roof_solar_model"))
    viz.render_3d(items, str(out / "figures" / "roof_3d.png"), title="Reconstructed roof, obstructions and panels")

    surf = val.surface_agreement(faces, z, sc.px, mask)
    chk = val.panel_checks(placements, model, obs, faces, cfg)
    metrics = dict(roof_faces=len(faces), obstacles=len(obs), solar_panel_count=n_pan, panel_wattage_w=PANEL_WATTS,
                   estimated_capacity_kw=round(total_w / 1000, 2), height_rmse_m=round(surf["height_rmse_m"], 3), panel_checks=chk,
                   faces=[dict(id=p.face_id, ok=p.ok, reason=p.reason, panels=len(p.panels_3d)) for p in placements])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nCOMPLETE | Faces: {len(faces)} | Obstacles: {len(obs)} | Panels: {n_pan} | Capacity: {total_w / 1000:.2f} kW")
    print(f"Overlay: {save_path}\nShapefile: {out / 'panels.shp'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3:
        main(a[0], a[1], a[2], a[3] if len(a) > 3 else None)
    else:
        main()

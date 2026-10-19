# 3D Roof Reconstruction & Solar Panel Placement (DSM + RGB)

Python pipeline: DSM (10 cm) + RGB aerial -> slope/aspect -> planar roof faces -> obstruction detection ->
geometrically consistent 3D roof -> standard-module solar layout, with quantitative validation.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
(`rasterio` ships wheels with PROJ for Windows/Linux/macOS; no manual PROJ_DATA setup needed.)

## Run
```bash
python main.py --rgb data/rgb.tif --dsm data/dsm_full.tif --clipped data/dsm_clipped.tif --out outputs
python -m pytest tests -q            # unit tests (Horn slope/aspect on known planes, frames, packing)
```
Every threshold in `roofpv/config.py` can be overridden on the CLI, e.g. `--panel-long 1.65 --panel-short 1.0 --edge-setback 0.5 --max-north-dev 45`.
`--no-orbit` skips the GIF. Runtime ~50 s.

## Outputs (`outputs/`)
`01_overview`, `02_elevation_slope_aspect`, `03_roof_faces` (+ residual map), `04_obstructions`, `05_panel_placement_2d`,
`06/07/08_roof_3d_*.png`, `roof_3d_orbit.gif`, `roof_solar_model.{glb,obj,ply}` (3D), `panels.csv` (3D corners),
`summary.json` (all metrics, per-face table, validation).

## Approach
| Stage | Method | Why |
|---|---|---|
| Missing data / noise | Laplace (harmonic) hole filling; median -> bilateral filter | Keeps ridges/eaves sharp (Gaussian would round them) |
| Roof mask | nDSM band (1.8-7 m) & inside clipped-DSM parcel & not vegetation (excess-green) -> opening, geodesic reconstruction | Separates house from hedges, walls, trees |
| Elevation/slope/aspect | Horn 3x3 with pixel size; aspect = down-slope azimuth from N | Validated on synthetic planes (`tests/`) |
| Faces | Sequential RANSAC: candidate planes from locally-planar seeds; inliers need height tolerance AND normal agreement AND connectivity; LSQ refit; coplanar merge; azimuths snapped to building axes (theta + k*90) | Prevents planes bridging across ridges; regularises noisy tilts |
| Face boundaries | Each remaining pixel -> face with minimum plane residual | Boundaries fall exactly on plane-intersection lines (ridge/hip/valley) |
| Geometry | Polygonise + Douglas-Peucker; shared vertices solved by regularised least squares on the incident planes | Neighbouring faces share 3D vertices; falls back to own plane when planes are inconsistent |
| Obstructions | Chimneys/dormers/vents: DSM residual > 0.30 m, compact blobs. Glazing: RGB anomaly vs same-orientation slate reference (Lab) + flush-with-plane DSM test; faces that are mostly glass = lantern | Skylights are nearly co-planar in the DSM, so colour is needed |
| Panels | Per face in its own plane frame (u horizontal along eave, v up-slope, true lengths); usable = face & planar support - 0.4 m setback - buffered obstructions; sweep grid offsets, portrait/landscape, max count, tie-break compactness; drop isolated modules | Real module size on the slope, not plan projection |
| Suitability | Reject glazing, north-ish (within 60 deg of N), flat (<5 deg, racks not modelled), >60 deg | Northern-hemisphere rule; all configurable |

## Results on the provided tile
16 faces (main pitch ~36 deg, azimuths 42.3 + k*90), 21 obstructions, **45 modules (1.72 x 1.13 m, 87.5 m2, ~18 kWp at 400 Wp)**.
Validation: height RMSE 5.5 cm; slope MAE 3.4 deg, aspect MAE 4.5 deg; 0 overlapping panels, 0 panels outside faces,
0 obstruction-clearance violations (min clearance 0.49 m); hold-out hole test (40 x 1.5 m patches) RMSE 4.3 cm (plane) vs 2.0 cm (Laplace).

## Assumptions & limitations (be upfront about these)
* DSM is already smooth (~1 m blur at ridges), so ridge/hip detail comes from plane intersections, not raw heights.
* ~16% of roof pixels (narrow gutters, ridge bars, dormer walls) are not planar and are not used for panels.
* Many mesh junctions remain unsnapped where faces are separated by non-planar strips (`vertex_snapping.n_fallback_nodes`).
* Glazing detection is heuristic (colour + flushness); some small false positives in dormer wells (harmless: those zones are unusable anyway). Small vents below DSM resolution may be missed.
* No shading/irradiance simulation; yield proxy is only an orientation factor. Flat roofs/tilt racks not modelled.
* Tree-covered roof edge (NW) is excluded rather than reconstructed.

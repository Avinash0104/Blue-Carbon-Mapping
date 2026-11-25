"""Export satellite predictors for the mangrove zone as GeoTIFFs on the mask's 10 m grid.

Usage:
    python satellite/export_satellite.py test <preview_dir>   small window: local preview + 2 Drive tasks
    python satellite/export_satellite.py full static          DEM + PALSAR (HH, HV), whole of Sabah
    python satellite/export_satellite.py full composite       Sentinel-2 + Sentinel-1, median of 2019-2023
    python satellite/export_satellite.py full 2020            same for one single year (fallback)
    python satellite/export_satellite.py testc                small window, 2019-2023 composite (Drive task)

Layout: one static file (DEM_x10, HH_x100, HV_x100) and one 2019-2023 composite file
(10 Sentinel-2 bands + VV_x100, VH_x100); both are int16, nodata -32768. Everything outside the mangrove
zone (mangrove + MARGIN_M) is nodata. The grid is the mask GeoTIFF's: EPSG:32650, 10 m.
The composite is the median of every clear scene over the whole period, not a median of yearly medians.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/collecting/ dir, so 'mask' is importable
import math
import sys
import urllib.request

import ee

import composites as c
from mask.mangrove_mask import build_mangrove_mask

CRS = "EPSG:32650"
MARGIN_M = 200
NODATA_I = -32768       # nodata of the static and yearly files (int16)
DEM_SCALE = 10          # DEM stored as metres x 10 (0.1 m steps)
SAR_SCALE = 100         # PALSAR stored as dB x 100 (0.01 dB steps), like Sentinel-1
S1_SCALE = 100          # Sentinel-1 is stored as int16 = dB * 100
YEARS = [2019, 2020, 2021, 2022, 2023]
FULL_GRID = (312050, 816500, 43995, 36200)      # x0, y0, width, height: same as the mask file
TEST_WINDOW = [117.7, 5.7, 118.1, 6.1]          # Sandakan Bay, ~44 km
PREVIEW_WINDOW = [118.00, 5.84, 118.08, 5.92]   # ~9 km, small enough to download directly


def grid_for(lonlat_rect):
    """Snap a lon/lat rectangle to the 10 m lattice (the mask grid is a multiple of 10 m)."""
    rect = ee.Geometry.Rectangle(lonlat_rect, proj="EPSG:4326", geodesic=False)
    ring = rect.transform(CRS, 1).coordinates().get(0).getInfo()
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    x0 = math.floor(min(xs) / 10) * 10
    y0 = math.ceil(max(ys) / 10) * 10
    return x0, y0, math.ceil((max(xs) - x0) / 10), math.ceil((y0 - min(ys)) / 10)


def region_for(grid):
    x0, y0, w, h = grid
    return ee.Geometry.Rectangle([x0, y0 - 10 * h, x0 + 10 * w, y0], proj=CRS, geodesic=False)


def transform_for(grid):
    x0, y0, _, _ = grid
    return [10, 0, x0, 0, -10, y0]


def static_image(zone):
    """DEM_x10 (m x 10), HH_x100, HV_x100 (dB x 100): all int16."""
    dem = c.dem_image().multiply(DEM_SCALE).round().toInt16().rename("DEM_x10")
    sar = (c.palsar_static().multiply(SAR_SCALE).round().toInt16()
           .rename(["HH_x100", "HV_x100"]))
    return ee.Image.cat([dem, sar]).updateMask(zone).unmask(NODATA_I)


def year_image(geom, zone, years=c.PERIOD):
    """10 Sentinel-2 bands (reflectance x 10000) + VV_x100, VH_x100: all int16.

    years: a single year (e.g. 2020) or a (first, last) range; default is the whole 2019-2023 period.
    """
    s2, _ = c.s2_composite(years, geom)
    s2 = s2.round().toInt16()
    s1 = (c.s1_composite(years, geom).multiply(S1_SCALE).round().toInt16()
          .rename(["VV_x100", "VH_x100"]))
    return ee.Image.cat([s2, s1]).updateMask(zone).unmask(NODATA_I)


def start_drive(img, name, folder, grid):
    task = ee.batch.Export.image.toDrive(
        image=img, description=name, folder=folder, fileNamePrefix=name,
        crs=CRS, crsTransform=transform_for(grid), dimensions=f"{grid[2]}x{grid[3]}",
        maxPixels=1e13, fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})
    task.start()
    print(f"Drive task {name}: {task.id}", flush=True)
    return task


def download(img, grid, path):
    url = img.getDownloadURL({"region": region_for(grid), "crs": CRS,
                              "crs_transform": transform_for(grid), "format": "GEO_TIFF"})
    urllib.request.urlretrieve(url, path)
    print("downloaded", path, flush=True)


if __name__ == "__main__":
    ee.Initialize(project="carbon-mapping-mangroves")
    sabah = c.get_sabah()
    zone = c.mangrove_zone(build_mangrove_mask(sabah), MARGIN_M)
    mode = sys.argv[1]

    if mode == "test":
        out_dir = sys.argv[2]
        pg = grid_for(PREVIEW_WINDOW)
        print("preview grid:", pg, flush=True)
        pgeom = region_for(pg)
        download(static_image(zone), pg, f"{out_dir}/static_preview.tif")
        download(year_image(pgeom, zone, 2020), pg, f"{out_dir}/year2020_preview.tif")

        tg = grid_for(TEST_WINDOW)
        print("test grid:", tg, flush=True)
        tgeom = region_for(tg)
        start_drive(static_image(zone), "test_static", "BlueCarbon_test", tg)
        start_drive(year_image(tgeom, zone, 2020), "test_2020", "BlueCarbon_test", tg)

    elif mode == "testc":
        tg = grid_for(TEST_WINDOW)
        print("test grid:", tg, flush=True)
        start_drive(year_image(region_for(tg), zone), "test_composite", "BlueCarbon_test", tg)

    elif mode == "full":
        what = sys.argv[2]
        if what == "static":
            start_drive(static_image(zone), "sabah_static", "BlueCarbon", FULL_GRID)
        elif what == "composite":
            start_drive(year_image(sabah.geometry(), zone), "sabah_composite_2019_2023",
                        "BlueCarbon", FULL_GRID)
        else:
            year = int(what)
            start_drive(year_image(sabah.geometry(), zone, year),
                        f"sabah_{year}", "BlueCarbon", FULL_GRID)

"""Sabah boundary + GMW v3/v4 inspection and comparison.

Analysis only: reproduces the areas used to choose the mask definition (option B).
The mask itself lives in mask.py and is exported by export_mask.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/collecting/ dir, so 'mask' is importable
import ee
import geemap
from mask.mangrove_mask import build_mangrove_mask

ee.Initialize(project="carbon-mapping-mangroves")

# --- Sabah boundary (FAO GAUL 2025 level 1; replaced geoBoundaries, which cut mangrove off the coast) ---
sabah = (
    ee.FeatureCollection("FAO/GAUL/2025/level1")
    .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia"))
    .filter(ee.Filter.eq("GAUL1_NAME", "Sabah"))
)
print("Sabah features:", sabah.size().getInfo())   # expect 1
print(sabah.first().getInfo()["properties"])

# --- GMW v3.0 (1996-2020, 25 m) and v4.0.19 (2020, 10 m) ---
gmw3 = ee.ImageCollection(
    "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/GMW_V3"
)
gmw4 = ee.ImageCollection(
    "projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020"
)

print("v3 size:", gmw3.size().getInfo())
print("v3 ids:", gmw3.aggregate_array("system:index").getInfo())
print("v4 size:", gmw4.size().getInfo())

g3 = gmw3.first()
print("v3 bands:", g3.bandNames().getInfo())
print("v3 scale (m):", g3.projection().nominalScale().getInfo())

g4 = gmw4.mosaic()
print("v4 bands:", g4.bandNames().getInfo())
# 111319 m is an artefact: mosaic() drops the tile projection. The real scale is 10 m.
print("v4 scale (m):", g4.projection().nominalScale().getInfo())


def to_binary(img):
    """1 = mangrove, 0 = everything else, within Sabah."""
    return img.select(0).gt(0).unmask(0).clip(sabah)


def area_km2(binary, scale, crs=None, geometry=None):
    kwargs = {"crs": crs} if crs else {}
    stats = binary.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry if geometry is not None else sabah.geometry(),
        scale=scale,
        maxPixels=1e13,
        **kwargs,
    )
    return ee.Number(stats.values().get(0)).divide(1e6).getInfo()


v3_2020 = to_binary(gmw3.filter(ee.Filter.stringContains("system:index", "2020")).mosaic())
v4_2020 = to_binary(g4)
print("v3 2020 mangrove area (km2):", area_km2(v3_2020, 25))
print("v4 2020 mangrove area (km2):", area_km2(v4_2020, 10))

# --- Comparison over 2015-2020 ---
# NOTE: this comparison uses unmask(0) on v3 and is evaluated on the default lat/lon grid.
# That works here, but unmask(0) on the global v3 rasters returns 0 when evaluated in UTM
# (see mask.py), so do not reuse this recipe for exports.
years = [2015, 2016, 2017, 2018, 2019, 2020]
v3_stack = gmw3.filter(ee.Filter.inList("system:index", [f"gmw_v3_{y}" for y in years]))
print("v3 epochs:", v3_stack.size().getInfo())     # expect 6

v3_bin = v3_stack.map(lambda im: im.select(0).gt(0).unmask(0))
v3_core = v3_bin.min().clip(sabah)    # mangrove in ALL six years
v3_union = v3_bin.max().clip(sabah)   # mangrove in ANY year
v4 = g4.select(0).gt(0).unmask(0).clip(sabah)

SCALE = 10
print("v3 core 2015-20 :", area_km2(v3_core, SCALE))
print("v3 union 2015-20:", area_km2(v3_union, SCALE))
print("v4 2020         :", area_km2(v4, SCALE))
print("v4 AND v3 core  :", area_km2(v4.And(v3_core), SCALE))
print("v4 NOT v3 union :", area_km2(v4.And(v3_union.Not()), SCALE))
print("v3 union NOT v4 :", area_km2(v3_union.And(v4.Not()), SCALE))

# --- The final mask (option B), built by mask.py ---
mangrove_mask = build_mangrove_mask(sabah)
print("Mask bands:", mangrove_mask.bandNames().getInfo())
print("Mask area (km2):", area_km2(mangrove_mask, 10, crs="EPSG:32650"))   # expect ~2385

# --- Coastline clipping check: how much mangrove does the boundary cut off? ---
sabah_buf = sabah.geometry().buffer(2000)   # 2 km
mask_raw = g4.select(0).gt(0).unmask(0).And(v3_bin.min())    # no clip
inside = area_km2(mask_raw, 10, geometry=sabah.geometry())
buffered = area_km2(mask_raw, 10, geometry=sabah_buf)
print("Mask inside boundary (km2):    ", inside)
print("Mask inside +2 km buffer (km2):", buffered)
print("Lost to clipping (km2):        ", buffered - inside)

# --- Map for visual inspection ---
sabah_outline = sabah.style(color="red", fillColor="00000000", width=2)
clipped_off = (mask_raw.clip(sabah_buf)
               .And(mask_raw.clip(sabah.geometry()).Not()).selfMask())

m = geemap.Map(basemap="SATELLITE")
m.centerObject(sabah, 8)
m.addLayer(g3.clip(sabah), {"min": 0, "max": 1, "palette": ["white", "green"]}, "GMW v3 first epoch", False)
m.addLayer(g4.clip(sabah), {"min": 0, "max": 1, "palette": ["white", "blue"]}, "GMW v4 2020", False)
m.addLayer(mangrove_mask.selfMask(), {"palette": ["yellow"]}, "Mask (1 only)")
m.addLayer(clipped_off, {"palette": ["magenta"]}, "Clipped off (within 2 km)")
m.addLayer(sabah_outline, {}, "Sabah boundary")
m.to_html("check.html")
print("Saved check.html")

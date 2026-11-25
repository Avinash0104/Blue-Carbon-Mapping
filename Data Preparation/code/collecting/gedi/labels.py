"""GEDI L2A exploration and diagnostics over Sabah mangrove (April 2019 - March 2023).

Analysis only: shows how many shots survive each quality filter and how many land in the
mangrove mask. The export itself is export_gedi.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/collecting/ dir, so 'mask' is importable
import ee
from export_gedi import MAX_RH98, MIN_RH98, filter_stages, label_image   # the export's own filters
from mask.mangrove_mask import build_mangrove_mask

ee.Initialize(project="carbon-mapping-mangroves")

sabah = (
    ee.FeatureCollection("FAO/GAUL/2025/level1")
    .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia"))
    .filter(ee.Filter.eq("GAUL1_NAME", "Sabah"))
)
geom = sabah.geometry()
mangrove = build_mangrove_mask(sabah)       # 0/1 inside Sabah, built in memory

START, END = "2019-04-01", "2023-04-01"      # end date is exclusive

# --- Monthly raster ---
gedi_m = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").filterBounds(geom)
print("Monthly images over Sabah:", gedi_m.size().getInfo())
first = gedi_m.first()
print("Band count:", first.bandNames().size().getInfo())
print("Scale (m):", first.projection().nominalScale().getInfo())
print("Projection:", first.projection().getInfo())    # EPSG:32650, 25 m

# --- Footprint vectors: per-granule tables found through the index ---
idx = (
    ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002_INDEX")
    .filterBounds(geom)
    .filter(ee.Filter.gte("time_start", START))
    .filter(ee.Filter.lt("time_start", END))
)
print("L2A tables in window over Sabah:", idx.size().getInfo())
table_id = idx.first().getInfo()["properties"]["table_id"]
print(ee.FeatureCollection(table_id).first().getInfo())     # shot properties incl. rh98


# --- Quality funnel: how many 25 m GEDI pixels survive each filter? ---
FUNNEL = ["raw"] + [name for name, _ in filter_stages(gedi_m.first())] + ["rh_range", "mangrove"]


def stage_counts(img):
    """Cumulative count of valid 25 m pixels after each filter (same filters as the export)."""
    rh = img.select("rh98")
    bands, keep = [rh.mask().rename("raw")], None
    for name, m in filter_stages(img):
        keep = m if keep is None else keep.And(m)
        bands.append(rh.updateMask(keep).mask().rename(name))
    keep = keep.And(rh.gte(MIN_RH98)).And(rh.lte(MAX_RH98))
    bands.append(rh.updateMask(keep).mask().rename("rh_range"))
    bands.append(rh.updateMask(keep).updateMask(mangrove.eq(1)).mask().rename("mangrove"))
    return ee.Feature(None, ee.Image.cat(bands).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=geom, scale=25, maxPixels=1e13))


raw = (ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
       .filterDate(START, END)
       .filterBounds(geom))
fc = raw.map(stage_counts)
for k in FUNNEL:
    print(k, fc.aggregate_sum(k).getInfo())


# --- Labels: rh98, filtered as in export_gedi.py, mangrove only ---
gedi_labels = raw.map(lambda img: label_image(img, mangrove).rename("rh98"))
print("Images (2 per month where two tiles overlap Sabah):", gedi_labels.size().getInfo())


# --- One mosaic per month, and valid pixels per month ---
def month_image(i):
    start = ee.Date(START).advance(i, "month")
    end = start.advance(1, "month")
    img = gedi_labels.filterDate(start, end).mosaic()
    return img.set({"month": start.format("YYYY-MM"), "system:time_start": start.millis()})


monthly = ee.ImageCollection(ee.List.sequence(0, 47).map(month_image))
print("Monthly mosaics:", monthly.size().getInfo())     # expect 48


def count_valid(img):
    n = img.mask().reduceRegion(
        ee.Reducer.sum(), geom, 25, maxPixels=1e13).get("rh98")
    return ee.Feature(None, {"month": img.get("month"), "valid": n})


for f in ee.FeatureCollection(monthly.map(count_valid)).getInfo()["features"]:
    print(f["properties"]["month"], f["properties"]["valid"])

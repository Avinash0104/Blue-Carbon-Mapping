import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/collecting/ dir, so 'mask' is importable
import ee
from mask.mangrove_mask import build_mangrove_mask

ee.Initialize(project="carbon-mapping-mangroves")

sabah = (ee.FeatureCollection("FAO/GAUL/2025/level1")
        .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia"))
        .filter(ee.Filter.eq("GAUL1_NAME", "Sabah")))

mask = build_mangrove_mask(sabah)
# Fixed grid (Project.md 2.1): covers the GAUL 2025 Sabah bounding box, edges on 50 m multiples.
X0, Y0, W, H = 312050, 816500, 43995, 36200

# Binary: 1 = mangrove (v4 2020 AND v3 mangrove in all of 2015-2020), 0 = everything else
# (including outside the Sabah polygon). No nodata value.
drive_task = ee.batch.Export.image.toDrive(
    image=mask.unmask(0).toUint8(), description="sabah_mangrove_mask_tif",
    folder="BlueCarbon", fileNamePrefix="sabah_mangrove_mask",
    crs="EPSG:32650", crsTransform=[10, 0, X0, 0, -10, Y0], dimensions=f"{W}x{H}", maxPixels=1e13,
    fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True})

drive_task.start()
print("Drive:", drive_task.id)

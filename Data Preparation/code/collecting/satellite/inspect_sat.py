"""Which satellite assets exist over Sabah? Metadata only, no exports."""
import ee

ee.Initialize(project="carbon-mapping-mangroves")

sabah = (ee.FeatureCollection("FAO/GAUL/2025/level1")
         .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia"))
         .filter(ee.Filter.eq("GAUL1_NAME", "Sabah")))
geom = sabah.geometry()
start, end = "2020-01-01", "2021-01-01"        # one sample year

# Sentinel-2 + Cloud Score+
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end)
cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(geom).filterDate(start, end)
print("S2 scenes 2020:", s2.size().getInfo(), "| CS+ images:", cs.size().getInfo())
print("S2 bands:", s2.first().bandNames().getInfo())

# Sentinel-1
s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(start, end)
      .filter(ee.Filter.eq("instrumentMode", "IW")))
print("S1 IW scenes 2020:", s1.size().getInfo())
print("S1 polarisations:", s1.aggregate_array("transmitterReceiverPolarisation").distinct().getInfo())
print("S1 passes:", s1.aggregate_array("orbitProperties_pass").distinct().getInfo())

# PALSAR-2 yearly mosaics
pal = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH").filterBounds(geom)
print("PALSAR images:", pal.aggregate_array("system:index").getInfo())
print("PALSAR bands:", pal.first().bandNames().getInfo(),
      "| scale:", pal.first().projection().nominalScale().getInfo())

# Copernicus DEM (GLO30 is deprecated; GLO30_2024_1 supersedes it)
dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(geom)
print("DEM tiles:", dem.size().getInfo(), "| bands:", dem.first().bandNames().getInfo(),
      "| scale:", dem.first().projection().nominalScale().getInfo())

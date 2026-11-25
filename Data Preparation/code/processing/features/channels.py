"""The 21 input channels (Project.md 2.4): names, order and where each one comes from.

One source of truth for every model. The feature table, the normalisation statistics and (later) the U-Net tile
stack must all use CHANNELS in this order, so a channel never ends up in the wrong place.

Each entry of SOURCES describes one family of GeoTIFF files:
    pattern    glob, relative to the project root (the 'Data Preparation' folder)
    nodata     value that marks missing data in these files (the raw exports do not tag it)
    file_bands band names in the order they are stored in the files (None = not used here)
    use        {file band name: (channel name, divisor)}; stored value / divisor = physical value
Adding the composite or the derived files later only means that the files exist; nothing else changes.
"""

CHANNELS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",     # Sentinel-2 reflectance
    "NDVI", "NDRE", "NDMI", "EVI",                                      # Sentinel-2 indices (derived locally)
    "VV", "VH", "VVVH_db",                                              # Sentinel-1 backscatter (dB), VV - VH
    "HH", "HV", "HVHH_db",                                              # PALSAR-2 backscatter (dB), HV - HH
    "DEM",                                                              # Copernicus DEM elevation (m)
]
assert len(CHANNELS) == 21 and len(set(CHANNELS)) == 21

_S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

SOURCES = [
    {"name": "static",
     "pattern": "data/raw/satellite/static/sabah_static-*.tif",
     "nodata": -32768,
     "file_bands": ["DEM_x10", "HH_x100", "HV_x100"],
     "use": {"DEM_x10": ("DEM", 10), "HH_x100": ("HH", 100), "HV_x100": ("HV", 100)}},
    {"name": "composite",
     "pattern": "data/raw/satellite/composite/sabah_composite_2019_2023*.tif",
     "nodata": -32768,
     "file_bands": _S2 + ["VV_x100", "VH_x100"],
     "use": {**{b: (b, 10000) for b in _S2}, "VV_x100": ("VV", 100), "VH_x100": ("VH", 100)}},
    {"name": "derived_composite",
     "pattern": "data/processed/satellite/sabah_composite_2019_2023*_derived.tif",
     "nodata": -9999,
     "file_bands": ["NDVI", "NDRE", "NDMI", "EVI", "VVVH_db"],
     "use": {b: (b, 1) for b in ["NDVI", "NDRE", "NDMI", "EVI", "VVVH_db"]}},
    {"name": "derived_static",
     "pattern": "data/processed/satellite/sabah_static-*_derived.tif",
     "nodata": -9999,
     "file_bands": ["HVHH_db", "slope_deg"],                 # slope is dropped (Project.md 2.4): not in `use`
     "use": {"HVHH_db": ("HVHH_db", 1)}},
]

# every channel must come from exactly one source
_made = [ch for s in SOURCES for ch, _ in s["use"].values()]
assert sorted(_made) == sorted(CHANNELS), "SOURCES and CHANNELS disagree"

"""Annual satellite composites over Sabah (Sentinel-2, Sentinel-1, PALSAR-2) and the DEM.

Functions only: nothing runs on import. Call ee.Initialize() first.
"""
import ee

S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CS_THRESHOLD = 0.6      # Cloud Score+ cs_cdf; the catalog suggests 0.5-0.65


def get_sabah():
    return (ee.FeatureCollection("FAO/GAUL/2025/level1")
            .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia"))
            .filter(ee.Filter.eq("GAUL1_NAME", "Sabah")))


PERIOD = (2019, 2023)   # first and last calendar year of the composite (Project.md 2.1)


def date_range(years):
    """('YYYY-01-01', 'YYYY-01-01' of the next year) for a year or a (first, last) pair, last inclusive."""
    first, last = (years, years) if isinstance(years, int) else years
    return f"{first}-01-01", f"{last + 1}-01-01"


def s2_collection(years, geom, threshold=CS_THRESHOLD):
    """Cloud-masked Sentinel-2 SR for a year or a (first, last) year range (Cloud Score+ linked by system:index)."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    start, end = date_range(years)
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start, end)
            .linkCollection(cs, ["cs_cdf"])
            .map(lambda im: im.updateMask(im.select("cs_cdf").gte(threshold))))


S2_10M = ["B2", "B3", "B4", "B8"]
S2_20M = ["B5", "B6", "B7", "B8A", "B11", "B12"]


def s2_composite(years, geom, threshold=CS_THRESHOLD):
    """Median of the cloud-masked scenes over a year or a (first, last) year range, 10 bands,
    plus the number of clear observations.

    The 6 native 20 m bands are medianed at 20 m, then interpolated to 10 m with bicubic and
    clamped at 0 (bicubic can overshoot near sharp edges; reflectance cannot be negative).
    Resampling every scene first, then taking the median, ran out of Earth Engine memory.
    """
    col = s2_collection(years, geom, threshold)
    med10 = col.select(S2_10M).median()
    med20 = (col.select(S2_20M).median()
             .reproject("EPSG:32650", None, 20)      # compute the median on the true 20 m grid
             .resample("bicubic").max(0))             # then interpolate to the output grid
    comp = ee.Image.cat([med10, med20]).select(S2_BANDS)
    n_obs = col.select("B8").count().rename("n_obs")
    return comp, n_obs


def s1_composite(years, geom):
    """Median VV and VH backscatter (already in dB in COPERNICUS/S1_GRD), IW mode, over a year
    or a (first, last) year range.

    Only descending passes exist over Sabah, so geometry is consistent.
    """
    start, end = date_range(years)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(geom)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]))
    return col.median()


def palsar_composite(year):
    """PALSAR-2 yearly mosaic: HH and HV in dB, land pixels only (qa == 255).

    The mosaic stores 16-bit DN: dB = 10*log10(DN^2) - 83.
    """
    raw = ee.Image(ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
                   .filter(ee.Filter.eq("system:index", str(year))).first())
    img = raw.resample("bicubic")       # 25 m -> 10 m: interpolate instead of copying blocks
    # max(1): bicubic can overshoot to <= 0 near sharp edges, and log10(0) is undefined
    db = (img.select(["HH", "HV"]).toFloat().max(1).pow(2).log10().multiply(10).subtract(83.0))
    # the mask uses the un-resampled qa band, so class codes are never interpolated
    return db.updateMask(raw.select("qa").eq(255)).rename(["HH_db", "HV_db"])


def palsar_static(years=(2019, 2020, 2021, 2022, 2023)):
    """One time-invariant PALSAR-2 layer: median HH_db and HV_db over the available years.

    The yearly mosaics have gaps over Sabah (2021 misses about 44% of the mangrove), but the
    gaps fall in different places each year and HV differs by only ~0.6 dB between years, so
    the median over all years covers 99.4% of mangrove pixels.
    """
    return ee.ImageCollection([palsar_composite(y) for y in years]).median()


def mangrove_zone(mask, margin_m=200):
    """1 within margin_m of mangrove, masked elsewhere. Built on a 100 m grid.

    mask is the 0/1 image from mask.build_mangrove_mask. The zone is computed coarsely (any
    mangrove inside a 100 m cell, then dilated by the margin) and used as a mask on 10 m data,
    so its edge is blocky at 100 m, which is fine for a margin.
    """
    coarse = (mask.eq(1)
              .setDefaultProjection(crs="EPSG:32650", scale=10)
              .reduceResolution(ee.Reducer.max(), maxPixels=256)
              .reproject(crs="EPSG:32650", scale=100))
    return coarse.focalMax(margin_m, "circle", "meters").selfMask().rename("zone")


def dem_image():
    """Copernicus GLO-30 surface model (30 m), mosaicked with its native projection restored."""
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
    return (col.map(lambda im: im.select("DEM").resample("bilinear"))   # 30 m -> 10 m
            .mosaic()
            .setDefaultProjection(col.first().projection())
            .rename("DEM"))

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/collecting/ dir, so 'mask' is importable
from functools import reduce

import ee
from mask.mangrove_mask import build_mangrove_mask

START = "2019-04-01"
END = "2023-04-01"      # exclusive
N_MONTHS = 48
NODATA = -1             # float GeoTIFFs can't hold a mask; safe because MIN_RH98 >= 0
MIN_RH98 = 0            # drop heights below this (m), so no real label can equal NODATA
MAX_RH98 = 50           # drop heights above this (m): not plausible for Sabah mangrove
SENS_MIN = 0.98         # project.md 2.3: sensitivity >= 0.98
POWER_BEAMS = [5, 6, 8, 11]   # full-power beams (BEAM0101, 0110, 1000, 1011); coverage beams are 0-3


def get_sabah():
    return (ee.FeatureCollection("FAO/GAUL/2025/level1")
            .filter(ee.Filter.eq("GAUL0_NAME", "Malaysia"))
            .filter(ee.Filter.eq("GAUL1_NAME", "Sabah")))


def filter_stages(img):
    """The GEDI shot filters of project.md 2.3, in order, as (name, boolean image) pairs."""
    sens = img.select("sensitivity")
    return [
        ("quality", img.select("quality_flag").eq(1)),
        ("degrade", img.select("degrade_flag").eq(0)),
        ("sensitivity", sens.gte(SENS_MIN).And(sens.lte(1))),
        ("power_beam", img.select("beam").remap(POWER_BEAMS, [1] * len(POWER_BEAMS), 0).eq(1)),
        ("night", img.select("solar_elevation").lt(0)),      # sun below the horizon
    ]


def label_image(img, mangrove):
    """rh98 of one monthly image where every filter passes and the pixel is mangrove.

    Uses the FULL mask on purpose. The 30 m inner mask (data/raw/mask/sabah_mangrove_inner_mask.tif,
    built by mask/inner_mask.py) is applied later, when the training labels are selected, so the
    shrink distance can change without another export. Count the labels after that step too.
    """
    good = reduce(lambda a, b: a.And(b), [m for _, m in filter_stages(img)])
    rh = img.select("rh98")
    return ee.Image(rh.updateMask(good)
                    .updateMask(mangrove.eq(1))
                    .updateMask(rh.gte(MIN_RH98))
                    .updateMask(rh.lte(MAX_RH98))
                    .copyProperties(img, ["system:time_start"]))


def build_stack(sabah):
    """48 monthly bands (rh98_YYYY_MM): filtered GEDI rh98, mangrove pixels only."""
    mangrove = build_mangrove_mask(sabah)

    labels = (ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
              .filterDate(START, END)
              .filterBounds(sabah.geometry())
              .map(lambda img: label_image(img, mangrove)))

    def month_band(i):
        start = ee.Date(START).advance(i, "month")
        end = start.advance(1, "month")
        name = ee.String("rh98_").cat(start.format("YYYY_MM"))
        return labels.filterDate(start, end).mosaic().rename(name).toFloat()

    # A Python list (not ee.List): ee.Image.cat cannot take a server-side list.
    return ee.Image.cat([month_band(i) for i in range(N_MONTHS)])


def gedi_grid():
    """25 m grid over the same extent as the 10 m mask (x 312,050-752,000, y 454,500-816,500).

    Every 25 m pixel edge lies on a 25 m multiple and on the mask's 50 m lattice.
    """
    return 312050, 816500, 17598, 14480      # x0, y0, width, height


if __name__ == "__main__":
    ee.Initialize(project="carbon-mapping-mangroves")
    sabah = get_sabah()
    stack = build_stack(sabah)
    names = stack.bandNames().getInfo()
    print(len(names), names[0], names[-1])        # expect 48, rh98_2019_04, rh98_2023_03

    x0, y0, w, h = gedi_grid()
    print("grid:", x0, y0, w, h)

    # Band order: rh98_2019_04 ... rh98_2023_03. Nodata = -1 (months with no valid
    # shot are all -1). Everything that is not a valid mangrove shot is -1.
    task = ee.batch.Export.image.toDrive(
        image=stack.unmask(NODATA),
        description="sabah_gedi_rh98_monthly_v2_tif",
        folder="BlueCarbon",
        fileNamePrefix="sabah_gedi_rh98_monthly_v2",
        crs="EPSG:32650",
        crsTransform=[25, 0, x0, 0, -25, y0],
        dimensions=f"{w}x{h}",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    print("Drive:", task.id)

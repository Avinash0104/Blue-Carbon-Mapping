import ee

def build_mangrove_mask(sabah):

    gmw3 = ee.ImageCollection("projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/GMW_V3")
    gmw4 = ee.ImageCollection("projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020")

    ids = [f"gmw_v3_{y}" for y in range(2015, 2021)]
    v3_flag = (gmw3.filter(ee.Filter.inList("system:index", ids))
                 .map(lambda im: im.select(0).gt(0)))
    v3_core = v3_flag.count().eq(6)

    v4 = gmw4.mosaic().select(0).gt(0).unmask(0)

    # v3_core is only defined near mangroves, so start from 0 everywhere in Sabah
    # and write 1 where both products agree (a masked test leaves the 0 untouched).
    mangrove = v4.And(v3_core)
    return (ee.Image(0).where(mangrove, 1)
            .toUint8().rename("mangrove").clip(sabah))
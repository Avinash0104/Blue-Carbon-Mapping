"""Stitch the GEDI pieces into one virtual raster (no gdalbuildvrt needed).

Usage: python build_gedi_vrt.py data/raw/gedi/sabah_gedi_rh98_monthly_v2.vrt data/raw/gedi/sabah_gedi_rh98_monthly_v2-*.tif
The VRT holds no pixels, only pointers to the pieces (relative paths), so keep them together.
"""
import os
import sys
from xml.sax.saxutils import quoteattr

import rasterio

out, pieces = sys.argv[1], sys.argv[2:]
srcs = []
for p in pieces:
    with rasterio.open(p) as s:
        srcs.append((os.path.relpath(p, os.path.dirname(out) or "."), s.transform, s.width, s.height))
        n, crs, names, nodata = s.count, s.crs, s.descriptions, -1   # nodata is not tagged in the exports
x0 = min(t.c for _, t, _, _ in srcs)
y0 = max(t.f for _, t, _, _ in srcs)
W = round(max(t.c + 25 * w for _, t, w, _ in srcs) - x0) // 25
H = round(y0 - min(t.f - 25 * h for _, t, _, h in srcs)) // 25

lines = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
         f"  <SRS>{crs.to_wkt()}</SRS>",
         f"  <GeoTransform>{x0}, 25, 0, {y0}, 0, -25</GeoTransform>"]
for b in range(n):
    lines += [f'  <VRTRasterBand dataType="Float32" band="{b + 1}">',
              f"    <NoDataValue>{nodata}</NoDataValue>",
              f"    <Description>{names[b]}</Description>"]
    for rel, t, w, h in srcs:
        xo, yo = round((t.c - x0) / 25), round((y0 - t.f) / 25)
        lines += ["    <SimpleSource>",
                  f"      <SourceFilename relativeToVRT=\"1\">{rel.replace(os.sep, '/')}</SourceFilename>",
                  f"      <SourceBand>{b + 1}</SourceBand>",
                  f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
                  f'      <DstRect xOff="{xo}" yOff="{yo}" xSize="{w}" ySize="{h}"/>',
                  "    </SimpleSource>"]
    lines.append("  </VRTRasterBand>")
lines.append("</VRTDataset>")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"wrote {out}: {W}x{H}, {n} bands, {len(srcs)} pieces")

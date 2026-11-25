"""Build the inner mask locally: the full mangrove mask shrunk by a distance (default 30 m).

Used to SELECT GEDI training labels (CLAUDE.md 2.2/2.3); the full mask is still used for mapping.
A pixel is inner when every pixel within `metres` of it is mangrove in the full mask.

Usage (from the Data Preparation folder):
    python code/processing/mask/inner_mask.py [metres]
    -> data/raw/mask/sabah_mangrove_inner_mask.tif   (the distance is stored in the file as the tag INNER_MASK_METRES)

Output (same grid as the full mask, uint8, binary, no nodata value):
    1   mangrove at least `metres` from any non-mangrove pixel (the inner mask)
    0   everything else (including mangrove closer to the edge)
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_erosion

SRC = "data/raw/mask/sabah_mangrove_mask.tif"
CHUNK = 2048            # rows per block (multiple of the 512 px output tile size)

if __name__ == "__main__":
    metres = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = "data/raw/mask/sabah_mangrove_inner_mask.tif"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(SRC) as src:
        if metres % int(src.res[0]):
            sys.exit(f"{metres} m is not a multiple of the {src.res[0]:.0f} m pixel size")
        px = metres // int(src.res[0])                         # distance in pixels
        yy, xx = np.mgrid[-px:px + 1, -px:px + 1]
        disc = (xx ** 2 + yy ** 2) <= px ** 2

        profile = src.profile | {"tiled": True, "blockxsize": 512, "blockysize": 512,
                                 "compress": "lzw", "nodata": None}
        n_in = n_full = 0
        with rasterio.open(out, "w", **profile) as dst:
            for r0 in range(0, src.height, CHUNK):
                r1 = min(r0 + CHUNK, src.height)
                a0, a1 = max(r0 - px, 0), min(r1 + px, src.height)      # halo so edges are exact
                blk = src.read(1, window=((a0, a1), (0, src.width)))
                full = blk == 1
                inner = binary_erosion(full, structure=disc, border_value=0)
                lo, hi = r0 - a0, r0 - a0 + (r1 - r0)
                res = inner[lo:hi].astype("uint8")
                dst.write(res, 1, window=((r0, r1), (0, src.width)))
                n_in += int((res == 1).sum())
                n_full += int(full[lo:hi].sum())
            dst.set_band_description(1, "mangrove_inner")
            dst.update_tags(INNER_MASK_METRES=str(metres))

    print(f"{out}: inner {n_in} px ({n_in * 1e-4:.0f} km2) of {n_full} full mangrove px "
          f"({100 * n_in / max(n_full, 1):.1f}% kept; {metres} m = {px} px)")

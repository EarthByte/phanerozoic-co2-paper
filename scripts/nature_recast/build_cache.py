#!/usr/bin/env python3
"""Build the land-only cache for the Nature-recast weathering analysis (one zip archive).

For each of the 109 ages it collects, on the union of the two emergent masks:
erosion-deposition for both surfaces, both masks, the `rain` forcing channel,
HadCM3 surface-air temperature, both palaeo-elevations, cell area, the flat cell
index, and the HadCM3 runoff ratio (surface + subsurface runoff over
precipitation, scotese_spinupa suite of Valdes et al. 2021, from the table written by
fetch_bridge_runoff.py), filled to the 0.1 degree grid by nearest model land cell and
bilinear interpolation. Ages already in the archive are skipped.

    python3 build_cache.py            # all ages
    python3 build_cache.py 0 300 420  # selected ages
"""
import sys, numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

import _config as cfg
import _cache
ROOT = cfg.ZENODO
_RUN = None
R_EARTH = 6.371e6


def runoff_ratio_field(age, lat, lon):
    global _RUN
    if _RUN is None:
        with np.load(cfg.RUNOFF) as r:
            _RUN = {k: r[k] for k in r.files}
    k = int(np.flatnonzero(_RUN['ages'] == age)[0])
    P = _RUN['precip'][k].astype(float); R = _RUN['runoff'][k].astype(float)
    la = _RUN['lat'].astype(float); lo = _RUN['lon'].astype(float)
    land = np.isfinite(R)
    ratio = np.where(land, np.clip(R / np.maximum(P, 1e-12), 0.0, 1.0), np.nan)
    # nearest-land fill, periodic in longitude (tile x3)
    t = np.concatenate([ratio, ratio, ratio], axis=1)
    idx = ndimage.distance_transform_edt(~np.isfinite(t), return_distances=False, return_indices=True)
    tf = t[tuple(idx)]
    nlo = lo.size
    filled = tf[:, nlo:2 * nlo]
    # to ascending lat, lon in -180..180 with wrap padding
    if la[0] > la[-1]:
        la = la[::-1]; filled = filled[::-1]; land = land[::-1]
    lo2 = np.where(lo >= 180, lo - 360, lo); o = np.argsort(lo2); lo2 = lo2[o]; filled = filled[:, o]
    lo3 = np.concatenate([[lo2[-1] - 360], lo2, [lo2[0] + 360]])
    f3 = np.concatenate([filled[:, -1:], filled, filled[:, :1]], axis=1)
    itp = RegularGridInterpolator((la, lo3), f3, bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    out = itp(np.stack([LA.ravel(), LO.ravel()], axis=1)).reshape(LA.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32), str(_RUN['puma_id'][k])


def main(ages):
    done = _cache.ages_present()
    for age in ages:
        if age in done:
            continue
        d = np.load(ROOT / f'surrogate_erodep/erodep_{age}Ma.npz')
        ms = d['mask_scotese']; mz = d['mask_zhou']; u = ms | mz
        lat = d['lat'].astype(float); lon = d['lon'].astype(float)
        t = np.load(ROOT / f'valdes_temperature/{age}Ma.npz')
        T = t['temp_q'].astype(np.float32) * float(t['scale']) + float(t['offset'])
        ratio, src = runoff_ratio_field(age, lat, lon)
        dg = np.deg2rad(0.1)
        area = (R_EARTH ** 2 * dg * dg * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        # the grid repeats the dateline column (lon -180 and +180) and the pole rows have zero area;
        # halve the two duplicate columns so integrals are not double counted
        area[:, 0] *= 0.5; area[:, -1] *= 0.5
        ii = np.flatnonzero(u.ravel()).astype(np.uint32)
        _cache.append(age, idx=ii, shape=np.array(u.shape),
                 ero_s=d['ero_scotese'].ravel()[ii].astype(np.float32),
                 ero_z=d['ero_zhou'].ravel()[ii].astype(np.float32),
                 ms=ms.ravel()[ii], mz=mz.ravel()[ii],
                 rain=d['rain'].ravel()[ii].astype(np.float32),
                 T=T.ravel()[ii].astype(np.float32),
                 ratio=ratio.ravel()[ii],
                 z_s=d['zpaleo_scotese'].ravel()[ii].astype(np.float32),
                 z_z=d['zpaleo_zhou'].ravel()[ii].astype(np.float32),
                 area=area.ravel()[ii].astype(np.float32),
                 lat=lat.astype(np.float32), lon=lon.astype(np.float32), age=age, puma_id=src)
        print(age, src, ii.size, float(ratio.ravel()[ii].mean()), flush=True)


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or list(range(0, 545, 5)))

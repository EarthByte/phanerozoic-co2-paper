#!/usr/bin/env python3
"""Runoff ratio of every HadCM3L simulation on the model grid, extended over the ocean from the nearest
model land cell, in ascending latitude and longitude -180..180 with one wrap column on either side.

Writes ONE table, data/nature_recast/hadcm3l_runoff_ratio_filled.npz (ages, lat, lon, ratio), from
hadcm3l_runoff_scotese_spinupa.npz. Bilinear interpolation of this table (_evaluate.ratio_at) is the
runoff-ratio field of build_cache.py; the table lets experiments run where scipy is not installed."""
import numpy as np
from scipy import ndimage
import _config as cfg
from _evaluate import RATIO_FILLED

with np.load(cfg.RUNOFF) as r:
    ages, la, lo, P, R = r['ages'], r['lat'].astype(float), r['lon'].astype(float), r['precip'].astype(float), r['runoff'].astype(float)
flip = la[0] > la[-1]
lo2 = np.where(lo >= 180, lo - 360, lo); o = np.argsort(lo2); lo2 = lo2[o]
out = []
for k in range(ages.size):
    land = np.isfinite(R[k])
    ratio = np.where(land, np.clip(R[k] / np.maximum(P[k], 1e-12), 0.0, 1.0), np.nan)
    t = np.concatenate([ratio, ratio, ratio], axis=1)
    idx = ndimage.distance_transform_edt(~np.isfinite(t), return_distances=False, return_indices=True)
    f = t[tuple(idx)][:, lo.size:2 * lo.size]
    if flip:
        f = f[::-1]
    f = f[:, o]
    out.append(np.concatenate([f[:, -1:], f, f[:, :1]], axis=1).astype(np.float32))
lat = la[::-1] if flip else la
lon = np.concatenate([[lo2[-1] - 360], lo2, [lo2[0] + 360]])
np.savez_compressed(RATIO_FILLED, ages=ages, lat=lat.astype(np.float32), lon=lon.astype(np.float32), ratio=np.stack(out))
print('wrote', RATIO_FILLED, np.stack(out).shape)

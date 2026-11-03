#!/usr/bin/env python3
"""Scan the adjusted surface for mountain belts and rank them by their position relative to the
local drawdown optimum. Used to choose the two end-member orogenic episodes of Fig. 4."""
import sys, ast, numpy as np
from pathlib import Path
from scipy import ndimage
import weathering_core as wc
import run_weathering as rw
import _cache

import _config as cfg
OUT = cfg.OUT
amp_txt = open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')
AMPS = ast.literal_eval(amp_txt[0])
fitsets = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}
qgrid = np.logspace(0, 4, 81)
eopt_q = rw.eopt_table(fitsets, AMPS, 'kinetic', qgrid)
ZMIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0
ages = [int(a) for a in sys.argv[2:]] or [0, 20, 50, 80, 100, 150, 200, 220, 240, 260, 280, 300, 320, 340, 360, 400, 420, 450, 480]
print('age reg lat lon area_Mkm2 zmean Emean qmean x_med beyond S_sil S_net Fnet_density share_of_global_net')
for age in ages:
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    m = d['mz']; A = d['area'].astype(float); T = d['T'] + 273.15
    q = d['rain'].astype(float) * 1000 * d['ratio']; E = np.where(m, np.maximum(-d['ero_z'].astype(float), 0), 0) * 2.0
    sil = dsil = sulf = dsulf = 0
    for r in rw.SETS:
        t = rw.region_terms(E, q, T, fitsets[r], 'kinetic')
        sil = sil + AMPS[r]['sil'] * t['sil'] / 3; dsil = dsil + AMPS[r]['sil'] * t['dsil'] / 3
        sulf = sulf + AMPS[r]['sulf'] * t['sulf'] / 3; dsulf = dsulf + AMPS[r]['sulf'] * t['dsulf'] / 3
    Eopt = np.interp(np.log(np.maximum(q, 1)), np.log(qgrid), eopt_q)
    netg = float(np.sum(((sil - sulf) * A)[m]))
    z = np.full(shp[0] * shp[1], -9999.0); z[idx] = np.where(m, d['z_z'], -9999.0); z = z.reshape(shp)
    zs = ndimage.uniform_filter(np.where(z > -9000, z, 0), size=15, mode='wrap')
    lab, n = ndimage.label(zs > ZMIN)
    labf = lab.ravel()[idx]
    lat = d['lat']; lon = d['lon']
    LA = np.repeat(lat, shp[1])[idx]; LO = np.tile(lon, shp[0])[idx]
    for k in range(1, n + 1):
        s = (labf == k) & m & (E > 0)
        a = A[s].sum() / 1e12
        if a < 0.4:
            continue
        w = A[s]
        x = (E / Eopt)[s]
        o = np.argsort(x); xm = x[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]
        print(age, k, f"{np.average(LA[s], weights=w):6.1f} {np.average(LO[s], weights=w):7.1f} {a:5.2f} "
              f"{np.average(d['z_z'][s], weights=w):6.0f} {np.average(E[s], weights=w):6.3f} {np.average(q[s], weights=w):6.0f} "
              f"{xm:6.2f} {w[(dsil - dsulf)[s] <= 0].sum() / w.sum():5.2f} "
              f"{np.sum((dsil * A)[s]) / np.sum((sil * A)[s]):6.2f} {np.sum(((dsil - dsulf) * A)[s]) / np.sum(((sil - sulf) * A)[s]):6.2f} "
              f"{np.sum(((sil - sulf) * A)[s]) / A[s].sum():7.3f} {np.sum(((sil - sulf) * A)[s]) / netg:6.3f}")

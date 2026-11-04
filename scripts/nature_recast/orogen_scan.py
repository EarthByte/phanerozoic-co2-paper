#!/usr/bin/env python3
"""Scan the adjusted surface for mountain belts and rank them by their position relative to the
local drawdown optimum. Used to choose the two end-member orogenic episodes of Fig. 4.

An exploratory survey behind the choice of the humid (Caledonian, 420 Ma) and arid (Central Pangaean, 300 Ma) belts
of Fig. 4b,c. At each age the land elevation of the main model (rock-chemistry-adjusted surface, 0.1-degree grid) is
smoothed with a 15 x 15-cell (1.5-degree) moving mean, sea counting as 0 m, and every connected region whose smoothed
elevation exceeds ZMIN is a candidate belt. For each candidate with at least 0.4 million km2 of eroding land, the
weathering laws of the main run (three regional parameter sets of Bufe et al. 2024 with the calibration amplitudes of
kin_bridge_s2, central estimate = mean of the three) give its position relative to the local optimum and its
erosion sensitivities.

Input: the land-only cache (nature_recast_cache.zip, build_cache.py) and data/nature_recast/kin_bridge_s2_amplitudes.txt.
Output: a table on standard output, one row per age and candidate belt, nothing written to disk. Columns:
    age (Ma), reg (label of the connected region), lat, lon (area-weighted centre, degrees), area_Mkm2 (eroding land,
    million km2), zmean (m), Emean (erosion rate, mm/yr), qmean (runoff, mm/yr), x_med (area-weighted median of
    E/E_opt), beyond (share of eroding area at or beyond the local optimum), S_sil and S_net (erosion sensitivities of
    the silicate sink and the net inorganic sink of the region), Fnet_density (net inorganic drawdown per unit area,
    mol C m-2 yr-1), share_of_global_net (share of the global net inorganic drawdown of that age).

    python3 orogen_scan.py                      # ZMIN = 1,500 m at 19 ages between 0 and 480 Ma
    python3 orogen_scan.py 2000 300 420         # ZMIN = 2,000 m at 300 and 420 Ma
"""
import sys, ast, numpy as np
from pathlib import Path
from scipy import ndimage
import weathering_core as wc
import run_weathering as rw
import _cache

import _config as cfg
OUT = cfg.OUT
# calibration of the main run: line 1 of the amplitudes file holds the silicate and sulfide amplitudes of each set
amp_txt = open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')
AMPS = ast.literal_eval(amp_txt[0])
fitsets = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}
qgrid = np.logspace(0, 4, 81)                                   # runoff 1-10,000 mm/yr for the E_opt table
eopt_q = rw.eopt_table(fitsets, AMPS, 'kinetic', qgrid)          # E_opt (mm/yr) against runoff, as in the main run
ZMIN = float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0       # m; smoothed elevation that outlines a belt
ages = [int(a) for a in sys.argv[2:]] or [0, 20, 50, 80, 100, 150, 200, 220, 240, 260, 280, 300, 320, 340, 360, 400, 420, 450, 480]
print('age reg lat lon area_Mkm2 zmean Emean qmean x_med beyond S_sil S_net Fnet_density share_of_global_net')
for age in ages:
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    # land mask, cell area (m2) and temperature (degrees C -> K) of the adjusted surface
    m = d['mz']; A = d['area'].astype(float); T = d['T'] + 273.15
    # runoff (mm/yr) = precipitation channel (m/yr) x runoff ratio; erosion rate (mm/yr) = NeuroLEM net erosion,
    # deposition set to zero, times the calibration factor 2.0
    q = d['rain'].astype(float) * 1000 * d['ratio']; E = np.where(m, np.maximum(-d['ero_z'].astype(float), 0), 0) * 2.0
    sil = dsil = sulf = dsulf = 0
    for r in rw.SETS:
        # per-cell silicate sink and sulfide source (mol C m-2 yr-1) and their derivatives d/d ln E, mean of the sets
        t = rw.region_terms(E, q, T, fitsets[r], 'kinetic')
        sil = sil + AMPS[r]['sil'] * t['sil'] / 3; dsil = dsil + AMPS[r]['sil'] * t['dsil'] / 3
        sulf = sulf + AMPS[r]['sulf'] * t['sulf'] / 3; dsulf = dsulf + AMPS[r]['sulf'] * t['dsulf'] / 3
    Eopt = np.interp(np.log(np.maximum(q, 1)), np.log(qgrid), eopt_q)
    netg = float(np.sum(((sil - sulf) * A)[m]))                 # global net inorganic drawdown (mol C/yr)
    # elevation on the full grid, sea and cells outside the cache at -9999 m
    z = np.full(shp[0] * shp[1], -9999.0); z[idx] = np.where(m, d['z_z'], -9999.0); z = z.reshape(shp)
    # 1.5-degree moving mean with sea at 0 m; connected regions above ZMIN are the candidate belts
    zs = ndimage.uniform_filter(np.where(z > -9000, z, 0), size=15, mode='wrap')
    lab, n = ndimage.label(zs > ZMIN)
    labf = lab.ravel()[idx]                                      # region label of each cache cell
    lat = d['lat']; lon = d['lon']
    LA = np.repeat(lat, shp[1])[idx]; LO = np.tile(lon, shp[0])[idx]
    for k in range(1, n + 1):
        s = (labf == k) & m & (E > 0)                            # eroding land of region k
        a = A[s].sum() / 1e12                                    # m2 -> million km2
        if a < 0.4:
            continue
        w = A[s]
        x = (E / Eopt)[s]
        o = np.argsort(x); xm = x[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)]     # area-weighted median
        # beyond: d(net inorganic)/d ln E <= 0; S = sum of derivatives / sum of fluxes over the region
        print(age, k, f"{np.average(LA[s], weights=w):6.1f} {np.average(LO[s], weights=w):7.1f} {a:5.2f} "
              f"{np.average(d['z_z'][s], weights=w):6.0f} {np.average(E[s], weights=w):6.3f} {np.average(q[s], weights=w):6.0f} "
              f"{xm:6.2f} {w[(dsil - dsulf)[s] <= 0].sum() / w.sum():5.2f} "
              f"{np.sum((dsil * A)[s]) / np.sum((sil * A)[s]):6.2f} {np.sum(((dsil - dsulf) * A)[s]) / np.sum(((sil - sulf) * A)[s]):6.2f} "
              f"{np.sum(((sil - sulf) * A)[s]) / A[s].sum():7.3f} {np.sum(((sil - sulf) * A)[s]) / netg:6.3f}")

#!/usr/bin/env python3
"""Share of eroding land beyond the local optimum, and erosion sensitivities, at fixed CO2 on each palaeogeography.
Climate at the prescribed CO2 comes from the per-cell pattern-scaling emulator (see run_reactivity.py).

Separates the effect of palaeogeography and relief from that of CO2 (Methods, "Land-surface reactivity
and the weathering feedback"; Supplementary Information; number audit). Adjusted surface, erosion at the
main-model calibration factor (2), three-set mean with the amplitudes of the main run (cellfields.py).

Output: data/nature_recast/beyond_vs_co2.csv, one row per age, with co2_ref (ppm) and for C = 280, 560 and
1120 ppm: beyond_<C> (area share of eroding land at or beyond the optimum of the net inorganic drawdown),
S_sil_<C> and S_net_<C> (global erosion sensitivities of the silicate and net inorganic sinks) and q_<C>
(area-weighted land runoff, mm/yr).

    python3 beyond_vs_co2.py
"""
import csv, numpy as np
import _config as cfg, run_weathering as rw, cellfields as cf
import _cache
SENS = np.load(cfg.SENS); slat = SENS['lat'].astype(float); slon = SENS['lon'].astype(float); sage = list(SENS['age'])
# main-run amplitudes; the E_opt table is loaded but not needed here, because the share beyond the optimum is taken
# from the sign of the local derivative of the net inorganic drawdown
amp = cf.amps(); eopt = cf.eopt(); rows = []
for age in range(0, 545, 5):
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    lat = d['lat'].astype(float); lon = d['lon'].astype(float)
    # nearest emulator cell of every cached land cell (longitude distance modulo 360)
    ilat = np.abs(slat[:, None] - lat[None, :]).argmin(0); ilon = np.abs(((slon[:, None] - lon[None, :]) + 180) % 360 - 180).argmin(0)
    ri, ci = np.unravel_index(idx, shp); k = sage.index(age); cref = float(SENS['co2_ref'][k])
    lam = SENS['lambda_T'][k][ilat[ri], ilon[ci]].astype(float); kap = SENS['kappa_P'][k][ilat[ri], ilon[ci]].astype(float)
    # adjusted surface: land mask, cell area (m2), erosion rate (mm/yr) at the main-model calibration factor
    m = d['mz']; A = d['area'].astype(float); E = np.where(m, np.maximum(-d['ero_z'].astype(float), 0), 0) * 2.0; ero = m & (E > 0)
    row = dict(age=age, co2_ref=cref)
    for C in (280.0, 560.0, 1120.0):
        # emulated temperature (K) and runoff (mm/yr): precipitation perturbed with the 10 mm/yr offset, times the
        # HadCM3L runoff ratio, which is held
        dbl = np.log2(C / cref); T = d['T'].astype(float) + 273.15 + lam * dbl
        q = np.maximum((d['rain'].astype(float) * 1000 + 10) * np.exp(kap * dbl) - 10, 0) * d['ratio'].astype(float)
        sil = dsil = sulf = dsulf = 0
        for r in rw.SETS:
            t = rw.region_terms(E, q, T, cf.FITS[r], 'kinetic')
            sil = sil + amp[r]['sil'] * t['sil'] / 3; dsil = dsil + amp[r]['sil'] * t['dsil'] / 3; sulf = sulf + amp[r]['sulf'] * t['sulf'] / 3; dsulf = dsulf + amp[r]['sulf'] * t['dsulf'] / 3
        row[f'beyond_{int(C)}'] = float(A[ero & ((dsil - dsulf) <= 0)].sum() / A[ero].sum())
        row[f'S_sil_{int(C)}'] = float(np.sum((dsil * A)[m]) / np.sum((sil * A)[m])); row[f'S_net_{int(C)}'] = float(np.sum(((dsil - dsulf) * A)[m]) / np.sum(((sil - sulf) * A)[m]))
        row[f'q_{int(C)}'] = float(np.sum((q * A)[m]) / A[m].sum())
    rows.append(row); print(row, flush=True)
with open(cfg.OUT / 'beyond_vs_co2.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator='\n'); w.writeheader(); w.writerows(rows)

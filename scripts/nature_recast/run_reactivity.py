#!/usr/bin/env python3
"""Land-surface reactivity: the global silicate (and net inorganic) weathering flux that each
palaeogeography and relief would deliver at a fixed atmospheric CO2.

Climate at the chosen CO2 is obtained per grid cell from the pattern-scaling emulator fitted to the
PhanDA HadCM3L prior ensemble (climate_uncertainty/phanda_emulator.py):
    T(x)     = T_ref(x) + lambda_T(x) * log2(C / C_ref)
    P'(x)    = (P_ref(x) + P0) * exp(kappa_P(x) * log2(C / C_ref)) - P0,   P0 = 10 mm/yr
with C_ref the CO2 of the simulation that supplied the forcing. The HadCM3 runoff ratio is held.
"""
import ast, csv, sys, numpy as np
from pathlib import Path
import weathering_core as wc
import run_weathering as rw
import _cache

import _config as cfg
SENS = np.load(cfg.SENS)
OUT = cfg.OUT
wc.check_form(OUT / 'kin_bridge_s2_amplitudes.txt')
AMPS = ast.literal_eval(open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[0])
CO2 = [280.0, 560.0, 1120.0]
fitsets = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}
slat = SENS['lat'].astype(float); slon = SENS['lon'].astype(float); sage = list(SENS['age'])


def nearest_index(lat, lon):
    i = np.abs(slat[:, None] - lat[None, :]).argmin(0) if slat.size * lat.size < 5e6 else None
    return i


rows = []
ages = [int(a) for a in sys.argv[1:]] or list(range(0, 545, 5))
for age in ages:
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    lat = d['lat'].astype(float); lon = d['lon'].astype(float)
    ilat = np.abs(slat[:, None] - lat[None, :]).argmin(0)
    dl = np.abs(((slon[:, None] - lon[None, :]) + 180) % 360 - 180); ilon = dl.argmin(0)
    r_i, c_i = np.unravel_index(idx, shp)
    k = sage.index(age)
    lam = SENS['lambda_T'][k][ilat[r_i], ilon[c_i]].astype(float)
    kap = SENS['kappa_P'][k][ilat[r_i], ilon[c_i]].astype(float)
    cref = float(SENS['co2_ref'][k])
    A = d['area'].astype(float); T0 = d['T'].astype(float) + 273.15; P0 = d['rain'].astype(float) * 1000.0
    ratio = d['ratio'].astype(float)
    row = dict(age=age, co2_ref=cref)
    for case, ek, mk in (('adj', 'ero_z', 'mz'), ('ref', 'ero_s', 'ms')):
        m = d[mk]; E = np.where(m, np.maximum(-d[ek].astype(float), 0.0), 0.0) * 2.0
        for n_c, C in enumerate([cref] + CO2):      # the first entry is the CO2 of the simulation itself; a prescribed level equal to it is still evaluated under its own name
            dbl = np.log2(C / cref)
            T = T0 + lam * dbl
            P = np.maximum((P0 + 10.0) * np.exp(kap * dbl) - 10.0, 0.0)
            q = P * ratio
            sil = sulf = 0.0
            for r in rw.SETS:
                t = rw.region_terms(E, q, T, fitsets[r], 'kinetic')
                sil += AMPS[r]['sil'] * float(np.sum((t['sil'] * A)[m])) * wc.GTC / 3
                sulf += AMPS[r]['sulf'] * float(np.sum((t['sulf'] * A)[m])) * wc.GTC / 3
            tag = 'own' if n_c == 0 else f'{int(C)}'
            row[f'F_sil_{tag}_{case}'] = sil; row[f'F_sulf_{tag}_{case}'] = sulf
            if case == 'adj':
                row[f'q_land_{tag}'] = float(np.sum((q * A)[m]) / A[m].sum()); row[f'T_land_{tag}'] = float(np.sum((T * A)[m]) / A[m].sum()) - 273.15
    rows.append(row)
    print(age, cref, {k: round(v, 4) for k, v in row.items() if k.startswith('F_sil') and k.endswith('adj')}, flush=True)
with open(OUT / 'reactivity_fixed_co2.csv', 'w', newline='') as fh:
    keys = sorted({k for r_ in rows for k in r_}, key=lambda k: (k != 'age', k)); w = csv.DictWriter(fh, fieldnames=keys, restval='', lineterminator='\n'); w.writeheader(); w.writerows(rows)

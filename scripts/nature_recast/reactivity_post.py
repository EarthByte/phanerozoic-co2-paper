#!/usr/bin/env python3
"""Reactivity, intrinsic feedback strength and balance pCO2 from reactivity_fixed_co2.csv."""
import numpy as np, pandas as pd
from _load import OUT, series, degassing
import _config as cfg
r = pd.read_csv(OUT / 'reactivity_fixed_co2.csv').set_index('age')
a = series()
t = r.index.values.astype(float)
# intrinsic feedback strength of the simulations, n = dlnF/dlnC between 280 and 1120 ppm
r['n_model'] = np.log(r.F_sil_1120_adj / r.F_sil_280_adj) / np.log(4.0)
# solar forcing (Gough 1981), W m-2 relative to present, and its CO2 equivalent
L = 1.0 / (1.0 + 0.4 * (t / 1000.0) / 4.57)
dFsun = (L - 1.0) * 1361.0 / 4.0 * (1 - 0.29)
r['dF_sun'] = dFsun
# reactivity at fixed total forcing: CO2 raised to compensate the fainter Sun, scaled with the intrinsic n
Cfix = 560.0 * np.exp(-dFsun / 5.35)
r['C_fix'] = Cfix
r['F_fixforcing'] = r.F_sil_560_adj * (Cfix / 560.0) ** r.n_model
r['k_fixco2'] = r.F_sil_560_adj / r.F_sil_560_adj.loc[0]
r['k_fixforcing'] = r.F_fixforcing / r.F_fixforcing.loc[0]
D = degassing()
for k in ('min', 'mean', 'max'):
    d5 = np.array([D[k].loc[max(0, a0 - 2):a0 + 2].mean() for a0 in t])
    r[f'D_{k}'] = d5
r['D_rel'] = r.D_mean / r.D_mean.loc[0]
G = a.Gamma_sil_adj.reindex(r.index); g = (G / G.loc[0]).values
for n in (0.2, 0.4, 0.6):
    ne = n * g
    r[f'pco2_constD_n{n}'] = 280.0 * np.exp(-dFsun / 5.35) * (1.0 / r.k_fixforcing) ** (1.0 / ne)
    r[f'pco2_D_n{n}'] = 280.0 * np.exp(-dFsun / 5.35) * (r.D_rel / r.k_fixforcing) ** (1.0 / ne)
r['Gamma_rel'] = g
r.to_csv(OUT / 'reactivity_summary.csv')
pd.set_option('display.width', 250)
print(r.loc[::20, ['co2_ref', 'F_sil_own_adj', 'F_sil_560_adj', 'n_model', 'dF_sun', 'k_fixco2', 'k_fixforcing', 'D_rel', 'pco2_constD_n0.4', 'pco2_D_n0.4', 'pco2_D_n0.6']].round(3))
print('n_model range', r.n_model.min(), r.n_model.max(), r.n_model.mean())
print('k range', r.k_fixforcing.min(), r.k_fixforcing.idxmin(), r.k_fixforcing.max(), r.k_fixforcing.idxmax())
# proxies
f = np.load(cfg.PCO2['foster']); j = np.load(cfg.PCO2['judd'])
def at(x, ages): return np.interp(ages, x[:, 0], x[:, 1])
for nm, x, lim in (('Foster', f, 420), ('Judd', j, 485)):
    sel = t <= lim
    for col in ('k_fixforcing', 'pco2_constD_n0.4', 'pco2_D_n0.4'):
        y = np.log(r[col].values[sel]); p = np.log(at(x, t[sel]))
        print(nm, col, 'r =', round(np.corrcoef(y, p)[0, 1], 3))

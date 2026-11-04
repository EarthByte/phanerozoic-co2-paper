#!/usr/bin/env python3
"""Reactivity, intrinsic feedback strength and balance pCO2 from reactivity_fixed_co2.csv.

Second step of the reactivity calculation (Methods, "Land-surface reactivity and the weathering
feedback"; Fig. 5, Extended Data Fig. 9, number audit). Reads the fixed-CO2 fluxes written by
run_reactivity.py, the main-model table (kin_bridge_s2_ens.csv) and the gross solid-Earth degassing of
Muller et al. (2024) (extract_degassing.py), and writes data/nature_recast/reactivity_summary.csv with
every column of the input plus, per age:

  n_model        intrinsic feedback strength n = dlnF_sil/dlnC of the simulations, 280 to 1120 ppm (-)
  dF_sun         solar forcing relative to the present (Gough, 1981), W m-2
  C_fix          CO2 (ppm) that offsets dF_sun relative to 560 ppm, with a CO2 forcing of 5.35 ln(C/C0) W m-2
  F_fixforcing   silicate sink at constant total forcing, F_sil(560 ppm) x (C_fix/560)^n_model, Gt C/yr
  k_fixco2, k_fixforcing   silicate sink at 560 ppm and at constant forcing, relative to the present (reactivity)
  D_min, D_mean, D_max     gross degassing averaged over the 5-Myr window centred on the age, Mt C/yr; D_rel = D_mean
                 relative to the present
  Gamma_rel      flux-weighted kinetically limited share of the silicate sink (main model) relative to the present
  pco2_constD_n<n>, pco2_D_n<n>   CO2 (ppm) at which the sink balances constant or reconstructed degassing, for
                 present-day feedback strengths n = 0.2, 0.4, 0.6 scaled at each age by Gamma_rel

It also prints a summary and the correlation of the logarithms of these series with the proxy CO2 records
of Foster et al. (2017, to 420 Ma) and Judd et al. (2024, to 485 Ma).

    python3 reactivity_post.py
"""
import numpy as np, pandas as pd
from _load import OUT, series, degassing
import _config as cfg
r = pd.read_csv(OUT / 'reactivity_fixed_co2.csv').set_index('age')
a = series()
t = r.index.values.astype(float)
# intrinsic feedback strength of the simulations, n = dlnF/dlnC between 280 and 1120 ppm
r['n_model'] = np.log(r.F_sil_1120_adj / r.F_sil_280_adj) / np.log(4.0)
# solar forcing (Gough 1981), W m-2 relative to present, and its CO2 equivalent: L is the solar luminosity relative to
# today, L = 1 / (1 + 0.4 t/4.57 Gyr) with t the age in Gyr; forcing = (L - 1) x S0/4 x (1 - albedo), S0 = 1361 W m-2,
# planetary albedo 0.29
L = 1.0 / (1.0 + 0.4 * (t / 1000.0) / 4.57)
dFsun = (L - 1.0) * 1361.0 / 4.0 * (1 - 0.29)
r['dF_sun'] = dFsun
# reactivity at fixed total forcing: CO2 raised to compensate the fainter Sun, scaled with the intrinsic n
# (5.35 W m-2 per e-fold of CO2, the logarithmic CO2 forcing of Myhre et al. (1998))
Cfix = 560.0 * np.exp(-dFsun / 5.35)
r['C_fix'] = Cfix
r['F_fixforcing'] = r.F_sil_560_adj * (Cfix / 560.0) ** r.n_model
r['k_fixco2'] = r.F_sil_560_adj / r.F_sil_560_adj.loc[0]
r['k_fixforcing'] = r.F_fixforcing / r.F_fixforcing.loc[0]
# degassing at 1-Myr steps, averaged over age - 2 to age + 2 Myr (5 values; label-based slicing is inclusive)
D = degassing()
for k in ('min', 'mean', 'max'):
    d5 = np.array([D[k].loc[max(0, a0 - 2):a0 + 2].mean() for a0 in t])
    r[f'D_{k}'] = d5
r['D_rel'] = r.D_mean / r.D_mean.loc[0]
# the feedback acts through the kinetically limited part of the sink, so a present-day strength n is scaled at each
# age by the kinetic share relative to today; the balance CO2 is then the solar-compensated 280 ppm times
# (relative degassing / relative reactivity)^(1/n_effective)
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
# proxies: two-column arrays (age in Ma, CO2 in ppm), interpolated to the model ages; correlations of logarithms over
# the span each record covers
f = np.load(cfg.PCO2['foster']); j = np.load(cfg.PCO2['judd'])
def at(x, ages): return np.interp(ages, x[:, 0], x[:, 1])
for nm, x, lim in (('Foster', f, 420), ('Judd', j, 485)):
    sel = t <= lim
    for col in ('k_fixforcing', 'pco2_constD_n0.4', 'pco2_D_n0.4'):
        y = np.log(r[col].values[sel]); p = np.log(at(x, t[sel]))
        print(nm, col, 'r =', round(np.corrcoef(y, p)[0, 1], 3))

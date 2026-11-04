#!/usr/bin/env python3
"""Recompute every number quoted in the Nature manuscript from the result tables.

The number audit of the paper: each quantity quoted in the main text, Methods and Supplementary Information is
recomputed from the archived result tables and written, with a short description, to one markdown table, so that
every number in the text can be checked against the code that produced it. No per-cell fields are needed.

Units follow the result tables: fluxes F_* in Gt C/yr (multiplied by 1e3 where Mt C/yr are quoted), erosion flux
D_Gt in Gt/yr, erosion rates in mm/yr, runoff in mm/yr, land areas in 10^6 km2, degassing (D_mean, D_min, D_max)
in Mt C/yr, shares as fractions. Suffix _adj is the rock-chemistry-adjusted surface (main model), _ref the
PALEOMAP reference surface and _adjc the adjusted surface on the land common to both.

Inputs (data/nature_recast/ unless stated):
  required  kin_bridge_s2_{ens,Taiwan,Sichuan,NewZealand}.csv (main model), highland_stats.csv,
            reactivity_summary.csv, beyond_vs_co2.csv, data/foster.npy and data/judd.npy (CO2 histories, age and
            pCO2 columns), and the lithium-isotope compilation in data/proxies/
  optional  (each block is skipped when its table is missing) the alternative runs kin_bridge_s1, kin_bridge_s4,
            out_bridge_s2, kin_coef_s2, out_coef_s2; the widened distributions dst_k{1.3,1.5,1.8}_ens.csv; the
            uncertainty experiments (ensemble_sensitivity_*, climate_sensitivity_*, gospl_*_sensitivity.csv); the
            dynamic-topography tables (dyntopo_*.csv); orogen_sensitivity_events.csv;
            palaeoelevation_validation.csv; and the ensemble member table in the data archive
Output:     data/nature_recast/number_audit.md (also printed)

    python3 verify_numbers.py
"""
import numpy as np, pandas as pd
from _load import series, OUT, SETS
import _config as cfg
# main model (mean of the three parameter sets) and each parameter set separately; all indexed by age (Ma)
a = series(); S = {s: series('kin_bridge_s2', s) for s in SETS}
h = pd.read_csv(OUT / 'highland_stats.csv').set_index('age'); r = pd.read_csv(OUT / 'reactivity_summary.csv').set_index('age')
L = []                       # (description, value) rows of the audit table
# put: add one row to the audit; rg: minimum, maximum (with their ages), mean and present-day value of column c of the main model
def put(claim, val): L.append((claim, val))
def rg(c, f='{:.3f}'): v = a[c]; return f"min {f.format(v.min())} @{v.idxmin()} Ma, max {f.format(v.max())} @{v.idxmax()} Ma, mean {f.format(v.mean())}, present {f.format(v.loc[0])}"
put('S silicate (adjusted, central)', rg('S_sil_adj')); put('S net inorganic', rg('S_net_adj'))
put('buffering capacity silicate (1-S) range', f"{1 - a.S_sil_adj.max():.2f}-{1 - a.S_sil_adj.min():.2f}")
put('share of eroding land at/beyond optimum', rg('beyond_frac_eroding_adj'))
put('first age (going forward in time) with share >= 0.50', int(a.index[a.beyond_frac_eroding_adj >= 0.5].max()))   # the oldest such age
b = a.beyond_frac_eroding_adj; put('share range 340-70 Ma', f"{b.loc[70:340].min():.3f}-{b.loc[70:340].max():.3f}"); put('share range 540-420 Ma', f"{b.loc[420:540].min():.3f}-{b.loc[420:540].max():.3f}")
put('median E/Eopt 540-420 Ma; present', f"{a.x_p50_adj.loc[420:540].min():.2f}-{a.x_p50_adj.loc[420:540].max():.2f}; {a.x_p50_adj.loc[0]:.2f}")
put('ages with S_net < 0', str(list(a.index[a.S_net_adj < 0])))
put('ages with S_sil <= present value', str(list(a.index[a.S_sil_adj <= a.S_sil_adj.loc[0] + 0.002])))   # 0.002 tolerance for rounding
put('denudation Gt/yr', rg('D_Gt_adj', '{:.1f}')); put('denudation max/min', f"{a.D_Gt_adj.max() / a.D_Gt_adj.min():.2f}")
# ages of minimum and maximum erosion flux: how much of the change comes from eroding area and how much from the mean rate
_i0, _i1 = a.D_Gt_adj.idxmin(), a.D_Gt_adj.idxmax(); put('denudation min->max: eroding area ratio; mean rate of eroding land ratio', f"{_i0}->{_i1} Ma: {a.eroding_Mkm2_adj[_i1] / a.eroding_Mkm2_adj[_i0]:.2f}; {a.E_mean_eroding_mm_adj[_i1] / a.E_mean_eroding_mm_adj[_i0]:.2f}")
put('silicate sink GtC/yr', rg('F_sil_adj')); put('silicate max/min', f"{a.F_sil_adj.max() / a.F_sil_adj.min():.2f}")
put('net inorganic sink', rg('F_net_adj')); put('mean runoff mm/yr', rg('q_mean_mm_adj', '{:.0f}'))
put('median E range; p95 range', f"{a.E_p50_adj.min():.3f}-{a.E_p50_adj.max():.3f}; {a.E_p95_adj.min():.2f}-{a.E_p95_adj.max():.2f}")
put('mean local optimum mm/yr', rg('Eopt_mean_mm_adj')); put('modern mean erosion mm/yr, denudation Gt/yr', f"{a.E_mean_mm_adj.loc[0]:.3f}, {a.D_Gt_adj.loc[0]:.1f}")
put('kinetically limited share', rg('Gamma_sil_adj')); put('weathering intensity', rg('WI_adj'))
put('WI change 65->0 Ma', f"{100 * (a.WI_adj.loc[0] / a.WI_adj.loc[65] - 1):.0f}%")
put('d7Li_riv Ordovician (485-445) mean; Permian-Jurassic (300-145) mean; min; max', f"{a.d7Li_riv_adj.loc[445:485].mean():.1f}; {a.d7Li_riv_adj.loc[145:300].mean():.1f}; {a.d7Li_riv_adj.min():.1f}; {a.d7Li_riv_adj.max():.1f}")
# runoff volume (mean runoff x land area) and least-squares fits of ln F_sil on ln runoff volume and ln erosion flux
Q = a.q_mean_mm_adj * a.land_Mkm2_adj; y = np.log(a.F_sil_adj)
# r2: coefficient of determination and coefficients (intercept first) of an ordinary least-squares fit of y on X
def r2(X): X = np.c_[np.ones(len(y)), X]; bb = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - ((y - X @ bb) ** 2).sum() / ((y - y.mean()) ** 2).sum(), bb
put('R2 ln F_sil ~ ln runoff volume', f"{r2(np.log(Q))[0]:.3f}"); put('R2 ln F_sil ~ ln denudation', f"{r2(np.log(a.D_Gt_adj))[0]:.3f}")
rr, bb = r2(np.c_[np.log(Q), np.log(a.D_Gt_adj)]); put('joint fit R2, exponents (runoff, denudation)', f"{rr:.3f}, {bb[1]:.2f}, {bb[2]:.2f}")
# Pearson correlations of the silicate sink with gross degassing and with the erosion flux, all ages and since 300 Ma
# (masks built from the index of a are applied to r, so the two tables must list the same ages in the same order)
for lim in (540, 300):
    s = a.index <= lim
    put(f'r(F_sil, degassing) 0-{lim} Ma; r(F_sil, denudation)', f"{np.corrcoef(a.F_sil_adj[s], r.D_mean[s])[0, 1]:.2f}; {np.corrcoef(a.F_sil_adj[s], a.D_Gt_adj[s])[0, 1]:.2f}")
# share beyond the optimum against the logarithm of two proxy CO2 histories (Foster et al. 2017, 0-420 Ma; Judd et al.
# 2024, 0-485 Ma), interpolated to the model ages
for nm, key, lim in (('Foster', 'foster', 420), ('Judd', 'judd', 485)):
    x = np.load(cfg.PCO2[key]); s = a.index <= lim; p = np.log(np.interp(a.index[s], x[:, 0], x[:, 1]))
    put(f'r(share beyond, ln pCO2 {nm})', f"{np.corrcoef(a.beyond_frac_eroding_adj[s], p)[0, 1]:.2f}")
# the two palaeoelevation histories: adjusted against PALEOMAP, on own land and on common land
for c in ('D_Gt', 'F_sil', 'F_net'):
    put(f'{c}: adjusted/reference Phanerozoic mean; common cells; max ratio @age', f"{a[c + '_adj'].mean() / a[c + '_ref'].mean():.3f}; {a[c + '_adjc'].mean() / a[c + '_ref'].mean():.3f}; {(a[c + '_adj'] / a[c + '_ref']).max():.2f} @{(a[c + '_adj'] / a[c + '_ref']).idxmax()}")
# realised sensitivity: relative change of the time-mean flux divided by the relative change of the time-mean
# erosion flux between the two surfaces (common land)
dD = a.D_Gt_adjc.mean() / a.D_Gt_ref.mean() - 1
put('realised sensitivity (common cells): silicate; net', f"{(a.F_sil_adjc.mean() / a.F_sil_ref.mean() - 1) / dD:.2f}; {(a.F_net_adjc.mean() / a.F_net_ref.mean() - 1) / dD:.2f}")
# erosion-rate distribution widened to the cosmogenic spread (distribution_tests.py; Methods, main text)
for _k in ('1.3', '1.5', '1.8'):
    _p = OUT / f'dst_k{_k}_ens.csv'
    if _p.exists():
        # _o orders ages from oldest to youngest, so np.diff gives the change over each 5-Myr step forward in time
        _d = pd.read_csv(_p).set_index('age').sort_index(); _Q = _d.q_mean_mm_adj * _d.land_Mkm2_adj; _o = np.sort(_d.index.values)[::-1]
        _sD, _sF, _sQ = (np.diff(np.log(v.loc[_o].values)) for v in (_d.D_Gt_adj, _d.F_sil_adj, _Q))
        _ch = lambda v, t0, t1: f"{100 * (v.loc[t1] / v.loc[t0] - 1):+.0f}%"
        put(f'widened k={_k}: S_sil mean (min-max); S_net mean, ages S_net<0; share mean, 480, 0 Ma',
            f"{_d.S_sil_adj.mean():.3f} ({_d.S_sil_adj.min():.3f}-{_d.S_sil_adj.max():.3f}); {_d.S_net_adj.mean():.3f}, {int((_d.S_net_adj < 0).sum())}; "
            f"{_d.beyond_frac_eroding_adj.mean():.3f}, {_d.beyond_frac_eroding_adj.loc[480]:.3f}, {_d.beyond_frac_eroding_adj.loc[0]:.3f}")
        put(f'widened k={_k}: F_sil mean / main; change 480->0, 350->285, 190->105; step R2 runoff, erosion',
            f"{_d.F_sil_adj.mean() / a.F_sil_adj.mean():.3f}; {_ch(_d.F_sil_adj, 480, 0)}, {_ch(_d.F_sil_adj, 350, 285)}, {_ch(_d.F_sil_adj, 190, 105)}; "
            f"{np.corrcoef(_sQ, _sF)[0, 1] ** 2:.2f}, {np.corrcoef(_sD, _sF)[0, 1] ** 2:.2f}")
# net sink with the organic pathways (Fig. 5a dashed line, from 400 Ma)
tot = a.F_net_adj - a.F_ocp_adj + a.F_ocb_adj; s400 = a.index <= 400     # biospheric burial counted from 400 Ma (land plants)
put('net sink with organic pathways / net inorganic sink, 0-400 Ma: min; max; present', f"{(tot / a.F_net_adj)[s400].min():.3f}; {(tot / a.F_net_adj)[s400].max():.3f}; {(tot / a.F_net_adj).loc[0]:.3f}")
# sinks in Gt C/yr x 1e3 against degassing in Mt C/yr
put('sink / degassing 0-400 Ma mean: with organic pathways; net inorganic; silicate', f"{(1e3 * tot[s400] / r.D_mean[s400]).mean():.2f}; {(1e3 * a.F_net_adj[s400] / r.D_mean[s400]).mean():.2f}; {(1e3 * a.F_sil_adj[s400] / r.D_mean[s400]).mean():.2f}")
# actual changes through time (summary paragraph, runoff section)
QV = a.q_mean_mm_adj * a.land_Mkm2_adj
for t0, t1 in ((480, 0), (350, 285), (190, 105), (50, 0)):
    ch = lambda v: f"{100 * (v.loc[t1] / v.loc[t0] - 1):+.0f}%"
    put(f'change {t0}->{t1} Ma: erosion flux; silicate sink; net sink; runoff volume; share beyond (ratio)',
        f"{a.D_Gt_adj.loc[t0]:.1f}->{a.D_Gt_adj.loc[t1]:.1f} Gt/yr ({ch(a.D_Gt_adj)}); {ch(a.F_sil_adj)}; {ch(a.F_net_adj)}; {ch(QV)}; "
        f"{a.beyond_frac_eroding_adj.loc[t1] / a.beyond_frac_eroding_adj.loc[t0]:.2f}")
# changes over the 108 five-million-year steps, oldest to youngest: how much of the variance of dlnF_sil each driver explains
o = np.sort(a.index.values)[::-1]
sD, sF, sQ = (np.diff(np.log(v.loc[o].values)) for v in (a.D_Gt_adj, a.F_sil_adj, QV))
put('5-Myr steps: n; R2 dlnF_sil~dlnQ; R2 dlnF_sil~dlnD; share of steps with D and F_sil of opposite sign',
    f"{len(sD)}; {np.corrcoef(sQ, sF)[0, 1] ** 2:.2f}; {np.corrcoef(sD, sF)[0, 1] ** 2:.2f}; {np.mean(np.sign(sD) != np.sign(sF)):.2f}")
put('uniform-perturbation S for reference surface: silicate; net (means)', f"{a.S_sil_ref.mean():.2f}; {a.S_net_ref.mean():.2f}")
# the three parameter sets and the alternative runs (Extended Data Figs 4, 5)
for s in SETS: put(f'parameter set {s}: mean S_sil; max S_sil; beyond range', f"{S[s].S_sil_adj.mean():.2f}; {S[s].S_sil_adj.max():.2f}; {S[s].beyond_frac_eroding_adj.min():.2f}-{S[s].beyond_frac_eroding_adj.max():.2f}")
for c in ('kin_bridge_s1', 'kin_bridge_s4', 'out_bridge_s2', 'kin_coef_s2', 'out_coef_s2'):
    try:
        x = series(c); put(f'config {c}: S_sil mean/min/max; beyond mean/min/max', f"{x.S_sil_adj.mean():.2f}/{x.S_sil_adj.min():.2f}/{x.S_sil_adj.max():.2f}; {x.beyond_frac_eroding_adj.mean():.2f}/{x.beyond_frac_eroding_adj.min():.2f}/{x.beyond_frac_eroding_adj.max():.2f}")
        allS = [series(c, s).S_sil_adj.max() for s in SETS]; put(f'config {c}: max S_sil over parameter sets', f"{max(allS):.2f}")
    except FileNotFoundError: put(f'config {c}', 'not run yet')
# organic-carbon pathways (Extended Data Fig. 7): sensitivities normalised by the silicate sink
put('OC: S(net+OCpetro)/F_sil', rg('S_net_oc_rel_sil_adj')); put('OC: beyond share incl. OCpetro', rg('beyond_oc_frac_eroding_adj'))
put('OC: burial derivative / petro derivative (present; mean)', f"{a.dF_ocb_adj.loc[0] / a.dF_ocp_adj.loc[0]:.2f}; {(a.dF_ocb_adj / a.dF_ocp_adj).mean():.2f}")
put('OC: ages with S_total > 0', str(list(a.index[a.S_total_rel_sil_adj > 0])))
# land above 1,500 m (highland_stats.py; Extended Data Fig. 8)
for c in ('high_share_eroding', 'high_share_denudation', 'high_share_net', 'high_beyond', 'low_beyond', 'high_S_net'):
    put(f'mountains {c}', f"min {h[c].min():.3f}, max {h[c].max():.3f}, mean {h[c].mean():.3f}")
# Mt C/yr per 10^6 km2 = t C km-2 yr-1
put('continental mean net drawdown tC/km2/yr at 420, 300 Ma', f"{1e3 * a.F_net_adj.loc[420] / a.land_Mkm2_adj.loc[420]:.2f}, {1e3 * a.F_net_adj.loc[300] / a.land_Mkm2_adj.loc[300]:.2f}")
put('Cenozoic: denudation 40 Ma, 0 Ma; runoff 40, 50, 0 Ma', f"{a.D_Gt_adj.loc[40]:.1f}, {a.D_Gt_adj.loc[0]:.1f}; {a.q_mean_mm_adj.loc[40]:.0f}, {a.q_mean_mm_adj.loc[50]:.0f}, {a.q_mean_mm_adj.loc[0]:.0f}")
put('mean erosion rate of eroding land 50-0 Ma', f"{a.E_mean_eroding_mm_adj.loc[0:50].min():.3f}-{a.E_mean_eroding_mm_adj.loc[0:50].max():.3f}")
# reactivity and the weathering feedback (reactivity_post.py, beyond_vs_co2.py; Extended Data Fig. 9)
put('intrinsic feedback n (model)', f"{r.n_model.min():.2f} to {r.n_model.max():.2f}, mean {r.n_model.mean():.2f}")
put('balance pCO2 (n=0.4), 5-95%', f"{np.percentile(r['pco2_D_n0.4'], 5):.0f}-{np.percentile(r['pco2_D_n0.4'], 95):.0f} ppm")
b = pd.read_csv(OUT / 'beyond_vs_co2.csv').set_index('age')
put('share beyond at constant 560 ppm: 480 Ma; present; mean change per CO2 doubling (points)', f"{b.beyond_560.loc[480]:.3f}; {b.beyond_560.loc[0]:.3f}; {100 * (b.beyond_560 - b.beyond_280).mean():.2f} / {100 * (b.beyond_1120 - b.beyond_560).mean():.2f}")
put('r(mountain net drawdown density, mountain runoff); r(..., mountain erosion rate)', f"{np.corrcoef(h.high_net_density, h.high_q)[0, 1]:.2f}; {np.corrcoef(h.high_net_density, h.high_E)[0, 1]:.2f}")
put('eroding share of land: min, max, mean', f"{(a.eroding_Mkm2_adj / a.land_Mkm2_adj).min():.2f}, {(a.eroding_Mkm2_adj / a.land_Mkm2_adj).max():.2f}, {(a.eroding_Mkm2_adj / a.land_Mkm2_adj).mean():.2f}")
# effective number of independent samples of a correlation between two autocorrelated series (Bretherton et al. 1999):
# N (1 - r1 r2) / (1 + r1 r2), with r1, r2 the lag-one autocorrelations; returns (N_eff, r1, r2)
def neff(x, y):
    x = np.asarray(x); y = np.asarray(y); r1 = np.corrcoef(x[:-1], x[1:])[0, 1]; r2 = np.corrcoef(y[:-1], y[1:])[0, 1]; return len(x) * (1 - r1 * r2) / (1 + r1 * r2), r1, r2
for lim in (540, 300):
    s_ = a.index <= lim; ne, r1, r2 = neff(a.F_sil_adj[s_], r.D_mean[s_]); put(f'effective sample size, F_sil vs degassing, 0-{lim} Ma (lag-1 autocorrelations)', f"{ne:.1f} ({r1:.2f}, {r2:.2f})")
put('sink / degassing: mean 540-300 Ma; mean 300-0 Ma', f"{(1e3 * a.F_sil_adj / r.D_mean).loc[300:540].mean():.2f}; {(1e3 * a.F_sil_adj / r.D_mean).loc[0:300].mean():.2f}")
_rs = (a.F_sil_adj / a.F_sil_adj.loc[0]) / (r.D_mean / r.D_mean.loc[0])      # Fig. 5a compares histories relative to the present
put('sink / degassing, each relative to present: mean 0-300 Ma; 300-540 Ma; present gross degassing (mean, min-max) Mt C/yr', f"{_rs.loc[0:300].mean():.2f}; {_rs.loc[300:540].mean():.2f}; {r.D_mean.loc[0]:.1f} ({r.D_min.loc[0]:.0f}-{r.D_max.loc[0]:.0f})")
put('raw (unscaled) modern mean erosion mm/yr', f"{a.E_mean_mm_adj.loc[0] / 2:.4f}")      # main model uses erosion factor 2
# F scales as C^n, so a doubling multiplies the sink by 2^n
put('sink change per CO2 doubling (from n)', f"{100 * (2 ** r.n_model.min() - 1):.1f}% to {100 * (2 ** r.n_model.max() - 1):.1f}%")
put('weathering intensity ranges: Cm-O (540-445); D-J (415-145); K (140-70)', f"{a.WI_adj.loc[445:540].min():.3f}-{a.WI_adj.loc[445:540].max():.3f}; {a.WI_adj.loc[145:415].min():.3f}-{a.WI_adj.loc[145:415].max():.3f}; {a.WI_adj.loc[70:140].min():.3f}-{a.WI_adj.loc[70:140].max():.3f}")
put('reactivity at constant forcing, relative to present', f"{r.k_fixforcing.min():.2f}-{r.k_fixforcing.max():.2f}")
# uncertainty experiments (elevation ensemble, climate states, direct and published goSPL; Extended Data Fig. 6)
def _rd(n):
    # read an optional table from data/nature_recast/; None if it has not been written
    p = OUT / n
    return pd.read_csv(p) if p.exists() else None
eh = _rd('ensemble_sensitivity_headline.csv'); ch = _rd('climate_sensitivity_headline.csv'); gv = _rd('gospl_validation_sensitivity.csv'); gp = _rd('gospl_published_reference_sensitivity.csv'); es = _rd('ensemble_sensitivity_age_summary.csv'); cb = _rd('climate_sensitivity_bands.csv')
if eh is not None:
    q = lambda m, st: eh[(eh.metric == m) & (eh.statistic == st)].iloc[0]     # one row of the headline table: metric, statistic
    put('elevation ensemble: time-mean S_sil p05-p95; share beyond p05-p95', f"{q('S_sil','time_mean').p05:.3f}-{q('S_sil','time_mean').p95:.3f}; {q('beyond_frac_eroding','time_mean').p05:.3f}-{q('beyond_frac_eroding','time_mean').p95:.3f}")
    put('elevation ensemble: max single-age 5-95% width of S_sil; of share beyond', f"{(es.S_sil_p95 - es.S_sil_p05).max():.3f}; {(es.beyond_frac_eroding_p95 - es.beyond_frac_eroding_p05).max():.3f}")
    put('elevation ensemble: members with higher share beyond today than at 480 Ma', f"{int(eh[eh.metric == 'members_with_beyond_share_higher_today_than_at_480Ma'].iloc[0]['mean'])}")
if ch is not None:
    q = lambda m, st: ch[(ch.metric == m) & (ch.statistic == st) & (ch.surface == 'adj')].iloc[0]
    put('climate states: time-mean S_sil p05-p95; share beyond p05-p95', f"{q('S_sil','time_mean').p05:.3f}-{q('S_sil','time_mean').p95:.3f}; {q('beyond_frac_eroding','time_mean').p05:.3f}-{q('beyond_frac_eroding','time_mean').p95:.3f}")
    put('climate states: median single-age 5-95% width of S_sil; share beyond; adj/ref F_sil ratio', f"{q('S_sil','p05_p95_width_median_over_ages').p50:.3f}; {q('beyond_frac_eroding','p05_p95_width_median_over_ages').p50:.3f}; {ch[(ch.metric == 'ratio_F_sil') & (ch.statistic == 'p05_p95_width_median_over_ages')].iloc[0].p50:.3f}")
if gv is not None:
    piv = lambda m, c: gv[(gv.metric == m) & (gv.case == c)].set_index('age')  # case 'central': adjusted surface; 'reference': PALEOMAP
    for m in ('S_sil', 'beyond_frac_eroding', 'D_Gt', 'F_sil'):
        g = gv[gv.metric == m]
        put(f'direct goSPL minus NeuroLEM, {m}: median; range', f"{g.difference.median():+.3f}; {g.difference.min():+.3f} to {g.difference.max():+.3f}")
    d = piv('F_sil', 'central'); r_ = piv('F_sil', 'reference')
    put('direct goSPL: adj/ref F_sil ratio, goSPL vs NeuroLEM, max abs relative difference', f"{((d.goSPL / r_.goSPL) / (d.NeuroLEM / r_.NeuroLEM) - 1).abs().max():.3f}")
    put('direct goSPL: adj minus ref S_sil per age', '; '.join(f"{int(a)}: {v:+.3f}" for a, v in (piv('S_sil', 'central').goSPL - piv('S_sil', 'reference').goSPL).items()))
    put('direct goSPL: S_sil adj at 220, 240 (goSPL / NeuroLEM); max goSPL S_sil any case', f"{piv('S_sil','central').loc[220].goSPL:.2f}/{piv('S_sil','central').loc[220].NeuroLEM:.2f}, {piv('S_sil','central').loc[240].goSPL:.2f}/{piv('S_sil','central').loc[240].NeuroLEM:.2f}; {gv[gv.metric == 'S_sil'].goSPL.max():.3f}")
if gp is not None:
    dd = gp.S_sil - gp.S_sil_NeuroLEM
    put('published goSPL fields: S_sil minus NeuroLEM mean; range; r; max goSPL S_sil', f"{dd.mean():+.3f}; {dd.min():+.3f} to {dd.max():+.3f}; {np.corrcoef(gp.S_sil, gp.S_sil_NeuroLEM)[0, 1]:.3f}; {gp.S_sil.max():.3f}")
    put('published goSPL fields: share beyond minus NeuroLEM mean; range; denudation relative', f"{(gp.beyond_frac_eroding - gp.beyond_frac_eroding_NeuroLEM).mean():+.3f}; {(gp.beyond_frac_eroding - gp.beyond_frac_eroding_NeuroLEM).min():+.3f} to {(gp.beyond_frac_eroding - gp.beyond_frac_eroding_NeuroLEM).max():+.3f}; {(gp.D_Gt.sum() / gp.D_Gt_NeuroLEM.sum() - 1):+.3f}")
# lithium isotopes: the smoothed seawater curve of the compilation interpolated to the model ages, against simulated riverine d7Li
sw_ = pd.read_csv(cfg.PROX / 'd7Li_seawater_smoothed_OuyangLi2025_0-3000Ma.csv').set_index('age_Ma').d7Li_seawater_permil
swi = np.interp(a.index, sw_.index, sw_.values); ix = a.index.values
put('compiled seawater d7Li: Ordovician (445-485) mean; Permian (252-299) mean; rise; simulated rivers same', f"{swi[(ix>=445)&(ix<=485)].mean():.1f}; {swi[(ix>=252)&(ix<=299)].mean():.1f}; {swi[(ix>=252)&(ix<=299)].mean()-swi[(ix>=445)&(ix<=485)].mean():+.1f}; {a.d7Li_riv_adj.loc[252:299].mean()-a.d7Li_riv_adj.loc[445:485].mean():+.1f}")
put('r(simulated river d7Li, smoothed seawater) 540-200 Ma (n); 0-200 Ma', f"{np.corrcoef(a.d7Li_riv_adj.loc[200:], swi[ix>=200])[0,1]:.2f} ({int((ix>=200).sum())}); {np.corrcoef(a.d7Li_riv_adj.loc[:200], swi[ix<=200])[0,1]:.2f}")
put('compiled seawater d7Li rise 200->145 Ma; 65->0 Ma; simulated rivers 65->0', f"{swi[ix==145][0]-swi[ix==200][0]:+.1f}; {swi[ix==0][0]-swi[ix==65][0]:+.1f}; {a.d7Li_riv_adj.loc[0]-a.d7Li_riv_adj.loc[65]:+.1f}")
put('compilation: offset to seawater; n values; n Phanerozoic', f"{pd.read_csv(cfg.PROX / 'd7Li_seawater_smoothed_OuyangLi2025_0-3000Ma.csv').offset_to_seawater_permil.iloc[0]:.2f}; {len(pd.read_csv(cfg.PROX / 'd7Li_carbonate_compilation_OuyangLi2025_0-3000Ma.csv'))}; {int((pd.read_csv(cfg.PROX / 'd7Li_carbonate_compilation_OuyangLi2025_0-3000Ma.csv').age_Ma <= 541).sum())}")
# dynamic topography, 5-250 Ma (dyntopo_sensitivity.py; Methods, main text, Supplementary Information)
_db = OUT / 'dyntopo_sensitivity_by_age.csv'
if _db.exists():
    _w = pd.read_csv(_db).set_index('age').loc[5:250]     # ages with a dynamic-topography term (it vanishes at 0 Ma)
    # suffix _shoreline: shorelines of the reconstruction kept; _free: shorelines free to move; _common: land of both
    put('dyntopo shoreline: S_sil change mean, min, ages lower; share beyond pp mean, max, ages higher; F_sil % mean, range',
        f"{_w.S_sil_shoreline.mean():+.4f}, {_w.S_sil_shoreline.min():+.4f}, {(_w.S_sil_shoreline < 0).sum()}/{len(_w)}; "
        f"{_w.beyond_pp_shoreline.mean():+.2f}, {_w.beyond_pp_shoreline.max():+.2f}, {(_w.beyond_pp_shoreline > 0).sum()}/{len(_w)}; "
        f"{_w.F_sil_pct_shoreline.mean():+.2f}, {_w.F_sil_pct_shoreline.min():+.2f} to {_w.F_sil_pct_shoreline.max():+.2f}")
    put('dyntopo shoreline change / |adjusted - reference| (mean abs): S_sil; share beyond',
        f"{_w.S_sil_shoreline.abs().mean() / _w.adj_minus_ref_S_sil.abs().mean():.2f}; {_w.beyond_pp_shoreline.abs().mean() / _w.adj_minus_ref_beyond_pp.abs().mean():.2f}")
    put('dyntopo free: S_sil, share pp, F_sil % (own land); land of both: S_sil, share pp, F_sil %',
        f"{_w.S_sil_free.mean():+.4f}, {_w.beyond_pp_free.mean():+.2f}, {_w.F_sil_pct_free.mean():+.2f}; {_w.S_sil_free_common.mean():+.4f}, {_w.beyond_pp_free_common.mean():+.2f}, {_w.F_sil_pct_free_common.mean():+.2f}")
    put('dyntopo lowland: share beyond pp mean, range; median E/E_opt ratio mean, range; below-optimum eroding land below 1,500 m',
        f"{_w.lowland_beyond_pp_shoreline.mean():+.2f}, {_w.lowland_beyond_pp_shoreline.min():+.2f} to {_w.lowland_beyond_pp_shoreline.max():+.2f}; "
        f"{_w.lowland_x_median_ratio_shoreline.mean():.3f}, {_w.lowland_x_median_ratio_shoreline.min():.3f}-{_w.lowland_x_median_ratio_shoreline.max():.3f}; "
        f"{100 * _w.below_opt_lowland_frac_adjusted.mean():.1f}% ({100 * _w.below_opt_lowland_frac_adjusted.min():.1f}-{100 * _w.below_opt_lowland_frac_adjusted.max():.1f}%)")
    put('dyntopo shoreline: erosion flux % mean, max (age); S_sil change range',
        f"{_w.D_pct_shoreline.mean():+.1f}, {_w.D_pct_shoreline.max():+.1f} ({_w.D_pct_shoreline.idxmax()}); {_w.S_sil_shoreline.min():+.4f} to {_w.S_sil_shoreline.max():+.4f}")
    _dr = OUT / 'dyntopo_gld428_allcrust.csv'
    if _dr.exists():
        # silicate sink of the adjusted surface and of the surface with the term added (shorelines kept)
        _r0 = pd.read_csv(_dr); _ra = _r0[_r0.surface == 'adjusted'].set_index('age').F_sil; _rs_ = _r0[_r0.surface == 'shoreline'].set_index('age').F_sil
        put('dyntopo shoreline: r(log F_sil, adjusted), 5-250 Ma; silicate sink 190->105 Ma shoreline / adjusted',
            f"{np.corrcoef(np.log(_ra.loc[5:250]), np.log(_rs_.loc[5:250]))[0, 1]:.3f}; x{_rs_[105] / _rs_[190]:.2f} / x{_ra[105] / _ra[190]:.2f}")
    _dc = OUT / 'dyntopo_controls_by_age.csv'
    if _dc.exists():
        _c = pd.read_csv(_dc).set_index('age').loc[5:250]   # amplitude-matched control terms, one column pair per control
        put('dyntopo controls, 5-250 Ma mean change S_sil / share pp', '; '.join(f"{c[6:]}: {_c[c].mean():+.4f} / {_c['beyond_pp_' + c[6:]].mean():+.2f}" for c in _c.columns if c.startswith('S_sil_')))
# silicate sink at a constant 560 ppm (circularity of the runoff control; degassing comparison; main text, Methods)
_rs = OUT / 'reactivity_summary.csv'
if _rs.exists():
    _r = pd.read_csv(_rs).set_index('age')
    _ch = lambda c, t0, t1: 100 * (_r[c].loc[t1] / _r[c].loc[t0] - 1)    # per cent change from age t0 to age t1
    # F_sil_560_adj: sink at a constant 560 ppm; F_sil_own_adj: sink with the simulated CO2 forcing of each age
    put('constant 560 ppm: silicate sink 350->285 Ma; 190->105 Ma (simulated forcing in brackets)',
        f"{_ch('F_sil_560_adj', 350, 285):+.0f}% ({_ch('F_sil_own_adj', 350, 285):+.0f}%); {_ch('F_sil_560_adj', 190, 105):+.0f}% ({_ch('F_sil_own_adj', 190, 105):+.0f}%)")
    _m3 = _r.index <= 300
    put('constant 560 ppm: r(log sink, log simulated sink), 109 ages; r(sink, gross degassing) 0-300 Ma',
        f"{np.corrcoef(np.log(_r.F_sil_560_adj), np.log(_r.F_sil_own_adj))[0, 1]:.2f}; {np.corrcoef(_r.D_mean[_m3], _r.F_sil_560_adj[_m3])[0, 1]:.2f}")
# mountain-building events (Fig. 4e,f; main text, Methods)
_ev = OUT / 'orogen_sensitivity_events.csv'
if _ev.exists():
    # events shown in Fig. 4e,f: belts whose erosion flux rose at least 1.5-fold (as orogen_tracks.MIN_RATIO)
    _e = pd.read_csv(_ev); _e = _e[_e.ero_belt_Mt / _e.ero_before_Mt >= 1.5]
    _sea = _e.land_share_before < 0.5          # crust largely below sea level before uplift
    rng = lambda v: f"{v.min():.2f} to {v.max():.2f}"
    put('events: number shown; raised largely from the sea (land before < 50%)', f"{len(_e)}; {', '.join(_e.belt[_sea])}")
    put('events, fixed climate: a_sulf; a_ocp (all)', f"{rng(_e.a_sulf_fixed)}; {rng(_e.a_ocp_fixed)}")
    # a_*: event-scale sensitivity ln(X1/X0)/ln(D1/D0); _fixed: climate held fixed. The count uses 0.095-0.225 so that
    # values rounding to 0.10-0.22 are included
    put('events, fixed climate: a_sil raised from sea; standing land (n in 0.10-0.22)',
        f"{rng(_e.a_sil_fixed[_sea])}; {rng(_e.a_sil_fixed[~_sea])} ({int(_e.a_sil_fixed[~_sea].between(0.095, 0.225).sum())} of {int((~_sea).sum())})")
    _st = _e[~_sea]
    put('events, t C km-2 yr-1: exposure raised from sea; exposure standing land; climate standing land',
        f"{rng(_e.dnet_exposure[_sea])}; {rng(_st.dnet_exposure)}; {rng(_st.dnet_climate)}")
    # 'climate reversed sign': the total change has the opposite sign to the exposure part
    put('events, standing land: climate larger than exposure; climate reversed sign; exposure with OC < 0',
        f"{int((_st.dnet_climate.abs() > _st.dnet_exposure.abs()).sum())}; {int((np.sign(_st.dnet_exposure) != np.sign(_st.net_belt - _st.net_before)).sum())}; {int((_st.dnet_oc_exposure < 0).sum())} of {len(_st)}")
# independent tests of the adjusted palaeoelevation (Methods, Supplementary Methods; palaeoelevation_validation.py)
_pv = OUT / 'palaeoelevation_validation.csv'
if _pv.exists():
    _v = pd.read_csv(_pv)
    for _s in ('land, within 150 km of a training sample', 'land, all distances'):
        _g = _v[(_v.subset == _s)].set_index('metric')
        put(f'held-out rock chemistry, {_s}: RMS PALEOMAP -> adjusted (m, %); bias; r (n)',
            f"{_g.loc['RMS (m)', 'paleomap']:.0f} -> {_g.loc['RMS (m)', 'adjusted']:.0f} ({_g.loc['RMS (m)', 'change_pct']:+.0f}%); "
            f"{_g.loc['bias (m)', 'paleomap']:+.0f} -> {_g.loc['bias (m)', 'adjusted']:+.0f}; {_g.loc['correlation', 'paleomap']:.2f} -> {_g.loc['correlation', 'adjusted']:.2f} ({int(_g['n'].iloc[0])})")
    _g = _v[_v.test.str.startswith('sedimentary')]
    put('GST-1 basins (> 2 km): AUC PALEOMAP / adjusted, 540-5 and 250-5 Ma; share held above 500 m (%)',
        '; '.join(f"{r.subset}: {r.paleomap:g} / {r.adjusted:g}" for r in _g.itertuples()))
# number of NeuroLEM erosion fields (summary paragraph, main text, Methods)
_n = {'main': 2 * len(a)}               # two palaeoelevation histories at every age
_m = cfg.ZENODO / 'geochem_uncertainty_ensemble' / 'nature_recast' / 'ensemble_sensitivity_members.csv'
if _m.exists():
    # rows with member -2 (reference surface) and -1 (mean correction) are not ensemble members and are not counted
    _mm = pd.read_csv(_m, usecols=['age', 'member']); _n['ensemble'] = int((_mm.member >= 0).sum())
_cn = OUT / 'climate_sensitivity_nodes.csv'
if _cn.exists():
    _n['climate'] = len(pd.read_csv(_cn, usecols=['age']))       # one field per age, climate state and surface
_dt = [OUT / 'dyntopo_gld428_allcrust.csv'] + [OUT / f'dyntopo_control_{c}.csv' for c in ('tsh075', 'tsh150', 'tsh225', 'lon060', 'lon120', 'lon180')]
# the dynamic-topography run and its six control terms; rows of the unperturbed adjusted surface are not new fields
_n['dyntopo'] = sum(int((pd.read_csv(f, usecols=['surface']).surface != 'adjusted').sum()) for f in _dt if f.exists())
put('NeuroLEM erosion fields: main, ensemble, climate, dynamic topography; total', ', '.join(f'{k} {v:,}' for k, v in _n.items()) + f'; total {sum(_n.values()):,}')
# write the audit as one markdown table
out = '| Quantity | Value |\n|---|---|\n' + '\n'.join(f'| {c} | {v} |' for c, v in L)
(OUT / 'number_audit.md').write_text(out + '\n'); print(out)

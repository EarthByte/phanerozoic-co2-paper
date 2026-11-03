#!/usr/bin/env python3
"""Bring the three uncertainty experiments together: one figure, one table of statements.

Reads whatever is complete of
  ensemble_sensitivity_age_summary.csv / ensemble_sensitivity_headline.csv   (palaeoelevation ensemble)
  climate_sensitivity_bands.csv / climate_sensitivity_headline.csv           (27 climate states)
  gospl_validation_sensitivity.csv                                           (direct goSPL runs)
and writes
  figures/nature/ed_uncertainty.png, .pdf     a  erosion sensitivity of the silicate sink
                                              b  share of eroding land at or beyond the optimum
                                              c  adjusted-to-reference ratio of the silicate sink
  data/nature_recast/uncertainty_statements.md   every number the Methods paragraph on uncertainty quotes

    python3 finalise_uncertainty.py
"""
import numpy as np, pandas as pd
import _nature_style as ns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from _load import series, FIG, OUT


def read(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


main = series(); t = main.index.values
ens = read('ensemble_sensitivity_age_summary.csv'); ens_h = read('ensemble_sensitivity_headline.csv')
cli = read('climate_sensitivity_bands.csv'); cli_h = read('climate_sensitivity_headline.csv')
gos = read('gospl_validation_sensitivity.csv')
pub = read('gospl_published_reference_sensitivity.csv')
if pub is not None:
    pub = pub.sort_values('age')
n_members = int(ens.n_members.max()) if ens is not None else 0
if ens is not None:
    ens = ens[ens.n_members == n_members].set_index('age')
if cli is not None:
    cli = cli.set_index('age')

fig, axs = plt.subplots(1, 3, figsize=(ns.DOUBLE, 2.55), gridspec_kw=dict(left=0.065, right=0.995, top=0.95, bottom=0.2, wspace=0.3))
panels = (('S_sil', 1.0, 'Erosion sensitivity, silicate sink', (0.0, 0.45)),
          ('beyond_frac_eroding', 100.0, 'Land at or beyond optimum (%)', (0, 100)),
          ('ratio_F_sil', 1.0, 'Silicate sink, adjusted / reference', (0.9, 1.6)))
for ax, (k, f, label, ylim), letter in zip(axs, panels, 'abc'):
    if cli is not None:
        lo, hi = (cli[f'{k}_p05'], cli[f'{k}_p95']) if k.startswith('ratio') else (cli[f'{k}_adj_p05'], cli[f'{k}_adj_p95'])
        ax.fill_between(cli.index, f * lo, f * hi, color=ns.C['sky'], alpha=0.35, lw=0)
    if ens is not None:
        ax.fill_between(ens.index, f * ens[f'{k}_p05'], f * ens[f'{k}_p95'], color=ns.C['adj'], alpha=0.45, lw=0)
    if k.startswith('ratio'):
        ax.plot(t, main.F_sil_adj / main.F_sil_ref, color=ns.C['adj'], lw=1.0); ax.axhline(1, color='k', lw=0.5)
    else:
        ax.plot(t, f * main[f'{k}_adj'], color=ns.C['adj'], lw=1.0)
        ax.plot(t, f * main[f'{k}_ref'], color=ns.C['ref'], lw=0.8, ls=':')
        if pub is not None:
            ax.plot(pub.age, f * pub[k], color=ns.C['ref'], lw=0.7, alpha=0.9)
        if gos is not None:
            for case, col in (('central', ns.C['adj']), ('reference', ns.C['ref'])):
                g = gos[(gos.metric == k) & (gos.case == case)]
                ax.plot(g.age, f * g.goSPL, ls='none', marker='o', ms=3.2, mfc='white', mec=col, mew=0.9, zorder=6)
    if k.startswith('ratio'):
        top = max([float((main.F_sil_adj / main.F_sil_ref).max())] + ([float(ens[f'{k}_p95'].max())] if ens is not None else []))
        ylim = (0.9, max(ylim[1], 1.05 * top))
    ax.set_xlim(540, 0); ax.set_ylim(*ylim); ax.set_xlabel('Age (Ma)'); ax.set_ylabel(label)
    ns.timescale(ax, height=0.07); ns.panel_letter(ax, letter, x=-0.2, y=0.96)
handles = [Line2D([], [], color=ns.C['adj'], lw=1.0, label='adjusted'), Line2D([], [], color=ns.C['ref'], lw=0.8, ls=':', label='Scotese–Wright')]
if ens is not None:
    handles.append(Patch(color=ns.C['adj'], alpha=0.45, lw=0, label='elevation ensemble'))
if cli is not None:
    handles.append(Patch(color=ns.C['sky'], alpha=0.35, lw=0, label='climate states'))
if pub is not None:
    handles.append(Line2D([], [], color=ns.C['ref'], lw=0.7, label='published goSPL, Scotese–Wright'))
if gos is not None:
    handles.append(Line2D([], [], ls='none', marker='o', ms=3.2, mfc='white', mec='k', mew=0.9, label='direct goSPL'))
axs[0].legend(handles=handles, loc='upper right', handlelength=1.5, borderaxespad=0.1, labelspacing=0.25)
fig.savefig(FIG / 'ed_uncertainty.png', dpi=400); fig.savefig(FIG / 'ed_uncertainty.pdf')

# ---------------------------------------------------------------- statements
L = ['# Uncertainty experiments: numbers for the Methods paragraph', '',
     'Central model (adjusted surface): time-mean S_sil {:.3f}, S_net {:.3f}, share beyond optimum {:.1%}.'.format(
         main.S_sil_adj.mean(), main.S_net_adj.mean(), main.beyond_frac_eroding_adj.mean()), '']


def h(tab, metric, stat, surface=None):
    q = (tab.metric == metric) & (tab.statistic == stat)
    if surface is not None:
        q &= tab.surface == surface
    return tab[q].iloc[0]


if ens is not None:
    na = len(ens); L += [f'## Palaeoelevation ensemble ({n_members} members, {na} of 109 ages)', '']
    for k, name in (('S_sil', 'time-mean S_sil'), ('S_net', 'time-mean S_net'), ('beyond_frac_eroding', 'time-mean share beyond optimum'),
                    ('ratio_D_Gt', 'time-mean denudation ratio, adjusted/reference'), ('ratio_F_sil', 'time-mean silicate-sink ratio'),
                    ('ratio_F_net', 'time-mean net-sink ratio'), ('ratio_F_sil_common', 'same, common land')):
        x = h(ens_h, k, 'time_mean'); L.append(f'- {name}: p05 {x.p05:.3f}, p50 {x.p50:.3f}, p95 {x.p95:.3f} (members {x.min_member:.3f} to {x.max_member:.3f})')
    x = h(ens_h, 'S_sil', 'time_max'); L.append(f'- highest S_sil at any age: p95 across members {x.p95:.3f}, largest member {x.max_member:.3f}')
    w = ens.S_sil_p95 - ens.S_sil_p05; wb = ens.beyond_frac_eroding_p95 - ens.beyond_frac_eroding_p05
    L.append(f'- p05-p95 width per age: S_sil median {w.median():.3f} (max {w.max():.3f} at {int(w.idxmax())} Ma); share beyond optimum median {wb.median():.3f} (max {wb.max():.3f} at {int(wb.idxmax())} Ma)')
    L.append(f'- adjusted-minus-reference S_sil exceeds the ensemble width at {int(((ens.S_sil_reference - ens.S_sil_p50).abs() > w).sum())} of {na} ages')
    for m in ('members_with_S_sil_below_0.5_at_every_age', 'members_with_beyond_share_above_0.5_at_every_age_since_300Ma', 'members_with_beyond_share_higher_today_than_at_480Ma'):
        r = ens_h[ens_h.metric == m]
        if len(r):
            L.append(f'- {m.replace("_", " ")}: {int(r.iloc[0]["mean"])} of {n_members}')
    L.append('')
if cli is not None:
    L += [f'## Climate states ({len(cli)} of 109 ages; CO2 envelope and model spread)', '']
    for k, name in (('S_sil', 'time-mean S_sil'), ('S_net', 'time-mean S_net'), ('beyond_frac_eroding', 'time-mean share beyond optimum')):
        x = h(cli_h, k, 'time_mean', 'adj'); w = h(cli_h, k, 'p05_p95_width_median_over_ages', 'adj')
        L.append(f'- {name}: p05 {x.p05:.3f}, p50 {x.p50:.3f}, p95 {x.p95:.3f}; median p05-p95 width per age {w.p50:.3f}')
    for k in ('D_Gt', 'F_sil', 'F_net'):
        w = h(cli_h, f'ratio_{k}', 'p05_p95_width_median_over_ages'); x = h(cli_h, f'ratio_{k}', 'time_mean')
        L.append(f'- adjusted/reference ratio of {k}: time-mean p05 {x.p05:.3f}, p50 {x.p50:.3f}, p95 {x.p95:.3f}; median width per age {w.p50:.3f}')
    r = cli_h[cli_h.metric == 'members_with_beyond_share_higher_today_than_at_480Ma']
    if len(r):
        L.append(f'- members with a higher share beyond the optimum today than at 480 Ma: {int(r.iloc[0].p50)}')
    L.append('')
if gos is not None:
    L += ['## Direct goSPL simulations (goSPL minus NeuroLEM)', '']
    for k in ('S_sil', 'S_net', 'beyond_frac_eroding', 'F_sil', 'F_net', 'D_Gt'):
        g = gos[(gos.metric == k) & np.isfinite(gos.difference)]
        L.append(f'- {k}: median {g.difference.median():+.4f}, range {g.difference.min():+.4f} to {g.difference.max():+.4f}; relative {100 * g.relative_difference.min():+.1f}% to {100 * g.relative_difference.max():+.1f}% (n = {len(g)})')
    for case in ('central', 'reference'):
        g = gos[(gos.metric == 'S_sil') & (gos.case == case)]
        L.append(f'- S_sil, {case}: ' + '; '.join(f'{int(r.age)} Ma goSPL {r.goSPL:.3f} / NeuroLEM {r.NeuroLEM:.3f}' for r in g.itertuples()))
    piv = gos[gos.metric == 'S_sil'].pivot(index='age', columns='case', values='goSPL')
    if {'central', 'reference'} <= set(piv.columns):
        L.append('- sign of adjusted-minus-reference S_sil in goSPL: ' + ', '.join(f'{int(a)} Ma {v:+.3f}' for a, v in (piv.central - piv.reference).items()))
    L.append('')
if pub is not None:
    L += [f'## Published goSPL fields, reference surface ({len(pub)} of 109 ages; goSPL minus NeuroLEM)', '']
    wh = pub.withheld_from_training == 1
    for k in ('S_sil', 'S_net', 'beyond_frac_eroding', 'D_Gt', 'F_sil'):
        dlt = pub[k] - pub[f'{k}_NeuroLEM']
        L.append(f'- {k}: mean {dlt.mean():+.4f}, range {dlt.min():+.4f} to {dlt.max():+.4f}; withheld ages mean {dlt[wh].mean():+.4f}, range {dlt[wh].min():+.4f} to {dlt[wh].max():+.4f}; '
                 f'time-mean goSPL {pub[k].mean():.3f}, NeuroLEM {pub[f"{k}_NeuroLEM"].mean():.3f}; goSPL range {pub[k].min():.3f} to {pub[k].max():.3f}')
    L.append(f'- correlation of the two S_sil time series: {np.corrcoef(pub.S_sil, pub.S_sil_NeuroLEM)[0, 1]:.3f}; of the share beyond optimum: {np.corrcoef(pub.beyond_frac_eroding, pub.beyond_frac_eroding_NeuroLEM)[0, 1]:.3f}')
    L.append('')
missing = [n for n, x in (('palaeoelevation ensemble', ens), ('climate states', cli), ('direct goSPL', gos)) if x is None]
if missing:
    L += ['Not yet available: ' + ', '.join(missing) + '.', '']
(OUT / 'uncertainty_statements.md').write_text('\n'.join(L))
print('\n'.join(L)); print('wrote', OUT / 'uncertainty_statements.md', 'and', FIG / 'ed_uncertainty.png')

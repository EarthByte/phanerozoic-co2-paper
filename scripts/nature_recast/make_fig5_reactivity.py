#!/usr/bin/env python3
"""Fig. 5: the history of the silicate sink against gross solid-Earth degassing (both relative to the present); the weathering CO2 sources and sinks
against degassing in absolute terms; the silicate sink against runoff
and erosion; and weathering intensity against the lithium-isotope record.

Four stacked time-series panels, 540 Ma to the present, main model (kin_bridge_s2, adjusted surface):
  a  silicate sink (band: range of the three parameter sets), net inorganic sink, net sink with the organic
     pathways (rock organic-carbon oxidation released, biospheric organic-carbon burial added; dashed, from
     400 Ma) and gross solid-Earth degassing of Müller et al. (2024) (preferred model, 5-Myr mean, band: model
     range), each divided by its own present-day value
  b  the same fluxes in Mt C/yr: degassing, oxidative weathering (sulfide plus rock organic carbon), silicate
     sink and biospheric burial (from 400 Ma)
  c  silicate sink, continental runoff volume (mean runoff times land area) and erosion flux, relative to present
  d  delta7Li: carbonate compilation of Ouyang et al. (2025) shifted to seawater (grey dots), its smoothed
     seawater curve with +-2 standard errors, and simulated riverine delta7Li (band: parameter sets); weathering
     intensity on the inverted right axis
With NET_BUDGET=full (environment) panel a shows the full net sink instead, with the inorganic sink dotted.

Inputs:  data/nature_recast/kin_bridge_s2_{ens,Taiwan,Sichuan,NewZealand}.csv (run_weathering.py),
         data/nature_recast/reactivity_summary.csv (degassing columns D_min, D_mean, D_max in Mt C/yr;
         reactivity_post.py), data/proxies/d7Li_carbonate_compilation_OuyangLi2025_0-3000Ma.csv and
         d7Li_seawater_smoothed_OuyangLi2025_0-3000Ma.csv
Outputs: figures/nature/fig5_reactivity.png and .pdf

    python3 make_fig5_reactivity.py
"""
import numpy as np, pandas as pd
import _nature_style as ns
import matplotlib.pyplot as plt
from _load import series, FIG, OUT, PROX, SETS
import _config as cfg

a = series(); t = a.index.values                     # main model, mean of the three parameter sets; t = age (Ma)
r = pd.read_csv(OUT / 'reactivity_summary.csv').set_index('age')
def env(col):
    # min and max of col across the three calibrated parameter sets of the main run
    arr = np.array([series('kin_bridge_s2', s)[col].values for s in SETS]); return arr.min(0), arr.max(0)
fig, axs = plt.subplots(4, 1, figsize=(ns.SINGLE, ns.MAX_DEPTH), sharex=True, gridspec_kw=dict(left=0.17, right=0.83, top=0.985, bottom=0.065, hspace=0.12))
ax = axs[0]
# panel a compares histories, not magnitudes: the absolute level of the continental sink is set by its modern calibration and
# it is one of several long-term sinks, so every series is shown relative to its own present-day value (absolute sinks: panel b
# and Extended Data Fig. 6)
rel = lambda x: x / x.loc[0]
ax.axhline(1, color='k', lw=0.4)
# every set is calibrated to the same present-day sink, so dividing the band by the central present-day value is
# the same as dividing each set by its own
lo, hi = env('F_sil_adj'); ax.fill_between(t, lo / a.F_sil_adj.loc[0], hi / a.F_sil_adj.loc[0], color=ns.C['sil'], alpha=0.25, lw=0)
ax.plot(t, rel(a.F_sil_adj), color=ns.C['sil'], lw=1.2, label='silicate sink')
import cellfields as cf
PLANTS = cf.PLANTS_MA
org = t <= PLANTS                # ages with land plants, at which biospheric burial is drawn
if cf.BUDGET == 'full':     # NET_BUDGET=full: the net sink is the full budget at every age; the inorganic budget dotted for reference
    ax.plot(t, rel(cf.net_series(a)[0]), color='k', lw=1.0, label='net sink')
    ax.plot(t, rel(a.F_net_adj), color='k', lw=0.7, ls=':', label='inorganic net sink')
else:
  ax.plot(t, rel(a.F_net_adj), color='k', lw=0.9, label='net inorganic sink')
# net sink with the organic pathways: rock organic-carbon oxidation released, biospheric burial added; burial needs
# land plants and is drawn from the Early Devonian (400 Ma) on
if cf.BUDGET != 'full':
  ax.plot(t[org], rel(a.F_net_adj - a.F_ocp_adj + a.F_ocb_adj).values[org], color='k', lw=0.9, ls=(0, (3.5, 2)), label='net sink + organic carbon')
D0 = r.D_mean.loc[0]        # present-day gross degassing, Mt C/yr
ax.fill_between(t, r.D_min / D0, r.D_max / D0, color=ns.C['oc'], alpha=0.22, lw=0); ax.plot(t, r.D_mean / D0, color=ns.C['oc'], lw=1.1, label='solid-Earth degassing')
ax.set_ylabel('Relative to present'); ax.set_ylim(0, 2.45)
ax.legend(loc='upper left', handlelength=1.2, borderaxespad=0.1, handletextpad=0.4, ncol=2, columnspacing=0.6)
ns.panel_letter(ax, 'a', x=-0.2, y=0.93)

ax = axs[1]
# panel b: absolute fluxes (Mt C/yr); the weathering sources are the oxidation of sulfides and rock organic carbon, the sinks
# silicate weathering and the burial of biospheric organic carbon (from 400 Ma); degassing as in panel a
M = 1e3                     # Gt C/yr -> Mt C/yr
ax.fill_between(t, r.D_min, r.D_max, color=ns.C['oc'], alpha=0.22, lw=0); ax.plot(t, r.D_mean, color=ns.C['oc'], lw=1.1, label='solid-Earth degassing')
ax.plot(t, M * (a.F_sulf_adj + a.F_ocp_adj), color=ns.C['adj'], lw=1.1, label='oxidative weathering')
ax.plot(t, M * a.F_sil_adj, color=ns.C['sil'], lw=1.2, label='silicate sink')
ax.plot(t[org], M * a.F_ocb_adj.values[org], color=ns.C['sky'], lw=1.1, label='biospheric burial')
ax.set_ylim(0, 145); ax.set_ylabel('CO$_2$ flux (Mt C yr$^{-1}$)')
ax.legend(loc='upper left', handlelength=1.2, borderaxespad=0.1, handletextpad=0.4, ncol=2, columnspacing=0.6)
ns.panel_letter(ax, 'b', x=-0.2, y=0.93)

ax = axs[2]
# panel c: drivers of the silicate sink, each relative to the present
Q = a.q_mean_mm_adj * a.land_Mkm2_adj        # runoff volume, proportional to mean runoff times land area
ax.axhline(1, color='k', lw=0.4)
ax.plot(t, a.F_sil_adj / a.F_sil_adj.loc[0], color=ns.C['sil'], lw=1.2, label='silicate sink')
ax.plot(t, Q / Q.loc[0], color=ns.C['sky'], lw=1.1, label='continental runoff')
ax.plot(t, a.D_Gt_adj / a.D_Gt_adj.loc[0], color=ns.C['adj'], lw=1.1, label='erosion flux')
ax.set_ylim(0, 2.45); ax.set_ylabel('Relative to present')
ax.legend(loc='upper left', handlelength=1.2, borderaxespad=0.1, handletextpad=0.4, ncol=2, columnspacing=0.6)
ns.panel_letter(ax, 'c', x=-0.2, y=0.93)

ax = axs[3]
# panel d: lithium isotopes, restricted to the Phanerozoic (<= 545 Ma)
comp = pd.read_csv(PROX / 'd7Li_carbonate_compilation_OuyangLi2025_0-3000Ma.csv'); comp = comp[comp.age_Ma <= 545]
sw = pd.read_csv(PROX / 'd7Li_seawater_smoothed_OuyangLi2025_0-3000Ma.csv'); sw = sw[sw.age_Ma <= 545]
OFF = float(sw.offset_to_seawater_permil.iloc[0])     # carbonate-to-seawater offset of the compilation, per mil
ax.plot(comp.age_Ma, comp.d7Li_carbonate_permil + OFF, '.', ms=1.8, color='#B0B0B0', label='carbonates', zorder=1, rasterized=True)
ax.fill_between(sw.age_Ma, sw.d7Li_seawater_permil - 2 * sw.se_permil, sw.d7Li_seawater_permil + 2 * sw.se_permil, color='#555555', alpha=0.25, lw=0, zorder=2)
ax.plot(sw.age_Ma, sw.d7Li_seawater_permil, color='#333333', lw=1.0, label='seawater', zorder=3)
lo, hi = env('d7Li_riv_adj'); ax.fill_between(t, lo, hi, color=ns.C['sky'], alpha=0.3, lw=0, zorder=2)
ax.plot(t, a.d7Li_riv_adj, color=ns.C['ref'], lw=1.2, label='simulated rivers', zorder=3)
ax.set_ylim(-4, 38); ax.set_ylabel('δ$^7$Li (‰)')
# weathering intensity (silicate dissolved flux / total denudation) on an inverted right axis: intensity increases downwards
ax2 = ax.twinx(); ax2.spines['right'].set_visible(True)
ax2.plot(t, a.WI_adj, color=ns.C['sulf'], lw=1.1); ax2.set_ylim(0.26, 0.0); ax2.set_yticks([0, 0.05, 0.10, 0.15])
ax2.set_ylabel('Weathering intensity', color=ns.C['sulf']); ax2.tick_params(axis='y', colors=ns.C['sulf'])
ax.legend(loc='lower left', handlelength=1.2, borderaxespad=0.1, ncol=3, columnspacing=0.8, bbox_to_anchor=(0.0, 0.085), handletextpad=0.3)
ns.panel_letter(ax, 'd', x=-0.2, y=0.93)
ax.set_xlim(540, 0); ax.set_xlabel('Age (Ma)'); ns.timescale(ax, height=0.07)
fig.savefig(FIG / 'fig5_reactivity.png', dpi=400); fig.savefig(FIG / 'fig5_reactivity.pdf')

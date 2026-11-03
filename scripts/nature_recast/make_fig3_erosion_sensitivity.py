#!/usr/bin/env python3
"""Fig. 3: Phanerozoic erosion rates relative to the drawdown optimum and the erosion sensitivity of the carbon fluxes.
(a) erosion rates and the mean local optimum; (b) share of eroding land at or beyond the optimum; (c) erosion rate
relative to the local optimum; (d) erosion sensitivity of the silicate sink, the net inorganic sink, the sulfide source
and rock organic-carbon oxidation; (e) erosion flux and runoff; (f) kinetically limited share of the silicate sink."""
import numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
from _load import series, FIG, SETS

def env(col, cfgs):
    arr = []
    for c in cfgs:
        for s in SETS + ['ens']:
            try:
                arr.append(series(c, s)[col].values)
            except FileNotFoundError:
                pass
    arr = np.array(arr); return arr.min(0), arr.max(0)

a = series(); t = a.index.values
PAR = ['kin_bridge_s2']                                  # parameter sets
ALL = ['kin_bridge_s2', 'kin_bridge_s1', 'kin_bridge_s4', 'out_bridge_s2']   # + erosion scale and runoff placement
BOX = dict(fc='w', ec='none', pad=0.8, alpha=0.8)
fig, axs = plt.subplots(3, 2, figsize=(ns.DOUBLE, 6.3), sharex=True,
                        gridspec_kw=dict(left=0.085, right=0.915, top=0.975, bottom=0.075, hspace=0.12, wspace=0.55))

ax = axs[0, 0]                                           # a: erosion rates and the mean local optimum
ax.fill_between(t, a.E_p05_adj, a.E_p95_adj, color=ns.C['adj'], alpha=0.12, lw=0)
ax.fill_between(t, a.E_p25_adj, a.E_p75_adj, color=ns.C['adj'], alpha=0.30, lw=0)
ax.plot(t, a.E_p50_adj, color=ns.C['adj'], lw=1.2, label='median erosion rate')
ax.plot(t, a.Eopt_mean_mm_adj, color='k', lw=1.0, ls='--', label='mean local optimum, $E_{opt}$')
ax.set_yscale('log'); ax.set_ylim(1.5e-4, 1.5); ax.set_ylabel('Erosion rate, $E$ (mm yr$^{-1}$)')
ax.legend(loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)

ax = axs[0, 1]                                           # b: share at or beyond the optimum
lo, hi = env('beyond_frac_eroding_adj', ALL); ax.fill_between(t, 100 * lo, 100 * hi, color=ns.C['light'], alpha=0.35, lw=0)
lo, hi = env('beyond_frac_eroding_adj', PAR); ax.fill_between(t, 100 * lo, 100 * hi, color=ns.C['adj'], alpha=0.25, lw=0)
ax.plot(t, 100 * a.beyond_frac_eroding_adj, color=ns.C['adj'], lw=1.2, label='adjusted')
ax.plot(t, 100 * a.beyond_frac_eroding_ref, color=ns.C['ref'], lw=0.9, ls=':', label='PALEOMAP')
ax.set_ylim(0, 100); ax.set_ylabel('Eroding land at or beyond\nthe local optimum (%)')
ax.legend(loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)

ax = axs[1, 0]                                           # c: erosion rate relative to the local optimum
ax.axhline(1, color='k', lw=0.6)
ax.fill_between(t, a.x_p05_adj, a.x_p95_adj, color=ns.C['adj'], alpha=0.12, lw=0)
ax.fill_between(t, a.x_p25_adj, a.x_p75_adj, color=ns.C['adj'], alpha=0.30, lw=0)
ax.plot(t, a.x_p50_adj, color=ns.C['adj'], lw=1.2, label='adjusted')
ax.plot(t, a.x_p50_ref, color=ns.C['ref'], lw=0.9, ls=':', label='PALEOMAP')
ax.set_yscale('log'); ax.set_ylim(0.006, 60); ax.set_ylabel('Erosion rate relative to\nlocal optimum, $E/E_{opt}$')
ax.text(535, 28, 'beyond optimum', fontsize=ns.SMALL, ha='left', va='center')
ax.text(535, 0.085, 'rising limb', fontsize=ns.SMALL, ha='left', va='center', bbox=BOX)
ax.legend(loc='upper right', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)

ax = axs[1, 1]                                           # d: erosion sensitivities
ax.axhline(0, color='k', lw=0.5); ax.axhline(1, color='k', lw=0.5, ls=(0, (4, 2)))
lo, hi = env('S_sil_adj', ALL); ax.fill_between(t, lo, hi, color=ns.C['light'], alpha=0.35, lw=0)
lo, hi = env('S_sil_adj', PAR); ax.fill_between(t, lo, hi, color=ns.C['sil'], alpha=0.25, lw=0)
ax.plot(t, a.S_sil_adj, color=ns.C['sil'], lw=1.2, label='silicate sink, $S_{sil}$')
import cellfields as cf
FULL = cf.BUDGET == 'full'
ax.plot(t, cf.net_series(a, 'adj')[1], color=ns.C['net'], lw=1.2, label='net sink, $S_{net}$' if FULL else 'net inorganic sink, $S_{net}$')
ax.plot(t, a.S_sulf_adj, color=ns.C['sulf'], lw=1.1, label='sulfide source')
ax.plot(t, a.dF_ocp_adj / a.F_ocp_adj, color=ns.C['oc'], lw=1.1, label='rock OC oxidation')
ax.plot(t, a.S_sil_ref, color=ns.C['sil'], lw=0.8, ls=':')
ax.plot(t, cf.net_series(a, 'ref')[1], color=ns.C['net'], lw=0.8, ls=':')
YL = (-1.15, 1.45) if FULL else (-0.45, 1.45)
ax.set_ylim(*YL); ax.set_ylabel('Erosion sensitivity, $S$')
ax2 = ax.twinx(); ax2.spines['right'].set_visible(True); ax2.set_ylim(1 - YL[0], 1 - YL[1]); ax2.set_ylabel('Buffering capacity, 1 − $S$')
ax.legend(loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)

ax = axs[2, 0]                                           # e: erosion flux and runoff
ax.plot(t, a.D_Gt_adj, color=ns.C['adj'], lw=1.2, label='adjusted')
ax.plot(t, a.D_Gt_ref, color=ns.C['ref'], lw=0.9, ls=':', label='PALEOMAP')
ax.set_ylabel('Erosion flux (Gt yr$^{-1}$)'); ax.set_ylim(0, 36)
ax3 = ax.twinx(); ax3.spines['right'].set_visible(True)
ax3.plot(t, a.q_mean_mm_adj, color=ns.C['sky'], lw=1.0)
ax.plot([], [], color=ns.C['sky'], lw=1.0, label='runoff')
ax3.set_ylabel('Runoff (mm yr$^{-1}$)', color=ns.C['sky']); ax3.set_ylim(0, 900); ax3.tick_params(axis='y', colors=ns.C['sky'])
ax.legend(loc='upper left', ncol=3, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)

ax = axs[2, 1]                                           # f: kinetically limited share of the silicate sink
lo, hi = env('Gamma_sil_adj', PAR); ax.fill_between(t, 100 * lo, 100 * hi, color=ns.C['sky'], alpha=0.25, lw=0)
ax.plot(t, 100 * a.Gamma_sil_adj, color=ns.C['sky'], lw=1.2)
ax.set_ylim(0, 100); ax.set_ylabel('Kinetically limited share\nof the silicate sink (%)')

for k, ax in enumerate(axs.flat):
    ns.panel_letter(ax, 'abcdef'[k], x=-0.2, y=0.93)
for ax in axs[2]:
    ax.set_xlim(540, 0); ax.set_xlabel('Age (Ma)'); ns.timescale(ax, height=0.075)
fig.savefig(FIG / 'fig3_erosion_sensitivity.png', dpi=400); fig.savefig(FIG / 'fig3_erosion_sensitivity.pdf')

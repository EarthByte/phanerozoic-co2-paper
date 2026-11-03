#!/usr/bin/env python3
"""Extended Data Figs 4-8: runoff treatment, parameter sets and erosion calibration, the two
palaeoelevation histories, organic-carbon pathways, and reactivity with the intrinsic weathering feedback."""
import sys, numpy as np, pandas as pd
import _nature_style as ns
import matplotlib.pyplot as plt
from _load import series, FIG, OUT, SETS
import _config as cfg

a = series(); t = a.index.values
def fin(ax, letter, ylabel, x=-0.17):
    ax.set_xlim(540, 0); ax.set_xlabel('Age (Ma)'); ax.set_ylabel(ylabel); ns.timescale(ax, height=0.06); ns.panel_letter(ax, letter, x=x, y=0.98)
GS = dict(left=0.085, right=0.985, top=0.965, bottom=0.10, hspace=0.34, wspace=0.27)
which = sys.argv[1:] or ['4', '5', '6', '7', '8', '9']

if '4' in which:
    runs = [('kin_bridge_s2', 'kinetic term, simulated runoff ratio', ns.C['adj'], '-'), ('out_bridge_s2', 'whole flux, simulated runoff ratio', ns.C['ref'], '-'),
           ('kin_coef_s2', 'kinetic term, uniform ratio 0.4', ns.C['adj'], ':'), ('out_coef_s2', 'whole flux, uniform ratio 0.4', ns.C['ref'], ':')]
    fig, axs = plt.subplots(2, 2, figsize=(ns.DOUBLE, 4.6), gridspec_kw=GS)
    ax = axs[0, 0]; ax.plot(t, a.q_mean_mm_adj / (series('kin_coef_s2').q_mean_mm_adj / 0.4), color=ns.C['sky'], lw=1.2); ax.axhline(0.4, color='k', lw=0.6, ls=':')
    ax.set_ylim(0, 0.75); ax.text(535, 0.415, 'uniform 0.4', fontsize=ns.SMALL, va='bottom'); fin(ax, 'a', 'Land runoff ratio')
    for ax, col, lab, L, sc in ((axs[0, 1], 'F_sil_adj', 'Silicate sink (Mt C yr$^{-1}$)', 'b', 1e3), (axs[1, 0], 'S_sil_adj', 'Erosion sensitivity, silicate sink', 'c', 1), (axs[1, 1], 'beyond_frac_eroding_adj', 'Eroding land with $E \\geq E_{opt}$ (%)', 'd', 100)):
        for c, name, colr, ls in runs:
            ax.plot(t, sc * series(c)[col], color=colr, ls=ls, lw=1.1, label=name)
        fin(ax, L, lab)
    axs[0, 1].set_ylim(0, 200); axs[1, 0].set_ylim(0, 0.5); axs[1, 1].set_ylim(0, 80)     # data: sink 31-125 Mt C/yr, S_sil 0.07-0.40, share 9-65%
    axs[0, 1].legend(loc='upper left', ncol=1, handlelength=1.8, borderaxespad=0.1)
    fig.savefig(FIG / 'ed4_runoff.png', dpi=400); fig.savefig(FIG / 'ed4_runoff.pdf')

if '5' in which:
    fig, axs = plt.subplots(2, 2, figsize=(ns.DOUBLE, 4.6), gridspec_kw=GS)
    colS = {'Taiwan': '#009E73', 'Sichuan': '#CC79A7', 'NewZealand': '#E69F00', 'ens': 'k'}; nameS = {'Taiwan': 'Taiwan', 'Sichuan': 'Sichuan', 'NewZealand': 'New Zealand', 'ens': 'mean of three'}
    for ax, col, lab, L, sc in ((axs[0, 0], 'S_sil_adj', 'Erosion sensitivity, silicate sink', 'a', 1), (axs[0, 1], 'beyond_frac_eroding_adj', 'Eroding land with $E \\geq E_{opt}$ (%)', 'b', 100)):
        for s in SETS + ['ens']:
            ax.plot(t, sc * series('kin_bridge_s2', s)[col], color=colS[s], lw=1.3 if s == 'ens' else 1.0, label=nameS[s])
        fin(ax, L, lab)
    axs[0, 0].set_ylim(-0.1, 0.6); axs[0, 1].set_ylim(0, 80);     # data: S_sil -0.03-0.50, share 15-75%
    axs[0, 0].legend(loc='upper right', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4)
    for ax, col, lab, L, sc in ((axs[1, 0], 'S_sil_adj', 'Erosion sensitivity, silicate sink', 'c', 1), (axs[1, 1], 'beyond_frac_eroding_adj', 'Eroding land with $E \\geq E_{opt}$ (%)', 'd', 100)):
        for c, nm, ls in (('kin_bridge_s1', '× 1', ':'), ('kin_bridge_s2', '× 2 (main model)', '-'), ('kin_bridge_s4', '× 4', '--')):
            ax.plot(t, sc * series(c)[col], color=ns.C['adj'], ls=ls, lw=1.1, label=nm)
        fin(ax, L, lab)
    axs[1, 0].set_ylim(-0.1, 0.6); axs[1, 1].set_ylim(0, 90);     # data: S_sil -0.03-0.41, share 13-81%
    axs[1, 0].legend(title='Erosion calibration factor', loc='upper right', ncol=3, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4, title_fontsize=ns.SMALL)
    fig.savefig(FIG / 'ed5_parameters.png', dpi=400); fig.savefig(FIG / 'ed5_parameters.pdf')

if '6' in which:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rd = lambda n: pd.read_csv(OUT / n) if (OUT / n).exists() else None
    ens, cli, gos, pub = rd('ensemble_sensitivity_age_summary.csv'), rd('climate_sensitivity_bands.csv'), rd('gospl_validation_sensitivity.csv'), rd('gospl_published_reference_sensitivity.csv')
    fig, axs = plt.subplots(3, 2, figsize=(ns.DOUBLE, 6.55), gridspec_kw=dict(left=0.085, right=0.985, top=0.975, bottom=0.07, hspace=0.30, wspace=0.42))
    for ax, col, lab, L, sc in ((axs[0, 0], 'D_Gt', 'Erosion flux (Gt yr$^{-1}$)', 'a', 1), (axs[0, 1], 'F_sil', 'Silicate sink (Mt C yr$^{-1}$)', 'b', 1e3), (axs[1, 0], 'F_net', 'Net inorganic sink (Mt C yr$^{-1}$)', 'c', 1e3)):
        ax.plot(t, sc * a[col + '_ref'], color=ns.C['ref'], lw=1.1, label='PALEOMAP'); ax.plot(t, sc * a[col + '_adj'], color=ns.C['adj'], lw=1.1, label='adjusted')
        ax.plot(t, sc * a[col + '_adjc'], color=ns.C['adj'], lw=0.9, ls='--', label='adjusted, on land common to both surfaces'); ax.set_ylim(0, None); fin(ax, L, lab)
    # eroding land area on a right-hand axis of a: the erosion flux is this area times the mean rate
    ax2 = axs[0, 0].twinx(); ax2.spines['right'].set_visible(True); ax2.plot(t, a.eroding_Mkm2_adj, color=ns.C['grey'], lw=0.9, ls=':', label='eroding land (right axis)')
    ax2.set_ylim(0, 200); ax2.set_ylabel('Eroding land (10$^6$ km$^2$)'); ax2.set_xlim(540, 0)
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[0, 0].legend(h0 + h2, l0 + l2, loc='upper left', ncol=1, handlelength=1.8, borderaxespad=0.1)
    ax = axs[1, 1]
    for col, c, nm in (('D_Gt', ns.C['grey'], 'erosion flux'), ('F_sil', ns.C['sil'], 'silicate sink'), ('F_net', 'k', 'net inorganic sink')):
        ax.plot(t, a[col + '_adjc'] / a[col + '_ref'], color=c, lw=1.1, label=nm)
    ax.axhline(1, color='k', lw=0.5); ax.set_ylim(0.8, 1.6); ax.legend(loc='upper right', handlelength=1.2, borderaxespad=0.1, handletextpad=0.4); fin(ax, 'd', 'Adjusted / PALEOMAP')
    # e, f: the same two diagnostics as Fig. 3 with the elevation ensemble, the climate states and goSPL itself
    for ax, k, f, lab, L, ylim in ((axs[2, 0], 'S_sil', 1.0, 'Erosion sensitivity, silicate sink', 'e', (0, 0.45)), (axs[2, 1], 'beyond_frac_eroding', 100.0, 'Eroding land with $E \\geq E_{opt}$ (%)', 'f', (0, 80))):
        if cli is not None:
            for s_, c_ in (('adj', ns.C['adj']), ('ref', ns.C['ref'])):
                ax.fill_between(cli.age, f * cli[f'{k}_{s_}_p05'], f * cli[f'{k}_{s_}_p95'], color=c_, alpha=0.18, lw=0)
        if ens is not None:
            e_ = ens[ens.n_members == ens.n_members.max()]
            ax.fill_between(e_.age, f * e_[f'{k}_p05'], f * e_[f'{k}_p95'], color=ns.C['adj'], alpha=0.6, lw=0)
        ax.plot(t, f * a[f'{k}_adj'], color=ns.C['adj'], lw=0.9); ax.plot(t, f * a[f'{k}_ref'], color=ns.C['ref'], lw=0.9)
        if pub is not None:
            pb = pub.sort_values('age'); ax.plot(pb.age, f * pb[k], color='k', lw=0.6)
        if gos is not None:
            for case, c_ in (('central', ns.C['adj']), ('reference', ns.C['ref'])):
                g = gos[(gos.metric == k) & (gos.case == case)]
                ax.plot(g.age, f * g.goSPL, ls='none', marker='o', ms=3.4, mfc='white', mec=c_, mew=1.0, zorder=6)
        ax.set_ylim(*ylim); fin(ax, L, lab)
    h = [Patch(color=ns.C['grey'], alpha=0.3, lw=0, label='climate states, 5–95%'), Patch(color=ns.C['adj'], alpha=0.6, lw=0, label='elevation ensemble, 5–95%'),
         Line2D([], [], color='k', lw=0.6, label='published goSPL fields'), Line2D([], [], ls='none', marker='o', ms=3.4, mfc='white', mec='k', mew=1.0, label='direct goSPL runs')]
    axs[2, 0].legend(handles=h, loc='upper right', ncol=1, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)
    fig.savefig(FIG / 'ed6_surfaces.png', dpi=400); fig.savefig(FIG / 'ed6_surfaces.pdf')

if '7' in which:
    ol = series('kin_bridge_s2_oclin')
    fig, axs = plt.subplots(1, 2, figsize=(ns.DOUBLE, 2.5), gridspec_kw=dict(left=0.085, right=0.985, top=0.94, bottom=0.18, wspace=0.27))
    ax = axs[0]
    org = t <= 400      # biospheric burial only once land plants supply biospheric carbon (as Fig. 5; v16)
    ax.plot(t, 1e3 * a.F_sil_adj, color=ns.C['sil'], lw=1.1, label='silicate sink'); ax.plot(t[org], 1e3 * a.F_ocb_adj.values[org], color=ns.C['sky'], lw=1.1, label='biospheric burial')
    ax.plot(t, -1e3 * a.F_sulf_adj, color=ns.C['sulf'], lw=1.1, label='sulfide oxidation'); ax.plot(t, -1e3 * a.F_ocp_adj, color=ns.C['oc'], lw=1.1, label='rock OC oxidation')
    ax.plot(t, -1e3 * ol.F_ocp_adj, color=ns.C['oc'], lw=0.8, ls=':')
    ax.axhline(0, color='k', lw=0.5); ax.set_ylim(-130, 200); ax.legend(loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.6, handletextpad=0.4, borderaxespad=0.3); fin(ax, 'a', 'CO$_2$ drawdown (Mt C yr$^{-1}$)')
    ax = axs[1]
    ax.plot(t, (a.dF_sil_adj - a.dF_sulf_adj) / a.F_sil_adj, color='k', lw=1.1, label='inorganic')
    ax.plot(t, a.S_net_oc_rel_sil_adj, color=ns.C['oc'], lw=1.1, label='inorganic + rock OC'); ax.plot(t, ol.S_net_oc_rel_sil_adj, color=ns.C['oc'], lw=0.8, ls=':')
    ax.plot(t[org], a.S_total_rel_sil_adj.values[org], color=ns.C['sky'], lw=1.1, label='inorganic + rock OC + burial')
    ax.axhline(0, color='k', lw=0.5); ax.set_ylim(-1.25, 0.5); ax.legend(loc='upper right', ncol=1, handlelength=1.2, columnspacing=0.6, handletextpad=0.4, borderaxespad=0.3)
    fin(ax, 'b', 'd(net sink)/dln$E$ / silicate sink')
    fig.savefig(FIG / 'ed7_organic_carbon.png', dpi=400); fig.savefig(FIG / 'ed7_organic_carbon.pdf')

if '9' in which:
    # reactivity and the intrinsic weathering feedback
    r = pd.read_csv(OUT / 'reactivity_summary.csv').set_index('age')
    fig, axs = plt.subplots(1, 2, figsize=(ns.DOUBLE, 2.4), gridspec_kw=dict(left=0.065, right=0.99, top=0.94, bottom=0.19, wspace=0.22))
    ax = axs[0]; ax.plot(t, r.k_fixco2, color=ns.C['sil'], lw=1.1, label='560 ppm'); ax.plot(t, r.k_fixforcing, color='k', lw=1.1, label='constant forcing')
    ax.plot(t, r.D_rel, color=ns.C['oc'], lw=1.1, label='solid-Earth degassing'); ax.set_ylim(0, 2.3); ax.legend(loc='upper left', ncol=1, handlelength=1.6, borderaxespad=0.1); fin(ax, 'a', 'Relative to present', x=-0.13)
    ax = axs[1]; ax.plot(t, r.n_model, color='k', lw=1.1); ax.set_ylim(-0.05, 0.2); fin(ax, 'b', 'Intrinsic feedback strength, $n$', x=-0.14)
    fig.savefig(FIG / 'ed9_reactivity.png', dpi=400); fig.savefig(FIG / 'ed9_reactivity.pdf')

if '8' in which:
    h = pd.read_csv(OUT / 'highland_stats.csv').set_index('age')
    fig, axs = plt.subplots(1, 3, figsize=(ns.DOUBLE, 2.4), gridspec_kw=dict(left=0.065, right=0.99, top=0.94, bottom=0.19, wspace=0.30))
    ax = axs[0]; ax.plot(t, 100 * h.high_beyond, color=ns.C['adj'], lw=1.1, label='above 1,500 m'); ax.plot(t, 100 * h.low_beyond, color=ns.C['ref'], lw=1.1, label='below 1,500 m')
    ax.set_ylim(0, 119); ax.legend(loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4); fin(ax, 'a', 'Eroding land with $E \\geq E_{opt}$ (%)', x=-0.2)
    ax = axs[1]; ax.plot(t, 100 * h.high_share_eroding, color=ns.C['grey'], lw=1.1, label='area'); ax.plot(t, 100 * h.high_share_denudation, color=ns.C['adj'], lw=1.1, label='erosion flux')
    ax.plot(t, 100 * h.high_share_net, color='k', lw=1.1, label='net drawdown'); ax.set_ylim(0, 52); ax.legend(loc='upper left', ncol=3, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4); fin(ax, 'b', 'Share from above 1,500 m (%)', x=-0.2)
    ax = axs[2]; ax.plot(t, h.high_S_net, color=ns.C['adj'], lw=1.1, label='above 1,500 m'); ax.plot(t, h.low_S_net, color=ns.C['ref'], lw=1.1, label='below 1,500 m'); ax.axhline(0, color='k', lw=0.5)
    ax.set_ylim(-1.4, 0.75); ax.legend(loc='upper left', ncol=2, handlelength=1.2, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.4); fin(ax, 'c', 'Erosion sensitivity, net sink', x=-0.22)
    ns.equal_gaps(fig, list(axs))
    fig.savefig(FIG / 'ed8_mountains.png', dpi=400); fig.savefig(FIG / 'ed8_mountains.pdf')

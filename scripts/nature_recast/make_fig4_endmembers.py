#!/usr/bin/env python3
"""Fig. 4: mountains. (a) Eroding land relative to the local optimum at three ages; (b, c) a humid
(Caledonian, 420 Ma) and an arid (Central Pangaean, 300 Ma) mountain belt; (d) their erosion-rate
distributions on the local drawdown curves; (e, f) thirteen Phanerozoic mountain belts followed through the plate
model: event-scale erosion sensitivity and the split of the change in net drawdown into exposure and climate
(orogen_tracks.py, OROGEN_SET=events; needs orogen_sensitivity_events.csv)."""
import os
os.environ['OROGEN_SET'] = 'events'          # panels e, f read the catalogue of orogen_catalogue.py
import numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import cellfields as cf, run_weathering as rw
from _load import FIG, OUT

FH = 6.69                          # Nature maximum depth, 170 mm
fig = plt.figure(figsize=(ns.DOUBLE, FH))
# v13 layout, reading order a b / c d / e f: a and c on the left, b and d on the right, e and f beneath and taller
L, R, CW = 0.075, 0.585, 0.39; ROW1, ROW2, RH = 0.745, 0.455, 0.225
ax = fig.add_axes([L, ROW1, CW, RH])
H = np.load(OUT / 'kin_bridge_s2_hist.npz'); xb = H['xbins']; xc = 0.5 * (xb[1:] + xb[:-1]); ages = list(H['ages'])
for age, c in ((480, '#0072B2'), (200, '#E69F00'), (0, '#D55E00')):
    h = H['x_adj'][ages.index(age)]; h = h / h.sum() / (xb[1] - xb[0])
    ax.plot(10 ** xc, h, color=c, lw=1.2, label='Present' if age == 0 else f'{age} Ma')
ax.axvline(1, color='k', lw=0.6); ax.set_xscale('log'); ax.set_xlim(1e-2, 1e2); ax.set_ylim(0, 1.12)
ax.set_xlabel('Erosion rate relative to local optimum, $E/E_{opt}$'); ax.set_ylabel('Share of eroding land\n(per decade of $E/E_{opt}$)')
ax.legend(loc='upper left', handlelength=1.5, borderaxespad=0.1)
ax.text(1.2, 1.09, 'beyond optimum', fontsize=ns.SMALL, va='top'); ax.text(0.83, 1.09, 'rising limb', fontsize=ns.SMALL, va='top', ha='right')
ns.panel_letter(ax, 'a', x=-0.17, y=0.97)

from matplotlib.colors import Normalize
norm = Normalize(0, 6.0)               # one-sided: nearly all of both belts draws down CO2; net sources are drawn in orange (v13)
amp = cf.amps(); Eg = np.logspace(-3, 0.7, 400)
axd = fig.add_axes([R, ROW2, CW, RH])
BELTS = ((420, -10.6, -13.2, 'Caledonian belt, 420 Ma', '#0072B2'), (300, -5.9, -2.3, 'Central Pangaean belt, 300 Ma', '#D55E00'))
for k, (age, lat0, lon0, name, col) in enumerate(BELTS):
    g, d = cf.compute(age)
    la, lo = g['lat'], g['lon']
    i0, i1 = np.searchsorted(la, [lat0 - 16, lat0 + 16]); j0, j1 = np.searchsorted(lo, [lon0 - 30, lon0 + 30])
    ax = fig.add_axes([(R, L)[k], (ROW1, ROW2)[k], CW, RH])
    sub = lambda f: g[f][i0:i1, j0:j1]
    depo = (np.nan_to_num(sub('mask')) > 0) & (np.nan_to_num(sub('E'), nan=1.0) <= 0)      # land with net deposition: grey, as in Fig. 2 (v13)
    ax.pcolormesh(lo[j0:j1], la[i0:i1], np.ma.masked_where(~depo, np.ones(depo.shape)), cmap=ListedColormap(['#8A8A8A']), rasterized=True, shading='auto')
    netc = sub('net') * 12.011
    pm = ax.pcolormesh(lo[j0:j1], la[i0:i1], np.ma.masked_where(depo | (netc < 0), netc), cmap='Blues', norm=norm, rasterized=True, shading='auto')
    src = np.nan_to_num(netc, nan=0.0) < 0
    kern = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2); kern /= kern.sum()             # separable Gaussian, sigma 1.5 cells (0.15 deg)
    src_s = np.apply_along_axis(lambda v: np.convolve(v, kern, 'same'), 0, src.astype(float)); src_s = np.apply_along_axis(lambda v: np.convolve(v, kern, 'same'), 1, src_s)   # the net-source cells drawn as a smoothed region, not cell by cell
    if (src_s >= 0.5).any():
        ax.contourf(lo[j0:j1], la[i0:i1], src_s, levels=[0.5, 1.5], colors=['#D55E00'], antialiased=True)
    ax.contour(lo[j0:j1], la[i0:i1], np.nan_to_num(sub('z'), nan=-1), levels=[1500], colors='k', linewidths=0.8)
    ax.contour(lo[j0:j1], la[i0:i1], np.nan_to_num(sub('mask')), levels=[0.5], colors='k', linewidths=0.35)
    ax.set_aspect('equal'); ax.set_facecolor('white')     # sea white; the centre of the scale is near-white too, so the sea was grey before v13
    xt = [x for x in range(-180, 181, 20) if lo[j0] < x < lo[j1 - 1]]; yt = [y for y in range(-80, 81, 10) if la[i0] < y < la[i1 - 1]]
    ax.set_xticks(xt); ax.set_yticks(yt); ax.grid(color='#888888', lw=0.25, alpha=0.6)
    ax.set_xticklabels([f'{abs(x)}°{"W" if x < 0 else "E" if x > 0 else ""}' for x in xt]); ax.set_yticklabels([f'{abs(y)}°{"S" if y < 0 else "N" if y > 0 else ""}' for y in yt])
    ax.tick_params(length=2, labelsize=ns.SMALL, pad=1.5)
    for s in ax.spines.values(): s.set_visible(True)
    ax.text(0.015, 0.965, name, transform=ax.transAxes, va='top', fontsize=ns.BASE, bbox=dict(fc='w', ec='none', pad=1.2, alpha=0.9))
    ns.panel_letter(ax, 'bc'[k], x=-0.17, y=0.97)
    sel = (sub('z') > 1500) & (sub('E') > 0); A = sub('area')[sel]
    st = dict(E=np.average(sub('E')[sel], weights=A), q=np.average(sub('q')[sel], weights=A), net=np.average(sub('net')[sel], weights=A) * 12.011,
              beyond=np.average(sub('dnet')[sel] <= 0, weights=A), source=np.average(sub('net')[sel] < 0, weights=A), area=A.sum() / 1e12)
    print(name, {a_: round(float(b_), 3) for a_, b_ in st.items()})
    ax.text(0.985, 0.04, f"above 1,500 m: $E$ = {st['E']:.2f} mm yr$^{{-1}}$, runoff {round(st['q'], -1):,.0f} mm yr$^{{-1}}$,\nnet drawdown {st['net']:.2g} t C km$^{{-2}}$ yr$^{{-1}}$",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=ns.SMALL, bbox=dict(fc='w', ec='none', pad=1.2, alpha=0.9))
    # panel d: local curve at the belt's mean runoff, with its erosion-rate distribution
    sil = sulf = 0
    for r in rw.SETS:
        tt = rw.region_terms(Eg, np.full_like(Eg, st['q']), np.full_like(Eg, 288.15), cf.FITS[r], 'kinetic')
        sil = sil + amp[r]['sil'] * tt['sil'] / 3; sulf = sulf + amp[r]['sulf'] * tt['sulf'] / 3
    axd.plot(Eg, (sil - sulf) * 12.011, color=col, lw=1.2)
    h, b = np.histogram(np.log10(sub('E')[sel]), bins=np.linspace(-3, 0.7, 38), weights=A); h = h / h.max()
    axd.fill_between(10 ** (0.5 * (b[1:] + b[:-1])), -6 + 0 * h, -6 + 3.2 * h, color=col, alpha=0.35, lw=0, step='mid')
axd.axhline(0, color='k', lw=0.4); axd.set_xscale('log'); axd.set_xlim(1e-3, 5); axd.set_ylim(-6, 10.5)
axd.set_xlabel('Erosion rate, $E$ (mm yr$^{-1}$)'); axd.set_ylabel('Net CO$_2$ drawdown\n(t C km$^{-2}$ yr$^{-1}$)')
axd.text(0.0012, 9.9, 'curves: net drawdown at the mean runoff of each belt\nshading: erosion rates above 1,500 m', fontsize=ns.SMALL, va='top')
axd.text(0.02, 3.4, 'Caledonian', color='#0072B2', fontsize=ns.SMALL, ha='right'); axd.text(0.0045, -1.6, 'Central Pangaean', color='#D55E00', fontsize=ns.SMALL, ha='center')
ns.panel_letter(axd, 'd', x=-0.17, y=0.97)
cax = fig.add_axes([L, ROW2 - 0.05, 0.27, 0.012]); cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='max')
from matplotlib.patches import Rectangle
fig.patches.append(Rectangle((L + 0.30, ROW2 - 0.05), 0.014, 0.012, transform=fig.transFigure, fc='#D55E00', ec='none'))   # key for the net-source colour, beside the colour bar
fig.text(L + 0.318, ROW2 - 0.044, 'net CO$_2$ source', fontsize=ns.SMALL, va='center', ha='left')
cb.set_ticks([0, 2, 4, 6]); cb.set_label('Net CO$_2$ drawdown (t C km$^{-2}$ yr$^{-1}$)', fontsize=ns.SMALL, labelpad=1); cb.ax.tick_params(labelsize=ns.SMALL, length=2)
# panels e, f: thirteen mountain belts followed through the plate model, from the age before uplift to the age of the belt
import orogen_tracks as ot
axe = fig.add_axes([0.085, 0.165, 0.39, 0.18]); axf = fig.add_axes([0.61, 0.165, 0.375, 0.18])
ot.event_panels(axe, axf, compact=True)
ns.panel_letter(axe, 'e', x=-0.2, y=1.0); ns.panel_letter(axf, 'f', x=-0.29, y=1.0)
fig.savefig(FIG / 'fig4_endmembers.png', dpi=400); fig.savefig(FIG / 'fig4_endmembers.pdf', dpi=400)

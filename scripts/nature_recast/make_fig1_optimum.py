#!/usr/bin/env python3
"""Fig. 1: the drawdown optimum. Net inorganic CO2 drawdown as a function of erosion rate at three values of runoff
(central estimate of the three regional parameter sets, 15 °C), with the silicate sink and the sulfide source at
500 mm/yr. Cited in the introduction, before the global results."""
import numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
import cellfields as cf, run_weathering as rw
from _load import FIG

amp = cf.amps(); E = np.logspace(-3.3, 0.7, 500)
def curves(q):
    sil = sulf = 0
    for r in rw.SETS:
        tt = rw.region_terms(E, np.full_like(E, q), np.full_like(E, 288.15), cf.FITS[r], 'kinetic')
        sil = sil + amp[r]['sil'] * tt['sil'] / 3; sulf = sulf + amp[r]['sulf'] * tt['sulf'] / 3
    return sil * 12.011, sulf * 12.011

fig, ax = plt.subplots(figsize=(ns.SINGLE, 2.75), gridspec_kw=dict(left=0.17, right=0.97, top=0.97, bottom=0.165))
BOX = dict(fc='w', ec='none', pad=0.8, alpha=0.8)
for q, c, lab in ((150, '#E69F00', '150'), (500, '#009E73', '500'), (1500, '#0072B2', '1,500')):
    s, f = curves(q); n = s - f
    ax.plot(E, n, color=c, lw=1.3, label=f'{lab} mm yr$^{{-1}}$')
    i = np.argmax(n); ax.plot(E[i], n[i], 'o', ms=3.2, color=c, mec='k', mew=0.4, zorder=5)
    if q == 500:
        ax.plot(E, s, color=c, lw=0.7, ls='--'); ax.plot(E, -f, color=c, lw=0.7, ls=':')
        ax.text(4.2, 1.35, 'silicate sink', fontsize=ns.SMALL, color=c, ha='right', va='bottom', bbox=BOX)
        ax.text(0.30, -2.6, 'sulfide\nsource', fontsize=ns.SMALL, color=c, ha='right', va='top')
ax.axhline(0, color='k', lw=0.4); ax.set_xscale('log'); ax.set_xlim(E[0], E[-1]); ax.set_ylim(-5, 7)
ax.set_xlabel('Erosion rate, $E$ (mm yr$^{-1}$)'); ax.set_ylabel('Net CO$_2$ drawdown (t C km$^{-2}$ yr$^{-1}$)')
ax.legend(title='Net drawdown at a runoff of', loc='upper left', handlelength=1.5, borderaxespad=0.1, title_fontsize=ns.SMALL)
fig.savefig(FIG / 'fig1_optimum.png', dpi=400); fig.savefig(FIG / 'fig1_optimum.pdf')

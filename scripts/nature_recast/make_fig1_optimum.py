#!/usr/bin/env python3
"""Fig. 1 (single panel): the drawdown optimum.

Net inorganic CO2 drawdown per unit land area as a function of erosion rate at three values of runoff (150, 500
and 1,500 mm/yr), for the central estimate (mean of the three calibrated regional parameter sets of Bufe et al.
2024) at 15 °C. Each solid curve is silicate sink minus sulfide source; a dot marks its maximum, the local optimum
E_opt at that runoff. For 500 mm/yr the silicate sink (dashed) and the sulfide source (dotted, drawn negative) are
also shown. The figure is cited in the introduction, before the global results.

The curves are evaluated directly from the weathering law (run_weathering.region_terms, runoff in the kinetic
term) with the calibration amplitudes of the main run, so no per-cell fields are needed.

Inputs:  data/nature_recast/kin_bridge_s2_amplitudes.txt (calibration amplitudes, via cellfields.amps) and
         data/bufe_2024_params.json (fitted parameter sets, via cellfields.FITS)
Outputs: figures/nature/fig1_optimum.png and .pdf

    python3 make_fig1_optimum.py
"""
import numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
import cellfields as cf, run_weathering as rw
from _load import FIG

# erosion rates from 10^-3.3 to 10^0.7 mm/yr (about 0.0005-5 mm/yr), log-spaced
amp = cf.amps(); E = np.logspace(-3.3, 0.7, 500)
def curves(q):
    """Silicate CO2 sink and sulfide CO2 source (t C km-2 yr-1) along E at runoff q (mm/yr) and 15 °C (288.15 K).

    Each parameter set is scaled by its own silicate and sulfide calibration amplitude and the three are averaged
    (the central estimate). region_terms returns mol C m-2 yr-1; times 12.011 g/mol gives g C m-2 yr-1, which
    equals t C km-2 yr-1."""
    sil = sulf = 0
    for r in rw.SETS:
        tt = rw.region_terms(E, np.full_like(E, q), np.full_like(E, 288.15), cf.FITS[r], 'kinetic')
        sil = sil + amp[r]['sil'] * tt['sil'] / 3; sulf = sulf + amp[r]['sulf'] * tt['sulf'] / 3
    return sil * 12.011, sulf * 12.011

fig, ax = plt.subplots(figsize=(ns.SINGLE, 2.75), gridspec_kw=dict(left=0.17, right=0.97, top=0.97, bottom=0.165))
BOX = dict(fc='w', ec='none', pad=0.8, alpha=0.8)    # semi-transparent white backing for labels that cross curves
# one solid curve of net drawdown per runoff value; the dot is the maximum of that curve, i.e. the optimum E_opt
for q, c, lab in ((150, '#E69F00', '150'), (500, '#009E73', '500'), (1500, '#0072B2', '1,500')):
    s, f = curves(q); n = s - f
    ax.plot(E, n, color=c, lw=1.3, label=f'{lab} mm yr$^{{-1}}$')
    i = np.argmax(n); ax.plot(E[i], n[i], 'o', ms=3.2, color=c, mec='k', mew=0.4, zorder=5)
    if q == 500:      # the two components at the middle runoff: silicate sink dashed, sulfide source dotted and negative
        ax.plot(E, s, color=c, lw=0.7, ls='--'); ax.plot(E, -f, color=c, lw=0.7, ls=':')
        ax.text(4.2, 1.35, 'silicate sink', fontsize=ns.SMALL, color=c, ha='right', va='bottom', bbox=BOX)
        ax.text(0.30, -2.6, 'sulfide\nsource', fontsize=ns.SMALL, color=c, ha='right', va='top')
ax.axhline(0, color='k', lw=0.4); ax.set_xscale('log'); ax.set_xlim(E[0], E[-1]); ax.set_ylim(-5, 7)
ax.set_xlabel('Erosion rate, $E$ (mm yr$^{-1}$)'); ax.set_ylabel('Net CO$_2$ drawdown (t C km$^{-2}$ yr$^{-1}$)')
ax.legend(title='Net drawdown at a runoff of', loc='upper left', handlelength=1.5, borderaxespad=0.1, title_fontsize=ns.SMALL)
fig.savefig(FIG / 'fig1_optimum.png', dpi=400); fig.savefig(FIG / 'fig1_optimum.pdf')

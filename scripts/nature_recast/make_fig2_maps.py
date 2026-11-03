#!/usr/bin/env python3
"""Fig. 2: elevation, erosion, net inorganic CO2 drawdown and position relative to the local
drawdown optimum at four ages (rock-chemistry-adjusted surface)."""
import sys, numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
from _wintri_map import projected_mesh, decorate_global_map, draw_mask_contour
import cellfields as cf
from _load import FIG

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
DEPO = ListedColormap(['#8A8A8A'])     # mid grey, distinct from the pale ends of LXMAP
LXMAP = LinearSegmentedColormap.from_list('lx', ['#005F73', '#2BB5C8', '#F7F7F7', '#F2A552', '#B33A00'])   # cyan (rising limb) to orange (beyond), v13: purple was lost against the grey     # land with net deposition (no erosion rate, no E/E_opt): grey in panels b and d
ELEV = LinearSegmentedColormap.from_list('elev', ['#2E6B3E', '#8DB36B', '#E8DDA0', '#C49A5A', '#8C5A3C', '#F2EEEA'])
AGES = [int(a) for a in sys.argv[1:]] or [450, 300, 90, 0]
STEP = 3
rows = [('z', 'Elevation (m)', dict(cmap=ELEV, norm=Normalize(0, 4000))),
        ('E', 'Erosion rate (mm yr$^{-1}$)', dict(cmap='YlOrBr', norm=LogNorm(1e-3, 1.0))),
        ('netC', 'Net CO$_2$ drawdown (t C km$^{-2}$ yr$^{-1}$)', dict(cmap='RdBu', norm=TwoSlopeNorm(0, -1.5, 3.0))),
        ('lx', 'Erosion rate relative to local optimum, $E/E_{opt}$', dict(cmap=LXMAP, norm=TwoSlopeNorm(0, -2, 2)))]
fig = plt.figure(figsize=(ns.DOUBLE, 5.5))
W = 0.232; H = 0.185; X0 = 0.035; Y0 = 0.972; PITCH = 0.2425
for j, age in enumerate(AGES):
    g, d = cf.compute(age)
    g['netC'] = g['net'] * 12.011
    g['lx'] = np.log10(g['x'])
    depo = (np.nan_to_num(g['mask']) > 0) & (np.nan_to_num(g['E'], nan=1.0) <= 0)
    sl = (slice(None, None, STEP), slice(None, None, STEP))
    X, Y = projected_mesh(g['lon'][::STEP], g['lat'][::STEP])
    for i, (k, lab, kw) in enumerate(rows):
        ax = fig.add_axes([X0 + j * (W + 0.006), Y0 - i * PITCH - H, W, H])
        f = g[k][sl]
        if k == 'E':
            f = np.where(f > 0, f, np.nan)
        if k in ('E', 'lx'):
            ax.pcolormesh(X, Y, np.ma.masked_where(~depo[sl], np.ones(depo[sl].shape)), cmap=DEPO, shading='auto', rasterized=True)
        pm = ax.pcolormesh(X, Y, np.ma.masked_invalid(f), shading='auto', rasterized=True, **kw)
        draw_mask_contour(ax, X, Y, np.nan_to_num(g['mask'][sl]), lw=0.3)
        if k == 'netC':   # the humid land: runoff above 500 mm/yr (the middle curve of Fig. 1), dashed (v13)
            # the runoff field is smoothed (normalised Gaussian on land, sigma 1 deg) so that the contour outlines regions rather than
            # tracing every cell; closed contours enclosing less than about 4 deg x 4 deg are dropped
            qf = np.nan_to_num(g['q'][sl], nan=0.0); lm = np.isfinite(g['q'][sl]).astype(float)
            sig = 1.0 / abs(float(g['lat'][STEP] - g['lat'][0])); r = int(3 * sig)
            kern = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); kern /= kern.sum()
            sm = lambda f: np.apply_along_axis(lambda v: np.convolve(v, kern, 'same'), 1, np.apply_along_axis(lambda v: np.convolve(v, kern, 'same'), 0, f))
            qs = np.where(lm > 0, sm(qf * lm) / np.maximum(sm(lm), 1e-6), 0.0)
            cs = ax.contour(X, Y, qs, levels=[500.0], colors='#D55E00', linestyles='solid', linewidths=0.6)
            for coll in getattr(cs, 'collections', [cs]):
                pth = coll.get_paths() if hasattr(coll, 'get_paths') else []
                keep = []
                for pp in pth:
                    for poly in pp.to_polygons():
                        if len(poly) > 2 and np.ptp(poly[:, 0]) * np.ptp(poly[:, 1]) >= (4 * np.ptp(X) / 360) * (4 * np.ptp(Y) / 180):
                            keep.append(type(pp)(poly))
                if hasattr(coll, 'set_paths'):
                    coll.set_paths(keep)
        decorate_global_map(ax)
        if i == 0:
            ax.text(0.5, 1.0, f'{age} Ma' if age else 'Present', transform=ax.transAxes, ha='center', va='bottom', fontsize=ns.BASE)
        if j == 0:
            ns.panel_letter(ax, 'abcd'[i], x=-0.12, y=0.84)
        if j == len(AGES) - 1:
            cax = fig.add_axes([0.30, Y0 - i * PITCH - H - 0.020, 0.40, 0.008])
            cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
            cb.ax.tick_params(labelsize=ns.SMALL, length=2, pad=1.5)
            if k == 'lx':
                cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['0.01', '0.1', '1', '10', '100'])
            fig.text(0.715, Y0 - i * PITCH - H - 0.016, lab, fontsize=ns.SMALL, va='center', ha='left')
fig.savefig(FIG / 'fig2_maps.png', dpi=400); fig.savefig(FIG / 'fig2_maps.pdf', dpi=400)

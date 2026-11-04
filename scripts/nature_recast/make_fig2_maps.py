#!/usr/bin/env python3
"""Fig. 2: elevation, erosion, net inorganic CO2 drawdown and position relative to the local
drawdown optimum at four ages (rock-chemistry-adjusted surface, main model).

Global maps (Winkel tripel) in a grid of four rows (panels a-d) by four columns (ages; 450, 300 and 90 Ma and the
present by default):
  a  palaeoelevation of the adjusted surface (m, 0-4,000)
  b  erosion rate of the calibrated field (mm/yr, log scale 0.001-1); land with net deposition in grey
  c  net inorganic CO2 drawdown, silicate sink minus sulfide source (t C km-2 yr-1; blue drawdown, red net
     source), with the 500 mm/yr runoff contour outlining the humid land
  d  erosion rate relative to the local optimum, E/E_opt (log scale 0.01-100; cyan on the rising limb below the
     optimum, orange beyond it); land with net deposition in grey
The per-cell fields come from cellfields.compute (erosion factor 2.0, calibration of the main run kin_bridge_s2);
panel c is the inorganic budget as long as the environment variable NET_BUDGET is unset or 'inorg' (the default).
Maps are drawn from every third 0.1° node (0.3°) to keep the file size manageable.

Inputs:  the land-only input cache (nature_recast_cache.zip, build_cache.py) and the calibration amplitudes in
         data/nature_recast/kin_bridge_s2_amplitudes.txt
Outputs: figures/nature/fig2_maps.png and .pdf

    python3 make_fig2_maps.py                 # the four ages of the paper
    python3 make_fig2_maps.py 480 200 0       # any other ages (Ma), one column each
"""
import sys, numpy as np
import _nature_style as ns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
from _wintri_map import projected_mesh, decorate_global_map, draw_mask_contour
import cellfields as cf
from _load import FIG

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
DEPO = ListedColormap(['#8A8A8A'])     # land with net deposition (no erosion rate, no E/E_opt): mid grey in panels b and d, distinct from the pale centre of LXMAP
LXMAP = LinearSegmentedColormap.from_list('lx', ['#005F73', '#2BB5C8', '#F7F7F7', '#F2A552', '#B33A00'])   # diverging: cyan below the optimum (rising limb), near-white at E = E_opt, orange beyond; no purple, which is hard to tell from the grey of depositing land
ELEV = LinearSegmentedColormap.from_list('elev', ['#2E6B3E', '#8DB36B', '#E8DDA0', '#C49A5A', '#8C5A3C', '#F2EEEA'])   # lowland green to highland brown and pale summits
AGES = [int(a) for a in sys.argv[1:]] or [450, 300, 90, 0]
STEP = 3                               # plot every third 0.1° node (0.3°)
# one entry per row (panel): field key in cellfields.compute output, colour-bar label, colour map and normalisation.
# netC: TwoSlopeNorm centred on zero so that white separates net drawdown (blue) from net source (red).
# lx: log10(E/E_opt), so -2..2 spans 0.01-100 with the optimum (0) at the centre.
rows = [('z', 'Elevation (m)', dict(cmap=ELEV, norm=Normalize(0, 4000))),
        ('E', 'Erosion rate (mm yr$^{-1}$)', dict(cmap='YlOrBr', norm=LogNorm(1e-3, 1.0))),
        ('netC', 'Net CO$_2$ drawdown (t C km$^{-2}$ yr$^{-1}$)', dict(cmap='RdBu', norm=TwoSlopeNorm(0, -1.5, 3.0))),
        ('lx', 'Erosion rate relative to local optimum, $E/E_{opt}$', dict(cmap=LXMAP, norm=TwoSlopeNorm(0, -2, 2)))]
fig = plt.figure(figsize=(ns.DOUBLE, 5.5))
# map width and height, left margin, top of the first row and row pitch, all in figure fractions
W = 0.232; H = 0.185; X0 = 0.035; Y0 = 0.972; PITCH = 0.2425
for j, age in enumerate(AGES):          # one column per age
    g, d = cf.compute(age)
    g['netC'] = g['net'] * 12.011       # mol C m-2 yr-1 -> t C km-2 yr-1
    g['lx'] = np.log10(g['x'])          # x = E/E_opt in eroding cells, NaN elsewhere
    depo = (np.nan_to_num(g['mask']) > 0) & (np.nan_to_num(g['E'], nan=1.0) <= 0)    # land cells with no net erosion
    sl = (slice(None, None, STEP), slice(None, None, STEP))
    X, Y = projected_mesh(g['lon'][::STEP], g['lat'][::STEP])
    for i, (k, lab, kw) in enumerate(rows):
        ax = fig.add_axes([X0 + j * (W + 0.006), Y0 - i * PITCH - H, W, H])
        f = g[k][sl]
        if k == 'E':
            f = np.where(f > 0, f, np.nan)      # log colour scale: only eroding cells carry a rate
        if k in ('E', 'lx'):                    # depositing land drawn grey underneath the field
            ax.pcolormesh(X, Y, np.ma.masked_where(~depo[sl], np.ones(depo[sl].shape)), cmap=DEPO, shading='auto', rasterized=True)
        pm = ax.pcolormesh(X, Y, np.ma.masked_invalid(f), shading='auto', rasterized=True, **kw)
        draw_mask_contour(ax, X, Y, np.nan_to_num(g['mask'][sl]), lw=0.3)
        if k == 'netC':   # the humid land: runoff above 500 mm/yr (the middle curve of Fig. 1), as a solid orange-red line
            # the runoff field is smoothed (normalised Gaussian on land, sigma 1 deg) so that the contour outlines regions rather than
            # tracing every cell; closed contours enclosing less than about 4 deg x 4 deg are dropped
            qf = np.nan_to_num(g['q'][sl], nan=0.0); lm = np.isfinite(g['q'][sl]).astype(float)
            # sigma of 1 deg in nodes of the subsampled grid; kernel truncated at 3 sigma.
            # Smoothing q*land and dividing by the smoothed land mask keeps the sea from diluting coastal runoff.
            sig = 1.0 / abs(float(g['lat'][STEP] - g['lat'][0])); r = int(3 * sig)
            kern = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); kern /= kern.sum()
            sm = lambda f: np.apply_along_axis(lambda v: np.convolve(v, kern, 'same'), 1, np.apply_along_axis(lambda v: np.convolve(v, kern, 'same'), 0, f))
            qs = np.where(lm > 0, sm(qf * lm) / np.maximum(sm(lm), 1e-6), 0.0)
            cs = ax.contour(X, Y, qs, levels=[500.0], colors='#D55E00', linestyles='solid', linewidths=0.6)
            # drop small closed contours: keep a polygon only if its bounding box in projected units is at least the
            # projected size of 4 deg x 4 deg (whole-map extent scaled by 4/360 and 4/180). Works both with Matplotlib
            # versions whose ContourSet has .collections and with those where the ContourSet holds the paths itself.
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
        if i == 0:      # age above the top row; panel letters on the first column; one colour bar per row beside the last column
            ax.text(0.5, 1.0, f'{age} Ma' if age else 'Present', transform=ax.transAxes, ha='center', va='bottom', fontsize=ns.BASE)
        if j == 0:
            ns.panel_letter(ax, 'abcd'[i], x=-0.12, y=0.84)
        if j == len(AGES) - 1:
            cax = fig.add_axes([0.30, Y0 - i * PITCH - H - 0.020, 0.40, 0.008])
            cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
            cb.ax.tick_params(labelsize=ns.SMALL, length=2, pad=1.5)
            if k == 'lx':       # ticks in log10 units, labelled with E/E_opt itself
                cb.set_ticks([-2, -1, 0, 1, 2]); cb.set_ticklabels(['0.01', '0.1', '1', '10', '100'])
            fig.text(0.715, Y0 - i * PITCH - H - 0.016, lab, fontsize=ns.SMALL, va='center', ha='left')
fig.savefig(FIG / 'fig2_maps.png', dpi=400); fig.savefig(FIG / 'fig2_maps.pdf', dpi=400)

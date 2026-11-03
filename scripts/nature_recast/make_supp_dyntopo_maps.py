#!/usr/bin/env python3
"""Supplementary Fig. 2: maps of the dynamic-topography term added in the sensitivity test, 250 Ma to the present.

The term is the one the test adds to the rock-chemistry-adjusted surface (Paleotopo_data_assimilation,
outputs_scotese_reach500_land/dt_term_all_crust/dt_term_all_lagr_0-250Ma.nc, variable term_robust, which holds the term on
all continental crust): the change in gld428 dynamic topography since the present day, carried point by point onto every
continental cell (dyntopo_term_lagrangian.py), low-passed with a 600-km Gaussian filter, with the mean level of each
continental block removed. Land without a term would be drawn grey (none is expected; the count is printed and the grey
legend entry appears only if there is any). The line is the shoreline of the adjusted surface on the same 1-degree grid
(dem_scotese_adjusted/<age>Ma_corrected_SW.nc, M_corrected > 0). Purple lines are the plate boundaries of the reconstruction
that drove the mantle-flow model, subduction zones with teeth on the overriding plate, placed on the map by the rotation
that best aligns the nearby continents of the two reconstructions (Paleotopo scripts_Scotese/flowmodel_boundaries_paleomap.py,
outputs_scotese_reach500_land/dt_term_all_crust/flowmodel_boundaries_paleomap_0-250Ma.npz; the PALEOMAP topologies end at
about 100 Ma and are not the boundaries that made the field).

    python3 make_supp_dyntopo_maps.py                  # 250, 200, 150, 100, 50, 25 Ma -> figures/nature/supp_dyntopo_term_maps.{png,pdf}
    python3 make_supp_dyntopo_maps.py 40 80 120 160 200 240
"""
import os, sys
from pathlib import Path
import numpy as np
import _nature_style as ns
import _config as cfg
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from _wintri_map import projected_mesh, decorate_global_map, draw_mask_contour, _project_line

PALEO = Path(os.environ.get('PALEO_ROOT', cfg.REPO.parents[1] / 'Paleotopo_data_assimilation'))
TERM = PALEO / 'outputs_scotese_reach500_land' / 'dt_term_all_crust' / 'dt_term_all_lagr_0-250Ma.nc'
BND = PALEO / 'outputs_scotese_reach500_land' / 'dt_term_all_crust' / 'flowmodel_boundaries_paleomap_0-250Ma.npz'
DEM = cfg.ZENODO / 'dem_scotese_adjusted'
BND_COLOR = '#6A1B9A'                           # purple: distinct from every colour of the roma scale and from the black shorelines
TOOTH_MM = (0.75, 0.42, 1.7)                    # tooth height, half base and spacing along the trench, in mm on the page
AGES = [int(a) for a in sys.argv[1:]] or [250, 200, 150, 100, 50, 25]
LIM = 500.0                                    # colour scale +-500 m; the term is mostly within +-400 m, larger values take the end colours
NOTERM = ListedColormap(['#D4D4D4'])
# Crameri's scientific colour map roma (doi:10.5281/zenodo.1243862), 33 anchor colours, reversed so that a positive change
# (crust higher in the past than today) is brown and a negative one blue; embedded so the script needs no colour-map package
ROMA = ['#7e1700', '#862c06', '#8f3c0c', '#964b12', '#9d5818', '#a3651e', '#a97223', '#b07f2a', '#b68c32', '#bd9b3c', '#c4aa4a',
        '#cbba5d', '#d0ca72', '#d2d78a', '#d1e2a1', '#cbe8b4', '#c0eac3', '#b1e9ce', '#9fe4d4', '#8bdcd7', '#76d1d7', '#62c4d4',
        '#51b7d0', '#43a9cb', '#389cc6', '#308fc1', '#2b82bb', '#2676b6', '#2269b0', '#1d5bab', '#184da4', '#0f3f9e', '#033198']
ROMA_R = LinearSegmentedColormap.from_list('roma_r', ROMA[::-1])


def pieces(arr):
    """(N, 2) lat/lon with NaN separators -> list of segments, also split where they cross the dateline."""
    out, cur = [], []
    for la, lo in arr:
        if not np.isfinite(la):
            if len(cur) > 1: out.append(np.array(cur))
            cur = []; continue
        if cur and abs(lo - cur[-1][1]) > 180:
            if len(cur) > 1: out.append(np.array(cur))
            cur = []
        cur.append((la, lo))
    if len(cur) > 1: out.append(np.array(cur))
    return out


def draw_boundaries(ax, bnd, age, mm_per_unit):
    """Plate boundaries of age `age`: other boundaries thin, subduction zones thicker with filled teeth on the side of the
    overriding plate (side +1 left of the segment direction, -1 right)."""
    for seg in pieces(bnd[f'other_{age}']):
        x, y = _project_line(seg[:, 1], seg[:, 0]); ax.plot(x, y, color=BND_COLOR, lw=0.35, zorder=5, solid_capstyle='round')
    h, b, sp = (v / mm_per_unit for v in TOOTH_MM)
    tri = []
    sides = bnd[f'sub_side_{age}']; k = 0
    raw = bnd[f'sub_{age}']; starts = [0] + [i + 1 for i in np.flatnonzero(~np.isfinite(raw[:, 0]))][:-1]
    for j, st in enumerate(starts):
        en = st + int(np.argmax(~np.isfinite(raw[st:, 0])))
        side = int(sides[j]) if j < len(sides) else 0
        for seg in pieces(raw[st:en + 1]):
            x, y = _project_line(seg[:, 1], seg[:, 0]); x = np.asarray(x); y = np.asarray(y)
            ax.plot(x, y, color=BND_COLOR, lw=0.7, zorder=5, solid_capstyle='round')
            if side == 0 or len(x) < 2:
                continue
            ds = np.hypot(np.diff(x), np.diff(y)); s = np.r_[0, np.cumsum(ds)]
            for s0 in np.arange(sp / 2, s[-1], sp):
                i = min(int(np.searchsorted(s, s0)) - 1, len(ds) - 1); i = max(i, 0)
                if ds[i] == 0: continue
                f = (s0 - s[i]) / ds[i]; px, py = x[i] + f * (x[i + 1] - x[i]), y[i] + f * (y[i + 1] - y[i])
                dx, dy = (x[i + 1] - x[i]) / ds[i], (y[i + 1] - y[i]) / ds[i]; nx, ny = -dy * side, dx * side
                tri.append([(px - dx * b, py - dy * b), (px + dx * b, py + dy * b), (px + nx * h, py + ny * h)])
    if tri:
        ax.add_collection(PolyCollection(tri, facecolors=BND_COLOR, edgecolors='none', zorder=5))


def read(path, keys):
    try:
        from netCDF4 import Dataset
        with Dataset(str(path)) as d:
            d.set_auto_mask(False)
            return {k: np.asarray(d.variables[k][:]) for k in keys}
    except ImportError:
        import h5py
        with h5py.File(str(path), 'r') as h:
            return {k: np.asarray(h[k][:]) for k in keys}


t = read(TERM, ('age', 'lat', 'lon', 'term_robust', 'robust'))
ages = np.rint(t['age']).astype(int).tolist()
X, Y = projected_mesh(t['lon'], t['lat'])
bnd = np.load(BND)
if not set(AGES) <= set(bnd['ages'].tolist()):
    raise SystemExit(f'plate boundaries missing for some of {AGES} in {BND}')

if len(AGES) != 6:
    raise SystemExit('six ages expected (two rows of three maps)')
fig = plt.figure(figsize=(ns.DOUBLE, 87 * ns.MM))
W, H = 0.318, 0.336                         # map axes, about the Winkel-Tripel proportions of a 58-mm map
n_grey = 0
for n, age in enumerate(AGES):
    i = ages.index(age)
    d = read(DEM / f'{age}Ma_corrected_SW.nc', ('lat', 'lon', 'M_corrected'))
    if not (np.allclose(d['lat'], t['lat']) and np.allclose(d['lon'], t['lon'])):
        raise SystemExit(f'grid of {age} Ma surface differs from the term grid')
    term = t['term_robust'][i].astype(float)
    rob = t['robust'][i].astype(bool) & np.isfinite(term)
    land = d['M_corrected'] > 0
    grey = (land | np.isfinite(term)) & ~rob
    n_grey += int((grey & land).sum())
    r, c = divmod(n, 3)
    ax = fig.add_axes([0.012 + c * 0.330, 0.594 - r * 0.414, W, H])
    ax.pcolormesh(X, Y, np.ma.masked_where(~grey, np.ones(grey.shape)), cmap=NOTERM, shading='nearest', rasterized=True)
    pm = ax.pcolormesh(X, Y, np.ma.masked_where(~rob, term), cmap=ROMA_R, norm=Normalize(-LIM, LIM),
                       shading='nearest', rasterized=True)
    draw_mask_contour(ax, X, Y, land, color='#202020', lw=0.45)
    draw_boundaries(ax, bnd, age, mm_per_unit=W * ns.DOUBLE / ns.MM / (np.nanmax(X) - np.nanmin(X)))
    decorate_global_map(ax)
    ax.text(0.0, 1.0, 'abcdef'[n], transform=ax.transAxes, fontsize=ns.PANEL, fontweight='bold', ha='left', va='bottom')
    ax.text(0.07, 1.0, f'{age} Ma', transform=ax.transAxes, fontsize=ns.BASE, ha='left', va='bottom')

cax = fig.add_axes([0.08, 0.115, 0.38, 0.025])
cb = fig.colorbar(pm, cax=cax, orientation='horizontal', extend='both')
cb.set_ticks([-500, -250, 0, 250, 500])
cb.ax.tick_params(labelsize=ns.SMALL, length=2)
cb.set_label('Change in dynamic topography since the present day (m)', fontsize=ns.BASE, labelpad=2)
print(f'land cells without the term over the {len(AGES)} maps: {n_grey}')
handles = ([Patch(facecolor='#D4D4D4', edgecolor='none', label='Land without the term')] if n_grey else []) + \
          [Line2D([], [], color='#202020', lw=0.6, label='Shoreline of the adjusted surface'),
           Line2D([], [], color=BND_COLOR, lw=0.7, marker='^', markersize=3.2, label='Subduction zone (teeth on the overriding plate)'),
           Line2D([], [], color=BND_COLOR, lw=0.35, label='Other plate boundary')]
fig.legend(handles=handles,
           loc='lower left', bbox_to_anchor=(0.56, 0.05), fontsize=ns.SMALL, frameon=False, handlelength=1.6)
out = cfg.FIG / 'supp_dyntopo_term_maps'
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=300)
print('wrote', out)

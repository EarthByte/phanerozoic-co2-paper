"""Small Winkel Tripel plotting helpers shared by the map figures and videos.

The global maps of the paper (Fig. 2, Extended Data Figs 2 and 3, Supplementary
Fig. 2 and Supplementary Videos 1-4) are drawn in the Winkel Tripel projection, the
projection of the published GMT palaeoelevation movies.  These helpers reproduce it
with PROJ (through pyproj), so the maps keep the same global geometry in matplotlib
without a GMT installation.

Typical use:
    x, y = projected_mesh(lon, lat)          # 1-D axes in degrees -> 2-D metres
    ax.pcolormesh(x, y, field, shading="nearest")
    decorate_global_map(ax)                  # graticule and outline
    draw_mask_contour(ax, x, y, land_mask)   # coastline of the displayed surface
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

# Winkel Tripel centred on the 0-degree meridian, projected coordinates in metres
WINTRI = CRS.from_proj4("+proj=wintri +lon_0=0 +datum=WGS84 +units=m +no_defs")
WGS84 = CRS.from_epsg(4326)
# always_xy: inputs are (lon, lat) in that order, whatever the CRS axis convention
TRANSFORMER = Transformer.from_crs(WGS84, WINTRI, always_xy=True)


def projected_mesh(lon, lat):
    """Return projected 2-D cell-centre coordinates for 1-D lon/lat axes.

    lon (nlon,) and lat (nlat,) in degrees.  Returns (x, y), each (nlat, nlon), in
    metres, matching a (lat, lon)-ordered field for pcolormesh or contour.
    """
    lo, la = np.meshgrid(np.asarray(lon), np.asarray(lat))
    return TRANSFORMER.transform(lo, la)


def _project_line(lon, lat):
    """Project matching 1-D lon/lat sequences (degrees) to x, y (metres)."""
    return TRANSFORMER.transform(np.asarray(lon), np.asarray(lat))


def decorate_global_map(ax, *, graticule=True):
    """Draw a plain frame and restrained 60°/30° graticule.

    Meridians every 60 degrees (-120 to 120) and parallels every 30 degrees (-60 to
    60) are drawn as densely sampled projected lines, followed by the map outline;
    the axes are set to equal aspect and hidden.
    """
    if graticule:
        for lo in np.arange(-120, 180, 60):
            # stop just short of the poles, where all meridians converge
            la = np.linspace(-89.5, 89.5, 361)
            x, y = _project_line(np.full_like(la, lo), la)
            ax.plot(x, y, color="#AEB6BA", lw=0.35, alpha=0.65, zorder=3)
        for la0 in np.arange(-60, 90, 30):
            lo = np.linspace(-180, 180, 721)
            x, y = _project_line(lo, np.full_like(lo, la0))
            ax.plot(x, y, color="#AEB6BA", lw=0.35, alpha=0.65, zorder=3)

    # Projected outline, sampled along all four geographic boundaries.
    lo = np.linspace(-180, 180, 721)
    la = np.linspace(-90, 90, 361)
    segments = [
        (lo, np.full_like(lo, -90)),
        (lo, np.full_like(lo, 90)),
        (np.full_like(la, -180), la),
        (np.full_like(la, 180), la),
    ]
    for los, las in segments:
        x, y = _project_line(los, las)
        ax.plot(x, y, color="#30363A", lw=0.75, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def draw_mask_contour(ax, x, y, mask, *, color="#4B5155", lw=0.45,
                      alpha=0.9, linestyle="-"):
    """Draw the zero-elevation/emergent-cell contour for a displayed surface.

    ``mask`` is a boolean (nlat, nlon) land mask on the grid of x, y; the 0.5
    contour of its 0/1 values traces the boundary between land and sea cells.
    """
    ax.contour(x, y, np.asarray(mask, dtype=float), levels=[0.5], colors=[color],
               linewidths=lw, alpha=alpha, linestyles=linestyle, zorder=6)

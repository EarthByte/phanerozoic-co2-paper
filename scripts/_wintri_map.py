"""Small Winkel–Tripel plotting helpers used by the CO2 figure and videos.

The publication paleoDEM movies and Figs. 1–2 use GMT's Winkel–Tripel
projection.  These helpers reproduce that projection with PROJ so the new
weathering maps keep the same global-map geometry even on systems without a
working GMT executable.
"""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

WINTRI = CRS.from_proj4("+proj=wintri +lon_0=0 +datum=WGS84 +units=m +no_defs")
WGS84 = CRS.from_epsg(4326)
TRANSFORMER = Transformer.from_crs(WGS84, WINTRI, always_xy=True)


def projected_mesh(lon, lat):
    """Return projected 2-D cell-centre coordinates for 1-D lon/lat axes."""
    lo, la = np.meshgrid(np.asarray(lon), np.asarray(lat))
    return TRANSFORMER.transform(lo, la)


def _project_line(lon, lat):
    return TRANSFORMER.transform(np.asarray(lon), np.asarray(lat))


def decorate_global_map(ax, *, graticule=True):
    """Draw a plain frame and restrained 60°/30° graticule."""
    if graticule:
        for lo in np.arange(-120, 180, 60):
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
    """Draw the zero-elevation/emergent-cell contour for a displayed surface."""
    ax.contour(x, y, np.asarray(mask, dtype=float), levels=[0.5], colors=[color],
               linewidths=lw, alpha=alpha, linestyles=linestyle, zorder=6)

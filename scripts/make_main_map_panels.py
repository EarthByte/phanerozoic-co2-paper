#!/usr/bin/env python3
"""Map panels of the two palaeoelevation histories and of the surrogate erosion field.

Writes two figures in the Winkel Tripel projection (see _wintri_map.py), each a grid of
three rows by four ages (540, 240, 150 and 40 Ma):

  fig1_paleoelev.{png,pdf}    Extended Data Fig. 2: elevation of the PALEOMAP reference
                              surface, elevation of the rock-chemistry-adjusted surface,
                              and their difference (adjusted - PALEOMAP), m
  fig2_erodep_temp.{png,pdf}  NeuroLEM signed erosion-deposition rate on the adjusted
                              surface, its change against the PALEOMAP surface, and
                              HadCM3 surface-air temperature (degC), land cells only

The four ages span the Phanerozoic: 540 Ma shows the distributed early Palaeozoic
response and 240 Ma a representative Pangaea configuration.

Inputs (paths from _paths.py; the large grids are in the Zenodo archive)
  surrogate_erodep/erodep_<age>Ma.npz   zpaleo_scotese, zpaleo_zhou (m), delta (m),
                                        ero_scotese, ero_zhou (signed rate, mm/yr,
                                        negative = erosion, unscaled), mask_scotese,
                                        mask_zhou, lat, lon on the 0.1-degree grid
  valdes_temperature/<age>Ma.npz        HadCM3 surface-air temperature
Fields are subsampled every STEP = 5 cells (0.5 degree) for plotting.
Output goes to the repository figures/ directory.

    python3 make_main_map_panels.py

In the nature_recast workflow this runs as part of ``run_uncertainty_m3.sh main``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import _epsl_style  # noqa: F401  EPSL sizing/lettering
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ERODEP_DIR, TEMP_DIR, FIGURES, require
from _wintri_map import projected_mesh, decorate_global_map, draw_mask_contour
from bufe_co2_from_dem import _load_temp_K

AGES = [540, 240, 150, 40]
STEP = 5                 # plot every 5th cell of the 0.1-degree grid (0.5 degree)
INK = "#30363A"
# Elevation colour map, used with a norm centred on sea level: blues below 0 m
# (position 0.5), greens, browns and white with height
EARTH = LinearSegmentedColormap.from_list(
    "paper_earth",
    [(0.00, "#17365D"), (0.26, "#4378A8"), (0.49, "#B8D6E8"),
     (0.50, "#E8E1B6"), (0.58, "#8BAA63"), (0.74, "#8B6A47"),
     (1.00, "#F6F6F4")],
)


def map_panel(ax, x, y, values, *, cmap, norm, mask, title=None):
    """Draw one global map with graticule, outline and the coastline of ``mask``;
    return the mesh artist for a colour bar."""
    im = ax.pcolormesh(x, y, values, shading="nearest", cmap=cmap, norm=norm,
                       rasterized=True, zorder=1)
    decorate_global_map(ax)
    draw_mask_contour(ax, x, y, mask, color="#50575B", lw=0.35)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=INK, pad=3)
    return im


def row_label(fig, axes, text):
    """Write a vertical row label to the left of the first axes of a row."""
    box = axes[0].get_position()
    fig.text(box.x0 - 0.012, (box.y0 + box.y1)/2, text, rotation=90,
             ha="right", va="center", fontsize=8, fontweight="bold", color=INK)


def add_shared_cbar(fig, mappable, axes, row, label, extend="both"):
    """Add one horizontal colour bar under row ``row``, spanning 84% of its width."""
    box0, box1 = axes[row][0].get_position(), axes[row][-1].get_position()
    y = box0.y0 - 0.027
    cax = fig.add_axes([box0.x0 + 0.08*(box1.x1-box0.x0), y,
                        0.84*(box1.x1-box0.x0), 0.014])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(label, fontsize=8, labelpad=1)


def load(age):
    """Load and subsample the elevation, erosion and mask fields of one age."""
    with np.load(require(ERODEP_DIR / f"erodep_{age}Ma.npz"), allow_pickle=True) as d:
        sl = (slice(None, None, STEP), slice(None, None, STEP))
        return dict(
            lat=d["lat"][::STEP].astype(float), lon=d["lon"][::STEP].astype(float),
            zs=d["zpaleo_scotese"][sl].astype(float),
            zz=d["zpaleo_zhou"][sl].astype(float),
            delta=d["delta"][sl].astype(float),
            es=d["ero_scotese"][sl].astype(float),
            ez=d["ero_zhou"][sl].astype(float),
            ms=d["mask_scotese"][sl], mz=d["mask_zhou"][sl],
        )


def fig1(data, x, y):
    """Extended Data Fig. 2: PALEOMAP and adjusted elevation and their difference."""
    fig, axes = plt.subplots(3, 4, figsize=_epsl_style.size(_epsl_style.DOUBLE, 15.2, 8.1))
    fig.patch.set_facecolor("white")
    elev_norm = TwoSlopeNorm(vmin=-4000, vcenter=0, vmax=4000)
    delta_norm = TwoSlopeNorm(vmin=-2000, vcenter=0, vmax=2000)
    ims = []
    for c, age in enumerate(AGES):
        d = data[age]
        ims.append(map_panel(axes[0, c], x, y, d["zs"], cmap=EARTH,
                             norm=elev_norm, mask=d["ms"], title=f"{age} Ma"))
        map_panel(axes[1, c], x, y, d["zz"], cmap=EARTH,
                  norm=elev_norm, mask=d["mz"])
        # the difference is shown wherever either surface is land
        map_panel(axes[2, c], x, y, np.where(d["ms"] | d["mz"], d["delta"], np.nan),
                  cmap="RdBu_r", norm=delta_norm, mask=d["mz"])
    fig.subplots_adjust(left=0.075, right=0.99, top=0.95, bottom=0.145,
                        wspace=0.035, hspace=0.22)
    for a in axes[2]:                      # difference row lowered to clear the elevation colour bar and its label
        b = a.get_position(); a.set_position([b.x0, b.y0 - 0.04, b.width, b.height])
    for r, text in enumerate(("PALEOMAP",
                              "Adjusted",
                              "Difference")):
        b = axes[r, 0].get_position()
        fig.text(b.x0 - 0.010, (b.y0+b.y1)/2, text, rotation=90,
                 ha="right", va="center", fontsize=8,
                 fontweight="bold", color=INK)
    add_shared_cbar(fig, ims[0], axes, 1, "elevation (m)")
    add_shared_cbar(fig, axes[2, 0].collections[0], axes, 2,
                    "elevation difference: adjusted − PALEOMAP (m)")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig1_paleoelev.{ext}", dpi=240 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)


def fig2(data, x, y):
    """Erosion-deposition of the adjusted surface, its change, and temperature."""
    # Taller canvas than fig1 because every row here carries its own colour bar.
    # The extra height goes into the gaps between rows (0.45 in rather than
    # 0.27 in), so the maps keep the same size (~0.95 in) and each colour-bar label
    # stays clear of the next row.
    fig, axes = plt.subplots(3, 4, figsize=_epsl_style.size(_epsl_style.DOUBLE, 15.2, 9.0))
    fig.patch.set_facecolor("white")
    n_ero = TwoSlopeNorm(vmin=-150, vcenter=0, vmax=150)
    n_dero = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    n_temp = Normalize(-30, 40)
    imrows = [None, None, None]
    for c, age in enumerate(AGES):
        d = data[age]
        temp = _load_temp_K(TEMP_DIR, age)[::STEP, ::STEP] - 273.15      # K -> degC
        # The archived erosion fields are in mm/yr, so the factor 1000 gives m/Myr;
        # the colour ranges (+-150 and +-60) are in those units.
        imrows[0] = map_panel(
            axes[0, c], x, y, np.where(d["mz"], d["ez"]*1000.0, np.nan),
            cmap="RdBu_r", norm=n_ero, mask=d["mz"], title=f"{age} Ma")
        imrows[1] = map_panel(
            axes[1, c], x, y, np.where(d["ms"] | d["mz"],
                                      (d["ez"]-d["es"])*1000.0, np.nan),
            cmap="RdBu_r", norm=n_dero, mask=d["mz"])
        imrows[2] = map_panel(
            axes[2, c], x, y, np.where(d["mz"], temp, np.nan),
            cmap="turbo", norm=n_temp, mask=d["mz"])
    fig.subplots_adjust(left=0.075, right=0.99, top=0.95, bottom=0.105,
                        wspace=0.035, hspace=0.47)
    labels = ["Rock-chemistry", "Difference",
              "Temperature"]
    for r, text in enumerate(labels):
        b = axes[r, 0].get_position()
        fig.text(b.x0 - 0.010, (b.y0+b.y1)/2, text, rotation=90,
                 ha="right", va="center", fontsize=8, fontweight="bold", color=INK)
    add_shared_cbar(fig, imrows[0], axes, 0,
                    "erosion (negative, blue) / deposition (positive, red), mm yr⁻¹")
    add_shared_cbar(fig, imrows[1], axes, 1,
                    "change in erosion/deposition: rock-chemistry − Scotese (mm yr⁻¹)")
    add_shared_cbar(fig, imrows[2], axes, 2, "surface-air temperature (°C)")
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig2_erodep_temp.{ext}", dpi=240 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)


def main():
    """Load the four ages and write both figures."""
    FIGURES.mkdir(exist_ok=True)
    data = {age: load(age) for age in AGES}
    # all ages share the same grid, so one projected mesh serves every panel
    x, y = projected_mesh(data[AGES[0]]["lon"], data[AGES[0]]["lat"])
    fig1(data, x, y)
    fig2(data, x, y)
    print("wrote fig1_paleoelev and fig2_erodep_temp for", AGES)


if __name__ == "__main__":
    main()

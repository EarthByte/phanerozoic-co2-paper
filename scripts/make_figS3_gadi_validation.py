#!/usr/bin/env python3
"""Extended Data Fig. 3: direct goSPL simulations against NeuroLEM on identical inputs.

The 20 direct goSPL runs (PALEOMAP reference, adjusted surface and two ensemble members at
60, 220, 240, 460 and 520 Ma) were post-processed by analyse_gadi_validation.py.

  a  goSPL against NeuroLEM gross erosion of every run, km3/yr, with the 1:1 line
  b  adjusted-to-reference ratios of gross erosion, goSPL against NeuroLEM
  c  240 Ma map of the change in signed erosion-deposition rate, adjusted - PALEOMAP, mm/yr
  d  240 Ma map of the change in net inorganic CO2 flux, adjusted - PALEOMAP, mmol C m-2 yr-1
Colour marks the age and marker shape the case; in c and d the solid grey contour is the
PALEOMAP coastline and the dashed red contour that of the adjusted surface (1-degree cells at
least half land).

Inputs
  panels a, b  neurolem_comparison.csv in --derived (the output directory of
               analyse_gadi_validation.py); if that is absent, the repository copy
               data/nature_recast/ed3_neurolem_comparison.csv. The repository copy was
               reconstructed from the marker positions in the vector PDF of the figure and
               checked against direct_gospl_skill.csv (Continental_drainage/validation):
               after one constant axis offset of 0.006 km3/yr, the goSPL/NeuroLEM differences
               agree with the tabulated ones within 0.02 percentage points.
  panels c, d  validation_maps_1deg.nc in --derived, or in ../Continental_drainage/validation.

Output: --out (default figures/figS3_gadi_validation.png) and a PDF beside it.
The figure carries panel letters only; the description belongs in the caption.

    python3 make_figS3_gadi_validation.py [--derived DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _epsl_style  # noqa: F401  EPSL sizing/lettering + collision auditor
from _wintri_map import decorate_global_map, draw_mask_contour, projected_mesh


AGES = (60, 220, 240, 460, 520)
CASES = ("reference", "central", "qnet_p05", "qnet_p95")
# One colour per age, validated as a set on all pairs against a white background:
# the closest pair for normal vision is dE 16.3 (floor 15), and lightness and chroma
# lie within bounds. The closest pair under colour-vision deficiency is dE 6.9, which
# is acceptable because the age is also named in the legend and tabulated in the
# Supplementary Information; the same table covers aqua and yellow, which sit below
# 3:1 contrast. Any substitute palette should be re-validated, not chosen by eye.
AGE_COLOURS = {60: "#1baf7a", 220: "#e34948", 240: "#eda100",
               460: "#2a78d6", 520: "#4a3aa7"}
CASE_LABELS = {"reference": "PALEOMAP", "central": "adjusted", "qnet_p05": "member p05", "qnet_p95": "member p95"}
CASE_MARKERS = {"reference": "o", "central": "s", "qnet_p05": "v", "qnet_p95": "^"}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def identity(ax, values: list[float]) -> None:
    """Draw the 1:1 line and set equal, square limits padded 7% beyond the data."""
    low, high = min(values), max(values)
    span = high - low
    low -= 0.07 * span
    high += 0.07 * span
    ax.plot([low, high], [low, high], color="#4D555A", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D9DEE1", lw=0.5)


def add_map(ax, x, y, data, mask_ref, mask_new, title, label, cmap, limit) -> None:
    """Draw one Winkel Tripel difference map with a symmetric colour scale.

    x, y are projected cell centres; data is centred on zero and scaled to
    +-limit; mask_ref and mask_new are drawn as solid and dashed coastlines.
    ``title`` is not used (panels carry letters only).
    """
    image = ax.pcolormesh(
        x, y, data, shading="nearest", cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        rasterized=True, zorder=1,
    )
    decorate_global_map(ax)
    draw_mask_contour(ax, x, y, mask_ref, color="#3F474C", lw=0.5)
    draw_mask_contour(ax, x, y, mask_new, color="#D55E4D", lw=0.55, linestyle="--")
    bar = plt.colorbar(image, ax=ax, orientation="horizontal", pad=0.025, fraction=0.06, aspect=32)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.set_label(label, fontsize=8)


def main() -> int:
    """Build the four panels and write PNG and PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived", type=Path,
        default=Path("gadi_runs/nature_validation_CO2/derived"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "figS3_gadi_validation.png",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    csv_path = args.derived / "neurolem_comparison.csv"
    if not csv_path.exists():
        csv_path = repo / "data" / "nature_recast" / "ed3_neurolem_comparison.csv"
    nc_path = args.derived / "validation_maps_1deg.nc"
    if not nc_path.exists():
        nc_path = repo.parent / "Continental_drainage" / "validation" / "validation_maps_1deg.nc"
    print("panels a, b:", csv_path, "| panels c, d:", nc_path)
    comparisons = read_rows(csv_path)
    by_key = {(int(row["age_Ma"]), row["case"]): row for row in comparisons}

    fig, axes = plt.subplots(2, 2, figsize=_epsl_style.size(_epsl_style.DOUBLE, 12.2, 8.4))
    fig.patch.set_facecolor("white")
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    all_xy: list[float] = []
    for row in comparisons:
        age = int(row["age_Ma"])
        case = row["case"]
        x = float(row["NeuroLEM_gross_erosion_surface_km3_yr"])
        y = float(row["goSPL_gross_erosion_surface_km3_yr"])
        all_xy.extend((x, y))
        ax_a.scatter(
            x, y, s=47, marker=CASE_MARKERS[case], color=AGE_COLOURS[age],
            edgecolor="white", linewidth=0.5, zorder=3,
        )
    identity(ax_a, all_xy)
    ax_a.set_xlabel("NeuroLEM gross erosion (km³ yr⁻¹)")
    ax_a.set_ylabel("goSPL gross erosion (km³ yr⁻¹)")

    ratio_xy: list[float] = []
    for age in AGES:
        ref = by_key[(age, "reference")]
        for case in CASES[1:]:
            row = by_key[(age, case)]
            x = float(row["NeuroLEM_gross_erosion_surface_km3_yr"]) / float(
                ref["NeuroLEM_gross_erosion_surface_km3_yr"]
            )
            y = float(row["goSPL_gross_erosion_surface_km3_yr"]) / float(
                ref["goSPL_gross_erosion_surface_km3_yr"]
            )
            ratio_xy.extend((x, y))
            ax_b.scatter(
                x, y, s=52, marker=CASE_MARKERS[case], color=AGE_COLOURS[age],
                edgecolor="white", linewidth=0.5, zorder=3,
            )
    identity(ax_b, ratio_xy)
    ax_b.set_xlabel("NeuroLEM, adjusted/PALEOMAP")
    ax_b.set_ylabel("goSPL, adjusted/PALEOMAP")

    age_handles = [
        plt.Line2D([], [], marker="o", ls="", color=AGE_COLOURS[age], label=f"{age} Ma")
        for age in AGES
    ]
    case_handles = [
        plt.Line2D([], [], marker=CASE_MARKERS[case], ls="", color="#4D555A", label=CASE_LABELS[case])
        for case in CASES
    ]
    age_legend = ax_a.legend(handles=age_handles, frameon=False, fontsize=8, loc="upper left")
    ax_a.add_artist(age_legend)
    ax_a.legend(handles=case_handles, frameon=False, fontsize=8, loc="lower right")

    # maps at 240 Ma, adjusted ("central") minus reference
    with nc.Dataset(nc_path) as ds:
        ages = list(map(int, ds.variables["age"][:]))
        cases = [str(value) for value in ds.variables["case"][:]]
        ai, ri, ci = ages.index(240), cases.index("reference"), cases.index("central")
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        erodep = np.asarray(ds.variables["erosion_deposition_rate"][ai], dtype=float)
        # mol C m-2 yr-1 -> mmol C m-2 yr-1
        fco2 = np.asarray(ds.variables["net_inorganic_CO2_flux"][ai], dtype=float) * 1000.0
        land = np.asarray(ds.variables["initial_land_fraction"][ai], dtype=float)
    x, y = projected_mesh(lon, lat)
    # show cells with any appreciable land on either surface
    union = (land[ri] > 0.05) | (land[ci] > 0.05)
    de = np.where(union, erodep[ci] - erodep[ri], np.nan)
    df = np.where(union, fco2[ci] - fco2[ri], np.nan)
    # symmetric colour limits at the 98.5th percentile of |difference|, with a floor,
    # so a few extreme cells do not wash out the pattern
    de_lim = max(0.02, float(np.nanpercentile(np.abs(de), 98.5)))
    df_lim = max(0.2, float(np.nanpercentile(np.abs(df), 98.5)))
    add_map(
        ax_c, x, y, de, land[ri] >= 0.5, land[ci] >= 0.5,
        None,
        "erosion–deposition, adjusted − PALEOMAP (mm yr⁻¹)", "RdBu_r", de_lim,
    )
    add_map(
        ax_d, x, y, df, land[ri] >= 0.5, land[ci] >= 0.5,
        None,
        "net CO₂ flux, adjusted − PALEOMAP (mmol C m⁻² yr⁻¹)", "RdBu", df_lim,
    )
    # No figure title: the caption carries the description, and the top margin
    # goes to the panels.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.985, wspace=0.17, hspace=0.18)
    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.text(-0.3 if letter in "ab" else 0.0, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")
    # The equal-aspect maps sit in boxes taller than the maps, which leaves a wide gap under the top row. Close it: measure
    # the gap between the top row's lowest artist (the x labels) and the top of the maps, keep GAP_IN of it, and take the
    # rest out of the figure height, so the top row moves down onto the maps and nothing else changes size.
    GAP_IN = 0.22
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top_axes = [ax_a, ax_b]
    bottom_axes = [a for a in fig.axes if a not in top_axes]
    w_in, h_in = fig.get_size_inches()
    low_top = min(a.get_tightbbox(r).y0 for a in top_axes) / fig.dpi
    def drawn_top(a):        # top of the drawn map (its outline), not of the axes box
        return max(l.get_window_extent(r).y1 for l in a.lines) / fig.dpi if a.lines else a.get_window_extent(r).y1 / fig.dpi
    map_top = max(drawn_top(a) for a in (ax_c, ax_d))
    cut = max(0.0, (low_top - map_top) - GAP_IN)
    if cut > 0:
        boxes = {a: a.get_position() for a in fig.axes}
        new_h = h_in - cut
        for a, b in boxes.items():
            y0_in, hh_in = b.y0 * h_in, b.height * h_in
            if a in top_axes:
                y0_in -= cut
            a.set_position([b.x0, y0_in / new_h, b.width, hh_in / new_h])
        fig.set_size_inches(w_in, new_h)
        print(f"closed the gap under the top row by {cut:.2f} in (figure height {h_in:.2f} -> {new_h:.2f} in)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        target = args.out.with_suffix(f".{extension}")
        # No bbox_inches='tight': it widens the saved figure past the printed
        # width, so the typesetter scales it back down and the lettering shrinks.
        fig.savefig(target, dpi=260 if extension == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out.with_suffix('.png')} and PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

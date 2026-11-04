#!/usr/bin/env python3
"""
build_pco2_envelope.py -- assemble the per-age pCO2 uncertainty envelope
(``age,co2_lo,co2_hi``) that drives ``phanda_emulator.py envelope`` / ``ensemble``,
``mc_driver.py`` and the climate-state uncertainty ensemble of the paper
(``nature_recast/climate_sensitivity.py``; Methods, "Uncertainty and sensitivity",
item 6: CO2 levels spanning the proxy envelope of Foster et al. 2017 and Judd et
al. 2024; Extended Data Fig. 6).

The envelope comes from OUTSIDE the model chain being perturbed wherever possible.
PhanDA (Judd et al. 2024) assimilates proxies into the same HadCM3L priors the
emulator is fitted on, so an envelope taken from it is not independent of the climate
model whose response is being perturbed.  Foster, Royer & Lunt (2017) is a pure proxy
compilation and is therefore the primary source; Judd et al. (2024) is used only where
Foster et al. (2017) does not reach.

THREE SEGMENTS (on the 109 plate ages 0..540 Ma, 5 Myr steps)
--------------
  0 - 420 Ma    Foster et al. (2017) LOESS band (Supplementary Data 2, sheet
                "LOESS Fit"): lw68/up68 or lw95/up95 about the probability
                maximum, at 0.5 Myr resolution.
  420 - 487 Ma  Judd et al. (2024) PhanDA CO2 percentiles (CO2_16/CO2_84 or
                CO2_05/CO2_95, per stage).  This segment's envelope is not
                independent of the HadCM3L priors.  ``--no-judd`` replaces it with
                the carried-width tail below.
  > 487 Ma      No proxy compilation or assimilation product covers the Cambrian
                and earliest Ordovician.  The envelope is centred on the model's
                OWN prescribed CO2 -- introducing no bias in the value -- and given
                the half-width (in doublings) carried forward from the median over
                ``--tail-window`` of the Foster et al. (2017) band, capped at
                ``--tail-width-cap``.  This is an extrapolation of the UNCERTAINTY,
                not of the data.  With the defaults (68% band, 350-420 Ma window)
                the carried half-width is 0.95 doublings, within the 0.9-1.24
                doublings that the Foster et al. (2017) band itself spans over the
                Devonian and Carboniferous.

Records are interpolated onto the plate ages linearly in log CO2.  Where two segments
meet (``--foster-max-age``, ``--judd-max-age``), ages within ``--blend-myr`` of the
boundary at which both segments have values are mixed in log space with a cubic
Hermite weight running from 0 at boundary - blend to 1 at boundary + blend.  Because
each record is cut at its boundary, this mixing acts only on ages at or younger than
the boundary.

RELATIVE MODE (the default)
---------------------------
The reference simulation was forced with the smoothed CO2 curve of Valdes et al.
(2021), which departs from the proxy central estimate by more than half a doubling at
many ages.  ``--mode relative`` keeps the band's half-width in doublings and centres
it on the reference run's own prescribed CO2 (CO2_ref, from ``phanda_experiments.csv``
for ``scotese_spinupa``), so the central member is the simulation the paper's results
were computed with.  ``--mode absolute`` uses the record's own band, which re-centres
the calculation on a different CO2 history.  The model-minus-record offset is written
to the provenance file and printed, but does not enter the envelope in relative mode.

Lower bounds below ``--floor-ppm`` are clamped and counted (Foster et al.'s 95% lower
bound, a LOESS fit in linear pCO2 space, goes negative beyond ~410 Ma; the 68% band
does not need clamping), and every upper bound is kept at least 1% above its lower
bound.

INPUTS
  --foster-xlsx   Foster et al. (2017) Supplementary Data 2
                  (Foster_etal_carbon_cycles_420Ma_to_present_NatComms2017-s3.xlsx)
  --judd-xlsx     PhanDA_GMSTandCO2_percentiles.xlsx, from the outputs of the PhanDA
                  code of Judd et al. (2024) (optional)
  phanda_experiments.csv beside phanda_emulator.py, for CO2_ref per age

OUTPUTS
  --out (pco2_envelope.csv), one row per plate age:
      age [Ma], co2_lo, co2_hi [ppm] (the band), co2_central [ppm] (CO2_ref in
      relative mode, the record's central estimate in absolute mode),
      co2_ref_model [ppm] (prescribed CO2 of the reference simulation)
  <out stem>_provenance.csv, one row per plate age:
      age, source (foster / judd / carried-width / blend), mode, band_percent,
      half_width_doublings, model_minus_record_doublings (log2 CO2_ref / record
      central), ref_inside_band (0/1)
  --plot (optional PNG): the envelope by source segment, CO2_ref and the record's
      central estimate against age

USAGE
  python build_pco2_envelope.py \
      --foster-xlsx Foster_etal_carbon_cycles_420Ma_to_present_NatComms2017-s3.xlsx \
      --judd-xlsx   PhanDA_GMSTandCO2_percentiles.xlsx \
      --band 68 --out pco2_envelope.csv --plot pco2_envelope.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # import phanda_emulator from this directory
from phanda_emulator import load_experiments, REFERENCE_SUITE  # noqa: E402

AGES = np.arange(0, 545, 5)          # the 109 plate ages of the basegrid


def _sheet(path, name=None):
    """All rows of worksheet ``name`` (default: the first sheet) as tuples of cell values."""
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("openpyxl is required: pip install openpyxl")
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[name] if name else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    return rows


def read_foster(path, band: int):
    """(age, central, lo, hi) from Supplementary Data 2, sheet 'LOESS Fit'.

    Returns an (n, 4) array sorted by age: age [Ma], the probability maximum and the
    lower / upper bound of the ``band`` (68 or 95) % interval [ppm].
    """
    rows = _sheet(path, "LOESS Fit")
    hdr = [str(c).strip() if c is not None else "" for c in rows[1]]   # header is the second row
    want = {"age": "Age (Ma)", "mid": "pCO2 probability maximum",
            "lo": f"lw{band}%", "hi": f"up{band}%"}
    idx = {}
    for k, name in want.items():
        if name not in hdr:
            raise SystemExit(f"{path}: column {name!r} not found; header is {hdr}")
        idx[k] = hdr.index(name)
    out = []
    for r in rows[2:]:
        if r[idx["age"]] is None:
            continue
        try:
            out.append([float(r[idx[k]]) for k in ("age", "mid", "lo", "hi")])
        except (TypeError, ValueError):
            continue                   # skip rows with blank or non-numeric cells
    a = np.array(out, float)
    return a[np.argsort(a[:, 0])]


def read_judd(path, band: int):
    """(age, central, lo, hi) from PhanDA_GMSTandCO2_percentiles.xlsx.

    Returns an (n, 4) array sorted by age: stage mean age [Ma], median CO2 and the
    16/84 (band 68) or 5/95 (band 95) percentiles [ppm], one row per stage.
    """
    rows = _sheet(path)
    hdr = [str(c).strip() if c is not None else "" for c in rows[0]]
    lo_c, hi_c = (f"CO2_{'05' if band == 95 else '16'}",
                  f"CO2_{'95' if band == 95 else '84'}")
    for name in ("AverageAge", "CO2_50", lo_c, hi_c):
        if name not in hdr:
            raise SystemExit(f"{path}: column {name!r} not found; header is {hdr}")
    i = [hdr.index(n) for n in ("AverageAge", "CO2_50", lo_c, hi_c)]
    out = []
    for r in rows[1:]:
        try:
            out.append([float(r[k]) for k in i])
        except (TypeError, ValueError):
            continue
    a = np.array(out, float)
    return a[np.argsort(a[:, 0])]


def _interp(rec, ages, tol=6.0):
    """Linear interpolation of (central, lo, hi) in log space onto *ages*.

    ``rec`` is an (n, 4) record from ``read_foster`` / ``read_judd``; returns an
    (n_age, 3) array [ppm], NaN outside the record's age range.  Values are floored
    at 1e-3 ppm before taking logarithms.

    ``tol`` lets an age snap to the record's endpoint when it is within that many
    Myr of it -- the Foster series starts at 0.0039 Ma and ends at 419.5 Ma, so a
    strict range test silently drops the 0 Ma and 420 Ma slices.
    """
    m = (ages >= rec[0, 0] - tol) & (ages <= rec[-1, 0] + tol)
    out = np.full((ages.size, 3), np.nan)
    for j in range(3):
        out[m, j] = np.exp(np.interp(ages[m], rec[:, 0], np.log(np.maximum(rec[:, j + 1], 1e-3))))
    return out


def _hermite(t):
    """Cubic Hermite (smoothstep) weight on t in [0, 1]: 3t^2 - 2t^3."""
    return t * t * (3.0 - 2.0 * t)


def main(argv=None):
    """Build the envelope and write the CSV, provenance CSV and optional plot."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--foster-xlsx", required=True)
    p.add_argument("--judd-xlsx")
    p.add_argument("--band", type=int, choices=[68, 95], default=68,
                   help="which published interval to use (default 68; the Foster 95%% "
                        "lower bound goes negative beyond ~410 Ma and needs clamping)")
    p.add_argument("--foster-max-age", type=float, default=420.0)
    p.add_argument("--judd-max-age", type=float, default=487.0)
    p.add_argument("--tail-window", type=float, nargs=2, default=[350.0, 420.0],
                   metavar=("A0", "A1"),
                   help="age window whose relative band width is carried into the "
                        "unconstrained tail beyond --judd-max-age")
    p.add_argument("--tail-width-cap", type=float, default=2.5,
                   help="cap on the carried half-width, in CO2 doublings")
    p.add_argument("--floor-ppm", type=float, default=100.0)
    p.add_argument("--blend-myr", type=float, default=20.0)
    p.add_argument("--no-judd", action="store_true",
                   help="use the carried-width tail for everything beyond Foster, so the "
                        "envelope is wholly independent of the PhanDA priors")
    p.add_argument("--mode", choices=["relative", "absolute"], default="relative",
                   help="'relative' (default) keeps the published band's WIDTH and applies "
                        "it about the model's own prescribed CO2, so the central member "
                        "stays the reference simulation; 'absolute' uses the record's own "
                        "band, which re-centres the calculation on a different CO2 history")
    p.add_argument("--out", default="pco2_envelope.csv")
    p.add_argument("--plot")
    a = p.parse_args(argv)

    ages = AGES.astype(float)
    # CO2_ref: prescribed CO2 of the reference simulation at each plate age (ppm)
    ref = {e.plate_age: e.co2_ppm for e in load_experiments() if e.suite == REFERENCE_SUITE}
    co2_ref = np.array([ref[int(x)] for x in ages])

    foster = read_foster(a.foster_xlsx, a.band)
    print(f"Foster:  {len(foster)} rows, {foster[0,0]:.2f}-{foster[-1,0]:.2f} Ma, "
          f"{a.band}% band")
    F = _interp(foster, ages)                                       # (n_age, 3): central, lo, hi
    F[ages > a.foster_max_age] = np.nan

    if a.judd_xlsx and not a.no_judd:
        judd = read_judd(a.judd_xlsx, a.band)
        print(f"Judd:    {len(judd)} stages, {judd[0,0]:.2f}-{judd[-1,0]:.2f} Ma, "
              f"{a.band}% band")
        J = _interp(judd, ages)
        # Judd et al. (2024) is used only between the end of Foster et al. (2017)
        # and --judd-max-age
        J[(ages < a.foster_max_age) | (ages > a.judd_max_age)] = np.nan
    else:
        J = np.full_like(F, np.nan)

    # ---- carried relative width for the unconstrained tail -----------------------
    w = (ages >= a.tail_window[0]) & (ages <= a.tail_window[1]) & np.isfinite(F[:, 0])
    if not w.any():
        raise SystemExit("tail window contains no Foster data")
    # half-width in CO2 doublings: 0.5 * log2(hi / lo), median over the window
    halfwidth = np.median(0.5 * np.log2(F[w, 2] / np.maximum(F[w, 1], 1e-6)))
    halfwidth = float(min(halfwidth, a.tail_width_cap))
    print(f"tail:    carried half-width {halfwidth:.2f} doublings "
          f"(x/÷{2**halfwidth:.2f}) from the {a.tail_window[0]:.0f}-{a.tail_window[1]:.0f} Ma window")
    # tail segment at every age: (central, lo, hi) = CO2_ref, CO2_ref / 2^hw, CO2_ref * 2^hw
    T = np.stack([co2_ref, co2_ref / 2 ** halfwidth, co2_ref * 2 ** halfwidth], 1)

    # ---- splice, blending in log space at each boundary --------------------------
    # segment per age: 0 = Foster where defined, else 1 = Judd, else 2 = carried width
    seg = np.where(np.isfinite(F[:, 0]), 0, np.where(np.isfinite(J[:, 0]), 1, 2))
    stack = np.stack([F, J, T])                                    # (3 segments, nage, central/lo/hi)
    out = np.array([stack[seg[k], k] for k in range(ages.size)])
    src = np.array(["foster", "judd", "carried-width"])[seg]

    # boundaries: Foster -> Judd (or -> carried width without Judd), then Judd ->
    # carried width.  Only ages where both segments are finite are mixed (see the
    # module docstring).
    for b, (i_from, i_to) in ((a.foster_max_age, (0, 1 if not a.no_judd and a.judd_xlsx else 2)),
                              (a.judd_max_age, (1, 2))):
        if i_from == 1 and (a.no_judd or not a.judd_xlsx):
            continue
        m = np.abs(ages - b) <= a.blend_myr
        if m.sum() < 2:
            continue
        t = _hermite(np.clip((ages[m] - (b - a.blend_myr)) / (2 * a.blend_myr), 0, 1))
        A, B = stack[i_from][m], stack[i_to][m]
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if ok.any():
            blended = np.exp((1 - t[ok, None]) * np.log(A[ok]) + t[ok, None] * np.log(B[ok]))
            out[np.nonzero(m)[0][ok]] = blended
            src[np.nonzero(m)[0][ok]] = "blend"

    # ---- re-centre on the model's own CO2, if asked (the default) -----------------
    #
    # The reference simulation was forced with the Valdes smoothed CO2 curve, which
    # departs substantially from the Foster central estimate at many ages (the
    # Valdes paper says so explicitly for the Triassic and Jurassic).  Using the
    # record's ABSOLUTE band would therefore re-centre the whole calculation on a
    # different CO2 history: the "central" member would no longer be the simulation
    # the paper's result was computed with, and the emulator would be extrapolating
    # its locally fitted slope a long way from CO2_ref.  That is a different
    # experiment, not an uncertainty band.
    #
    # In RELATIVE mode we keep the published band's WIDTH -- how uncertain pCO2 is
    # at that age -- and apply it about the model's own prescribed CO2.  The central
    # member stays the actual reference run, and the band answers the question the
    # paper is asking: how far could the result move given pCO2 uncertainty?
    # model minus record central estimate, in doublings (zero by construction on
    # the carried-width segment, which is centred on CO2_ref)
    offset = np.log2(co2_ref / np.maximum(out[:, 0], 1e-6))
    rec_central = out[:, 0].copy()          # the record's own central estimate, for the plot
    if a.mode == "relative":
        # keep the half-width in doublings, centre on CO2_ref
        hw = 0.5 * np.log2(np.maximum(out[:, 2], 1e-6) / np.maximum(out[:, 1], 1e-6))
        out = np.stack([co2_ref, co2_ref / 2 ** hw, co2_ref * 2 ** hw], 1)
    else:
        print("\nWARNING: --mode absolute re-centres the calculation on the record's own "
              "pCO2 history.\n  The central member is then NOT the reference simulation. "
              "Only use this if you\n  intend a re-forced experiment rather than an "
              "uncertainty band.")

    # ---- floor and sanity --------------------------------------------------------
    floored = int((out[:, 1] < a.floor_ppm).sum())
    out[:, 1] = np.maximum(out[:, 1], a.floor_ppm)
    out[:, 2] = np.maximum(out[:, 2], out[:, 1] * 1.01)         # keep hi above a clamped lo
    if floored:
        print(f"floor:   {floored} ages had a lower bound below {a.floor_ppm:g} ppm "
              f"(a LOESS artefact in linear pCO2 space) and were clamped")

    # envelope CSV (ppm); phanda_emulator.load_envelope reads age, co2_lo, co2_hi
    with open(a.out, "w", newline="") as f:
        w_ = csv.writer(f, lineterminator="\n")
        w_.writerow(["age", "co2_lo", "co2_hi", "co2_central", "co2_ref_model"])
        for k in range(ages.size):
            w_.writerow([int(ages[k])] + [f"{v:.4f}" for v in
                                          (out[k, 1], out[k, 2], out[k, 0], co2_ref[k])])
    prov = Path(a.out).with_name(Path(a.out).stem + "_provenance.csv")
    with open(prov, "w", newline="") as f:
        w_ = csv.writer(f, lineterminator="\n")
        w_.writerow(["age", "source", "mode", "band_percent", "half_width_doublings",
                     "model_minus_record_doublings", "ref_inside_band"])
        for k in range(ages.size):
            hw = 0.5 * np.log2(out[k, 2] / out[k, 1])
            w_.writerow([int(ages[k]), src[k], a.mode, a.band, f"{hw:.3f}",
                         f"{offset[k]:.3f}", int(out[k, 1] <= co2_ref[k] <= out[k, 2])])

    # summary table at selected ages
    print(f"\n{'age':>5}{'source':>15}{'lo':>9}{'central':>9}{'hi':>9}"
          f"{'model ref':>11}{'half-width':>12}")
    for age in (0, 100, 200, 300, 400, 420, 450, 485, 500, 540):
        k = int(np.argmin(np.abs(ages - age)))
        print(f"{int(ages[k]):>5}{src[k]:>15}{out[k,1]:>9.0f}{out[k,0]:>9.0f}{out[k,2]:>9.0f}"
              f"{co2_ref[k]:>11.0f}{0.5*np.log2(out[k,2]/out[k,1]):>12.2f}")
    fin = np.isfinite(offset)
    big = int((np.abs(offset[fin]) > 0.5).sum())
    print(f"\ndiagnostic -- the reference run's prescribed CO2 vs the record's central "
          f"estimate:\n  median offset {np.median(offset[fin]):+.2f} doublings, range "
          f"{offset[fin].min():+.2f} to {offset[fin].max():+.2f}; "
          f"|offset| > 0.5 doublings at {big}/{fin.sum()} ages.")
    if a.mode == "relative":
        print("  In relative mode this offset does NOT enter the envelope -- it is reported "
              "because\n  it is a real difference between the forcing and the proxy record, "
              "and belongs in the text.")
    inside = int(sum(out[k, 1] <= co2_ref[k] <= out[k, 2] for k in range(ages.size)))
    print(f"  the reference CO2 lies inside the envelope at {inside}/{ages.size} ages.")
    print(f"wrote {a.out} and {prov.name}")

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib unavailable -- plot skipped", file=sys.stderr)
            return 0
        fig, ax = plt.subplots(figsize=(11, 4.6))
        cols = {"foster": "#2c5f8a", "judd": "#7d5ba6", "carried-width": "#c0392b",
                "blend": "#8d8d8d"}
        shown = set()
        k0 = 0
        for k in range(1, ages.size + 1):                    # contiguous runs, with a
            if k < ages.size and src[k] == src[k0]:          # one-point overlap so the
                continue                                     # segments join seamlessly
            sl = slice(k0, min(k + 1, ages.size))
            nm = src[k0]
            ax.fill_between(ages[sl], out[sl, 1], out[sl, 2], color=cols.get(nm, "#999"),
                            alpha=0.25, lw=0,
                            label=None if nm in shown else f"{nm} ({a.band}%)")
            shown.add(nm)
            k0 = k
        ax.plot(ages, co2_ref, color="k", lw=1.5,
                label="prescribed CO$_2$ of the reference run (= envelope centre)"
                      if a.mode == "relative" else "prescribed CO$_2$ of the reference run")
        ax.plot(ages, rec_central, color="#7d5ba6", lw=1.3, ls="--",
                label="central estimate of the proxy record")
        ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("Age (Ma)"); ax.set_ylabel("pCO$_2$ (ppm)")
        ax.set_title(f"pCO₂ envelope driving the climate-forcing uncertainty "
                     f"({a.band}% band, {a.mode} mode)", loc="left", fontsize=11)
        ax.grid(alpha=0.25, which="both"); ax.legend(frameon=False, fontsize=8.5, ncol=2)
        fig.tight_layout(); fig.savefig(a.plot, dpi=200)
        print(f"wrote {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

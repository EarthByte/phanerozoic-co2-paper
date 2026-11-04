#!/usr/bin/env bash
# Move the whole analysis onto the adopted adjusted surface: rebuild the archived erosion fields with
# NeuroLEM on the product grids, replace the archive's palaeoelevation products, preserve what the direct
# goSPL comparison must keep, and set aside the tables that the re-run must not resume from.
#
# Role in the paper: the adjusted palaeoelevation surface used throughout (Methods, "Palaeoelevation") is
# the product grid set corrected_scotese_reach500_land of the palaeotopography assimilation. This script
# makes the data archive carry that surface: the NeuroLEM erosion fields (surrogate_erodep/), the shipped
# 1-degree grids (dem_scotese_adjusted/) and the ensemble summaries. Every table and figure is then
# recomputed from the archive by run_uncertainty_m3.sh. It is the "rebuild-surface" step of that script.
#
#   ./rebuild_surface.sh                 the product named in PRODUCT (corrected_scotese_reach500_land)
#   ./rebuild_surface.sh --dry-run       print what would move and run; change nothing
#
# Environment (all optional):
#   NEUROLEM_CO2_DATA   data archive (default: neurolem_CO2_zenodo_data beside the repository)
#   PALEO_ROOT          Paleotopo_data_assimilation checkout holding data/<PRODUCT> and the ensemble summaries
#   PRODUCT             product grid set to adopt (default: corrected_scotese_reach500_land)
#   PREVIOUS_SURFACE    name of the surface being replaced, used to label the _to_delete/ folders
#   NEUROLEM_PY         Python with torch and netCDF4, for apply_dem.py
#   ANALYSIS_PY         Python for check_surface_provenance.py (default: python3)
#
# What it does, in order (nothing is deleted; superseded files go to _to_delete/ folders named after
# the surface they belong to and a time stamp):
#   0. checks that 109 product grids and 109 ensemble summaries exist and that NEUROLEM_PY has torch and
#      netCDF4, and stops without changes if check_surface_provenance.py already passes at 220 Ma
#   1. preserves the NeuroLEM main run and ensemble table of the surface the direct goSPL runs used, in
#      data/nature_recast/gospl_surfaces/, so gospl_validation_sensitivity.py keeps pairing goSPL with the
#      surrogate on identical inputs (the goSPL runs are not repeated)
#   2. runs apply_dem.py on the product grids into a fresh directory (109 ages; the reference arm
#      ero_scotese is reused from the present archive and checked identical; GPU, about 15 min),
#      compares the old and new archives age by age (surface_change_<previous>_to_<product>.csv),
#      then swaps the new one in as surrogate_erodep/
#   3. replaces dem_scotese_adjusted/ with copies of the product grids and
#      geochem_uncertainty_ensemble/elevation_summary/ with the ensemble summaries built on them;
#      the NeuroLEM member tables of the earlier ensemble go to _to_delete/
#   4. moves every derived table of the earlier surface out of data/nature_recast/ (main-run tables,
#      ensemble, climate, goSPL, reactivity, audit) and sets aside the input cache; the runoff-ratio and
#      degassing inputs stay
#   5. runs check_surface_provenance.py, which must pass before anything else is computed
# Then: ./run_uncertainty_m3.sh main, runoff-tests, ensemble, climate, gospl-published, gospl,
# reactivity, finalise (or ./run_uncertainty_m3.sh resume-surface for the whole sequence).
#
# Exit codes: 3 NEUROLEM_PY unusable, 4 missing product grids, summaries or archive, 5 incomplete rebuild
# (the present archive is then left untouched).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
ROOT="$(cd "$HERE/../.." && pwd)"                                     # CO2_paper
NEU="$(cd "$ROOT/.." && pwd)"                                         # NeuroLEM
ZEN="${NEUROLEM_CO2_DATA:-$NEU/neurolem_CO2_zenodo_data}"
PALEO_ROOT="${PALEO_ROOT:-$(cd "$NEU/../Paleotopo_data_assimilation" && pwd)}"
PRODUCT="${PRODUCT:-corrected_scotese_reach500_land}"
PRODUCT_DIR="$PALEO_ROOT/data/$PRODUCT"
ENSEMBLE_SUMMARIES="$PALEO_ROOT/data/geochem_elevation_ensemble_reach500_land"
PY="${NEUROLEM_PY:-python3}"
APY="${ANALYSIS_PY:-python3}"
OUT="$ROOT/data/nature_recast"
STAMP="$(date +%Y%m%d_%H%M%S)"
PREVIOUS="${PREVIOUS_SURFACE:-nocraton}"                              # name of the surface being replaced, for the _to_delete folders
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
# run CMD...: execute, or in --dry-run only print the command
run() { if [[ $DRY -eq 1 ]]; then echo "  would: $*"; else "$@"; fi; }

echo "Rebuild on $PRODUCT"
echo "  product grids:      $PRODUCT_DIR"
echo "  ensemble summaries: $ENSEMBLE_SUMMARIES"
echo "  data archive:       $ZEN"
# 0. preconditions: complete product and summary sets, a working NeuroLEM Python, an existing archive
n_prod=$(ls "$PRODUCT_DIR"/*Ma_corrected_SW.nc 2>/dev/null | wc -l | tr -d ' ')
n_sum=$(ls "$ENSEMBLE_SUMMARIES"/*Ma_geochem_ensemble_summary.nc 2>/dev/null | wc -l | tr -d ' ')
[[ "$n_prod" -eq 109 ]] || { echo "expected 109 product grids in $PRODUCT_DIR, found $n_prod" >&2; exit 4; }
[[ "$n_sum" -eq 109 ]] || { echo "expected 109 ensemble summaries in $ENSEMBLE_SUMMARIES, found $n_sum" >&2; exit 4; }
if [[ $DRY -eq 0 ]]; then
    [[ -x "$PY" ]] || PY="$(command -v python3 || true)"
    [[ -n "$PY" && -x "$PY" ]] || { echo "no usable Python: set NEUROLEM_PY" >&2; exit 3; }
    "$PY" -c 'import torch, numpy, netCDF4' || { echo "NEUROLEM_PY lacks torch or netCDF4" >&2; exit 3; }
fi
[[ -d "$ZEN/surrogate_erodep" && -e "$ZEN/surrogate_erodep/erodep_220Ma.npz" ]] || { echo "no present archive at $ZEN/surrogate_erodep" >&2; exit 4; }
# a quick provenance check at one age: if the archive already carries the product, there is nothing to do
if "$APY" check_surface_provenance.py --ages 220 --out "${TMPDIR:-/tmp}/surface_provenance_220.csv" > /dev/null 2>&1; then
    echo "the archive already carries $PRODUCT (check_surface_provenance.py passes at 220 Ma); nothing to rebuild"; exit 0
fi

# 1. preserve the paired NeuroLEM run for the direct goSPL comparison
PAIRED="$OUT/gospl_surfaces"
if [[ ! -e "$PAIRED/kin_bridge_s2_ens.csv" ]]; then
    echo "==> preserving the main run and ensemble table of the goSPL surfaces in $PAIRED"
    run mkdir -p "$PAIRED"
    for f in NewZealand.csv Sichuan.csv Taiwan.csv amplitudes.txt ens.csv hist.npz; do run cp -p "$OUT/kin_bridge_s2_$f" "$PAIRED/"; done
    run cp -p "$ZEN/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv" "$PAIRED/ensemble_sensitivity_members.csv"
    if [[ $DRY -eq 0 ]]; then
        cat > "$PAIRED/README.md" <<EOF
# NeuroLEM run on the surfaces of the direct goSPL simulations

The 20 direct goSPL runs (reference surface, adjusted surface, two ensemble members at 60, 220, 240, 460
and 520 Ma) were made on the adjusted surface of 2026-08, \`corrected_scotese_${PREVIOUS}\`, and are not
repeated. This folder preserves the NeuroLEM main run (\`kin_bridge_s2_*\`, the same weathering
calibration the goSPL evaluation uses) and the ensemble member table computed on that surface, so that
\`gospl_validation_sensitivity.py\` pairs every goSPL run with the surrogate on identical inputs. The
paper's adjusted surface is \`$PRODUCT\`; the reference surface is the same in both.
Preserved $STAMP.
EOF
    fi
else
    echo "==> $PAIRED already holds the paired run; kept"
fi

# 2. NeuroLEM on the product grids ("delta" mode: the 1-degree increment added to the goSPL physiography),
#    written to a fresh time-stamped directory so an interrupted run cannot touch the present archive
NEW="$ZEN/surrogate_erodep.rebuild_$STAMP"
echo "==> NeuroLEM on $PRODUCT, 109 ages -> $NEW (reference arm reused from the present archive)"
run "$PY" "$ROOT/scripts/apply_dem.py" --zhou-dir "$PRODUCT_DIR" --salles-dir "$ZEN/gospl_salles" \
    --ckpt "$ZEN/surrogate_model/best.pt" --out "$NEW" --mode delta --reuse-baseline-dir "$ZEN/surrogate_erodep"
if [[ $DRY -eq 0 ]]; then
    n_new=$(ls "$NEW"/erodep_*Ma.npz | wc -l | tr -d ' ')
    [[ "$n_new" -eq 109 ]] || { echo "rebuild wrote $n_new of 109 ages; the present archive is untouched at $ZEN/surrogate_erodep" >&2; exit 5; }
    # Per age, in the embedded Python below: the reference arm and inputs must be bit-identical; the table
    # records the land-mean and largest adjustment of both surfaces, their land-cell counts and the RMS
    # change of the adjusted erosion field (mm/yr) over the land of either surface.
    echo "  comparing the two archives age by age -> $OUT/surface_change_${PREVIOUS}_to_${PRODUCT}.csv"
    "$PY" - "$ZEN/surrogate_erodep" "$NEW" "$OUT/surface_change_${PREVIOUS}_to_${PRODUCT}.csv" "$PREVIOUS" "$PRODUCT" <<'PYCHECK'
import sys, csv, numpy as np
old, new, out, prev, prod = sys.argv[1:6]
rows = []
for age in range(0, 545, 5):
    with np.load(f'{old}/erodep_{age}Ma.npz') as a, np.load(f'{new}/erodep_{age}Ma.npz') as b:
        assert np.array_equal(a['ero_scotese'], b['ero_scotese']), f'{age} Ma: reference arm differs'
        assert np.array_equal(a['rain'], b['rain']) and np.array_equal(a['zpaleo_scotese'], b['zpaleo_scotese']), f'{age} Ma: inputs differ'
        ma, mb = a['mask_zhou'], b['mask_zhou']
        rows.append(dict(age=age, adjustment_landmean_previous_m=float(a['delta'][ma].mean()), adjustment_landmean_product_m=float(b['delta'][mb].mean()),
                         adjustment_maxabs_previous_m=float(np.abs(a['delta'][ma]).max()), adjustment_maxabs_product_m=float(np.abs(b['delta'][mb]).max()),
                         land_previous_cells=int(ma.sum()), land_product_cells=int(mb.sum()),
                         rmse_adjusted_erosion_change_mm_yr=float(np.sqrt(np.mean((b['ero_zhou'] - a['ero_zhou'])[ma | mb] ** 2)))))
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
p = np.array([r['adjustment_landmean_previous_m'] for r in rows]); q = np.array([r['adjustment_landmean_product_m'] for r in rows])
print('  reference arm identical at all 109 ages')
print(f'  adjustment over emergent land, mean over ages: {prev} {p.mean():+.1f} m, {prod} {q.mean():+.1f} m; per age {q.min():+.1f} to {q.max():+.1f} m')
PYCHECK
fi
# swap: the present archive goes to _to_delete/, the rebuilt one takes its name
run mkdir -p "$ZEN/_to_delete"
run mv -n "$ZEN/surrogate_erodep" "$ZEN/_to_delete/surrogate_erodep_${PREVIOUS}_$STAMP"
run mv -n "$NEW" "$ZEN/surrogate_erodep"
echo "  archive swapped in; superseded fields in $ZEN/_to_delete/surrogate_erodep_${PREVIOUS}_$STAMP"

# 3. palaeoelevation products of the archive
echo "==> replacing dem_scotese_adjusted/ and the ensemble summaries"
[[ -d "$ZEN/dem_scotese_adjusted" ]] && run mv -n "$ZEN/dem_scotese_adjusted" "$ZEN/_to_delete/dem_scotese_adjusted_${PREVIOUS}_$STAMP"
run mkdir -p "$ZEN/dem_scotese_adjusted"
if [[ $DRY -eq 0 ]]; then cp -p "$PRODUCT_DIR"/*Ma_corrected_SW.nc "$ZEN/dem_scotese_adjusted/"; else echo "  would: cp product grids -> $ZEN/dem_scotese_adjusted/"; fi
GE="$ZEN/geochem_uncertainty_ensemble"
[[ -d "$GE/elevation_summary" ]] && run mv -n "$GE/elevation_summary" "$ZEN/_to_delete/ensemble_elevation_summary_${PREVIOUS}_$STAMP"
[[ -d "$GE/neurolem_results" ]] && run mv -n "$GE/neurolem_results" "$ZEN/_to_delete/ensemble_neurolem_results_${PREVIOUS}_$STAMP"
[[ -e "$GE/nature_recast/ensemble_sensitivity_members.csv" ]] && run mv -n "$GE/nature_recast/ensemble_sensitivity_members.csv" "$ZEN/_to_delete/ensemble_sensitivity_members_${PREVIOUS}_$STAMP.csv"
run mkdir -p "$GE/elevation_summary"
if [[ $DRY -eq 0 ]]; then cp -p "$ENSEMBLE_SUMMARIES"/*Ma_geochem_ensemble_summary.nc "$ENSEMBLE_SUMMARIES"/geochem_elevation_ensemble_summary.csv "$GE/elevation_summary/"; else echo "  would: cp ensemble summaries -> $GE/elevation_summary/"; fi

# 4. derived tables of the earlier surface (moved, never deleted; the glob list is matched with nullglob so a
#    pattern without matches is skipped)
TD="$OUT/_to_delete/${PREVIOUS}_$STAMP"
echo "==> moving the derived tables of the earlier surface to $TD"
run mkdir -p "$TD"
shopt -s nullglob
for f in "$OUT"/kin_*_* "$OUT"/out_*_* "$OUT"/rft_* "$OUT"/log_*.txt \
         "$OUT"/ensemble_sensitivity_*.csv "$OUT"/climate_sensitivity_*.csv "$OUT"/gospl_*.csv \
         "$OUT"/reactivity_*.csv "$OUT"/archive_consistency.csv "$OUT"/runoff_form_tests.csv "$OUT"/surface_provenance.csv \
         "$OUT"/highland_stats.csv "$OUT"/beyond_vs_co2.csv "$OUT"/number_audit.md "$OUT"/uncertainty_statements.md; do
    run mv -n "$f" "$TD/"
done
shopt -u nullglob
[[ -e "$ZEN/nature_recast_cache.zip" ]] && run mv -n "$ZEN/nature_recast_cache.zip" "$ZEN/_to_delete/nature_recast_cache_${PREVIOUS}_$STAMP.zip"
echo "  kept: hadcm3l_runoff_*.npz, muller2024_gross_degassing.csv, gospl_surfaces/"

# 5. the archive must now be the product
echo "==> provenance"
run "$APY" check_surface_provenance.py
echo "Rebuilt on $PRODUCT. Next: ./run_uncertainty_m3.sh resume-surface   (main, runoff-tests, ensemble, climate, gospl-published, gospl, reactivity, finalise)"

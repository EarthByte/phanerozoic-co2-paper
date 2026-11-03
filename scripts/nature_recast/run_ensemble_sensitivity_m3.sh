#!/usr/bin/env bash
# Palaeoelevation ensemble under the weathering formulation of the Nature-format manuscript.
#
# For every age: generate the geochemically permitted elevation members (gplately environment),
# run NeuroLEM on each and evaluate fluxes, erosion sensitivities and the share of land beyond the
# optimum (neurolem-mps environment), delete the temporary 1-degree member cube. The whole experiment
# writes ONE member table; ages already complete in it are skipped, so the command is resumable.
#
#   ./run_ensemble_sensitivity_m3.sh --ages 220 --members 12      smoke test (minutes)
#   ./run_ensemble_sensitivity_m3.sh --ages 220                   pilot: 200 members
#   ./run_ensemble_sensitivity_m3.sh                              all 109 ages, 200 members
#
# Two checks guard every age: the generator's unperturbed member chain must reproduce the production
# grid of the central directory (else nothing is generated), and NeuroLEM on that production correction
# must reproduce the archived adjusted erosion field and the main-run table (else the age is rejected).
#
# Options: --members N  --seed S  --ages A B C ...  --workers W  --no-verify  --force
# Environment overrides: PALEOTOPO_PY, NEUROLEM_PY, NEUROLEM_CO2_DATA, PALEO_ROOT, PALEO_CENTRAL_DIR, ENSEMBLE_CSV
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                                   # CO2_paper
ZENODO_ROOT="${NEUROLEM_CO2_DATA:-$ROOT/../neurolem_CO2_zenodo_data}"
PALEO_ROOT="${PALEO_ROOT:-$(cd "$ROOT/../../Paleotopo_data_assimilation" && pwd)}"
PALEOTOPO_PY="${PALEOTOPO_PY:-python3}"
NEUROLEM_PY="${NEUROLEM_PY:-python3}"
CENTRAL_DIR="${PALEO_CENTRAL_DIR:-$PALEO_ROOT/data/corrected_scotese_reach500_land}"
OUT_CSV="${ENSEMBLE_CSV:-$ZENODO_ROOT/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv}"
GENERATOR="$ROOT/scripts/build_geochem_elevation_ensemble.py"
APPLIER="$HERE/apply_ensemble_sensitivity.py"
MAIN_RUN="$ROOT/data/nature_recast/kin_bridge_s2"

MEMBERS=200; SEED=20260814; AGES=(); EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --members) MEMBERS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --workers) EXTRA+=(--workers "$2"); shift 2 ;;
        --no-verify) EXTRA+=(--no-verify); shift ;;
        --force) EXTRA+=(--force); FORCE=1; shift ;;
        --ages) shift; while [[ $# -gt 0 && "$1" != --* ]]; do AGES+=("$1"); shift; done ;;
        -h|--help) awk 'NR == 1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
FORCE="${FORCE:-0}"
if [[ ${#AGES[@]} -eq 0 ]]; then for ((age=0; age<=540; age+=5)); do AGES+=("$age"); done; fi

for required in "$GENERATOR" "$APPLIER" "${MAIN_RUN}_amplitudes.txt" "${MAIN_RUN}_hist.npz" \
                "$ROOT/data/nature_recast/hadcm3l_runoff_scotese_spinupa.npz" "$ZENODO_ROOT/surrogate_model/best.pt"; do
    [[ -e "$required" ]] || { echo "Required input missing: $required" >&2; exit 4; }
done
for required in "$ZENODO_ROOT/gospl_salles" "$ZENODO_ROOT/valdes_temperature" "$CENTRAL_DIR"; do
    [[ -d "$required" ]] || { echo "Required directory missing: $required" >&2; exit 4; }
done
[[ -x "$PALEOTOPO_PY" ]] || { echo "PALEOTOPO_PY is not executable: $PALEOTOPO_PY" >&2; exit 3; }
[[ -x "$NEUROLEM_PY" ]] || { echo "NEUROLEM_PY is not executable: $NEUROLEM_PY (run_geochem_uncertainty_m3.sh creates it)" >&2; exit 3; }
"$PALEOTOPO_PY" -c 'import numpy, pandas, scipy, netCDF4, pygplates' || { echo "PALEOTOPO_PY lacks a required package" >&2; exit 3; }
"$NEUROLEM_PY" -c 'import numpy, pandas, scipy, netCDF4, torch' || { echo "NEUROLEM_PY lacks a required package" >&2; exit 3; }

mkdir -p "$(dirname "$OUT_CSV")"
TASK_TMP="$(mktemp -d "${TMPDIR:-/tmp}/ensemble-sensitivity.XXXXXX")"
trap 'rm -rf "$TASK_TMP"' EXIT INT TERM
DONE=" $("$NEUROLEM_PY" "$APPLIER" --list-done "$MEMBERS" --out-csv "$OUT_CSV") "

echo "Palaeoelevation ensemble -> NeuroLEM -> erosion sensitivity"
echo "  members:      $MEMBERS (seed $SEED)"
echo "  ages:         ${AGES[*]}"
echo "  central maps: $CENTRAL_DIR"
echo "  calibration:  $MAIN_RUN"
echo "  member table: $OUT_CSV"

START=$(date +%s); N=0
for age in "${AGES[@]}"; do
    if [[ "$DONE" == *" $age "* && "$FORCE" -eq 0 ]]; then echo "[$age Ma] already complete; skipping"; continue; fi
    MEMBER_NC="$TASK_TMP/${age}Ma_members.nc"
    echo "[$age Ma] generating $MEMBERS elevation members"
    PALEOTOPO_PROJECT_ROOT="$PALEO_ROOT" "$PALEOTOPO_PY" "$GENERATOR" "$age" --members "$MEMBERS" --seed "$SEED" \
        --central-dir "$CENTRAL_DIR" --out-dir "$TASK_TMP/elevation_summary" --write-members --members-out "$MEMBER_NC"
    "$NEUROLEM_PY" "$APPLIER" --members-nc "$MEMBER_NC" --out-csv "$OUT_CSV" --main-run "$MAIN_RUN" \
        --salles-dir "$ZENODO_ROOT/gospl_salles" --ckpt "$ZENODO_ROOT/surrogate_model/best.pt" ${EXTRA[@]+"${EXTRA[@]}"}
    rm -f "$MEMBER_NC" "$TASK_TMP/elevation_summary/${age}Ma_geochem_ensemble_summary.nc"
    N=$((N + 1)); echo "[$age Ma] done; $N ages in $(( ($(date +%s) - START) / 60 )) min this session"
done

"$NEUROLEM_PY" "$HERE/collate_ensemble_sensitivity.py" "$OUT_CSV"

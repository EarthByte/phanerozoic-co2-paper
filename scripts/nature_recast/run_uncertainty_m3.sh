#!/usr/bin/env bash
# One entry point for the computations of the paper: the main model, its sensitivity tests and the
# uncertainty experiments behind the Extended Data figures and the Supplementary Information (Methods,
# "Uncertainty and sensitivity"). Run from scripts/nature_recast:  ./run_uncertainty_m3.sh <step> [arguments for the step]
#
#   ./run_uncertainty_m3.sh status           what is complete, what to run next
#   ./run_uncertainty_m3.sh ensemble         palaeoelevation ensemble, 200 members x 109 ages (resumable)
#   ./run_uncertainty_m3.sh climate-pilot    27 climate states at 240 Ma, with both consistency checks
#   ./run_uncertainty_m3.sh climate          27 climate states x 109 ages (resumable), then the Monte Carlo bands
#   ./run_uncertainty_m3.sh gospl            direct goSPL runs on their own mesh (needs the volume holding the goSPL run bundle)
#   ./run_uncertainty_m3.sh cache            input cache for Figs 1 and 4 and the reactivity runs
#   ./run_uncertainty_m3.sh finalise         figure and statements from whatever is complete
#   ./run_uncertainty_m3.sh after-ensemble   climate, gospl, cache, finalise in sequence
#   ./run_uncertainty_m3.sh rebuild-surface  NeuroLEM on the adopted adjusted surface (corrected_scotese_reach500_land): new archive,
#                                            archive products replaced, earlier tables moved aside (GPU, about 15 min; rebuild_surface.sh)
#   ./run_uncertainty_m3.sh runoff-tests     the two runoff forks against the main model, one table (CPU, four runs)
#   ./run_uncertainty_m3.sh octopus          present-day erosion field against the OCTOPUS cosmogenic basins: table, summary, Extended Data Fig. 10 (CPU; downloads 3 zips once)
#   ./run_uncertainty_m3.sh distribution-tests   results with the simulated erosion-rate distribution widened (k = 1.3, 1.5, 1.8), one table (CPU, three runs)
#   ./run_uncertainty_m3.sh martin-lamb      present-day eroding/depositing land against the Martin & Lamb (2026) process domains; weathering with more land withheld (CPU)
#   ./run_uncertainty_m3.sh dyntopo-pilot    dynamic topography (gld428, all continental crust): both checks at 0 Ma and one perturbed age, 150 Ma (GPU, minutes)
#   ./run_uncertainty_m3.sh dyntopo          the same at all 51 ages, 0-250 Ma (resumable), then the comparison and the decision rule (GPU, about 10 min;
#                                            needs Paleotopo_data_assimilation beside NeuroLEM, or PALEO_ROOT; surfaces in <archive>/dyntopo_gld428/)
#   ./run_uncertainty_m3.sh dyntopo-controls the six control terms of the dynamic-topography test (time shifts first), then their reading (GPU, about 40 min)
#   ./run_uncertainty_m3.sh orogen-tracks    four mountain belts followed through the plate model: before/after table, sensitivities, Supplementary Fig. 1 (CPU; track step needs pygplates)
#   ./run_uncertainty_m3.sh orogen-events    thirteen Phanerozoic orogens (orogen_catalogue.py): event-scale sensitivities, Fig. 4 (CPU; locate and track need pygplates)
#   ./run_uncertainty_m3.sh resume-surface   rebuild-surface, main, runoff-tests, distribution-tests, octopus, ensemble, climate, gospl-published, gospl, reactivity, finalise
#   ./run_uncertainty_m3.sh resume-experiments   the same from the ensemble on (main model and runoff tests already done)
#   ./run_uncertainty_m3.sh provenance       is every archived surface the adopted adjusted surface? (CPU, one table)
#   ./run_uncertainty_m3.sh archive-check    is every archived erosion field what NeuroLEM gives on the archived inputs? (GPU, 218 passes)
#   ./run_uncertainty_m3.sh archive-repair   the same, replacing fields that fail; superseded files go to _to_delete
#   ./run_uncertainty_m3.sh main             main-model tables, figures and number audit (rebuilds the cache if the archive is newer)
#   ./run_uncertainty_m3.sh reactivity       fixed-CO2 reactivity, its summary, number audit, Extended Data Figs 8 and 9 (CPU)
#   ./run_uncertainty_m3.sh aess             weathering feedback with the emulator scaled to the PhanDA apparent Earth-system sensitivity (CPU, about 6 min; resumable)
#   ./run_uncertainty_m3.sh gospl-published  published goSPL fields of the reference surface, all ages (CPU)
#   ./run_uncertainty_m3.sh resume           archive-repair, main, climate, finalise in sequence
#   ./run_uncertainty_m3.sh pure-west        every result that depends on the weathering form, recomputed from the erosion archive (named after
#                                            the West (2012) form of the sulfide runoff term): park-tables (unless the run has already started),
#                                            then main, runoff-tests, distribution-tests, martin-lamb, orogen-tracks, orogen-events, reactivity,
#                                            gospl-published and pure-west-gpu (GPU; most of a day). Erosion fields, OCTOPUS and the direct
#                                            goSPL comparison (paired with its preserved run, published exponent) do not depend on it and are not re-run.
#   ./run_uncertainty_m3.sh pure-west-gpu    the second half of pure-west alone (ensemble, climate, dyntopo with no surfaces kept, dyntopo-controls,
#                                            finalise, number audit, all figures, videos, pure-west-check): resumes the GPU experiments after an
#                                            interruption without repeating the CPU steps
#   ./run_uncertainty_m3.sh pure-west-check  lists the result tables that are missing or older than the start of the run (RUN_TAG)
#   ./run_uncertainty_m3.sh longterm         pure-west under the run tag 'longterm', with its own park folder (PARK_NAME); repeat the same
#                                            command to resume (most of a day)
#   ./run_uncertainty_m3.sh park-tables      copy every result table, the audit and the figures to PARK_ROOT/PARK_NAME as a record of the
#                                            previous results, and move there the resumable tables, so that their steps start afresh;
#                                            writes the start marker of RUN_TAG
#
# The ensemble and the climate experiment both use the GPU: run them one after the other, not together.
# Every step writes ONE table and can be repeated safely. Extra arguments are passed to the step.
#
# Environment: NEUROLEM_PY (Python with PyTorch, for the GPU steps), ANALYSIS_PY (Python for tables and figures;
# default python3, or NEUROLEM_PY if python3 lacks numpy, pandas, scipy or matplotlib), NEUROLEM_CO2_DATA (companion
# data archive), PALEO_ROOT (palaeotopography assimilation checkout), PARK_ROOT and PARK_NAME (where park-tables
# copies to), RUN_TAG (name of the start marker used by pure-west and pure-west-check; default pure_west).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PY="${NEUROLEM_PY:-python3}"
[[ -x "$PY" ]] || PY="$(command -v python3 || true)"          # fall back to python3 on the PATH
[[ -n "$PY" && -x "$PY" ]] || { echo "no usable Python: set NEUROLEM_PY" >&2; exit 3; }
# GPU steps use the PyTorch environment; tables and the figure use the Python on the PATH (override with ANALYSIS_PY)
APY="${ANALYSIS_PY:-python3}"
"$APY" -c 'import numpy, pandas, scipy, matplotlib' 2>/dev/null || APY="$PY"
STEP="${1:-status}"; shift || true

# the filled runoff-ratio table (fill_runoff_ratio.py) is built the first time a step needs it
need_ratio() { [[ -e "$HERE/../../data/nature_recast/hadcm3l_runoff_ratio_filled.npz" ]] || "$APY" fill_runoff_ratio.py; }

case "$STEP" in
    status)        "$APY" status.py ;;
    ensemble)      ./run_ensemble_sensitivity_m3.sh "$@" ;;
    climate-pilot) need_ratio; "$PY" -u climate_sensitivity.py nodes --ages 240 "$@" ;;
    climate)       need_ratio; "$PY" -u climate_sensitivity.py nodes "$@"; "$APY" climate_sensitivity.py sample ;;
    gospl)         need_ratio; "$APY" -u gospl_validation_sensitivity.py "$@" ;;
    cache)         "$APY" -u build_cache.py "$@" ;;
    finalise)      if [[ -e "$HERE/../../data/nature_recast/climate_sensitivity_nodes.csv" ]]; then "$APY" climate_sensitivity.py sample; fi
                   "$APY" collate_ensemble_sensitivity.py || true
                   "$APY" finalise_uncertainty.py
                   "$APY" make_ed_figures.py 6 ;;                  # Extended Data Fig. 6 carries the ensemble and climate bands
    provenance)    "$APY" -u check_surface_provenance.py "$@" ;;
    rebuild-surface) ./rebuild_surface.sh "$@" ;;
    runoff-tests)  "$APY" -u runoff_form_tests.py "$@" ;;
    octopus)       "$APY" -u octopus_comparison.py fetch; "$APY" -u octopus_comparison.py "$@" ;;
    distribution-tests) "$APY" -u distribution_tests.py "$@" ;;
    dyntopo-pilot) need_ratio; "$PY" -u dyntopo_sensitivity.py run --ages 0 150 "$@" ;;
    dyntopo-controls) need_ratio; CTL="${PALEO_ROOT:-$HERE/../../../../Paleotopo_data_assimilation}/outputs_scotese_reach500_land/dt_term_all_crust"
                   for c in tsh075 tsh150 tsh225 lon060 lon120 lon180; do
                       "$PY" -u dyntopo_sensitivity.py run --term "$CTL/dt_term_control_${c}_0-250Ma.nc" --tag "control_$c" --no-surfaces "$@"
                   done
                   "$APY" dyntopo_sensitivity.py controls ;;
    dyntopo)       need_ratio; "$PY" -u dyntopo_sensitivity.py run "$@"; "$APY" dyntopo_sensitivity.py analyse ;;
    martin-lamb)   [[ -e "$HERE/../../data/martin_lamb/martin_lamb_domains_0p1deg.npz" ]] || { "$APY" -u martin_lamb_comparison.py fetch; "$APY" -u martin_lamb_comparison.py aggregate; }
                   "$APY" -u martin_lamb_comparison.py ;;
    orogen-tracks) "$APY" -u orogen_tracks.py select
                   "$APY" -c "import pygplates" 2>/dev/null && "$APY" -u orogen_tracks.py track || echo "pygplates not available: keeping the existing orogen_tracks.npz"
                   "$APY" -u orogen_tracks.py evaluate; "$APY" -u orogen_tracks.py analyse; "$APY" -u orogen_tracks.py sensitivity; "$APY" -u orogen_tracks.py figure ;;
    orogen-events) "$APY" -u orogen_catalogue.py scan
                   "$APY" -c "import pygplates" 2>/dev/null && "$APY" -u orogen_catalogue.py locate || echo "pygplates not available: keeping the existing orogen_catalogue.csv"
                   OROGEN_SET=events "$APY" -u orogen_tracks.py select
                   "$APY" -c "import pygplates" 2>/dev/null && OROGEN_SET=events "$APY" -u orogen_tracks.py track || echo "pygplates not available: keeping the existing orogen_tracks_events.npz"
                   for s in evaluate analyse sensitivity; do OROGEN_SET=events "$APY" -u orogen_tracks.py $s; done; "$APY" make_fig4_endmembers.py ;;
    resume-surface) "$0" rebuild-surface; "$0" main || echo "main step reported an error (see above); going on"
                   "$0" runoff-tests; "$0" distribution-tests; "$0" octopus || echo "octopus step reported an error (geopandas installed?); going on"
                   "$0" ensemble; "$0" climate; "$0" gospl-published
                   "$0" gospl || echo "gospl step skipped (direct goSPL run bundle not found?)"; "$0" reactivity; "$0" finalise; "$0" provenance ;;
    resume-experiments) "$0" ensemble; "$0" climate; "$0" gospl-published
                   "$0" gospl || echo "gospl step skipped (direct goSPL run bundle not found?)"; "$0" reactivity; "$0" finalise; "$0" provenance ;;
    archive-check)  "$PY" -u check_archive.py "$@" ;;
    archive-repair) "$PY" -u check_archive.py --repair "$@" ;;
    main)          ZEN="${NEUROLEM_CO2_DATA:-$HERE/../../../neurolem_CO2_zenodo_data}"; ZIP="$ZEN/nature_recast_cache.zip"
                   if [[ -e "$ZIP" && -n "$(find "$ZEN/surrogate_erodep" -name 'erodep_*Ma.npz' -newer "$ZIP" -print -quit)" ]]; then
                       mkdir -p "$ZEN/_to_delete"; OLD="$ZEN/_to_delete/nature_recast_cache_superseded_$(date +%Y%m%d_%H%M%S).zip"
                       mv -n "$ZIP" "$OLD"; echo "archive is newer than the input cache: cache moved to $OLD and rebuilt"
                   fi
                   PATH="$(dirname "$(command -v "$APY")"):$PATH" bash run_all.sh
                   "$APY" ../make_main_map_panels.py ;;          # Extended Data Fig. 2, the two palaeoelevation histories, from the archive
    aess)          "$APY" -u aess_feedback_test.py ;;
    reactivity)    "$APY" -u run_reactivity.py > "$HERE/../../data/nature_recast/log_reactivity.txt"; "$APY" reactivity_post.py
                   "$APY" verify_numbers.py > /dev/null; "$APY" make_ed_figures.py 8 9 ;;
    gospl-published) need_ratio; "$APY" -u gospl_published_sensitivity.py "$@" ;;
    resume)        "$0" archive-repair; "$0" main || echo "main step reported an error (see above); going on with the climate run"
                   "$0" climate; "$0" finalise ;;
    after-ensemble) "$0" climate; "$0" gospl || echo "gospl step skipped (direct goSPL run bundle not found?)"; "$0" cache; "$0" finalise ;;
    park-tables)   PARK="${PARK_ROOT:-parked_tables}/${PARK_NAME:-tables_before_$(date +%Y%m%d)}"
                   [[ -d "$(dirname "$PARK")" ]] || { echo "folder $(dirname "$PARK") not found: set PARK_ROOT" >&2; exit 4; }
                   D="$HERE/../../data/nature_recast"; ZEN="${NEUROLEM_CO2_DATA:-$HERE/../../../neurolem_CO2_zenodo_data}"
                   mkdir -p "$PARK/data_nature_recast" "$PARK/figures_nature" "$PARK/ensemble"
                   shopt -s nullglob
                   cp -p "$D"/*.csv "$D"/*.md "$D"/*.txt "$D"/*.npz "$PARK/data_nature_recast/"          # the record (inputs included; gospl_surfaces stays)
                   cp -p "$HERE"/../../figures/nature/*.png "$HERE"/../../figures/nature/*.pdf "$PARK/figures_nature/"
                   for f in "$D"/climate_sensitivity_nodes.csv "$D"/gospl_published_reference_sensitivity.csv "$D"/dyntopo_gld428_allcrust.csv \
                            "$D"/dyntopo_control_*.csv "$D"/rft_* "$D"/dst_*; do
                       mv -f "$f" "$PARK/data_nature_recast/"                                                  # resumable: must start empty
                   done
                   M="$ZEN/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv"
                   [[ -e "$M" ]] && mv -f "$M" "$PARK/ensemble/"
                   shopt -u nullglob
                   date +%s > "$D/.${RUN_TAG:-pure_west}_start"
                   echo "parked in $PARK" ;;
    pure-west)     [[ -e "$HERE/../../data/nature_recast/.${RUN_TAG:-pure_west}_start" ]] || "$0" park-tables
                   "$0" main || echo "main step reported an error (see above); going on"
                   "$0" runoff-tests --force; "$0" distribution-tests --force
                   "$0" martin-lamb || echo "martin-lamb step reported an error; going on"
                   "$0" orogen-tracks || echo "orogen-tracks step reported an error; going on"
                   "$0" orogen-events || echo "orogen-events step reported an error; going on"
                   "$0" reactivity; "$0" gospl-published
                   "$0" pure-west-gpu ;;
    pure-west-gpu) "$0" ensemble; "$0" climate                                   # resumable: re-run this step alone after an interruption
                   need_ratio; "$PY" -u dyntopo_sensitivity.py run --no-surfaces; "$APY" dyntopo_sensitivity.py analyse
                   "$0" dyntopo-controls; "$0" finalise
                   "$APY" verify_numbers.py > /dev/null; "$APY" make_ed_figures.py
                   for f in make_fig1_optimum.py make_fig2_maps.py make_fig3_erosion_sensitivity.py make_fig4_endmembers.py make_fig5_reactivity.py; do "$APY" "$f"; done
                   "$APY" make_videos.py || echo "videos not made (imageio-ffmpeg missing?): run python3 make_videos.py"
                   "$0" pure-west-check ;;
    longterm)      # pure-west under the run tag 'longterm', the previous tables parked under PARK_NAME; resume after an
                   # interruption with the same command (or RUN_TAG=longterm ./run_uncertainty_m3.sh pure-west-gpu)
                   export PARK_NAME="${PARK_NAME:-tables_before_longterm}" RUN_TAG=longterm; "$0" pure-west ;;
    pure-west-check) D="$HERE/../../data/nature_recast"; T0=$(cat "$D/.${RUN_TAG:-pure_west}_start")
                   stale=0
                   for f in kin_bridge_s2_ens.csv kin_bridge_s1_ens.csv kin_bridge_s4_ens.csv out_bridge_s2_ens.csv kin_coef_s2_ens.csv out_coef_s2_ens.csv \
                            kin_bridge_s2_oclin_ens.csv runoff_form_tests.csv distribution_tests.csv martin_lamb_comparison.csv orogen_tracks_before_after.csv \
                            reactivity_summary.csv highland_stats.csv beyond_vs_co2.csv gospl_published_reference_sensitivity.csv ensemble_sensitivity_headline.csv \
                            climate_sensitivity_headline.csv climate_sensitivity_bands.csv dyntopo_sensitivity_table.csv dyntopo_controls_by_age.csv \
                            uncertainty_statements.md number_audit.md; do
                       if [[ ! -e "$D/$f" ]]; then echo "MISSING $f"; stale=1
                       elif (( $(stat -f %m "$D/$f" 2>/dev/null || stat -c %Y "$D/$f") < T0 )); then echo "STALE   $f"; stale=1; fi
                   done
                   (( stale )) && echo "some tables were not recomputed: see above" || { echo "all tables recomputed (${RUN_TAG:-pure_west})"; mv -f "$D/.${RUN_TAG:-pure_west}_start" "$D/.${RUN_TAG:-pure_west}_done"; } ;;
    -h|--help)     awk 'NR == 1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0" ;;
    *) echo "Unknown step: $STEP" >&2; exit 2 ;;
esac

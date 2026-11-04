#!/bin/bash
# Full Nature-recast analysis, from the archived erosion fields to figures and the number audit.
# Run from CO2_paper/scripts/nature_recast:  bash run_all.sh
# Needs numpy, scipy, pandas, matplotlib, cartopy (netCDF4 only for fetch_bridge_runoff.py).
#
# Steps: the input cache (build_cache.py); the main model and its alternatives (run_weathering.py);
# land-surface reactivity (run_reactivity.py, reactivity_post.py); mountain statistics and the share
# beyond the optimum at fixed CO2; the number audit (verify_numbers.py, data/nature_recast/number_audit.md);
# Figs 1-5 and Extended Data Figs 1 and 4-9 (figures/nature/).
#
# Weathering runs, each writing data/nature_recast/<name>_{ens,Taiwan,Sichuan,NewZealand}.csv, <name>_hist.npz,
# <name>_amplitudes.txt and log_<name>.txt. Names give runoff placement (kin = in the rate constant, out = on
# the whole flux), runoff source (bridge = HadCM3L runoff ratio, coef = uniform 0.4) and erosion scale:
#   kin_bridge_s2        main model
#   kin_bridge_s1, _s4   erosion calibration factors 1 and 4
#   out_bridge_s2        runoff on the whole flux
#   kin_coef_s2, out_coef_s2   uniform runoff ratio of 0.4
#   kin_bridge_s2_oclin  constant oxidised fraction of rock organic carbon
set -e
cd "$(dirname "$0")"
O=../../data/nature_recast
python3 build_cache.py                                   # one archive in the companion data folder; skips ages already present
# run NAME [options]: one weathering run, output prefix and log named after the run
run() { name=$1; shift; python3 run_weathering.py "$@" --out $O/$name > $O/log_$name.txt; }
# two groups of runs in parallel; run_reactivity.py reads the calibration of the main model, so it follows it in the
# same group
( run kin_bridge_s2 --mode kinetic --qsrc bridge          # main model
  python3 run_reactivity.py > $O/log_reactivity.txt
  run kin_bridge_s1 --mode kinetic --qsrc bridge --scale 1
  run kin_bridge_s4 --mode kinetic --qsrc bridge --scale 4 ) &
( run out_bridge_s2 --mode outer --qsrc bridge            # runoff on the whole flux
  run kin_coef_s2 --mode kinetic --qsrc coef              # uniform runoff coefficient
  run out_coef_s2 --mode outer --qsrc coef
  run kin_bridge_s2_oclin --mode kinetic --qsrc bridge --oc-form linear ) &
wait
# products of the main model: reactivity summary, mountain statistics, share beyond the optimum at fixed CO2
python3 reactivity_post.py
python3 highland_stats.py
python3 beyond_vs_co2.py
# number audit, then the figures
python3 verify_numbers.py > /dev/null
python3 make_fig1_optimum.py
python3 make_fig2_maps.py
python3 make_fig3_erosion_sensitivity.py
python3 make_fig4_endmembers.py
python3 make_fig5_reactivity.py
python3 make_ed1_workflow.py
python3 make_ed_figures.py
echo done

#!/bin/bash
# Full Nature-recast analysis, from the archived erosion fields to figures and the number audit.
# Run from CO2_paper/scripts/nature_recast:  bash run_all.sh
# Needs numpy, scipy, pandas, matplotlib, cartopy (netCDF4 only for fetch_bridge_runoff.py).
set -e
cd "$(dirname "$0")"
O=../../data/nature_recast
python3 build_cache.py                                   # one archive in the companion data folder; skips ages already present
run() { name=$1; shift; python3 run_weathering.py "$@" --out $O/$name > $O/log_$name.txt; }
( run kin_bridge_s2 --mode kinetic --qsrc bridge          # main model
  python3 run_reactivity.py > $O/log_reactivity.txt
  run kin_bridge_s1 --mode kinetic --qsrc bridge --scale 1
  run kin_bridge_s4 --mode kinetic --qsrc bridge --scale 4 ) &
( run out_bridge_s2 --mode outer --qsrc bridge            # runoff on the whole flux
  run kin_coef_s2 --mode kinetic --qsrc coef              # uniform runoff coefficient
  run out_coef_s2 --mode outer --qsrc coef
  run kin_bridge_s2_oclin --mode kinetic --qsrc bridge --oc-form linear ) &
wait
python3 reactivity_post.py
python3 highland_stats.py
python3 beyond_vs_co2.py
python3 verify_numbers.py > /dev/null
python3 make_fig1_optimum.py
python3 make_fig2_maps.py
python3 make_fig3_erosion_sensitivity.py
python3 make_fig4_endmembers.py
python3 make_fig5_reactivity.py
python3 make_ed1_workflow.py
python3 make_ed_figures.py
echo done

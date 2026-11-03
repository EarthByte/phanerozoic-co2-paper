# Erosion sensitivity of the continental carbon sink: analysis workflow

Scripts behind Müller et al., *Mountain uplift did not strengthen the weathering CO₂ sink over 540 million years*
(submitted to *Nature*).
They take the archived NeuroLEM erosion fields, evaluate the weathering laws of Bufe et al. (2024)
in every 0.1° land cell at 109 ages, and produce the result tables, figures and number audit.

The adjusted palaeoelevation surface is `Paleotopo_data_assimilation/data/corrected_scotese_reach500_land`
(craton samples excluded, 400 km support, residual kernel tapered to zero between 250 and 500 km from the
nearest sample, samples on PALEOMAP sea dropped), with `_p05` and `_p95` as its bounds. `rebuild_surface.sh`
moves the whole analysis onto a new adopted surface (NeuroLEM on the product grids into a fresh archive,
archive products replaced, earlier tables set aside, provenance measured), and `check_surface_provenance.py`
measures every archived surface against the product at any time; both are steps of `run_uncertainty_m3.sh`.

## What is computed

| Quantity | Definition |
|---|---|
| Silicate, sulfide, net inorganic CO₂ flux | West (2012) law with the three regional parameter sets of Bufe et al. (2024), each kept intact and calibrated separately to the modern silicate (0.099 Gt C/yr) and sulfide (0.020 Gt C/yr) fluxes; the central estimate is the mean of the three predictions |
| Runoff | `rain` channel × runoff ratio (surface + subsurface runoff over precipitation) of the same HadCM3L simulation |
| Runoff dependence | inside the kinetic term (`--mode kinetic`); on the whole flux as alternative (`--mode outer`) |
| Local optimum *E*opt | erosion rate that maximises net drawdown at the runoff of the cell |
| Erosion sensitivity *S* | d ln *F* / d ln λ for erosion scaled uniformly by λ, from the analytic derivative of the law; buffering capacity is 1 − *S* |
| Rock organic carbon | Zondervan et al. (2023) mass balance scaled to 68 Mt C/yr, oxidised fraction declining with sediment yield (`--oc-form linear` for a constant fraction); biospheric burial after Galy et al. (2015) |
| Weathering intensity, riverine δ⁷Li | Caves Rugenstein et al. (2019) |
| Reactivity and feedback strength | fluxes recomputed at 280, 560 and 1,120 ppm with the PhanDA pattern-scaling emulator |

## Layout

| Path | Content |
|---|---|
| `scripts/nature_recast/` | this folder |
| `data/nature_recast/` | result tables (`<run>_{ens,Taiwan,Sichuan,NewZealand}.csv`, histograms, calibration amplitudes), `hadcm3l_runoff_scotese_spinupa.npz`, `muller2024_gross_degassing.csv`, `number_audit.md` |
| `data/proxies/` | lithium-, strontium- and beryllium-isotope compilations with provenance (`README.md` there) |
| `figures/nature/` | main and Extended Data figures (PNG and PDF) |
| `../neurolem_CO2_zenodo_data/nature_recast_cache.zip` | land-only cache of all input fields, one archive, written by `build_cache.py` (7.8 GB, not tracked) |

Runs are named by runoff placement, runoff source and erosion scale: `kin_bridge_s2` is the main model
(kinetic placement, HadCM3L runoff ratio, erosion scale 2.0); `kin_bridge_s1`, `kin_bridge_s4`, `out_bridge_s2`,
`kin_coef_s2`, `out_coef_s2` and `kin_bridge_s2_oclin` are the alternatives shown in Extended Data.
Column suffixes: `_adj` rock-chemistry-adjusted surface (main model), `_ref` Scotese–Wright surface,
`_adjc` adjusted surface on the land common to both.

## Running

Everything, from the archived erosion fields to figures:

```bash
cd scripts/nature_recast
bash run_all.sh
```

Step by step:

```bash
python3 check_surface_provenance.py          # every archived surface against the adopted product grids; one table
python3 runoff_form_tests.py                 # the two runoff forks (sulfide exponent, Sichuan reference) against the main model
python3 octopus_comparison.py fetch          # once: the OCTOPUS v2 basin shapes (INT, AUS, XXL) from Zenodo into data/octopus/
python3 octopus_comparison.py                # present-day field against the cosmogenic basins: table, summary, Extended Data Fig. 10 (needs geopandas)
python3 distribution_tests.py                # the results with the simulated erosion-rate distribution widened (k = 1.3, 1.5, 1.8); three runs
python3 martin_lamb_comparison.py fetch       # once: the Martin & Lamb (2026) process-domain map (250 m, 269 MB) into data/martin_lamb/
python3 martin_lamb_comparison.py aggregate   # once: area of every domain in each 0.1° cell, one file (needs rasterio, pyproj)
python3 martin_lamb_comparison.py             # present-day eroding/depositing land against the domains; weathering with more land withheld
python3 dyntopo_sensitivity.py run --ages 0 150   # pilot of the dynamic-topography test (GPU; checks against the archive and the main run)
python3 dyntopo_sensitivity.py run                # all 51 ages, 0-250 Ma; resumable
python3 dyntopo_sensitivity.py analyse            # changes, lowland measures, comparators, decision rule
python3 dyntopo_sensitivity.py run --term <control>.nc --tag control_<name> --no-surfaces   # each control term
python3 dyntopo_sensitivity.py controls           # the control terms against the real term
python3 orogen_tracks.py select && python3 orogen_tracks.py track && python3 orogen_tracks.py evaluate && python3 orogen_tracks.py analyse && python3 orogen_tracks.py figure
                                             # four mountain belts followed through the PALEOMAP plate model (track needs pygplates)
python3 build_cache.py                       # once; writes one archive, skips ages already present
python3 run_weathering.py --mode kinetic --qsrc bridge --out ../../data/nature_recast/kin_bridge_s2
python3 run_reactivity.py
python3 reactivity_post.py
python3 highland_stats.py
python3 beyond_vs_co2.py
python3 verify_numbers.py                    # writes data/nature_recast/number_audit.md
python3 make_fig1_optimum.py
python3 make_fig2_maps.py
python3 make_fig3_erosion_sensitivity.py
python3 make_fig4_endmembers.py
python3 make_fig5_reactivity.py
python3 make_ed1_workflow.py
python3 make_ed_figures.py                   # Extended Data Figs 4-9; pass numbers to select
```

A pilot on three ages:

```bash
python3 build_cache.py 0 300 480
python3 run_weathering.py --mode kinetic --qsrc bridge --ages 0 300 480 --out /tmp/pilot
```

The figure scripts and `verify_numbers.py` read the tables in `data/nature_recast/` and run without the cache;
`make_fig2_maps.py`, `make_fig4_endmembers.py`, `run_*.py`, `highland_stats.py`, `beyond_vs_co2.py` and `orogen_scan.py` need it.

`fetch_bridge_runoff.py` rebuilds the runoff table from the BRIDGE server and `extract_degassing.py` rebuilds the
degassing table from the supplementary workbook of Müller et al. (2024); both tables ship with the repository.

## Palaeoelevation ensemble

`run_ensemble_sensitivity_m3.sh` passes the 200 geochemically permitted elevation members of every age through
NeuroLEM and through the same weathering evaluation as the main model (`apply_ensemble_sensitivity.py`), with the
calibration amplitudes and the *E*opt(runoff) table read from the main run, so members and main model differ in
elevation only. Each age adds 202 rows to ONE member table: the reference surface (member −2), the mean correction
(member −1) and the members. Two checks guard every age: the generator's member chain, run on the unperturbed
observations, must reproduce the product grid (`build_geochem_elevation_ensemble.py`, parity recorded in every
summary), and NeuroLEM on the production correction must reproduce the archived adjusted erosion field and the
main-run table, so the main model sits inside the experiment it is bracketed by. The run is resumable.

```bash
cd CO2_paper/scripts/nature_recast
./run_ensemble_sensitivity_m3.sh --ages 220 --members 12     # smoke test
./run_ensemble_sensitivity_m3.sh --ages 220                  # pilot: 200 members
./run_ensemble_sensitivity_m3.sh                             # all 109 ages
python3 collate_ensemble_sensitivity.py                      # runs automatically at the end; safe to repeat
```

| Output | Content |
|---|---|
| `../neurolem_CO2_zenodo_data/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv` | one row per age and member: fluxes, sensitivities, share beyond the optimum, on own and on common land, per parameter set |
| `data/nature_recast/ensemble_sensitivity_age_summary.csv` | per age: reference, mean-correction surface, main model, p05 / p50 / mean / p95 across members, member-to-reference ratios |
| `data/nature_recast/ensemble_sensitivity_headline.csv` | across members: quantiles of the time-mean, minimum and maximum of every metric |

Needs the two environments of `../run_geochem_uncertainty_m3.sh` (gplately with pygplates; neurolem-mps with PyTorch),
the pinned checkout of the palaeotopography assimilation, `gospl_salles/` and `surrogate_model/best.pt`.

## Climate states and direct goSPL runs

`climate_sensitivity.py` evaluates 27 climate states per age (nine CO₂ levels across the proxy envelope × three draws of
the climate-model spread, PhanDA pattern-scaling emulator), running NeuroLEM on both surfaces with the perturbed
precipitation, and interpolates Monte Carlo members from that grid. `gospl_validation_sensitivity.py` evaluates the
direct goSPL simulations on their own mesh with exact vertex areas and pairs each with the NeuroLEM run on the same
surfaces, preserved in `data/nature_recast/gospl_surfaces/` (the goSPL runs were made on the adjusted surface of
2026-08 and are not repeated; the reference surface is the same). Both use `_evaluate.py`, the evaluation of the main
model packaged for any grid or mesh (numpy only), and the filled runoff-ratio table written by `fill_runoff_ratio.py`.
`finalise_uncertainty.py` draws the three experiments against the main model and writes the numbers the Methods quote.

`RUNBOOK.md` gives the order of work, machine times, built-in checks and recovery; `./run_uncertainty_m3.sh status`
reports what is complete and prints the next command.

| Output (`data/nature_recast/`) | Content |
|---|---|
| `climate_sensitivity_nodes.csv` | one row per age, climate state and surface |
| `climate_sensitivity_bands.csv`, `climate_sensitivity_headline.csv` | p05 / p50 / p95 per age and of the time-mean |
| `gospl_validation_sensitivity.csv` | goSPL against NeuroLEM on identical inputs, per age, case and metric |
| `runoff_form_tests.csv` | the runoff forks against the main model |
| `octopus_comparison.csv`, `octopus_summary.md`, `../../figures/nature/ed10_octopus` | simulated present-day erosion rate against the ¹⁰Be rate of every OCTOPUS basin; ratios by rate, area, slope and region |
| `distribution_tests.csv` | the main-model results with the erosion-rate distribution widened in log space, flux held |
| `martin_lamb_comparison.csv`, `martin_lamb_summary.md` | the present-day simulation against the Martin & Lamb (2026) process domains: domain shares in depositing cells, erosion rate and flux by domain, their evaluation points, and the present-day sensitivities with river corridors, lakes and flatlands withheld |
| `orogen_belt_cells.csv`, `orogen_tracks.npz`, `orogen_tracks_points.npz` | crust of four belts at their age, its position at every other age, and the main-model fields along it |
| `orogen_tracks_summary.csv`, `orogen_tracks_before_after.csv/.md`, `../../figures/nature/supp1_orogen_tracks` | trajectories of the tracked crust; before uplift against the belt age, with the change in net drawdown split into erosion, climate and emergence |
| `dyntopo_gld428_robust.csv` | per age, the adjusted surface and the two dynamic-topography surfaces (shorelines kept, shorelines free): fluxes, sensitivities, share beyond the optimum, lowland (< 1,500 m) measures |
| `dyntopo_sensitivity_by_age.csv`, `dyntopo_sensitivity_table.csv`, `dyntopo_sensitivity_summary.md` | changes against the main model per age and over 5–250 Ma, with the climate-state, elevation-ensemble and two-surface comparators and the outcome of the decision rule |
| `dyntopo_control_<name>.csv`, `dyntopo_controls_by_age.csv`, `dyntopo_controls_summary.md` | the same for six amplitude-matched control terms (the term of another age on the same crust; the term shifted in longitude) and their comparison with the real term |
| `surface_provenance.csv` | every archived surface against the product grids |
| `uncertainty_statements.md`, `../../figures/nature/ed_uncertainty.png` | numbers and figure for the uncertainty paragraph |

## Paths

Defaults follow the repository layout (`_config.py`). Override with `NEUROLEM_CO2_REPO`, `NEUROLEM_CO2_DATA`,
`NATURE_CACHE`, `NATURE_RUNOFF`, `NATURE_OUT`, `NATURE_FIG`, `NATURE_PROX`; the dynamic-topography test also reads `PALEO_ROOT`
(default: `Paleotopo_data_assimilation` beside `NeuroLEM`) and writes its surfaces to `NATURE_DYNTOPO` (default: `<data archive>/dyntopo_gld428`).

## Figure standard

`_nature_style.py` sets Nature artwork sizes (89 and 183 mm widths, 5–8 pt lettering, no titles) and wraps
`savefig` with the text-collision and minimum-size audit of `../_epsl_style.py`; every figure must report `clean`.

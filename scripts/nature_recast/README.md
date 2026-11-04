# Erosion sensitivity of the continental carbon sink: analysis workflow

Scripts behind Müller et al., *Mountain uplift did not strengthen the weathering CO₂ sink over 540 million years*
(submitted to *Nature*).
They take the archived NeuroLEM erosion fields, evaluate the weathering laws of Bufe et al. (2024)
in every 0.1° land cell at 109 ages, and produce the result tables, figures and number audit.

The adjusted palaeoelevation surface is the product grid set `corrected_scotese_reach500_land` of the
palaeoelevation-assimilation code (cratonic samples excluded, 400 km support, residual kernel tapered to zero
between 250 and 500 km from the nearest sample, samples on PALEOMAP sea dropped), with `_p05` and `_p95` as its
bounds. The data archive carries its NeuroLEM erosion fields, and `check_surface_provenance.py` measures every
archived surface against it.

## What is computed

| Quantity | Definition |
|---|---|
| Erosion rate | eroding part of the NeuroLEM erosion–deposition field × 2.0, the factor that gives 20 Gt/yr of present-day sediment export; depositing cells carry no weathering |
| Silicate and sulfide fluxes | West (2012) law with the three regional parameter sets of Bufe et al. (2024), each kept intact and calibrated separately to a modern long-term silicate CO₂ sink of 72 Mt C/yr and a sulfide CO₂ source of 14.5 Mt C/yr (`run_weathering.TARGET`); the central estimate is the mean of the three predictions |
| Net inorganic CO₂ flux | *F*net = ½ (SCF/*M*cat − SF/*M*SO₄) in charge equivalents (Bufe et al., 2024, SM eq. 11): ½ mol C per equivalent of silicate cations (*M*cat = 30 g per equivalent, an effective value) and 1 mol C per mol of sulfide-derived sulfate (`weathering_core.co2_sil`, `co2_sulf`) |
| Runoff | `rain` channel (precipitation) × runoff ratio (surface + subsurface runoff over precipitation) of the same HadCM3L simulation |
| Runoff dependence | in the rate constant, as in West (2012), for both phases (`--mode kinetic`); on the whole flux as an alternative (`--mode outer`); the sulfide runoff exponent of Bufe et al. (2024), −0.75, as a test (`--beta-sulf -0.75`) |
| Local optimum *E*opt | erosion rate that maximises net inorganic drawdown at the runoff of the cell |
| Erosion sensitivity *S* | d ln *F* / d ln λ for erosion scaled uniformly by λ, from the analytic derivative of the law; buffering capacity is 1 − *S* |
| Rock organic carbon | Zondervan et al. (2023) mass balance scaled to 68 Mt C/yr, oxidised fraction declining with sediment yield (`--oc-form linear` for a constant fraction); biospheric burial after Galy et al. (2015), scaled to 60 Mt C/yr |
| Weathering intensity, riverine δ⁷Li | Caves Rugenstein et al. (2019) |
| Reactivity and feedback strength | fluxes recomputed at 280, 560 and 1,120 ppm with the PhanDA pattern-scaling emulator (`../climate_uncertainty/`) |

The calibration targets set the absolute fluxes only; the optimum, the shares beyond it and every sensitivity
are independent of them.

## Layout

| Path | Content |
|---|---|
| `scripts/nature_recast/` | this folder; `weathering_core.py` holds the weathering laws, `run_weathering.py` the main calculation |
| `data/nature_recast/` | result tables (`<run>_{ens,Taiwan,Sichuan,NewZealand}.csv`, histograms, calibration amplitudes, sensitivity tests, uncertainty experiments), the HadCM3L runoff table `hadcm3l_runoff_scotese_spinupa.npz`, the degassing table `muller2024_gross_degassing.csv`, `number_audit.md` |
| `data/proxies/` | lithium-, strontium- and beryllium-isotope compilations with provenance (`README.md` there) |
| `figures/nature/` | main, Extended Data and Supplementary figures (PNG and PDF) |
| `<data archive>/nature_recast_cache.zip` | land-only cache of all input fields, one archive, written by `build_cache.py` (7.8 GB; not in the repository) |

Runs are named by runoff placement, runoff source and erosion scale: `kin_bridge_s2` is the main model
(runoff in the rate constant, HadCM3L runoff ratio, erosion scale 2.0); `kin_bridge_s1`, `kin_bridge_s4`,
`out_bridge_s2`, `kin_coef_s2`, `out_coef_s2` (uniform runoff ratio of 0.4) and `kin_bridge_s2_oclin` are the
alternatives shown in Extended Data. Column suffixes: `_adj` rock-chemistry-adjusted surface (main model), `_ref`
PALEOMAP reference surface, `_adjc` adjusted surface on the land common to both.

## Running

The main model, its alternatives and the figures, from the archived erosion fields:

```bash
cd scripts/nature_recast
bash run_all.sh
```

Step by step:

```bash
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

The calibration is made at the first age given, so a run includes 0 Ma first.

The sensitivity tests and uncertainty experiments run through `./run_uncertainty_m3.sh <step>`; `RUNBOOK.md`
gives the order of work, machine times, built-in checks and recovery, and `./run_uncertainty_m3.sh status`
reports what is complete and prints the next command.

The figure scripts and `verify_numbers.py` read the tables in `data/nature_recast/` and run without the cache;
`make_fig2_maps.py`, `make_fig4_endmembers.py`, `run_*.py`, `highland_stats.py`, `beyond_vs_co2.py`,
`orogen_scan.py` and `make_videos.py` need it.

`fetch_bridge_runoff.py` rebuilds the runoff table from the BRIDGE server and `extract_degassing.py` rebuilds the
degassing table from the supplementary workbook of Müller et al. (2024); both tables ship with the repository.

## Palaeoelevation ensemble

`run_ensemble_sensitivity_m3.sh` passes the 200 geochemically permitted elevation members of every age through
NeuroLEM and through the same weathering evaluation as the main model (`apply_ensemble_sensitivity.py`), with the
calibration amplitudes and the *E*opt(runoff) table read from the main run, so members and main model differ in
elevation only. Each age adds 202 rows to one member table: the reference surface (member −2), the mean correction
(member −1) and the members. Two checks guard every age: the generator's member chain, run on the unperturbed
observations, must reproduce the production grid (`../build_geochem_elevation_ensemble.py`, parity recorded in
every summary), and NeuroLEM on the production correction must reproduce the archived adjusted erosion field and
the main-run table, so the main model sits inside the experiment that brackets it. The run is resumable.

```bash
cd scripts/nature_recast
./run_ensemble_sensitivity_m3.sh --ages 220 --members 12     # smoke test
./run_ensemble_sensitivity_m3.sh --ages 220                  # pilot: 200 members
./run_ensemble_sensitivity_m3.sh                             # all 109 ages
python3 collate_ensemble_sensitivity.py                      # runs automatically at the end; safe to repeat
```

| Output | Content |
|---|---|
| `<data archive>/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv` | one row per age and member: fluxes, sensitivities, share beyond the optimum, on own and on common land, per parameter set |
| `data/nature_recast/ensemble_sensitivity_age_summary.csv` | per age: reference, mean-correction surface, main model, p05 / p50 / mean / p95 across members, member-to-reference ratios |
| `data/nature_recast/ensemble_sensitivity_headline.csv` | across members: quantiles of the time-mean, minimum and maximum of every metric |

The generator needs a Python with pyGPlates (`PALEOTOPO_PY`) and the palaeoelevation-assimilation code and its
production grids (`PALEO_ROOT`, `PALEO_CENTRAL_DIR`), released separately; NeuroLEM needs a Python with PyTorch
(`NEUROLEM_PY`), the model definition `salles_unet.py` (released separately) and, from the data archive,
`gospl_salles/` and `surrogate_model/best.pt`.

## Climate states, direct goSPL runs and further tests

`climate_sensitivity.py` evaluates 27 climate states per age (nine CO₂ levels across the proxy envelope × three
draws of the climate-model spread, PhanDA pattern-scaling emulator), running NeuroLEM on both surfaces with the
perturbed precipitation, and interpolates 2,000 Monte Carlo members from that grid. `gospl_validation_sensitivity.py`
evaluates the 20 direct goSPL simulations on their own mesh with exact vertex areas and pairs each with the
NeuroLEM run on the same surfaces, kept in `data/nature_recast/gospl_surfaces/`: the goSPL runs were made on a
variant of the adjustment without the distance taper and with the samples on flooded regions retained
(Supplementary Information); the reference surface is the same. `gospl_published_sensitivity.py` evaluates the
published goSPL fields of the reference surface at all 109 ages. These use `_evaluate.py`, the evaluation of the
main model packaged for any grid or mesh (numpy only), and the filled runoff-ratio table written by
`fill_runoff_ratio.py`. `dyntopo_sensitivity.py` adds the change in gld428 dynamic topography since 250 Ma to
the adjusted surface and repeats the erosion and weathering calculation, with six control terms.
`finalise_uncertainty.py` draws the experiments against the main model and writes the numbers the Methods quote.

| Output (`data/nature_recast/`) | Content |
|---|---|
| `climate_sensitivity_nodes.csv` | one row per age, climate state and surface |
| `climate_sensitivity_bands.csv`, `climate_sensitivity_headline.csv` | p05 / p50 / p95 per age and of the time-mean |
| `gospl_validation_sensitivity.csv` | goSPL against NeuroLEM on identical inputs, per age, case and metric |
| `gospl_published_reference_sensitivity.csv` | the published goSPL fields of the reference surface against NeuroLEM, per age |
| `runoff_form_tests.csv` | the runoff forks (sulfide exponent, Sichuan water variable) against the main model |
| `distribution_tests.csv` | the main-model results with the erosion-rate distribution widened in log space, flux held |
| `octopus_comparison.csv`, `octopus_summary.md`, `../../figures/nature/ed10_octopus` | simulated present-day erosion rate against the ¹⁰Be rate of every OCTOPUS basin; ratios by rate, area, slope and region |
| `martin_lamb_comparison.csv`, `martin_lamb_summary.md` | the present-day simulation against the Martin & Lamb (2026) process domains: domain shares in depositing cells, erosion rate and flux by domain, their evaluation points, and the present-day sensitivities with river corridors, lakes and flatlands withheld |
| `orogen_belt_cells.csv`, `orogen_tracks.npz`, `orogen_tracks_points.npz` | crust of four belts at their age, its position at every other age, and the main-model fields along it |
| `orogen_tracks_summary.csv`, `orogen_tracks_before_after.csv/.md`, `../../figures/nature/supp1_orogen_tracks` | trajectories of the tracked crust; before uplift against the belt age, with the change in net drawdown split into erosion, climate and emergence |
| `orogen_catalogue.csv`, `orogen_sensitivity_events.csv` | the catalogue of orogens and their event-scale sensitivities (Fig. 4e,f) |
| `reactivity_fixed_co2.csv`, `reactivity_summary.csv`, `aess_feedback_test.csv` | fluxes at fixed CO₂, reactivity and feedback strength; the feedback at the apparent Earth-system sensitivity |
| `dyntopo_gld428_allcrust.csv` | per age, the adjusted surface and the two dynamic-topography surfaces (shorelines kept, shorelines free): fluxes, sensitivities, share beyond the optimum, lowland (< 1,500 m) measures |
| `dyntopo_sensitivity_by_age.csv`, `dyntopo_sensitivity_table.csv`, `dyntopo_sensitivity_summary.md` | changes against the main model per age and over 5–250 Ma, with the climate-state, elevation-ensemble and two-surface comparators and the outcome of the decision rule |
| `dyntopo_control_<name>.csv`, `dyntopo_controls_by_age.csv`, `dyntopo_controls_summary.md` | the same for six amplitude-matched control terms (the term of another age on the same crust; the term shifted in longitude) and their comparison with the real term |
| `surface_provenance.csv`, `archive_consistency.csv` | every archived surface against the product grids; every archived erosion field against NeuroLEM on the archived inputs |
| `uncertainty_statements.md`, `../../figures/nature/ed_uncertainty.png` | numbers and figure for the uncertainty paragraph |

## Paths

Defaults follow the repository layout (`_config.py`). Override with `NEUROLEM_CO2_REPO`, `NEUROLEM_CO2_DATA`,
`NATURE_CACHE`, `NATURE_RUNOFF`, `NATURE_OUT`, `NATURE_FIG`, `NATURE_PROX`. Steps that read the outputs of the
palaeoelevation-assimilation code (the ensemble generator, the dynamic-topography test, `make_supp_dyntopo_maps.py`,
`palaeoelevation_validation.py`) find them through `PALEO_ROOT` (default: `Paleotopo_data_assimilation` beside the
folder that holds the repository); the PALEOMAP plate model used by `orogen_tracks.py` and `orogen_catalogue.py`
is read from `PALEOMAP_PLATE_DIR`. The dynamic-topography test writes its surfaces to `NATURE_DYNTOPO` (default:
`<data archive>/dyntopo_gld428`). `NET_BUDGET=full` makes the map and mountain scripts add the organic pathways to
the net budget; the paper uses the default inorganic budget.

## Figure standard

`_nature_style.py` sets Nature artwork sizes (89 and 183 mm widths, 5–7 pt lettering, 8 pt panel letters, no
titles) and wraps `savefig` with the text-collision and minimum-size audit of `../_epsl_style.py`; every figure
must report `clean`.

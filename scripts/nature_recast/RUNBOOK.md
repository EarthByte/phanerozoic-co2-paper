# Runbook: what is left to compute for the Nature-format manuscript

## Rebuild on the adopted adjusted surface (2026-09-30)

The adjusted surface of the paper is `Paleotopo_data_assimilation/data/corrected_scotese_reach500_land`
(bounds `_p05`, `_p95`; ensemble summaries `data/geochem_elevation_ensemble_reach500_land`, all built
2026-09-30 on the corrected continent mask). Everything computed on `corrected_scotese_nocraton` is
superseded and is recomputed from the archive up; the direct goSPL simulations are not repeated (see below).
One command does the whole sequence, resumable at every step:

```zsh
cd ~/Documents/Software/NeuroLEM/CO2_paper/scripts/nature_recast
./rebuild_surface.sh --dry-run                       # what will move and run; changes nothing
caffeinate -i ./run_uncertainty_m3.sh resume-surface 2>&1 | tee -a ~/rebuild_surface.log
```

or step by step:

| # | Step | Command | Machine time | Writes |
|---|---|---|---|---|
| 1 | NeuroLEM on the product grids; archive products replaced; earlier tables moved to `_to_delete/` folders; provenance check | `./run_uncertainty_m3.sh rebuild-surface` | about 15 min (GPU) | `neurolem_CO2_zenodo_data/surrogate_erodep/` (109 ages, reference arm reused and checked identical), `dem_scotese_adjusted/`, `geochem_uncertainty_ensemble/elevation_summary/`, `data/nature_recast/gospl_surfaces/`, `surface_provenance.csv` |
| 2 | Input cache, main-model tables, figures, number audit, Extended Data Fig. 2 | `./run_uncertainty_m3.sh main` | about 1 h (CPU) | `nature_recast_cache.zip`, `kin_*`, `out_*`, figures |
| 3 | The two runoff forks against the main model | `./run_uncertainty_m3.sh runoff-tests` | four weathering runs (CPU) | `runoff_form_tests.csv` |
| 3a | The results with the erosion-rate distribution widened (k = 1.3, 1.5, 1.8; flux held) | `./run_uncertainty_m3.sh distribution-tests` | three weathering runs (CPU) | `distribution_tests.csv` |
| 3b | Present-day field against the OCTOPUS cosmogenic basins | `./run_uncertainty_m3.sh octopus` | minutes (CPU; needs geopandas; downloads three zips once) | `octopus_comparison.csv`, `octopus_summary.md`, `figures/nature/ed10_octopus` |
| 3c | Present-day eroding and depositing land against the Martin & Lamb process domains | `./run_uncertainty_m3.sh martin-lamb` | minutes (CPU; the 0.1° aggregate is in `data/martin_lamb/`, rebuilt from the 250-m map only if missing, which needs rasterio) | `martin_lamb_comparison.csv`, `martin_lamb_summary.md` |
| 3d | Four mountain belts followed through the plate model (Fig. 4e, f; Extended Data Fig. 11) | `./run_uncertainty_m3.sh orogen-tracks` | minutes (CPU; the track step needs pygplates, otherwise the committed `orogen_tracks.npz` is used) | `orogen_tracks_*`, `figures/nature/ed11_orogen_tracks`, `fig4_endmembers` |
| 3e | Dynamic topography, 0–250 Ma: the change in gld428 dynamic topography since the present day (retained plates; `dt_term_robust_0-250Ma.nc` from Paleotopo_data_assimilation) added to the adjusted surface, shorelines kept (primary) or free; NeuroLEM and weathering with the main-run calibration; decision rule against the climate-state widths | `./run_uncertainty_m3.sh dyntopo-pilot`, then `dyntopo`, then `dyntopo-controls` (six amplitude-matched control terms: three time shifts, three longitude shifts; their reading) | minutes, about 10 min, about 40 min (GPU) | `dyntopo_gld428_robust.csv`, `dyntopo_sensitivity_by_age.csv`, `dyntopo_sensitivity_table.csv`, `dyntopo_sensitivity_summary.md`, `dyntopo_control_<name>.csv`, `dyntopo_controls_by_age.csv`, `dyntopo_controls_summary.md`; surfaces and erosion fields in `neurolem_CO2_zenodo_data/dyntopo_gld428/` (two zips, about 1.5 GB) |
| 4 | Palaeoelevation ensemble | `./run_uncertainty_m3.sh ensemble` | 5–6 h (GPU) | `ensemble_sensitivity_members.csv`, then the two summaries |
| 5 | Climate states | `./run_uncertainty_m3.sh climate` | about 1.5 h (GPU) | `climate_sensitivity_*.csv` |
| 6 | Published goSPL fields, reference surface | `./run_uncertainty_m3.sh gospl-published` | CPU | `gospl_published_reference_sensitivity.csv` |
| 7 | Direct goSPL runs, paired with the preserved NeuroLEM run | `./run_uncertainty_m3.sh gospl` | minutes; needs the SSD | `gospl_validation_sensitivity.csv` |
| 8 | Reactivity, figure and statements, provenance | `./run_uncertainty_m3.sh reactivity`, `finalise`, `provenance` | CPU | `reactivity_*.csv`, `ed_uncertainty`, `uncertainty_statements.md`, `surface_provenance.csv` |

A pilot of the ensemble before the long run: `./run_ensemble_sensitivity_m3.sh --ages 220`. Two checks
guard every age of it: the generator's unperturbed member chain must reproduce the product grid (it
prints the parity; a value above 0.5 m stops everything), and NeuroLEM on the production correction must
reproduce the archived adjusted erosion field and the main-run table.

The direct goSPL runs stay on the surfaces they were made on (`corrected_scotese_nocraton` central and
two of its ensemble members; the reference surface is unchanged). `gospl_validation_sensitivity.py`
therefore reads its NeuroLEM counterparts from `data/nature_recast/gospl_surfaces/`, the main run and
ensemble table preserved from that surface, and Extended Data Fig. 3 (`figures/figS3_gadi_validation`)
is not regenerated. The comparison tests the surrogate against goSPL on identical inputs, which is what it
is for; the text says which surface it was made on.

After the runs: read `uncertainty_statements.md`, `number_audit.md`, `runoff_form_tests.csv`, `octopus_summary.md`, `distribution_tests.csv`, `martin_lamb_summary.md`, `orogen_tracks_before_after.md` and `dyntopo_sensitivity_summary.md`, and
update the Methods (palaeoelevation paragraph: the correction now has a residual reach of 250–500 km and
drops samples on PALEOMAP sea; uncertainty paragraph; goSPL paragraph), the main text numbers, the
Extended Data legends and `response_to_AB_notes.md`, then rebuild the manuscript (`build_manuscript.py`).


Everything goes through one entry point, from `CO2_paper/scripts/nature_recast`:

```zsh
cd ~/Documents/Software/NeuroLEM/CO2_paper/scripts/nature_recast
./run_uncertainty_m3.sh status
```

`status` lists what is complete and prints the next command. Every step writes ONE table, is resumable, and can be
repeated safely. The two GPU steps (ensemble, climate) run one after the other, never together.

## Order of work

| # | Step | Command | Machine time | Writes |
|---|---|---|---|---|
| 1 | Palaeoelevation ensemble, 200 members × 109 ages | `caffeinate -i ./run_uncertainty_m3.sh ensemble 2>&1 \| tee -a ~/uncertainty_ensemble.log` | about 3 min per age on the M3 (measured on the first ages), 5–6 h in all | `neurolem_CO2_zenodo_data/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv` |
| 2 | Climate pilot, 27 states at 240 Ma | `./run_uncertainty_m3.sh climate-pilot` | minutes | first 54 rows of `data/nature_recast/climate_sensitivity_nodes.csv` |
| 3 | Climate states, 109 ages, then Monte Carlo bands | `caffeinate -i ./run_uncertainty_m3.sh climate 2>&1 \| tee -a ~/uncertainty_climate.log` | 54 NeuroLEM passes per age against 202 for the ensemble: about a quarter of step 1 | `climate_sensitivity_nodes.csv`, `climate_sensitivity_bands.csv`, `climate_sensitivity_headline.csv` |
| 4 | Direct goSPL runs on their own mesh | `./run_uncertainty_m3.sh gospl` | minutes; CPU only; needs the direct goSPL run bundle | `data/nature_recast/gospl_validation_sensitivity.csv` |
| 5 | Input cache (Figs 1, 4, reactivity runs) | `./run_uncertainty_m3.sh cache` | 3 min; CPU only | `neurolem_CO2_zenodo_data/nature_recast_cache.zip` |
| 6 | Figure and statements | `./run_uncertainty_m3.sh finalise` | seconds; any time, uses whatever is complete | `figures/nature/ed_uncertainty.png/.pdf`, `data/nature_recast/uncertainty_statements.md` |

Steps 2–6 in one go once step 1 has finished:

```zsh
caffeinate -i ./run_uncertainty_m3.sh after-ensemble 2>&1 | tee -a ~/uncertainty_after_ensemble.log
```

Steps 4 and 5 use no GPU and can run in a second terminal while step 1 or 3 is going.

## Published goSPL fields, reference surface (CPU only, any time)

```zsh
python3 gospl_published_sensitivity.py
```

evaluates the published goSPL erosion fields of the reference surface at all 109 ages with the evaluation of the main
model and sets them beside the NeuroLEM values: what the surrogate does to the erosion sensitivity and to the share of land
beyond the optimum, age by age, five of the ages withheld from training. Resumable; writes
`data/nature_recast/gospl_published_reference_sensitivity.csv`.

## Surface provenance

`./run_uncertainty_m3.sh provenance` measures, at every age, the `delta` layer of the archived 0.1° surfaces, the
ensemble's stored production-central field and the 1° grids shipped with the archive against
`Paleotopo_data_assimilation/data/corrected_scotese_nocraton` (cratonic samples excluded, 400 km support), the only
valid rock-chemistry adjustment; `data/nature_recast/surface_provenance.csv` is the record. The land-mean is the
discriminant (a match agrees to the metre; the withdrawn 1,000-km set is 10–25 m off in the mean and up to 1.5 km at
single cells). Run it after any change to the archive or to the correction grids.

## Archive consistency

`./run_uncertainty_m3.sh archive-check` runs NeuroLEM on the archived elevation and precipitation of every age and both
surfaces (218 passes) and compares with the archived erosion field; `data/nature_recast/archive_consistency.csv` is the record.
`archive-repair` does the same and replaces a field that fails: the new `erodep_<age>Ma.npz` carries every other array over
unchanged, is read back and compared before it is installed, and the superseded file moves to
`neurolem_CO2_zenodo_data/_to_delete/`. `main` then rebuilds the input cache (the old one moves to `_to_delete/` as well),
all main-model tables, the figures and the number audit. In one go, ending with the climate run and the figure:

```zsh
caffeinate -i ./run_uncertainty_m3.sh resume 2>&1 | tee -a ~/uncertainty_resume.log
```

## What each step checks before it accepts its own result

| Step | Built-in check |
|---|---|
| Ensemble | every regenerated member reproduces the stored production gross erosion and land area of the first experiment (same 200 members, same seed); an age that fails is not written |
| Climate | at the unperturbed state NeuroLEM reproduces the archived erosion field of both surfaces, and the evaluation reproduces the main-run table; an age that fails stops the run |
| goSPL | the bundle's retrieval verification must have passed; NeuroLEM counterparts are read from the main-run table and, for the two ensemble members, from the ensemble table |
| All | calibration amplitudes and the *E*opt(runoff) table are read from the main run (`kin_bridge_s2`), never refitted, so each experiment differs from the main model only in the input it perturbs |

## If something stops

| Symptom | What to do |
|---|---|
| Terminal closed, Mac slept, run interrupted | run the same command again; completed ages are skipped, the interrupted age is redone |
| `Required directory missing: .../corrected_scotese_nocraton` | the ensemble reads the production correction from `Paleotopo_data_assimilation/data/corrected_scotese_nocraton`, the only valid set; if that tree lives elsewhere, `PALEO_CENTRAL_DIR=/path/to/corrected_scotese_nocraton ./run_uncertainty_m3.sh ensemble` |
| `does not reproduce production gross erosion` | the central maps or the seed differ from the first experiment; stop and check `PALEO_CENTRAL_DIR` before going on |
| `unperturbed state does not reproduce the archived ... erosion field` | that archived field is not what the checkpoint gives on the archived inputs: `./run_uncertainty_m3.sh resume` (archive-repair, main, climate, finalise) |
| `raw not found: is the SSD mounted?` | mount the volume or pass `--gadi-root` : `./run_uncertainty_m3.sh gospl --gadi-root /path/to/nature_validation_CO2` |
| Tables or the figure need a different Python | `ANALYSIS_PY=/path/to/python ./run_uncertainty_m3.sh finalise` (GPU steps use `NEUROLEM_PY`) |

## After the runs

1. `./run_uncertainty_m3.sh finalise`, then read `data/nature_recast/uncertainty_statements.md`: it holds every number the
   Methods paragraph on uncertainty quotes, and `figures/nature/ed_uncertainty.png` shows the three experiments against the main model.
2. The Methods sentence on the ensemble and the climate experiment, the legend of the uncertainty figure and the
   number audit are updated from those two files.
3. Commit from the terminal:

```zsh
cd ~/Documents/Software/NeuroLEM/CO2_paper
git add scripts/nature_recast data/nature_recast data/proxies figures/nature README.md
git commit -m "Nature recast: erosion sensitivity analysis, uncertainty experiments, figures, tables"
```

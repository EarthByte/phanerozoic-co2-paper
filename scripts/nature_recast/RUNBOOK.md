# Runbook: reproducing the tables and figures

The tables in `data/nature_recast/` and the figures in `figures/nature/` are computed from the archived
NeuroLEM erosion fields of the data archive (`surrogate_erodep/`, both palaeoelevation surfaces at 109 ages) and
the further inputs named below. All computations go through one entry point, run from `scripts/nature_recast`:

```bash
cd scripts/nature_recast
./run_uncertainty_m3.sh status          # what is complete, and the next command
./run_uncertainty_m3.sh --help          # every step
```

Each step writes one table (or one table set), skips work that is already complete, and can be repeated
safely. An interrupted step is resumed by running the same command again: completed ages are skipped and the
interrupted age is redone. The two long GPU experiments (ensemble, climate) run one after the other, never
together.

## Setting up

| Variable | Purpose |
|---|---|
| `NEUROLEM_CO2_DATA` | the extracted data archive (default: `neurolem_CO2_zenodo_data/` beside the repository) |
| `NEUROLEM_PY` | Python with PyTorch, for the GPU steps (default: `python3` on the PATH) |
| `ANALYSIS_PY` | Python for tables and figures (default `python3`; `NEUROLEM_PY` if `python3` lacks numpy, pandas, scipy or matplotlib) |
| `PYTHONPATH` | must include the directory holding `salles_unet.py`, the NeuroLEM model definition, for every step that runs NeuroLEM |
| `PALEO_ROOT` | the palaeoelevation-assimilation code and its outputs, for the steps that need them (below) |
| `PALEOMAP_PLATE_DIR` | PALEOMAP rotations and plate polygons, for the plate-model tracking of mountain belts |

GPU steps use CUDA or Apple-silicon MPS, with the CPU as a fallback; the times below were measured on an
Apple-silicon GPU. CPU steps need the packages of `requirements.txt`; some steps need more and say so.

The NeuroLEM model definition and the palaeoelevation-assimilation code are released separately. The steps
that need them are marked "NeuroLEM" and "assimilation" in the table.

## Order of work

| # | Step | Machine time | Writes |
|---|---|---|---|
| 1 | `provenance`: every archived surface against the adjusted surface of the paper | CPU | `surface_provenance.csv` |
| 2 | `archive-check`: NeuroLEM on the archived inputs of every age and surface against the archived erosion field (NeuroLEM) | 218 passes, GPU | `archive_consistency.csv` |
| 3 | `main`: input cache, main model and its alternatives (`run_all.sh`), reactivity, mountain statistics, number audit, Figs 1–5, Extended Data Figs 1, 2 and 4–9 | about 1 h, CPU | `<data>/nature_recast_cache.zip` (one archive), `kin_*`, `out_*`, `reactivity_*`, `highland_stats.csv`, `beyond_vs_co2.csv`, `number_audit.md`, figures |
| 4 | `runoff-tests`: the sulfide runoff exponent and the Sichuan water variable against the main model | four weathering runs, CPU | `runoff_form_tests.csv`, `rft_*` |
| 5 | `distribution-tests`: erosion-rate distribution widened (k = 1.3, 1.5, 1.8), erosion flux held | three weathering runs, CPU | `distribution_tests.csv`, `dst_k*` |
| 6 | `octopus`: present-day field against the OCTOPUS cosmogenic basins; downloads the basin shapefiles (95 MB) into `data/octopus/` once; needs geopandas and shapely 2 | minutes, CPU | `octopus_comparison.csv`, `octopus_summary.md`, Extended Data Fig. 10 |
| 7 | `martin-lamb`: depositing land against the Martin & Lamb (2026) process domains; downloads the 250-m domain map (269 MB) and aggregates it to 0.1° once (needs rasterio and pyproj); reads the article's supplementary data S1–S5, obtained from the journal, from `Papers/Martin-Lamb-Earth_is_mostly_diffusive_SciAdv2026_S1-S5.zip` beside the repository | minutes, CPU | `martin_lamb_comparison.csv`, `martin_lamb_summary.md` |
| 8 | `orogen-tracks`: four mountain belts followed through the plate model; the track stage needs pyGPlates, otherwise the shipped `orogen_tracks.npz` is used | minutes, CPU | `orogen_*`, Supplementary Fig. 1 |
| 9 | `orogen-events`: catalogue of thirteen orogens, event-scale sensitivities; the locate and track stages need pyGPlates | minutes, CPU | `orogen_catalogue.csv`, `orogen_*_events*`, Fig. 4 |
| 10 | `reactivity`: fixed-CO₂ fluxes, feedback strength, number audit | minutes, CPU | `reactivity_fixed_co2.csv`, `reactivity_summary.csv`, Extended Data Figs 8, 9 |
| 11 | `aess`: the feedback with the emulator scaled to the apparent Earth-system sensitivity | about 6 min, CPU | `aess_feedback_test.csv`, `aess_feedback_test_summary.md` |
| 12 | `gospl-published`: the published goSPL fields of the reference surface at all 109 ages | CPU | `gospl_published_reference_sensitivity.csv` |
| 13 | `ensemble`: 200 palaeoelevation members × 109 ages (NeuroLEM, assimilation) | about 3 min per age, 5–6 h, GPU | `<data>/geochem_uncertainty_ensemble/nature_recast/ensemble_sensitivity_members.csv`, `ensemble_sensitivity_age_summary.csv`, `ensemble_sensitivity_headline.csv` |
| 14 | `climate-pilot`, then `climate`: 27 climate states × 109 ages, then 2,000 Monte Carlo members (NeuroLEM) | minutes; about 1.5 h, GPU | `climate_sensitivity_nodes.csv`, `climate_sensitivity_bands.csv`, `climate_sensitivity_headline.csv` |
| 15 | `dyntopo-pilot`, `dyntopo`, `dyntopo-controls`: dynamic topography 0–250 Ma and six control terms (NeuroLEM, assimilation outputs) | minutes; about 10 min; about 40 min, GPU | `dyntopo_gld428_allcrust.csv`, `dyntopo_sensitivity_*`, `dyntopo_control_*.csv`, `dyntopo_controls_*`; surfaces in `<data>/dyntopo_gld428/` |
| 16 | `gospl`: the 20 direct goSPL runs on their own mesh, paired with NeuroLEM; pass the run bundle with `--gadi-root DIR` | minutes, CPU | `gospl_validation_sensitivity.csv` |
| 17 | `finalise`: the three uncertainty experiments against the main model | seconds, CPU | `uncertainty_statements.md`, `ed_uncertainty`, Extended Data Fig. 6 |

Tables are written to `data/nature_recast/` unless another folder is given; `<data>` is the data archive.
Steps 4–12 read the calibration of step 3 and can run in a second terminal while a GPU step is going.

Steps that need the palaeoelevation-assimilation outputs read them from `PALEO_ROOT`: the ensemble reads the
production grids of the adjusted surface (`PALEO_CENTRAL_DIR`, default
`<PALEO_ROOT>/data/corrected_scotese_reach500_land`) and runs its generator with the Python of `PALEOTOPO_PY`
(with pyGPlates); the dynamic-topography test reads the term files in
`<PALEO_ROOT>/outputs_scotese_reach500_land/dt_term_all_crust/`.

### One command for everything the weathering form touches

```bash
PARK_ROOT=/path/to/folder PARK_NAME=shipped_tables ./run_uncertainty_m3.sh pure-west
./run_uncertainty_m3.sh pure-west-check
```

`pure-west` first copies every shipped table, the audit and the figures to `PARK_ROOT/PARK_NAME` as a record
(so the recomputed results can be compared with them) and moves the resumable tables there so that their steps
start afresh. It then runs `main`, `runoff-tests`, `distribution-tests`, `martin-lamb`, `orogen-tracks`,
`orogen-events`, `reactivity` and `gospl-published` on the CPU, and `pure-west-gpu`: the ensemble, the climate
states, the dynamic-topography test (no surfaces kept) and its controls, `finalise`, the number audit, all
figures and the Supplementary Videos. Most of a day. After an interruption, `./run_uncertainty_m3.sh
pure-west-gpu` resumes the GPU part without repeating the CPU steps. `pure-west-check` lists every result table
that is missing or older than the start of the run. The erosion fields, the OCTOPUS comparison and the direct
goSPL comparison do not depend on the weathering form; run `octopus` and `gospl` on their own.

### Scripts outside the entry point

| Script | Product | Needs |
|---|---|---|
| `make_videos.py` | Supplementary Videos 1–4 (`figures/nature/videos/`) | input cache, imageio-ffmpeg |
| `make_supp_dyntopo_maps.py` | Supplementary Fig. 2 | term and plate-boundary files of the assimilation outputs |
| `palaeoelevation_validation.py` | `palaeoelevation_validation.csv` (held-out samples, sedimentary basins) | assimilation outputs |
| `../analyse_gadi_validation.py`, then `../make_figS3_gadi_validation.py --derived DIR` | Extended Data Fig. 3 (`figures/figS3_gadi_validation`) | the direct goSPL run bundle |
| (palaeoelevation-assimilation code) | Supplementary Fig. 3 (`figures/nature/supp_dyntopo_elevation_effect`) | released separately |
| `../climate_uncertainty/` | the emulator coefficients (`phanda_sensitivity.npz`, shipped in the archive) and the CO₂ envelope (`data/pco2_envelope.csv`) | see the README there |

`rebuild_surface.sh` (step `rebuild-surface`) regenerates the archived erosion fields with NeuroLEM from the
production grids of the adjusted surface (about 15 min, GPU), replaces the archive's palaeoelevation products,
sets the derived tables aside and checks provenance; `--dry-run` lists what it would move and run. It is needed
only to rebuild the archive itself.

## Built-in checks

| Step | Check |
|---|---|
| `provenance` | the land-mean of every archived adjustment must match the adopted surface (to the metre); the script exits non-zero otherwise |
| `archive-check` | RMSE of the inference against the archived erosion field at most 1e-5 mm/yr; `archive-repair` replaces a failing field after reading the new file back |
| `main` | the input cache is rebuilt when any archived erosion field is newer than it |
| every experiment | calibration amplitudes and the *E*opt(runoff) table are read from the main run (`kin_bridge_s2`), never refitted, so each experiment differs from the main model only in the input it perturbs |
| `reactivity`, `aess`, map and mountain statistics | a main run made with a sulfide runoff exponent other than that of `weathering_core.py` is refused |
| `ensemble` | the generator's unperturbed member chain must reproduce the production grid to 0.5 m (else nothing is generated), and NeuroLEM on the production correction must reproduce the archived adjusted erosion field and the main-run table (else the age is rejected); the rows of an age are written in one block after it finishes |
| `climate` | at the unperturbed state NeuroLEM reproduces the archived erosion field of both surfaces and the evaluation reproduces the main-run table; a failure stops the run |
| `dyntopo` | NeuroLEM on the adjusted surface reproduces the archived field and the re-evaluated surface reproduces the main run |
| `gospl` | the run bundle's verification report must read "passed"; NeuroLEM counterparts come from the main run and ensemble table computed on the goSPL surfaces (`data/nature_recast/gospl_surfaces/`) |
| `aess` | the unscaled column reproduces `reactivity_fixed_co2.csv` to a relative difference of 1e-6 |
| figures | every `savefig` passes the text-collision and minimum-lettering audit of `_nature_style.py` |

The direct goSPL runs were made on a variant of the adjustment (no distance taper, samples on flooded regions
retained; Supplementary Information). Their NeuroLEM counterparts on the same surfaces are kept in
`data/nature_recast/gospl_surfaces/`, so the comparison tests the surrogate against goSPL on identical inputs.

## If something stops

| Symptom | What to do |
|---|---|
| run interrupted | run the same command again |
| `no usable Python: set NEUROLEM_PY` | set `NEUROLEM_PY` to a Python with the packages of `requirements.txt` (see Setting up) |
| `Required directory missing: .../corrected_scotese_reach500_land` | set `PALEO_ROOT`, or `PALEO_CENTRAL_DIR` to the production grids |
| the ensemble's parity check fails, or `The archive and --central-dir are not the same surface` | `PALEO_CENTRAL_DIR` does not hold the production grids of the archived surface |
| `unperturbed state does not reproduce the archived ... erosion field` | the checkpoint or the archive differ from the published ones: `./run_uncertainty_m3.sh archive-check` |
| `raw not found` (step `gospl`) | pass the run bundle: `./run_uncertainty_m3.sh gospl --gadi-root /path/to/bundle` |
| `cannot locate salles_unet.py` | put the NeuroLEM model definition on `PYTHONPATH` |
| `main run made with sulfide exponent ...` | re-run `./run_uncertainty_m3.sh main` |

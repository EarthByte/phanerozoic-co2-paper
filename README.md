# Mountain uplift did not strengthen the weathering CO₂ sink over 540 million years

Analysis and figure code for

> Müller, R.D., Bufe, A., Caves Rugenstein, J.K., Hilton, R.G., Zhou, J., Dutkiewicz, A.
> **Mountain uplift did not strengthen the weathering CO₂ sink over 540 million years.** Submitted to *Nature*.

The study runs NeuroLEM, a neural-network emulator of the global landscape evolution model goSPL, on two
palaeoelevation histories at 109 ages from 540 Ma to the present: the PALEOMAP palaeo-digital elevation models
and a surface adjusted with crustal thickness from chemical mohometry. Field-calibrated weathering laws
(Bufe et al., 2024) are evaluated in every 0.1° land cell under the climate of each age to give silicate
weathering, sulfide oxidation, net inorganic CO₂ drawdown, the local drawdown optimum and the erosion
sensitivity of each flux; rock organic-carbon oxidation and biospheric burial are added as global scalings.
With the data archive, the repository recomputes the result tables, figures and the numbers quoted in the paper.

## Layout

| Path | Content |
|---|---|
| `scripts/nature_recast/` | the workflow: weathering laws, runs, sensitivity tests, uncertainty experiments, figures, number audit (`README.md` and `RUNBOOK.md` there) |
| `scripts/` | shared modules (figure style with automatic lettering and collision checks, map projection, paths, the single-fit weathering module) and the palaeoelevation, ensemble and direct-goSPL steps the workflow calls |
| `scripts/climate_uncertainty/` | the pattern-scaling climate emulator and the CO₂ envelope (`README.md` there) |
| `data/` | weathering-law parameters (Bufe et al., 2024), CO₂ reconstructions and envelope, isotope compilations (`data/proxies/`, provenance there) |
| `data/nature_recast/` | the result tables (main model `kin_bridge_s2_*`, alternatives, sensitivity tests, uncertainty experiments, number audit) |
| `figures/` | main, Extended Data and Supplementary figures (PNG and PDF) |

## Data

The gridded inputs and outputs (NeuroLEM erosion fields and palaeoelevation of both surfaces at 0.1°, the
published goSPL fields, HadCM3L temperature, the climate emulator coefficients, the 200-member elevation
ensemble, the dynamic-topography surfaces, the direct goSPL comparison, the NeuroLEM weights, the larger result
arrays and the supplementary videos) are in the companion Zenodo archive, https://doi.org/10.5281/zenodo.23909273.
Extract it beside this repository as `neurolem_CO2_zenodo_data/`, or set `NEUROLEM_CO2_DATA`:

    working folder/
      phanerozoic-co2-paper/          this repository
      neurolem_CO2_zenodo_data/       the Zenodo archive

## Running

```bash
pip install -r requirements.txt
cd scripts/nature_recast
bash run_all.sh                 # input cache, main model and alternatives, reactivity, number audit, figures
```

`run_all.sh` first writes a land-only cache of the inputs (one archive in the data folder) and then runs the main
model and its alternatives, the reactivity calculation, the mountain statistics, `verify_numbers.py` (every number
the paper quotes, recomputed from the tables into `data/nature_recast/number_audit.md`) and the figure scripts.
The figure scripts and the number audit read the tables in `data/nature_recast/` and run without the cache.

The sensitivity tests and uncertainty experiments run through `scripts/nature_recast/run_uncertainty_m3.sh`;
`RUNBOOK.md` there gives the order of the steps, their machine times, hardware and built-in checks. NeuroLEM
inference (the ensemble, climate-state and dynamic-topography tests) needs PyTorch and a GPU (CUDA or
Apple-silicon MPS); the plate-model tracking of mountain belts needs pyGPlates.

## Dependencies outside this repository

The NeuroLEM model definition used for this study is included (`scripts/paleomap_unet.py`); with the trained weights
of the data archive (`surrogate_model/best.pt`) it re-runs NeuroLEM inference for the elevation ensemble, the
climate states, the dynamic-topography test and the archive check. Training code and the rest of NeuroLEM are
released separately, as is the palaeoelevation-assimilation code (modules
`assimilate_scotese`, `paths_scotese`, `plate_model_utils_scotese` and `sw_io`) is imported only by
`build_geochem_elevation_ensemble.py`; its output grids are read by the ensemble and dynamic-topography steps
(`PALEO_ROOT`, see `RUNBOOK.md`).

## Licence

MIT (code). The data archive is CC BY 4.0. Please cite the paper and the archive.

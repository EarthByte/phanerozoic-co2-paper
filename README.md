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
The repository reproduces every result table, figure and number of the paper.

## Layout

| Path | Content |
|---|---|
| `scripts/nature_recast/` | the workflow: weathering laws, runs, sensitivity tests, figures, number audit (`README.md` and `RUNBOOK.md` there) |
| `scripts/` | shared modules (figure style with automatic lettering and collision checks, map projection, paths) and the palaeoelevation, ensemble and direct-goSPL steps the workflow calls |
| `scripts/climate_uncertainty/` | the pattern-scaling climate emulator and the CO₂ envelope |
| `data/` | weathering-law parameters (Bufe et al., 2024), CO₂ reconstructions, isotope compilations (`data/proxies/`, provenance there) |
| `data/nature_recast/` | the result tables (main model `kin_bridge_s2_*`, alternatives, sensitivity tests, number audit) |
| `figures/` | main, Extended Data and Supplementary figures (PNG and PDF) |

## Data

The gridded inputs and outputs (NeuroLEM erosion fields and palaeoelevation of both surfaces at 0.1°, the
published goSPL fields, HadCM3L temperature, the climate emulator coefficients, the 200-member elevation
ensemble, the dynamic-topography surfaces, the direct goSPL comparison, the NeuroLEM weights, the larger result
arrays and the supplementary videos) are in the companion Zenodo archive (DOI on publication). Extract it beside
this repository as `neurolem_CO2_zenodo_data/`, or set `NEUROLEM_CO2_DATA`:

    working folder/
      phanerozoic-co2-paper/          this repository
      neurolem_CO2_zenodo_data/       the Zenodo archive

## Running

```bash
pip install -r requirements.txt
cd scripts/nature_recast
python3 build_cache.py          # once: a land-only cache of the inputs, one archive, a few minutes
bash run_all.sh                 # the main model, the tests and the figures
python3 verify_numbers.py       # every number the paper quotes, from the tables
```

The figure scripts read the tables in `data/nature_recast/`; the runs and the map figures also need the cache.
`RUNBOOK.md` lists the steps, their machine times and their checks. NeuroLEM inference (the ensemble, climate-state
and dynamic-topography tests) needs PyTorch and a GPU or Apple-silicon MPS; the plate-model tracking of mountain
belts needs pyGPlates.

## Licence

MIT (code). The data archive is CC BY 4.0. Please cite the paper and the archive.

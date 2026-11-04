# `climate_uncertainty`: pattern-scaling emulator of the HadCM3L palaeoclimate forcing

The precipitation and surface-air temperature that force NeuroLEM and the weathering calculation come from the
HadCM3L simulations of Valdes et al. (2021), run along one smoothed proxy CO₂ history. This folder supplies the
climate of the same palaeogeography at other CO₂ levels: a per-cell, per-age pattern-scaling emulator fitted to
the PhanDA HadCM3L model-prior ensemble (Judd et al., 2024), and the proxy CO₂ envelope it is driven with. Its
fitted coefficients (`phanda_sensitivity.npz`, shipped in the data archive under `climate_uncertainty/`) are
read by

* the land-surface reactivity calculation (`nature_recast/run_reactivity.py`): the silicate sink at 280, 560
  and 1,120 ppm and the feedback strength *n* = d ln *F* / d ln *C* (Extended Data Fig. 9);
* the test of the feedback at the apparent Earth-system sensitivity (`nature_recast/aess_feedback_test.py`);
* the climate-state uncertainty ensemble (`nature_recast/climate_sensitivity.py`, which follows the design of
  `mc_driver.py`; Extended Data Fig. 6).

## Training data

The PhanDA model priors ([Zenodo 8237751](https://zenodo.org/records/8237751)) are 872 HadCM3L simulations on the
same 109 PALEOMAP time slices as the forcing:

| Suite | CO₂ | HadCM3L configuration |
|---|---|---|
| `scotese_solara` | 1 × pre-industrial | Original |
| `scotese_2co2a` | 2 × pre-industrial | Original |
| `scotese_4co2a` | 4 × pre-industrial | Original |
| `scotese_spinupa` | smoothed Foster et al. (2017) | Original (the forcing of this study) |
| `scotese_02` | Foster et al. (2017) | Original |
| `scotese_06` | Foster et al. (2017) | Update A |
| `scotese_07` | Foster et al. (2017) + Rae et al. (2021) | Update B |
| `scotese_08` | tuned to the GMST of Scotese et al. (2021) | Update B |

Each simulation is split into five 20-year sub-means (`M1`…`M5`). The emulator uses `tas` (2-m air temperature,
K) and `pr` (precipitation, mm/yr) on the HadCM3L grid of 96 × 73 cells (3.75° × 2.5°). The archive's
`ExperimentInfo.xlsx`, which maps every simulation onto the plate ages 0–540 Ma in 5-Myr steps, is vendored as
`phanda_experiments.csv` (872 rows).

The forcing of this study is the `scotese_spinupa` suite: at 240 Ma the packed temperature grid
(`valdes_temperature/240Ma.npz`) matches it to a root-mean-square difference of 0.011 °C, against 0.49 °C for the
next-closest suite. Its CO₂ at each age is the reference `CO2_ref` of the emulator.

Every PhanDA file carries two longitude axes: the Unified-Model `longitude` (0…356.25° E, for `temp_mm_1_5m` and
`precip_mm_srf`) and `lon` (−180…176.25°, for `tas` and `pr`), a 180° roll apart. `_read_nc_variable` always
uses the axis that belongs to the array it reads, and `selftest` checks the georeferencing.

## The emulator

Per grid cell and per age, fitted on the five Original suites (weighted by the inverse internal variance of each
suite, `--weighting internal`):

```
T(x)      = a_T(x) + lambda_T(x) * log2(CO2 / CO2_ref)        [K per doubling]
ln P'(x)  = a_P(x) + kappa_P(x)  * log2(CO2 / CO2_ref)        [ln units per doubling]
```

with `P' = P + P0`, `P0` = 10 mm/yr, which keeps hyper-arid cells finite. Fitting each age separately carries the
change of climate sensitivity through the Phanerozoic (Farnsworth et al., 2019) in the data. The emulator is
applied as a perturbation of the reference forcing: an additive temperature change and a multiplicative
precipitation factor,

```
R_hi / R_ref = (P'_hi / P'_ref) ** epsilon
```

The `rain` channel of the archived erosion files is precipitation, the same variable as PhanDA's `pr`. Over land,
against the annual climatologies of the same simulations at 0 and 240 Ma (channel land-mean 0.646 and
0.552 m/yr):

| Candidate | Land-mean, 0 / 240 Ma (m/yr) | *r* | Slope |
|---|---|---|---|
| `precip_mm_srf` | 0.649 / 0.532 | 0.997 / 0.982 | 1.01 / 0.97 |
| P − E over land | 0.103 / 0.007 | 0.52 / 0.40 | 0.68 / 0.64 |
| `srfRunoff + subsrfRunoff` | 0.205 / 0.194 | 0.73 / 0.85 | 1.65 / 1.56 |

The fractional perturbation therefore transfers one-to-one and `epsilon` is 1 (`--runoff-elasticity`, kept as a
sensitivity option). The runoff ratio that the weathering calculation applies to precipitation is not changed by
the emulator.

Four uncertainty terms are kept separate:

| Term | Source |
|---|---|
| `sigma_int` | spread of the five 20-year sub-means: internal variability |
| `sigma_fit` | regression residual: where a single log₂ slope describes the suite response poorly |
| `sigma_struct` | spread between model versions (Original, Update A, Update B) after normalising each suite to `CO2_ref` with the fitted slopes |
| CO₂ term | `lambda_T × log2(CO2_hi / CO2_ref)`: the forcing uncertainty, set by the pCO₂ envelope |

`sigma_struct` leaves out any suite whose CO₂ lies more than `--struct-tol` (0.5) doublings outside the fitted
range, so that normalising it does not extrapolate the slope; this drops `scotese_08` at ages where its
GMST-tuned CO₂ is far from the others. The three sigma terms are added in quadrature and applied as
±*z*·σ (*z* = 1.645 for a 5–95% band).

The emulator assumes a log-linear response to CO₂ per cell and per age (`sigma_fit`, `r2_T` and `r2_P` report
where that holds less well); both PhanDA and the forcing use the PALEOMAP reconstructions, so the coefficient
fields and the 0.1° grid share palaeo-coordinates at every age; the coefficients are interpolated bilinearly from
3.75° × 2.5° to 0.1° and add no orographic detail; and the precipitation factor is clipped
(`--runoff-scale-clip`, default 0.2–5.0), with the clipped fraction stored in every output file.

## Running

Needs numpy, and netCDF4 for `fit`. From `scripts/climate_uncertainty`:

```bash
# check the machinery without downloading anything
python phanda_emulator.py selftest

# download the tas and pr archives (16 of the 32 files, about 6.5 GB), with checksums
python phanda_emulator.py fetch --dest ~/data/phanda --dry-run
python phanda_emulator.py fetch --dest ~/data/phanda

# list what arrived and check the age mapping
python phanda_emulator.py index --phanda-dir ~/data/phanda

# fit the emulator, reading straight from the zips; resumable through per-age checkpoints
python phanda_emulator.py fit --phanda-dir ~/data/phanda --out phanda_sensitivity.npz

# the band from the CO2 term alone, then the full band
python phanda_emulator.py envelope --sensitivity phanda_sensitivity.npz \
    --envelope csv --envelope-csv ../../data/pco2_envelope.csv --sigma-terms "" --out-dir env_co2only
python phanda_emulator.py envelope --sensitivity phanda_sensitivity.npz \
    --envelope csv --envelope-csv ../../data/pco2_envelope.csv --out-dir env_full

# add the perturbations to the reference forcing
python phanda_emulator.py apply --envelope-dir env_full \
    --temp-dir   ../../../neurolem_CO2_zenodo_data/valdes_temperature \
    --erodep-dir ../../../neurolem_CO2_zenodo_data/surrogate_erodep \
    --out-dir forcing_bands

# Monte Carlo members of the forcing
python phanda_emulator.py ensemble --sensitivity phanda_sensitivity.npz \
    --envelope csv --envelope-csv ../../data/pco2_envelope.csv --n 200 --seed 0 --out-dir ens
```

`fit` reads 8 suites × 5 sub-means × 2 variables at each of the 109 ages (8,720 small netCDF files);
`--phanda-dir remote` streams them from Zenodo instead. `--envelope` defaults to `factor`; pass `--envelope csv`
for `--envelope-csv` to take effect.

### Outputs

| Command | Output |
|---|---|
| `fit` | `phanda_sensitivity.npz`, arrays (age, 73, 96) on the native grid: `lambda_T`, `kappa_P`, `intercept_T`, `intercept_lnP`, `sigma_{int,fit,struct}_{T,lnP}`, `r2_T`, `r2_P`, `T_ref_model`, `lnP_ref_model`; per age `co2_ref`, `ecs_area_weighted`, `dP_global_pct_per_doubling`, `n_struct_groups`, `n_fit_suites`, `co2_span_doublings`; `age`, `lat`, `lon`, `meta` |
| `envelope` | `climate_envelope_<age>Ma.npz`: int16-packed `dT_{lo,central,hi}` (K, added to the reference temperature) and `lnR_{lo,central,hi}` (exponentiated and multiplied into the reference precipitation), with `co2_{lo,ref,hi}`, the settings and the clipped fraction; native grid (`--emit coarse`, default) or 0.1° grid (`--emit full`); `envelope_summary.csv` |
| `apply` | `forcing_<age>Ma.npz`: packed `temp_{lo,hi}` (°C) and `lnrain_{lo,hi}` (ln m/yr) on the 0.1° grid; the central member is the unmodified reference forcing and is not rewritten; `forcing_summary.csv` |
| `ensemble` | `climate_ensemble_<age>Ma.npz`: (n, 73, 96) stacks of `dT` and `runoff_scale` with the CO₂ draw of each member |

## pCO₂ envelope (`build_pco2_envelope.py`)

`--envelope csv` reads `age,co2_lo,co2_hi`. The envelope used in the paper ships as `data/pco2_envelope.csv`,
with its provenance in `data/pco2_envelope_provenance.csv`. It is rebuilt from the Foster et al. (2017)
Supplementary Data 2 workbook and the PhanDA CO₂ percentiles of Judd et al. (2024):

```bash
python build_pco2_envelope.py \
    --foster-xlsx Foster_etal_carbon_cycles_420Ma_to_present_NatComms2017-s3.xlsx \
    --judd-xlsx   PhanDA_GMSTandCO2_percentiles.xlsx \
    --band 68 --out pco2_envelope.csv --plot pco2_envelope.png
```

The envelope is taken from outside the model chain wherever possible. PhanDA assimilates proxies into the same
HadCM3L priors the emulator is fitted on, so Foster et al. (2017), a proxy compilation, is the primary source and
Judd et al. (2024) fills only the interval it does not reach:

| Ages | Source |
|---|---|
| 0–420 Ma | Foster et al. (2017) LOESS band |
| 420–487 Ma | Judd et al. (2024) PhanDA CO₂ percentiles (`--no-judd` replaces them with the carried width below) |
| > 487 Ma | no proxy compilation or assimilation product; the half-width of the Foster et al. (2017) band over 350–420 Ma (0.95 doublings with the 68% band) carried forward about the model's own CO₂ |

The band is applied in `--mode relative` (default): its half-width in doublings is centred on the reference run's
prescribed CO₂, so the central member is the simulation the results were computed with. `--mode absolute`
re-centres the calculation on the record's own CO₂ history, a different experiment; the model-minus-record offset
is written to the provenance file in both modes. The 68% band is the default because the 95% lower bound of
Foster et al. (2017), a LOESS fit in linear pCO₂, goes negative beyond about 410 Ma; `--band 95` relies on
`--floor-ppm`. The `factor` and `records` envelope modes of `phanda_emulator.py` are for quick tests.

## Monte Carlo through NeuroLEM (`mc_driver.py`)

`mc_driver.py` propagates the forcing uncertainty through NeuroLEM and a weathering step, and defines the design
that `nature_recast/climate_sensitivity.py` uses for the paper's climate-state experiment (with the weathering
evaluation and calibration of the main model and a 9 × 3 node grid). Run on its own it bands the sediment-flux and
silicate-sink histories with a simpler weathering step (median parameter set, one amplitude per reaction).

The perturbation is a two-parameter family: the CO₂ draw `r = log2(pCO2 / pCO2_ref)` and the spatially coherent
draw `z` on the sigma field. Everything downstream is a smooth function of `(r, z)`, so the surrogate is run on a
small node grid per age and any number of members is interpolated from the integrated series. Interpolation error
is set by the node count, and `nodes` prints the curvature of each series in `r` to show whether it is enough.

```bash
# stage A: GPU, resumable
python mc_driver.py nodes --erodep-dir <data>/surrogate_erodep --temp-dir <data>/valdes_temperature \
    --sensitivity <data>/climate_uncertainty/phanda_sensitivity.npz --ckpt <data>/surrogate_model/best.pt \
    --params-json ../../data/bufe_2024_params.json \
    --envelope csv --envelope-csv ../../data/pco2_envelope.csv --out mc_nodes.npz

# stage B: seconds
python mc_driver.py sample --nodes mc_nodes.npz --n 2000 --seed 0 --data-dir ../../data --out mc_bands
```

`nodes` needs PyTorch and the NeuroLEM model definition `scripts/paleomap_unet.py` on the Python
path; `<data>` is the data archive.

**Reference-node check.** The `(r = 0, z = 0)` node is the unperturbed forcing, so the surrogate output there must
reproduce the archived `ero_scotese` and `ero_zhou` fields. `nodes` checks this at every age and stops if the RMSE
exceeds `--reference-tolerance` (1e-6 mm/yr).

**Calibration.** `--calibration per-member` (default) rescales every member so that its present-day sediment flux
equals the 20 Gt/yr target and its present-day silicate and sulfide fluxes equal those of the unperturbed member:
these are observational anchors that hold whatever the climate member. This removes the part of the uncertainty
that rescales both histories together and keeps the part that changes their shape and their difference.
`--calibration fixed` keeps the reference amplitudes; both are written to `mc_band_widths.csv`.

**Time correlation.** The CO₂ draw is independent between ages by default (`--co2-time-correlation independent`,
proxy error), and the `z` draw is shared by all ages of a member (`--sigma-time-correlation full`, the coherent
bias of one model configuration). Both can be changed; the setting is printed.

**Outputs.** `mc_nodes.npz` (stage A); `mc_bands.csv` (percentiles per age of sediment flux, silicate and net
flux for the reference surface, the adjusted surface and the adjusted surface on reference land, and of the
adjusted-minus-reference differences), `mc_band_widths.csv`, `mc_correlations.csv` (per-member Pearson *r* of the
silicate sink against the CO₂ records of Foster et al. (2017) and Judd et al. (2024)) and two quick-look plots.

`mc_driver.py selftest` checks the interpolator against an exactly bilinear function, the curvature diagnostic,
and that per-member calibration removes a pure amplitude perturbation while keeping a shape perturbation.
`nodes --stub-surrogate` runs the chain with an analytic stand-in so the code can be exercised without a GPU;
`sample` refuses stub nodes unless `--allow-stub` is given.

## Library use

```python
import numpy as np
from phanda_emulator import ClimateEmulator

emu = ClimateEmulator.load("phanda_sensitivity.npz")

# deterministic: 1,500 ppm at 240 Ma, CO2 term only
dT, rscale = emu.perturbation(240, 1500, terms=())

# Monte Carlo inside a loop
rng = np.random.default_rng(0)
for _ in range(200):
    dT, rscale = emu.draw(240, rng.uniform(800, 2500), rng)
```

`draw(correlated=True)` (default) uses one spatially uniform standard normal per member, so the sigma field acts as
a coherent bias and does not average away under global integration; `correlated=False` draws independent noise
per cell, which nearly cancels in a global sum and understates the band on integrated quantities.

## References

- Farnsworth, A. et al. (2019) *Geophys. Res. Lett.* **46**, 9880–9889. doi:10.1029/2019GL083574
- Foster, G.L., Royer, D.L., Lunt, D.J. (2017) *Nat. Commun.* **8**, 14845. doi:10.1038/ncomms14845
- Judd, E.J. et al. (2024) *Science* **385**, eadk3705. doi:10.1126/science.adk3705
- Judd, E.J. et al. (2023) PhanDA HadCM3L model priors. Zenodo. doi:10.5281/zenodo.8237751
- Valdes, P.J. et al. (2017) *Geosci. Model Dev.* **10**, 3715–3743. doi:10.5194/gmd-10-3715-2017
- Valdes, P.J., Scotese, C.R., Lunt, D.J. (2021) *Clim. Past* **17**, 1483–1506. doi:10.5194/cp-17-1483-2021

# `climate_uncertainty` — pattern-scaling emulator for the HadCM3 paleoclimate forcing

Puts uncertainty bands on the Valdes et al. (2021) HadCM3 runoff and temperature
fields that force NeuroLEM and the Bufe weathering step, using the PhanDA HadCM3L
model-prior ensemble (Judd et al. 2024) as the sensitivity training set.

## The problem

The Valdes et al. (2021) Phanerozoic archive is not a single run — it is two CO₂
sequences (`scotese_spinupa`, smoothed CO₂; `scotese_02`, Foster et al. 2017 CO₂) —
but it has no perturbed-parameter ensemble and no min/max member. Our forcing uses
one of them, so the paper currently has no climate-forcing uncertainty term at all.

## The data that solves it

The PhanDA model priors ([Zenodo 8237751](https://zenodo.org/records/8237751)) are
872 HadCM3L simulations on the *same* 109 Scotese time slices:

| suite | CO₂ | HadCM3L configuration |
|---|---|---|
| `scotese_solara` | 1 × PI (276 ppm) | Original |
| `scotese_2co2a` | 2 × PI (560 ppm) | Original |
| `scotese_4co2a` | 4 × PI (1120 ppm) | Original |
| `scotese_spinupa` | smoothed Foster et al. (2017) | Original ← **our forcing** |
| `scotese_02` | Foster et al. (2017) | Original |
| `scotese_06` | Foster et al. (2017) | Update A |
| `scotese_07` | Foster et al. (2017) + Rae et al. (2021) | Update B |
| `scotese_08` | tuned to Scotese et al. (2021) GMST | Update B |

Each simulation is split into 5 × 20-yr sub-averages (`M1`…`M5`), giving 4360
priors. Variables: `tas` (2 m air temperature, K), `pr` (precipitation, mm yr⁻¹),
plus `tos`/`so` which we do not use. Grid: HadCM3L 96 × 73 (3.75° × 2.5°).

The archive's `ExperimentInfo.xlsx` carries a **Plate Age** column that maps every
simulation onto the 0–540 Ma / 5 Myr paleogeography ages our basegrid uses — the
mapping is exact, no nearest-age fuzz. It is vendored here as
`phanda_experiments.csv` (872 rows), so nothing here depends on reading the xlsx.

CO₂ spread per age is 2.0–4.1 doublings, ample for a slope.

> **Longitude trap.** Every file carries *two* longitude axes: the Unified-Model
> `longitude` (0…356.25 °E, describing `temp_mm_1_5m` / `precip_mm_srf`) and the
> PhanDA-derived `lon` (−180…176.25, describing `tas` / `pr`). They differ by a
> 180° roll. Reading `tas` against the `longitude` axis puts every continent on
> the wrong side of the globe, and **zonal means are identical either way**, so
> the usual sanity checks will not catch it. `_read_nc_variable` always returns
> the axis belonging to the array it read, and `selftest` includes an asymmetric
> longitude check that fails on a half-globe roll.

**Which suite is our forcing?** `scotese_spinupa`, verified to RMSE 0.011 °C
against `valdes_temperature/240Ma.npz` (see caveat 2 below). Its per-age CO₂ is
the `CO2_ref` the emulator regresses about, e.g. 543 ppm at the 240 Ma slice
(PhanDA experiment 61, stage age 239 Ma — which matches the note in the project
CHANGELOG that nominal 240 Ma uses the published 239 Ma fields).

## What the emulator does

Per grid cell and per age, on the five Original-configuration suites:

```
T(x)      = a_T(x) + lambda_T(x) * log2(CO2 / CO2_ref)        [K per doubling]
ln P'(x)  = a_P(x) + kappa_P(x)  * log2(CO2 / CO2_ref)        [ln per doubling]
```

with `P' = P + P0` (`P0` = 10 mm yr⁻¹ keeps hyper-arid cells finite) and `CO2_ref`
the CO₂ of the suite that produced our forcing. Fitting per age means the drift of
climate sensitivity through the Phanerozoic (Farnsworth et al. 2019) is carried by
the data rather than assumed away.

Four uncertainty terms are kept separate so they can be reported separately:

| term | source | area-weighted size, all 109 ages (median) |
|---|---|---|
| `sigma_int` | spread across the 5 × 20-yr sub-means — internal variability, assumption-free | 0.14–0.30 K (0.17) |
| `sigma_fit` | regression residual — where a single log₂ slope is a poor description | 0.27–1.13 K (0.56) |
| `sigma_struct` | spread between model versions after normalising to a common CO₂ | **1.02–3.41 K (1.34) — 86 % of the variance** |
| CO₂ term | `lambda_T × log2(CO2_hi/CO2_ref)` — the forcing uncertainty proper | set by your pCO₂ record |

`sigma_struct` excludes any suite whose CO₂ sits more than `--struct-tol` doublings
outside the fitted range: normalising those would extrapolate the slope and charge
fit error to the model-error term. In practice this drops `scotese_08` at ages
where its GMST-tuned CO₂ is far from the rest.

The forcing channel is applied as a multiplicative factor rather than an absolute
field:

```
R_hi / R_ref = (P'_hi / P'_ref) ** epsilon
```

**`epsilon` is not a free assumption — it is exactly 1.** The `rain` channel in the
erodep NPZs is *precipitation* — not P − E, and not model runoff. Tested against the
BRIDGE annual climatologies of the same simulation (`texya` at 0 Ma, `texYw` at
240 Ma), land cells only:

| candidate | land-mean (0 / 240 Ma) | r | slope |
|---|---|---|---|
| **`precip_mm_srf`** | **0.649 / 0.532** | **0.997 / 0.982** | **1.01 / 0.97** |
| P − E_land | 0.103 / 0.007 | 0.52 / 0.40 | 0.68 / 0.64 |
| `srfRunoff + subsrfRunoff` | 0.205 / 0.194 | 0.73 / 0.85 | 1.65 / 1.56 |

against a `rain` land-mean of 0.646 / 0.552 m/yr. Precipitation matches at unit
slope; P − E is an order of magnitude too small at 240 Ma. So PhanDA's `pr` is the
*same variable* as our forcing channel, the fractional perturbation transfers
one-to-one, and the 0.4 runoff coefficient applied downstream in the Bufe step is
untouched. `--runoff-elasticity` is retained only as a sensitivity knob for anyone
who wants to ask what an amplified hydrological response would do; leave it at 1.

(This is a statement about the array the surrogate was trained on and the Bufe step
consumes. If a P − E step exists elsewhere in the upstream goSPL workflow, it is not
in this field.)

## Install and run

Needs `numpy` and `netCDF4` (the latter only for `fit`).

```bash
# 0. check the machinery without downloading anything (30 s)
python phanda_emulator.py selftest

# 1. download only the tas + pr archives -- 16 of the 32 files, ~6.5 GB
python phanda_emulator.py fetch --dest ~/data/phanda --dry-run
python phanda_emulator.py fetch --dest ~/data/phanda

# 2. confirm what arrived and that the age mapping is exact
python phanda_emulator.py index --phanda-dir ~/data/phanda

# 3. fit the emulator -- reads straight out of the zips, no extraction needed
python phanda_emulator.py fit --phanda-dir ~/data/phanda --out phanda_sensitivity.npz

# 4a. the band a reader wants first: the CO2 term alone
python phanda_emulator.py envelope --sensitivity phanda_sensitivity.npz \
    --envelope csv --envelope-csv pco2_envelope.csv --sigma-terms "" --out-dir env_co2only

# 4b. the full band, from a published pCO2 envelope
python phanda_emulator.py envelope --sensitivity phanda_sensitivity.npz \
    --envelope csv --envelope-csv pco2_envelope.csv --out-dir env_full

# 5. add the perturbations to the real HadCM3 reference forcing
python phanda_emulator.py apply --envelope-dir env_full \
    --temp-dir   ../../../neurolem_CO2_zenodo_data/valdes_temperature \
    --erodep-dir ../../../neurolem_CO2_zenodo_data/surrogate_erodep \
    --out-dir forcing_bands

# 6. Monte-Carlo members, for propagating end-to-end through the surrogate
python phanda_emulator.py ensemble --sensitivity phanda_sensitivity.npz \
    --envelope csv --envelope-csv pco2_envelope.csv --n 200 --seed 0 --out-dir ens
```

`fit` over all 109 ages reads 8 suites × 5 sub-means × 2 variables per age
(8720 small netCDFs in total) and writes a single ~35 MB file.

### pCO₂ envelopes — `build_pco2_envelope.py`

`--envelope csv` reads `age,co2_lo,co2_hi`; build that file with
`build_pco2_envelope.py` from the sources already in `data/`:

```bash
python build_pco2_envelope.py \
    --foster-xlsx ../../../data/Foster_CO2/Foster_..._NatComms2017-s3.xlsx \
    --judd-xlsx   ../../../data/Judd_CO2/..._github/5_Outputs/PhanDA_GMSTandCO2_percentiles.xlsx \
    --band 68 --out pco2_envelope.csv --plot pco2_envelope.png
```

**The envelope must come from outside the model chain being perturbed.** PhanDA
assimilates proxies into the same HadCM3L priors this emulator is fitted on, and
§S7 already correlates the silicate sink against Judd's pCO₂ — using it for the
forcing envelope too is circular twice over. Foster, Royer & Lunt (2017) is a pure
proxy compilation and is the primary source; Judd fills only what Foster cannot
reach. Three segments:

| ages | source | typical half-width |
|---|---|---|
| 0–420 Ma | Foster (2017) LOESS band, Supplementary Data 2 | 0.10–1.24 doublings, median 0.45 |
| 420–487 Ma | Judd (2024) PhanDA CO₂ percentiles — **flag the circularity** | median 0.49 |
| > 487 Ma | nothing exists; width carried forward from `--tail-window` | 0.95 |

Nothing constrains pCO₂ in the Cambrian and earliest Ordovician — no proxy
compilation, no assimilation product. Beyond 487 Ma the envelope is therefore
centred on the model's *own* prescribed CO₂ (so no bias is introduced in the value)
and given the relative width carried from the 350–420 Ma window. That width, 0.95
doublings, sits squarely inside the range Foster's own band spans over the
Devonian–Carboniferous (0.9–1.24), so it is not a soft assumption. Describe it as
an extrapolation of the *uncertainty*, not of the data.

**`--mode relative` is the default, and it matters.** The reference run was forced
with the Valdes smoothed CO₂ curve, which departs from the proxy central estimate
by more than half a doubling at 45 of 109 ages, and by up to 1.95 doublings (225 Ma,
Triassic — the discrepancy the Valdes paper itself flags). Using the record's
*absolute* band would re-centre the whole calculation on a different CO₂ history:
the central member would no longer be the simulation the paper's result was computed
with, and the emulator would extrapolate its locally fitted slope a long way from
`CO2_ref`. That is a different experiment, not an uncertainty band. Relative mode
keeps the published band's *width* and applies it about the model's own CO₂, so the
central member stays the reference run. The model-minus-record offset is written to
`pco2_envelope_provenance.csv` — it is a real and interesting difference, and it
belongs in the text rather than in the band.

Foster's 95% lower bound goes **negative** beyond ~410 Ma (a LOESS artefact in
linear pCO₂ space), so `--band 68` is the default; `--band 95` works but relies on
`--floor-ppm`. `--no-judd` produces an envelope wholly independent of the PhanDA
priors, at the cost of a wider 420–487 Ma segment.

The other two modes are for quick tests only: `--envelope factor` brackets
`CO2_ref / f … CO2_ref × f`, and `--envelope records` takes the spread between the
vendored Foster and Judd curves, which is a between-record difference rather than
an uncertainty.

## Outputs

`fit` → `phanda_sensitivity.npz`, all fitted ages on the native 96 × 73 grid:
`lambda_T`, `kappa_P`, `intercept_T`, `intercept_lnP`, `sigma_{int,fit,struct}_{T,lnP}`,
`r2_T`, `r2_P`, `co2_ref`, `ecs_area_weighted`, `dP_global_pct_per_doubling`,
`n_struct_groups`, `co2_span_doublings`, `age`, `lat`, `lon`, `meta`.

`envelope` → `climate_envelope_<age>Ma.npz`, int16-packed `dT_{lo,central,hi}` (K,
to be *added* to the reference temperature) and `lnR_{lo,central,hi}` (to be
exponentiated and *multiplied* into the reference runoff), plus `co2_{lo,ref,hi}`,
`sigma_multiplier`, `sigma_terms`, `runoff_elasticity`, `runoff_scale_clip` and
`clipped_fraction`. Default `--emit coarse` writes the native grid (~50 kB/age);
`--emit full` writes the 0.1° basegrid (~20 MB/age).

`apply` → `forcing_<age>Ma.npz`, packed `temp_{lo,hi}` (°C) and `lnrain_{lo,hi}`
(m yr⁻¹) on the basegrid, ready to substitute for the reference forcing. The
central member is by construction the unmodified reference, so it is not rewritten.

`ensemble` → `climate_ensemble_<age>Ma.npz`, `(n, 73, 96)` stacks of `dT` and
`runoff_scale` with the CO₂ draw that produced each.

## Monte-Carlo driver (`mc_driver.py`)

Pushes the forcing ensemble end-to-end through the NeuroLEM surrogate and the Bufe
weathering step, and bands Figures 3 and 7.

**Why it is cheap.** A naive Monte Carlo needs one surrogate pass per member per
age per DEM — 2000 × 109 × 2 = 436,000 passes at 1801 × 3601. Unnecessary: the
perturbation is a strictly *two-parameter* family, the CO₂ draw `r = log2(pCO₂/pCO₂_ref)`
and the spatially coherent noise draw `z`. Everything downstream is a smooth
function of `(r, z)` alone. So the expensive chain is evaluated on a small `(r, z)`
node grid per age and the integrated series are interpolated for as many members as
you like. With 5 × 3 nodes that is 3,270 passes and the member count is free.
Interpolation error is controlled by the node count, not the member count, and
`nodes` prints the curvature of each series in `r` so you can tell whether 5 nodes
is enough.

```bash
# stage A -- GPU, the only expensive part
python mc_driver.py nodes --erodep-dir .../surrogate_erodep --temp-dir .../valdes_temperature \
    --sensitivity phanda_sensitivity.npz --ckpt .../surrogate_model/best.pt \
    --params-json ../../data/bufe_2024_params.json \
    --envelope csv --envelope-csv pco2_envelope.csv --out mc_nodes.npz

# stage B -- laptop, seconds, rerun freely
python mc_driver.py sample --nodes mc_nodes.npz --n 2000 --seed 0 \
    --data-dir ../../data --out mc_bands
```

**Reference-node check.** The `(r=0, z=0)` node is the unperturbed forcing, so the
surrogate output there must reproduce the archived `ero_scotese` / `ero_zhou`
exactly. `nodes` asserts this at every age and **aborts** if the RMSE exceeds
`--reference-tolerance` (1e-6 m/yr). A failure means the checkpoint or the input
fields are not the ones that produced the erodep archive.

**Calibration.** `--calibration per-member` (default) re-derives the modern anchors
— 20 Gt/yr sediment export, ~0.10/0.02 Gt C/yr silicate/sulfide — inside each
member, because those are observations that hold whatever the climate member is.
That removes the part of the uncertainty which merely rescales both histories and
leaves the part that changes their shape and their difference. `--calibration fixed`
keeps the reference amplitudes and shows the raw sensitivity. **Both are always
written to `mc_band_widths.csv`, and the gap between them is the
amplitude-versus-shape statement.** In a stub-surrogate dry run the pattern is:

| | fixed calibration | per-member |
|---|---|---|
| band on the reference history | 48 % | 11 % |
| band on the informed − reference difference | 31 % | 12 % |

i.e. under fixed calibration the difference band is much narrower than the absolute
band — the common-mode cancellation — while per-member calibration has already
removed the amplitude, so little further cancellation remains. (Illustrative
numbers from the plumbing test, not results.)

**Time correlation** is a real choice and the two parameters get different
defaults: `--co2-time-correlation independent` (proxy pCO₂ error is largely
independent between distant stages) and `--sigma-time-correlation full` (a model
configuration's structural bias is coherent across the Phanerozoic). Both can be
flipped; state which you used.

**Outputs.** `mc_bands.csv` (percentiles per age for sediment, silicate and net
flux, all three cases, plus the informed − reference difference),
`mc_band_widths.csv` (both calibrations side by side), `mc_correlations.csv`
(per-member Pearson *r* against Foster and Judd, the Δ*r* distribution and
P(Δ*r* ≤ 0) — the §S7 statistic, now with forcing uncertainty in it),
`fig3_sediment_flux_banded.png` and `fig7_pco2_banded.png`.

`mc_driver.py selftest` checks the interpolator against an exactly bilinear
function, the curvature diagnostic, and — the one that matters — that per-member
calibration removes a pure amplitude perturbation (CV → 5e-17) while leaving a
shape perturbation intact (CV 0.17). `--stub-surrogate` runs the whole chain with a
cheap analytic stand-in so the plumbing can be exercised without a GPU; `sample`
refuses to treat stub nodes as a result unless you pass `--allow-stub`.

## Library use

```python
from phanda_emulator import ClimateEmulator, regrid_bilinear

emu = ClimateEmulator.load("phanda_sensitivity.npz")

# deterministic: what does 1500 ppm do at 240 Ma, CO2 term only?
dT, rscale = emu.perturbation(240, 1500, terms=())

# Monte Carlo inside your own loop -- no need to write 200 x 109 grids to disk
rng = np.random.default_rng(0)
for _ in range(200):
    dT, rscale = emu.draw(240, rng.uniform(800, 2500), rng, runoff_elasticity=2.0)
    ...
```

`draw(correlated=True)` (the default) uses one spatially uniform standard normal
per member, so the sigma field acts as a coherent bias and does **not** average
away under global integration. `correlated=False` draws white noise per cell, which
very nearly cancels in a global sum and will understate the band on integrated
quantities.

## Caveats to carry into the methods section

1. **Log-linear in CO₂**, per cell and per age. `sigma_fit` reports where that form
   fails; the median R² is 0.984 over all 109 ages and never falls below 0.954.
2. ~~HadCM3L stands in for HadCM3.~~ **Not an issue — verified.** Our forcing *is*
   `scotese_spinupa`. Comparing `valdes_temperature/240Ma.npz` against every suite
   at that age gives RMSE 0.011 °C and correlation 1.00000 for `scotese_spinupa`
   (the next closest, `scotese_2co2a`, is 0.49 °C away), so the grids Salles
   supplied are that simulation, and the residual is packing plus regridding
   error. The other seven suites are therefore exact siblings of our own reference
   run — same model, same paleogeography, same spin-up protocol — and the
   sensitivity is measured within the very family the forcing came from.
3. **Same paleogeographic frame.** Both PhanDA and our forcing use the Scotese
   reconstructions, so the coarse λ fields and the basegrid share paleo-coordinates
   at each age. This breaks for a Merdith-frame reconstruction.
4. **λ is interpolated 3.75° × 2.5° → 0.1°.** The perturbation is long-wavelength
   by construction; it adds no orographic detail and does not claim to.
5. **The precipitation scaling is clipped** (`--runoff-scale-clip`, default 0.2–5.0).
   Hyper-arid cells have a very large `sigma_lnP` and would otherwise produce
   meaningless factors. The clipped fraction is stored in every output file and
   printed; report it.
6. **`sigma_struct` is large at high latitudes**, where model versions disagree
   about sea ice. Quote the CO₂-only band and the full band separately rather than
   folding them into one number.
7. **Climate is identical in both members of the paired DEM experiment**, so most
   of this uncertainty is common-mode and largely cancels in Δ. Run the Monte Carlo
   on the difference as well as on each history; the band on Δ is the one that
   speaks to the headline result.

## References

- Judd, E.J. et al. (2024) *Science* **385**, eadk3705. doi:10.1126/science.adk3705
- Judd, E.J. et al. (2023) PhanDA HadCM3L model priors. Zenodo. doi:10.5281/zenodo.8237751
- Valdes, P.J., Scotese, C.R., Lunt, D.J. (2021) *Clim. Past* **17**, 1483–1506. doi:10.5194/cp-17-1483-2021
- Valdes, P.J. et al. (2017) *Geosci. Model Dev.* **10**, 3715–3743. doi:10.5194/gmd-10-3715-2017
- Farnsworth, A. et al. (2019) *Geophys. Res. Lett.* **46**, 9880–9889. doi:10.1029/2019GL083574
- Foster, G.L., Royer, D.L., Lunt, D.J. (2017) *Nat. Commun.* **8**, 14845. doi:10.1038/ncomms14845

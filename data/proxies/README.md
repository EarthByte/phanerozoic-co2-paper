# Weathering proxy datasets (Phanerozoic d7Li, 87Sr/86Sr, 10Be/9Be)

Assembled 2026-09-20 (all URLs accessed on that date). Nothing here was digitised from
figures. All numerical values are copied unchanged from the source files; the only
transformations are Ga -> Ma, column renaming, sorting by age, and the derived flag /
ratio columns explicitly listed below. Conversion scripts are in `scripts/`, untouched
downloads in `raw/`.

IMPORTANT for plotting: the deep-time d7Li files are **carbonate** d7Li, not seawater.
Kalderon-Asael et al. (2021) assume carbonate is 4 +/- 5 permil lighter than seawater
(range 0-10 permil depending on mineralogy/diagenesis); Adiatma et al. (2024, Fig. 6)
convert with a constant calcite offset of +5.5 permil. Only the Cenozoic foraminiferal
record (Misra & Froelich 2012) is a near-direct seawater record (modern forams ~1 permil
lighter than seawater).

## Tidy CSV files

| File | Rows | Age coverage (Ma) | Content |
|---|---|---|---|
| `d7Li_carbonate_KalderonAsael2021_Phanerozoic.csv` | 365 | 1-500 | New carbonate d7Li data, Phanerozoic subset |
| `d7Li_carbonate_KalderonAsael2021_all_0-3000Ma.csv` | 712 | 1-3000 | Same, full 3-Gyr record |
| `d7Li_carbonate_Cao2022_Carboniferous-Triassic.csv` | 59 | 248.8-301.3 | Carbonate (53) + brachiopod (6) d7Li |
| `d7Li_carbonate_Adiatma2024_Ordovician_TableS2.csv` | 29 | 456.9-486.2 | Least-altered Ordovician carbonate d7Li |
| `d7Li_seawater_forams_MisraFroelich2012_Cenozoic.csv` | 301 | 0.097-68.25 | Planktonic foram / bulk foram d7Li (+ Li/Ca, Mg/Ca, Sr/Ca, Mn/Ca, 87Sr/86Sr for 238 rows) |
| `d7Li_seawater_LOESS_Zhang2026_Cenozoic.csv` | 601 | 0-60 (0.1 Myr step) | LOESS fit + 95 % CI to Misra & Froelich data |
| `d7Li_carbonate_compilation_OuyangLi2025_0-3000Ma.csv` | 1222 | 0.1-3000 | Marine-carbonate d7Li compiled by Ouyang et al. (2025) from 11 published sources (900 Phanerozoic values); supersedes the four separate d7Li files above for plotting |
| `d7Li_seawater_smoothed_OuyangLi2025_0-3000Ma.csv` | 301 | 0-3000 (10 Myr step) | Gaussian-smoothed curve of that compilation (kernel s.d. = 20 Myr + 10% of age), SD and SE, and the seawater curve offset so that the modern value is 31 permil (offset 3.79 permil) |
| `Sr87_86_seawater_curve_SCION_geochem_data_2020.csv` | 541 | 0-540 (1 Myr step) | Smoothed seawater 87Sr/86Sr curve (NOT LOWESS 6; see below) |
| `Sr87_86_brachiopod_conodont_compilation_Cao2022_250-305Ma.csv` | 400 | 236.1-303.7 | Compiled brachiopod/conodont 87Sr/86Sr measurements |
| `Be10_Be9_marine_WillenbringVonBlanckenburg2010.csv` | 139 | 0-13.26 | Marine 10Be/9Be, Fe-Mn crusts (76) and sediment cores (63) |

## Provenance

### 1. Kalderon-Asael et al. 2021 (d7Li carbonates, 3 Gyr)
- Paper: Kalderon-Asael B. et al. (2021) Nature 595, 394-398, doi:10.1038/s41586-021-03612-1.
- Files: Supplementary Table 1 (sample descriptions) and Table 2 (geochemistry):
  `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-021-03612-1/MediaObjects/41586_2021_3612_MOESM2_ESM.xlsx`
  and `..._MOESM3_ESM.xlsx` (+ `..._MOESM1_ESM.pdf`, Supplementary Information, 52 MB).
- Licence: none stated on the files; article is (c) Springer Nature, supplementary files freely downloadable. Cite the paper.
- 712 rows = n_new = 712 quoted in the paper's Fig. 1 caption (576 Yale + 136 Oxford/UCL).
- NOTE: the Mendeley Data DOI printed in the paper's Data availability statement
  (10.17632/ztpkpbm43x.1) resolves to an unrelated dataset (Ebert et al., Dead Sea sediment
  magnetics, G-cubed 2020GC009176). The Li data are therefore only available as the Nature
  supplementary tables.
- The previously published data plotted as closed symbols in their Fig. 1a (total n = 1,396) are
  NOT in the supplementary tables and are not included here.
- Columns: `age_Ma` (= Age (Ga) x 1000; ages are unit-level assignments, many samples share an
  age), `d7Li_carb_permil`, `d7Li_2se_internal`, `d7Li_external_precision` (+ `_type`: 1sd for
  Yale, 2sd for Oxford/UCL, as labelled in the source), lab, sample codes, location,
  stratigraphic age, formation, lithology/fabric, mineralogy, Li ppm, d44/40Ca.
- Derived columns (mine, not the authors'): `Al_Ca_ppm_ppm` and `Mn_Sr_ppm_ppm` (computed from
  reported concentrations, or converted from molar ratios for the Oxford/UCL samples);
  `passes_AlCa_detrital_screen` = 1 if Al/Ca <= 0.00054 ppm/ppm (threshold in the Fig. 1b caption);
  `from_C_isotope_excursion_interval` = 1 for the rows listed in the source table under
  "Samples from known Carbon isotope excursions". These two flags give 542 rows (whole record),
  versus n_filtered = 525 in the paper, which additionally removed samples with "indications of
  diagenetic alteration"; the authors' exact filtered list is not tabulated.
- Sampling is very uneven in the Phanerozoic: dense in the Silurian (n = 147), Ordovician (61),
  Carboniferous (53, mostly 346-359 Ma), Devonian (41) and Neogene (32); only 5 Permian,
  14 Triassic (all 205 Ma), 0 Jurassic, 2 Cretaceous, 2 Paleogene samples; nothing between
  85 and 205 Ma or older than 500 Ma within the Phanerozoic.

### 2. Cao et al. 2022 (d7Li and 87Sr/86Sr, late Carboniferous-Early Triassic)
- Paper: Cao C. et al. (2022) Nature Geoscience 15, 832-838, doi:10.1038/s41561-022-01009-x.
- Files: Supplementary Tables S1-S4, `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41561-022-01009-x/MediaObjects/41561_2022_1009_MOESM{2,3,4,5}_ESM.xlsx`.
- Licence: none stated on the files. Ages used: column "Age calculated in this study (GTS 2020)";
  d7Li = column "Final". This dataset is one of the two inputs to the LOESS curve in
  Adiatma et al. (2024) Fig. 6 (the other being Kalderon-Asael et al. 2021).

### 3. Adiatma et al. 2024 (Ordovician d7Li)
- Paper: Adiatma Y.D., Saltzman M.R., Liu X.-M., Wang X.-K., Edwards C.T. (2024) EPSL 647, 119030,
  doi:10.1016/j.epsl.2024.119030.
- File: supplementary PDF `https://ars.els-cdn.com/content/image/1-s2.0-S0012821X2400462X-mmc1.pdf`.
  Table S2 (29 least-altered samples) exists only as an embedded IMAGE of a table. The 29 rows
  of sample ID, locality, d7Li and age were transcribed by hand from that image (rendered at
  200 dpi, clearly legible) - please spot-check against the PDF before publication. The
  elemental-ratio columns were not transcribed. Uncertainty column = long-term 2 sigma
  reproducibility of 1.3 permil quoted in the paper.
- The Phanerozoic LOESS curve and 10-Myr bin means of their Fig. 6 are NOT tabulated anywhere I
  could reach. The authors' code/data repository (https://github.com/datuadiatma/stochastic-lithium,
  MIT licence) could not be listed or cloned: github.com and api.github.com are blocked by this
  session's proxy (HTTP 403); raw.githubusercontent.com is reachable but file names are unknown.
  Their Fig. 6 can be re-created from files 1 + 2 + 3 with a +5.5 permil calcite offset.

### 4. Misra & Froelich 2012 (Cenozoic foraminiferal d7Li) - obtained indirectly
- Paper: Misra S., Froelich P.N. (2012) Science 335, 818-823, doi:10.1126/science.1214697.
- science.org returns HTTP 403 to the proxy, so the original Table S1 could not be downloaded.
  The data were instead taken from `d7Li.seawater.RData`, bundled with the CLiBeSO-W model code in
  the Supplementary Information of Caves Rugenstein J.K., Ibarra D.E., von Blanckenburg F. (2019)
  Nature 571, 99-102, doi:10.1038/s41586-019-1332-y:
  `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-019-1332-y/MediaObjects/41586_2019_1332_MOESM2_ESM.zip`
  (the R script header identifies it as "Seawater d7Li data from Misra and Frohlich 2012--Science").
- 301 rows, 0.097-68.25 Ma, d7Li 20.1-31.0 permil, `d7Li_2sigma` = source column `sigma2`.
  Second-hand copy: not checked line-by-line against the original Science table. Ages are those in
  the bundled file (original publication time scale). `material` = foram species or "Bulk" foram
  fraction. Licence: none stated.
- Also in `raw/`: `CavesRugenstein2019_MOESM1_ESM.xlsx` = CLiBeSO-W model output 0-16 Ma
  (modelled seawater/river d7Li, weathering intensity, erosion) - model results, not proxy data.

### 5. Zhang & Huang 2026 (LOESS of Cenozoic seawater d7Li)
- Paper: Zhang P., Huang K.-J. (2026) Geology, doi:10.1130/G54403.1.
- File: `G54403_SuppData.zip` from GSA figshare, doi:10.1130/GEOL.S.32896877
  (`https://ndownloader.figshare.com/files/66299204`); figshare item licence CC BY-NC 4.0, code package MIT.
- `input/input.xlsx` columns `dLi`, `CI_95_lower`, `CI_95_upper` = LOESS of Misra & Froelich (2012);
  per the paper's Fig. 1 caption the curve is offset by +0.5 permil to match modern seawater (31 permil).
  The same workbook also holds their seawater [Li], F_MOR, F_AOC inputs (not converted).

### 5b. Ouyang et al. 2025 (compiled carbonate d7Li, 3 Gyr) - from the authors' data archive

Ouyang, S., Ji, J., Li, C., Mills, B. J. W., Wei, G.-Y., Hedding, D. W. & Li, G. (2025). Phanerozoic onset of massive continental weathering. Geophys. Res. Lett. 52, e2025GL117072. https://doi.org/10.1029/2025GL117072

Source: `LiIsoNEW.mat` (1222 x 2: age in Ga, d7Li in permil) and `o_fig1_Liiso.m` in the authors' code and data archive (`NeuroLEM/data/Quyang_Li_2025/`), pointed out by a co-author. The compilation draws on Adiatma et al. 2024, Cao et al. 2022, Kalderon-Asael et al. 2021, Lechler et al. 2015, Liu et al. 2023, Misra & Froelich 2012, Pogge von Strandmann et al. 2013, Shan et al. 2024, Wang et al. 2023, 2024 and Yang et al. 2024 (their Figure 1 caption). The archive does not label the source of each value. The smoothed curve reproduces their script exactly: Gaussian kernel of standard deviation 20 Myr + 10% of the age, evaluated every 10 Myr; SE = SD / sqrt(sum of weights / central weight); the seawater curve adds 31 - smoothed(0) = 3.79 permil, as in their script. `scripts/ouyang_li_compilation.py` regenerates both files.

### 6. Seawater 87Sr/86Sr
- McArthur, Howarth, Shields & Zhou (2020) LOWESS 6 (Geologic Time Scale 2020, ch. 7,
  doi:10.1016/B978-0-12-824360-2.00007-3) is NOT openly available: the look-up table is
  distributed by J. McArthur on request and the book chapter is paywalled. No Zenodo/PANGAEA
  deposit was found. COPSE (github.com/sjdaines/COPSE) states that its evaluation datasets
  "are not part of the public release".
- Substitute: `sr_x`, `sr_y` from `data/geochem_data_2020.mat` in the SCION model repository
  (B.J.W. Mills; v1.3, October 2025):
  `https://raw.githubusercontent.com/bjwmills/SCION/main/data/geochem_data_2020.mat`.
  Repository README: "This code is free to use"; no formal licence file found.
  The .mat file carries no metadata on the origin of the curve. SCION: Mills B.J.W., Donnadieu Y.,
  Godderis Y. (2021) Gondwana Research 100, 73-86, doi:10.1016/j.gr.2021.02.011. The predecessor
  paper (Mills et al. 2019, Gondwana Research 67, 172-186, doi:10.1016/j.gr.2018.12.001, Fig. 8c)
  credits its Sr comparison curve to McArthur et al. (2012) [GTS2012, LOWESS 5] and Cox et al. (2016);
  the SCION curve is most probably the same LOWESS 5-based curve resampled at 1 Myr, but I could
  not confirm this. Treat it as a smoothed plotting curve on (probably) the GTS2012 time scale,
  without uncertainty; confirm provenance with B. Mills or request LOWESS 6 from J. McArthur
  before publication.
- `Sr87_86_brachiopod_conodont_compilation_Cao2022_250-305Ma.csv`: raw measurement compilation
  (Cao et al. 2022 Table S2, original sources in column `original_source`), Permian only.
- The Misra & Froelich file includes 87Sr/86Sr for 238 Cenozoic samples.

### 7. 10Be/9Be (Willenbring & von Blanckenburg 2010) - obtained indirectly
- Paper: Willenbring J.K., von Blanckenburg F. (2010) Nature 465, 211-214, doi:10.1038/nature09044.
  Original SI (tables only as PDF): `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnature09044/MediaObjects/41586_2010_BFnature09044_MOESM320_ESM.pdf` (saved in raw/).
- Machine-readable copy from `10-9Be.RData` in the same CLiBeSO-W zip as item 4.
- `age_Ma` = source column `Age.2` = "age from growth/sedimentation rate" in SI Tables A1/A2 (the age
  Caves Rugenstein et al. plot); `age_from_10Be_decay_Ma` = source `Age`; `Be10_Be9_initial` =
  decay-corrected initial ratio. Spot checks against the SI PDF agree, except one obvious typo in
  the bundled file: a Pacific sediment row has age_from_10Be_decay = -43.00 where the SI has -0.43.
  Use `age_Ma`, not the decay age.

## Not obtained
- McArthur et al. 2020 LOWESS 6 table (not open; see 6).
- Misra & Froelich 2012 original SOM (science.org 403); FSU repository copy of Misra's thesis (403).
- Adiatma et al. 2024 GitHub repository contents and their Fig. 6 LOESS/binned curve (github.com blocked).
- Literature d7Li compilation plotted in Kalderon-Asael et al. 2021 Fig. 1a (not tabulated by the authors).
- Cenozoic seawater 187Os/188Os: no single open compilation found (only site-level PANGAEA datasets,
  e.g. doi:10.1594/PANGAEA.713249, 10.1594/PANGAEA.785723); nothing downloaded.

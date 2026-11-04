#!/usr/bin/env python3
"""
bufe_co2_from_dem.py -- weathering law of Bufe et al. (2024) evaluated on the NeuroLEM
erosion fields with a single (median) parameter set.

Role in the published scripts
-----------------------------
The main analysis of the paper (scripts/nature_recast/, weathering_core.py and
run_weathering.py) keeps the three regional parameter sets of Bufe et al. (2024)
separate, places runoff in the rate constant and calibrates each set to the modern
fluxes.  This module holds the simpler single-fit form that the scripts one level
up import:

  analyse_gadi_validation.py    weathering_flux, median_fit, BETA, M_CAT, M_SO4,
                                REACTIONS_USED, DEFAULT_REGIONS: fluxes of the direct
                                goSPL runs on their own mesh (Extended Data Fig. 3)
  apply_geochem_ensemble.py     weathering_flux, median_fit, BETA, T_REF_K,
                                REACTIONS_USED, DEFAULT_REGIONS, _load_temp_K
  make_main_map_panels.py       _load_temp_K (HadCM3L temperature for the maps)

climate_uncertainty/mc_driver.py carries its own copy of these constants and of
weathering_flux.

Weathering law
--------------
Per cell and per reaction (silicate-cation release, sulfide-derived sulfate release),
the law of West (2012) as fitted by Bufe et al. (2024, eq. 6), with the runoff factor
applied to the whole flux:

    W = chi * E * (1 - exp(-a)) * (q / q_ref)**(beta + 1),
    a = K/(sigma+1) * f(T) * (z / E)**(sigma + 1),
    f(T) = exp[(Ea / R) (1/T_ref - 1/T)]

with E the denudation flux (t km-2 yr-1), q runoff and q_ref the reference runoff of
the region (mm/yr), T the surface-air temperature (K) and W in t km-2 yr-1, which is
numerically g m-2 yr-1.  beta is 0 for silicates and -0.75 for sulfides, the runoff
normalisation of Bufe et al. (2024) as published (BETA).

CO2 conversion (Bufe et al., 2024, SM eq. 11)
---------------------------------------------
On timescales of 10^4-10^7 yr the net inorganic drawdown is half the difference of
the silicate-cation and sulfate fluxes in charge equivalents:

    F_net = 1/2 (SCF / M_CAT) - SF / M_SO4        [mol C m-2 yr-1]

  * silicate: 1/2 mol C per equivalent of cations; M_CAT = 30 g per equivalent is an
    effective value for the silicate cations, so SCF / M_CAT is in eq m-2 yr-1;
  * sulfide: sulfate carries 2 equivalents per mol, so 1/2 mol C per equivalent is
    1 mol C per mol of sulfate; M_SO4 = 96.06 g mol-1.

Positive F_net is a sink of atmospheric CO2.  Carbonate weathering by carbonic acid is
neutral on these timescales and is not evaluated.  Area integrals are converted to
Gt C/yr with 12.011 g C per mol and 1e-15 Gt per g.

Command-line use
----------------
Run on its own, the module evaluates both palaeoelevation surfaces of every archived
age with the median parameter set and writes

  <out>/co2_series.csv            one row per age: net_*, scf_* and sf_* (net inorganic
                                  drawdown, silicate sink, sulfide source; Gt C/yr) for
                                  the reference surface (_reference), the adjusted
                                  surface on its own land (_informed) and on the land of
                                  the reference surface (_informed_common_reference_cells)
  <out>/fco2_maps_<age>Ma.npz     per-cell F_net (mol C m-2 yr-1) of the three cases at
                                  --map-age (fco2_scotese, fco2_zhou,
                                  fco2_informed_common_reference_cells), lat, lon, the
                                  union land mask and the age

Inputs: the archived NeuroLEM fields (surrogate_erodep/erodep_<age>Ma.npz: ero_scotese,
ero_zhou in mm/yr, negative = erosion; mask_scotese, mask_zhou; rain, precipitation in
m/yr; lat, lon), the packed HadCM3L temperature (valdes_temperature/<age>Ma.npz,
written by compress_valdes_temp.py) and data/bufe_2024_params.json.  Erosion is the
eroding part of the field times --erate-scale (the erosion calibration factor, 2);
runoff is precipitation times a uniform --runoff-coefficient (0.4); --f-sil and
--f-sulf scale the silicate and sulfide fluxes.  These are the settings of this
single-fit evaluation; the calibration of the paper's main model is that of
nature_recast/run_weathering.py.

USAGE (from the repository root, with the data archive beside it)
  python3 scripts/bufe_co2_from_dem.py \
      --erodep-dir  ../neurolem_CO2_zenodo_data/surrogate_erodep \
      --temp-dir    ../neurolem_CO2_zenodo_data/valdes_temperature \
      --params-json data/bufe_2024_params.json \
      --out         bufe_co2 --map-age 240
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
import numpy as np

# R_GAS: gas constant, J mol-1 K-1; T_REF_K: reference temperature of the Arrhenius factor, K;
# RHO: rock density, kg m-3 (Bufe et al., 2024), numerically the factor from an erosion rate in mm/yr
# to a denudation flux in t km-2 yr-1 (1 mm/yr x 2,500 kg m-3 = 2,500 t km-2 yr-1)
R_GAS = 8.3145; T_REF_K = 293.15; RHO = 2500.0
# M_SO4: molar mass of sulfate, g mol-1 (2 charge equivalents per mol, so 1 mol C per mol in F_net);
# M_CAT: effective mass of the silicate cations per charge equivalent, g eq-1 (1/2 mol C per equivalent)
M_SO4 = 96.06; M_CAT = 30.0
REACTIONS_USED = ("silicate", "sulfide")   # carbonate weathering is neutral on 10^4-10^7 yr (SM eq. 11) and is not evaluated
# runoff exponents of Bufe et al. (2024): W scales with (q/q_ref)**(beta + 1) on the whole flux
BETA = {"silicate": 0.0, "sulfide": -0.75}
# the three regional calibrations: Taiwan, the eastern Tibetan Plateau (Sichuan) and New Zealand
DEFAULT_REGIONS = ["Taiwan", "Sichuan", "NewZealand"]


def median_fit(raw, rxn, regions):
    """Median parameter set of one reaction across regions.

    raw      the parameter file as loaded from JSON, {reaction: {region: {key: value}}}
    rxn      'silicate' or 'sulfide'
    regions  region names; those absent from the file are skipped

    Returns {key: median value} for chi_m (-), K_over_sigp1, sigma_plus_1 (-), z_t_per_km2
    (t km-2), Ea_J_per_mol (J mol-1) and q_ref_mm_yr (mm/yr).  The median is taken parameter by
    parameter, so the result is not one of the fitted regional sets.
    """
    ks = ["chi_m", "K_over_sigp1", "sigma_plus_1", "z_t_per_km2", "Ea_J_per_mol", "q_ref_mm_yr"]
    fs = [raw[rxn][r] for r in regions if r in raw[rxn]]
    return {k: float(np.median([f[k] for f in fs])) for k in ks}


def weathering_flux(E_mm, q_mm, T_K, f, beta):
    """Weathering flux of one reaction, Bufe et al. (2024, eq. 6) with the runoff factor on the whole flux.

    E_mm  erosion rate, mm/yr (converted to a denudation flux with RHO)
    q_mm  runoff, mm/yr
    T_K   surface-air temperature, K
    f     one parameter set (an entry of median_fit)
    beta  runoff exponent (BETA): W is multiplied by (q / q_ref)**(beta + 1)

    Returns W in t km-2 yr-1 (= g m-2 yr-1) of silicate cations or of sulfate, before any amplitude
    factor.  Arrays broadcast against each other.
    """
    # floors keep (z/E)**s and q/q_ref finite where there is no erosion or no runoff; such cells carry W ~ 0
    E = np.maximum(E_mm, 1e-6) * RHO           # mm/yr -> t km-2 yr-1
    q = np.maximum(q_mm, 1e-3)
    fT = 1.0 if f["Ea_J_per_mol"] == 0 else np.exp(f["Ea_J_per_mol"] / R_GAS * (1.0 / T_REF_K - 1.0 / T_K))
    # a of the law; clipped at 700 so that exp(-a) stays representable (fully supply-limited beyond)
    arg = np.clip(f["K_over_sigp1"] * fT * np.power(f["z_t_per_km2"] / E, f["sigma_plus_1"]), 0.0, 700.0)
    W = f["chi_m"] * E * (1.0 - np.exp(-arg))
    return W * np.power(q / f["q_ref_mm_yr"], beta + 1.0)


def _area_m2(lat, shape):
    """Area (m2) of each cell of a 0.1-degree latitude-longitude grid, broadcast to `shape` (lat, lon)."""
    Rm = 6.371e6; dg = np.deg2rad(0.1)
    return np.broadcast_to((Rm * Rm * dg * dg) * np.cos(np.deg2rad(lat))[:, None], shape)


def _load_temp_K(temp_dir: Path, age: int):
    """HadCM3L surface-air temperature of one age in K, from the packed `<age>Ma.npz` of compress_valdes_temp.py
    (temp_q * scale + offset, in deg C)."""
    z = np.load(temp_dir / f"{age}Ma.npz", allow_pickle=True)
    return z["temp_q"].astype(np.float64) * float(z["scale"]) + float(z["offset"]) + 273.15


def _one_case(ero, mask, q_mm, T_K, area, fits, f_sil=1.0, f_sulf=1.0):
    """Fluxes of one surface over the land given by `mask`.

    ero     signed erosion-deposition rate already multiplied by the erosion scale, mm/yr (negative = erosion);
            depositing cells carry no weathering
    q_mm    runoff, mm/yr; T_K temperature, K; area cell area, m2; fits {'silicate': set, 'sulfide': set}
    f_sil, f_sulf   amplitude factors of the silicate and sulfide fluxes (1 = the fitted amplitudes)

    Returns (net, scf, sf, Fnet): the net inorganic drawdown, the silicate sink and the sulfide source, each in
    Gt C/yr, and the per-cell net flux Fnet in mol C m-2 yr-1 (zero off the mask).
    """
    E_mm = np.where(mask, np.maximum(-ero.astype(np.float64), 0.0), 0.0)
    qc = np.where(mask, q_mm, 0.0); Tc = np.where(mask, T_K, T_REF_K)
    # f_sil, f_sulf: one global amplitude factor per reaction (f = 1 keeps the fitted amplitudes)
    SCF = f_sil * np.where(mask, weathering_flux(E_mm, qc, Tc, fits["silicate"], BETA["silicate"]), 0.0)
    SF = f_sulf * np.where(mask, weathering_flux(E_mm, qc, Tc, fits["sulfide"], BETA["sulfide"]), 0.0)
    # g m-2 yr-1 divided by g per equivalent (cations) or g per mol (sulfate) gives eq or mol m-2 yr-1;
    # 1/2 mol C per cation equivalent, 1 mol C per mol sulfate (SM eq. 11)
    Fnet = np.where(mask, 0.5 * (SCF / M_CAT) - (SF / M_SO4), 0.0)   # mol C m-2 yr-1
    # area integrals: mol C m-2 yr-1 x m2 -> mol C yr-1; x 12.011 g C mol-1 x 1e-15 -> Gt C yr-1
    net = float(np.nansum(Fnet * area)) * 12.011 * 1e-15
    scf = float(np.nansum(SCF * area)) / M_CAT * 0.5 * 12.011 * 1e-15
    sf = float(np.nansum(SF * area)) / M_SO4 * 12.011 * 1e-15
    return net, scf, sf, Fnet


def main():
    """Command-line evaluation of every archived age; writes co2_series.csv and one map file (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--erodep-dir", required=True)
    ap.add_argument("--temp-dir", required=True, help="dir of packed <age>Ma.npz temperature grids")
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    ap.add_argument("--runoff-coefficient", type=float, default=0.4)
    ap.add_argument("--erate-scale", type=float, default=2.0,
                    help="scale surrogate erate onto Bufe catchment denudation (mm/yr). "
                         "Default 2 is CALIBRATED so present-day global denudation matches "
                         "the observed ~20 GT/yr sediment flux (Salles 2023) and ~0.05-0.1 "
                         "mm/yr cosmogenic rate. A literal m/yr->mm/yr cast (1000) overshoots "
                         "this by ~600x and is unphysical.")
    ap.add_argument("--f-sil", type=float, default=2.4,
                    help="silicate amplitude factor. Default 2.4 CALIBRATES the present-day "
                         "global silicate CO2 sink to the observed ~0.10 GtC/yr "
                         "(Gaillardet et al. 1999). f=1 = raw Bufe mountain-stream amplitude.")
    ap.add_argument("--f-sulf", type=float, default=0.1,
                    help="sulfide amplitude factor. Default 0.1 CALIBRATES the present-day "
                         "pyrite-oxidation CO2 source to ~0.02 GtC/yr (Torres et al. 2014). "
                         "With both defaults the present-day net is a ~0.08 GtC/yr SINK "
                         "(within modern degassing; Muller et al. 2024) and the informed "
                         "reconstruction ENHANCES the net drawdown. f_sil=f_sulf=1 recovers the "
                         "uncalibrated Bufe amplitudes (net a small source).")
    ap.add_argument("--map-age", type=int, default=240)
    ap.add_argument("--ages", nargs="*", type=int)
    a = ap.parse_args()

    ed = Path(a.erodep_dir); td = Path(a.temp_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = json.load(open(a.params_json))
    # one median parameter set per reaction
    fits = {rxn: median_fit(raw, rxn, a.regions) for rxn in REACTIONS_USED}
    print(f"regions={a.regions}  runoff_coeff={a.runoff_coefficient}  erate_scale={a.erate_scale}  "
          f"f_sil={a.f_sil}  f_sulf={a.f_sulf}")

    # archived erosion fields by age (Ma)
    files = {int(re.search(r"erodep_(\d+)Ma", f.name).group(1)): f for f in ed.glob("erodep_*Ma.npz")}
    ages = sorted(a.ages) if a.ages else sorted(files)
    rows = []
    for age in ages:
        tp = td / f"{age}Ma.npz"
        if not tp.exists():
            print(f"  [{age}Ma] no temp grid — skipped"); continue
        d = np.load(files[age], allow_pickle=True)
        lat = d["lat"].astype(np.float64); area = _area_m2(lat, d["ero_scotese"].shape)
        T_K = _load_temp_K(td, age)
        # runoff = precipitation (m/yr -> mm/yr) x a uniform runoff coefficient
        q_mm = d["rain"].astype(np.float64) * 1000.0 * a.runoff_coefficient
        # erosion calibration factor applied to the signed fields before the eroding part is taken
        es = d["ero_scotese"].astype(np.float64) * a.erate_scale
        ez = d["ero_zhou"].astype(np.float64) * a.erate_scale
        # three cases: reference surface on its own land, adjusted surface on its own land, adjusted surface on
        # the land of the reference surface
        net_s, scf_s, sf_s, F_s = _one_case(es, d["mask_scotese"], q_mm, T_K, area, fits, a.f_sil, a.f_sulf)
        net_z, scf_z, sf_z, F_z = _one_case(ez, d["mask_zhou"], q_mm, T_K, area, fits, a.f_sil, a.f_sulf)
        net_zc, scf_zc, sf_zc, F_zc = _one_case(
            ez, d["mask_scotese"], q_mm, T_K, area, fits, a.f_sil, a.f_sulf)
        rows.append((age, net_s, scf_s, sf_s, net_z, scf_z, sf_z,
                     net_zc, scf_zc, sf_zc))
        if age == a.map_age:
            np.savez_compressed(out / f"fco2_maps_{age}Ma.npz",
                                fco2_scotese=F_s.astype(np.float32), fco2_zhou=F_z.astype(np.float32),
                                fco2_informed_common_reference_cells=F_zc.astype(np.float32),
                                lat=d["lat"], lon=d["lon"],
                                mask=(d["mask_scotese"] | d["mask_zhou"]), age=age)
        print(f"  [{age:>4}Ma] net Scotese={net_s:+.3f} informed={net_z:+.3f} GtC/yr  "
              f"(informed {100*(net_z/net_s-1):+.1f}%)")

    with open(out / "co2_series.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["age", "net_reference", "scf_reference", "sf_reference",
                    "net_informed", "scf_informed", "sf_informed",
                    "net_informed_common_reference_cells",
                    "scf_informed_common_reference_cells",
                    "sf_informed_common_reference_cells"])
        w.writerows([[r[0]] + [f"{v:.6g}" for v in r[1:]] for r in rows])
    print(f"\nwrote {len(rows)} ages -> {out/'co2_series.csv'}  (net<0 = CO2 source). "
          f"map npz for {a.map_age} Ma.")


if __name__ == "__main__":
    main()

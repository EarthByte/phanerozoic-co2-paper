#!/usr/bin/env python3
"""Per-cell weathering fluxes, their erosion sensitivity and their climate sensitivity.

This module holds the weathering laws used by every script of the analysis (Methods, "Weathering
laws", "Rock organic carbon" and "Erosion sensitivity, local optimum and kinetic share"). It is a
library: it reads only the parameter file passed to `load_fits` and writes nothing.

Weathering law (West, 2012, as fitted by Bufe et al., 2024, eq. 6), per reaction (silicate-cation
release or sulfide-derived sulfate release) and per regional parameter set:

    W = A * chi * E * (1 - exp(-a)),      a = c * f(T) * f(q) * (z / E)**s

  E      denudation flux, t km-2 yr-1 (erosion rate in mm/yr times RHO; see `flux_and_sens`)
  chi    mass fraction of the weatherable phase in the eroded rock (-)
  z      mass of the weathering zone per unit area, t km-2
  s      sigma + 1, where sigma describes the decline of the reaction rate with time spent in the
         weathering zone
  c      K / (sigma + 1), the fitted rate constant (units as fitted by Bufe et al. (2024) with E in
         t km-2 yr-1)
  f(T)   Arrhenius factor, exp[(Ea / R) (1 / T_ref - 1 / T)], T in K, T_ref = 293.15 K
  f(q)   runoff factor, (q / q_ref)**(beta + 1), q and q_ref in mm/yr; beta = 0 for both phases in
         the main model, so f(q) = q / q_ref
  A      amplitude that calibrates the set to the modern global flux; it is not applied here but
         by the caller (run_weathering.py), because the optimum, the shares and every sensitivity
         are independent of it
  W      weathering flux, t km-2 yr-1 of silicate cations or of sulfate (numerically g m-2 yr-1)

For a >> 1 weathering is supply-limited, W -> chi * E; for a << 1 it is kinetically limited and
W ~ chi * c f(T) f(q) z**s E**(-sigma).

Two placements of the runoff term are implemented:

  'kinetic'  f(q) sits inside the exponential, with the rate constant, as in West (2012). Supply-
             limited weathering is then independent of runoff and kinetically limited weathering
             scales with f(q); the fitted curves of Bufe et al. (2024) are reproduced exactly at
             q = q_ref. This is the main model.
  'outer'    f(q) multiplies the whole flux, which makes supply-limited weathering scale with
             runoff too. This is the alternative of Extended Data Fig. 4.

Local erosion sensitivity:      eps = dlnW/dlnE = 1 - s * gamma
Local kinetic (climate) share:  gamma = dlnW/dln a = a e^-a / (1 - e^-a)
gamma = 0 for supply-limited weathering (eps = 1) and gamma = 1 for kinetically limited
weathering (eps = -sigma). In the 'kinetic' placement gamma is also the elasticity of W to runoff.

Fluxes are converted to CO2 with the net-flux formulation of Bufe et al. (2024),
F_net = 1/2 ([Cat_sil]eq - [SO4]eq) in charge equivalents (Bufe et al., 2024, SM eq. 11), i.e.
F_net = 1/2 SCF / M_cat - SF / M_SO4: 1/2 mol C per equivalent of silicate cations (M_cat = 30 g, an effective
mass per equivalent) and 1 mol C per mol of sulfide-derived sulfate (2 equivalents) (`co2_sil`, `co2_sulf`). Rock organic-carbon oxidation
and the burial of biospheric organic carbon follow `oc_petro` and `oc_bio_burial`.
"""
from __future__ import annotations
import json, re
import numpy as np

R_GAS = 8.3145          # J mol-1 K-1
T_REF_K = 293.15        # reference temperature of the Arrhenius factor, K
# Rock density, kg m-3 (Bufe et al., 2024). Numerically it is also the factor that converts an erosion
# rate in mm/yr into a denudation flux in t km-2 yr-1: 1 mm/yr x 2,500 kg m-3 = 2.5 kg m-2 yr-1 = 2,500 t km-2 yr-1.
RHO = 2500.0            # kg m-3, Bufe et al. (2024)
M_SO4 = 96.06           # g mol-1, molar mass of sulfate
M_CAT = 30.0            # g mol-1, effective molar mass of the silicate cations (Bufe et al., 2024)
# Runoff enters the rate constant as (q/q_ref)^(BETA + 1), the placement of West (2012), for both phases. The sulfide
# value -0.75 of Bufe et al. (2024) normalised the measured concentrations to q_ref across catchments that straddle both
# regimes; the West form produces that regime dependence itself, so it is not applied again in the rate constant.
# The published -0.75 is carried as a sensitivity test (run_weathering.py --beta-sulf -0.75; runoff_form_tests.py).
# Callers may overwrite BETA['sulfide'] at run time (run_weathering.py --beta-sulf, _evaluate.load_calibration).
BETA = {"silicate": 0.0, "sulfide": 0.0}


def check_form(amplitudes_txt):
    """Refuse a main run whose sulfide runoff exponent differs from BETA: fields and calibration would mix two forms.

    `amplitudes_txt` is the `<run>_amplitudes.txt` file written by run_weathering.py; its third line records the
    settings of the run, including `beta_sulf=<value>`. Exits with a message if the value differs from BETA['sulfide'].
    """
    lines = open(amplitudes_txt).read().split('\n')
    m = re.search(r'beta_sulf=(\S+)', lines[2]) if len(lines) > 2 else None
    beta = float(m.group(1)) if m else -0.75          # a file without the field is taken to have the published -0.75
    if beta != BETA['sulfide']:
        raise SystemExit(f'{amplitudes_txt}: main run made with sulfide exponent {beta:g}, weathering_core uses {BETA["sulfide"]:g}; '
                         're-run the main model (./run_uncertainty_m3.sh main) first')


# The three regional calibrations of Bufe et al. (2024): Taiwan, the eastern Tibetan Plateau (Sichuan) and New Zealand
REGIONS = ["Taiwan", "Sichuan", "NewZealand"]
# Parameters of one fit, as named in the parameter file: chi (-), K/(sigma+1), sigma+1 (-), z (t km-2),
# activation energy Ea (J mol-1) and reference runoff q_ref (mm/yr)
KEYS = ["chi_m", "K_over_sigp1", "sigma_plus_1", "z_t_per_km2", "Ea_J_per_mol", "q_ref_mm_yr"]
GTC = 12.011e-15        # mol C -> Gt C


def load_fits(path, regions=None):
    """Read the Bufe et al. (2024) parameter file (JSON, {reaction: {region: {key: value}}}).

    Returns {'silicate': {key: value}, 'sulfide': {key: value}} for the keys in KEYS. Each value is the median
    over `regions`; the analysis always passes a single region, so a set is used intact (its parameters covary
    within a fit and are never mixed between regions).
    """
    raw = json.load(open(path))
    regions = regions or REGIONS
    out = {}
    for rxn in ("silicate", "sulfide"):
        fs = [raw[rxn][r] for r in regions]
        out[rxn] = {k: float(np.median([f[k] for f in fs])) for k in KEYS}
    return out


def flux_and_sens(E_mm, q_mm, T_K, f, beta, mode="kinetic"):
    """Return W (t km-2 yr-1), dW/dlnE (same units) and gamma (-).

    E_mm  erosion (denudation) rate, mm/yr; any array shape
    q_mm  runoff, mm/yr; broadcastable to E_mm
    T_K   surface-air temperature, K; broadcastable to E_mm
    f     parameters of one reaction and one set (an entry of `load_fits`)
    beta  runoff exponent; f(q) = (q / q_ref)**(beta + 1)
    mode  'kinetic' (f(q) in the rate constant) or 'outer' (f(q) on the whole flux)

    The amplitude A is not applied. dW/dlnE is the analytic derivative for a fractional change of E at fixed
    climate; summed over cells and divided by the summed flux it gives the global erosion sensitivity S.
    """
    # floors keep (z/E)**s and q/q_ref finite in cells with no erosion or no runoff; such cells carry W ~ 0
    E = np.maximum(E_mm, 1e-6) * RHO                    # t km-2 yr-1
    q = np.maximum(q_mm, 1e-3)
    fT = 1.0 if f["Ea_J_per_mol"] == 0 else np.exp(f["Ea_J_per_mol"] / R_GAS * (1.0 / T_REF_K - 1.0 / T_K))
    fq = np.power(q / f["q_ref_mm_yr"], beta + 1.0)
    s = f["sigma_plus_1"]
    base = f["K_over_sigp1"] * fT * np.power(f["z_t_per_km2"] / E, s)
    # a: how far weathering proceeds while rock passes through the weathering zone; clipped at 700 so that exp(-a)
    # stays representable (beyond that the flux is fully supply-limited)
    a = np.clip(base * fq if mode == "kinetic" else base, 0.0, 700.0)
    em = np.exp(-a)
    one_m = -np.expm1(-a)                               # 1 - exp(-a), accurate for small a
    outer = 1.0 if mode == "kinetic" else fq
    W = f["chi_m"] * E * one_m * outer
    # dW/dlnE = W + chi E e^-a da/dlnE, with da/dlnE = -s a
    dW = W - f["chi_m"] * E * em * a * s * outer
    # gamma -> 1 as a -> 0 (kinetic limit)
    gamma = np.where(a > 0, a * em / np.maximum(one_m, 1e-300), 1.0)
    return W, dW, gamma


def co2_sil(W):   # t km-2 yr-1 of silicate cations -> mol C m-2 yr-1 consumed (1/2 SCF / M_cat; t km-2 = g m-2)
    return 0.5 * W / M_CAT


def co2_sulf(W):  # t km-2 yr-1 of sulfide-derived sulfate -> mol C m-2 yr-1 released: 1/2 per equivalent, 2 equivalents per mol
    return W / M_SO4


# ---------------------------------------------------------------- rock organic carbon
# Petrogenic organic-carbon oxidation, J = rho * [OC] * chi_ox * E (Zondervan et al., 2023, eq. 1).
# 'linear'    chi_ox constant: oxidation strictly supply-limited.
# 'declining' chi_ox falls with sediment yield Y (t km-2 yr-1), chi_ox = 1/(1 + 0.0037 Y**0.72),
#             the ratio implied by the oxidation regression of Zondervan et al. (2023;
#             0.19 Y**0.39) and the unoxidised petrogenic export of Galy et al. (2015; 0.0007 Y**1.11):
#             chi_ox = ox / (ox + unox) = 1 / (1 + (0.0007 / 0.19) Y**(1.11 - 0.39)).
# Both are scaled by one amplitude so that the modern global release matches a target (68 Mt C yr-1;
# run_weathering.TARGET), so rho, [OC] and units are absorbed in that amplitude: the functions below
# return quantities proportional to the flux, uniform rock organic-carbon content assumed.

def oc_petro(E_mm, form="declining"):
    """Rock organic-carbon oxidation up to a constant factor: J = Y * chi_ox(Y) and dJ/dlnE.

    E_mm is the erosion rate in mm/yr; Y = E_mm * RHO is the sediment yield in t km-2 yr-1.
    `form` is 'declining' (central case) or 'linear' (constant chi_ox).
    """
    Y = np.maximum(E_mm, 1e-6) * RHO
    if form == "linear":
        J = Y.copy(); dJ = Y.copy()
    else:
        chi = 1.0 / (1.0 + 0.0037 * np.power(Y, 0.72))
        J = Y * chi
        dJ = J * (1.0 - 0.72 * (1.0 - chi))      # dJ/dlnE
    return J, dJ


def oc_bio_burial(E_mm):
    """Erosion-driven burial of biospheric organic carbon, Y_Corg = 0.022 Y**0.66 (Galy et al., 2015).

    E_mm in mm/yr, Y = E_mm * RHO in t km-2 yr-1. Returns Y_Corg (scaled to the modern target by the caller)
    and its derivative dY_Corg/dlnE = 0.66 Y_Corg.
    """
    Y = np.maximum(E_mm, 1e-6) * RHO
    J = 0.022 * np.power(Y, 0.66)
    return J, 0.66 * J

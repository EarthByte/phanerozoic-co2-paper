#!/usr/bin/env python3
"""Per-cell weathering fluxes, their erosion sensitivity and their climate sensitivity.

Weathering law (West, 2012; Bufe et al., 2024, eq. 6), per reaction:

    W = A * chi * E * (1 - exp(-a)),      a = c * f(T) * f(q) * (z / E)**s

with E the denudation flux (t km-2 yr-1), s = sigma + 1, c = K/(sigma+1),
f(T) Arrhenius about 293.15 K and f(q) = (q / q_ref)**(beta + 1).

Two placements of the runoff term are implemented:

  'kinetic'  f(q) sits inside the exponential, with the rate constant, as in
             West (2012). Supply-limited weathering (a >> 1, W -> chi*E) is then
             independent of runoff and kinetically limited weathering (a << 1)
             scales with f(q), reproducing the Bufe et al. (2024) runoff
             correction exactly where their data lie.
  'outer'    f(q) multiplies the whole flux (the earlier implementation), which
             makes supply-limited weathering scale with runoff too.

Local erosion sensitivity:  eps = dlnW/dlnE = 1 - s * gamma,
local kinetic (climate) share: gamma = dlnW/dln a = a e^-a / (1 - e^-a),
gamma = 0 supply-limited, gamma = 1 kinetically limited.
"""
from __future__ import annotations
import json, re
import numpy as np

R_GAS = 8.3145
T_REF_K = 293.15
RHO = 2500.0            # kg m-3, Bufe et al. (2024)
M_SO4 = 96.06
M_CAT = 30.0
# Runoff enters the rate constant as (q/q_ref)^(BETA + 1) (West 2012 placement). Pure West form for both phases, adopted
# 2026-10-05: the sulfide value -0.75 of Bufe et al. (2024) normalised the measured concentrations to q_ref across catchments of
# both regimes, and the West form produces the regime dependence itself, so it is not applied again in the rate constant.
# The published -0.75 is a sensitivity test (run_weathering.py --beta-sulf -0.75; runoff_form_tests.py).
BETA = {"silicate": 0.0, "sulfide": 0.0}


def check_form(amplitudes_txt):
    """Refuse a main run whose sulfide runoff exponent differs from BETA: fields and calibration would mix two forms."""
    lines = open(amplitudes_txt).read().split('\n')
    m = re.search(r'beta_sulf=(\S+)', lines[2]) if len(lines) > 2 else None
    beta = float(m.group(1)) if m else -0.75          # runs written before the field existed used the published -0.75
    if beta != BETA['sulfide']:
        raise SystemExit(f'{amplitudes_txt}: main run made with sulfide exponent {beta:g}, weathering_core uses {BETA["sulfide"]:g}; '
                         're-run the main model (./run_uncertainty_m3.sh main) first')


REGIONS = ["Taiwan", "Sichuan", "NewZealand"]
KEYS = ["chi_m", "K_over_sigp1", "sigma_plus_1", "z_t_per_km2", "Ea_J_per_mol", "q_ref_mm_yr"]
GTC = 12.011e-15        # mol C -> Gt C


def load_fits(path, regions=None):
    raw = json.load(open(path))
    regions = regions or REGIONS
    out = {}
    for rxn in ("silicate", "sulfide"):
        fs = [raw[rxn][r] for r in regions]
        out[rxn] = {k: float(np.median([f[k] for f in fs])) for k in KEYS}
    return out


def flux_and_sens(E_mm, q_mm, T_K, f, beta, mode="kinetic"):
    """Return W (t km-2 yr-1), dW/dlnE (same units) and gamma (-)."""
    E = np.maximum(E_mm, 1e-6) * RHO
    q = np.maximum(q_mm, 1e-3)
    fT = 1.0 if f["Ea_J_per_mol"] == 0 else np.exp(f["Ea_J_per_mol"] / R_GAS * (1.0 / T_REF_K - 1.0 / T_K))
    fq = np.power(q / f["q_ref_mm_yr"], beta + 1.0)
    s = f["sigma_plus_1"]
    base = f["K_over_sigp1"] * fT * np.power(f["z_t_per_km2"] / E, s)
    a = np.clip(base * fq if mode == "kinetic" else base, 0.0, 700.0)
    em = np.exp(-a)
    one_m = -np.expm1(-a)
    outer = 1.0 if mode == "kinetic" else fq
    W = f["chi_m"] * E * one_m * outer
    dW = W - f["chi_m"] * E * em * a * s * outer
    gamma = np.where(a > 0, a * em / np.maximum(one_m, 1e-300), 1.0)
    return W, dW, gamma


def co2_sil(W):   # t km-2 yr-1 of silicate cations -> mol C m-2 yr-1 consumed
    return 0.5 * W / M_CAT


def co2_sulf(W):  # t km-2 yr-1 of sulfide-derived sulfate -> mol C m-2 yr-1 released
    return 0.5 * W / M_SO4


# ---------------------------------------------------------------- rock organic carbon
# Petrogenic organic-carbon oxidation, J = rho * [OC] * chi_ox * E (Zondervan et al., 2023, eq. 1).
# 'linear'   chi_ox constant: oxidation strictly supply-limited.
# 'declining' chi_ox falls with sediment yield Y (t km-2 yr-1), chi_ox = 1/(1 + 0.0037 Y**0.72),
#            the ratio implied by the oxidation regression of Zondervan et al. (2023;
#            0.19 Y**0.39) and the unoxidised petrogenic export of Galy et al. (2015; 0.0007 Y**1.11).
# Both are scaled by one amplitude so that the modern global release matches a target.

def oc_petro(E_mm, form="declining"):
    Y = np.maximum(E_mm, 1e-6) * RHO
    if form == "linear":
        J = Y.copy(); dJ = Y.copy()
    else:
        chi = 1.0 / (1.0 + 0.0037 * np.power(Y, 0.72))
        J = Y * chi
        dJ = J * (1.0 - 0.72 * (1.0 - chi))      # dJ/dlnE
    return J, dJ


def oc_bio_burial(E_mm):
    """Erosion-driven burial of biospheric organic carbon, Y_Corg = 0.022 Y**0.66 (Galy et al., 2015)."""
    Y = np.maximum(E_mm, 1e-6) * RHO
    J = 0.022 * np.power(Y, 0.66)
    return J, 0.66 * J

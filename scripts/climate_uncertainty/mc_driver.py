#!/usr/bin/env python3
"""
mc_driver.py — propagate the climate-forcing uncertainty end-to-end through the
NeuroLEM surrogate and the Bufe et al. (2024) weathering law, and put percentile
bands on the Phanerozoic time series of continental sediment delivery and of the
silicate weathering sink.

ROLE IN THE PAPER
-----------------
This script defines the two-parameter Monte Carlo design behind the climate-state
uncertainty test (Methods, "Uncertainty and sensitivity", item 6: climate states
from the pattern-scaling emulator over a grid of CO2 levels spanning the proxy
envelope and draws of the climate-model spread).  The paper's climate-state tables
are written by ``nature_recast/climate_sensitivity.py``, which follows the same
design under the weathering formulation of the main model and imports
``Surrogate``, ``_node_axes`` and ``_perturb_coarse`` from this module
(``dyntopo_sensitivity.py`` and ``check_archive.py`` also use ``Surrogate``).  Run
on its own, this script bands the sediment-flux and silicate-sink histories with the
simpler weathering step defined below (median parameter set, one amplitude per
reaction, runoff factor on the whole flux).

WHY THIS IS CHEAP
-----------------
A naive Monte Carlo would need one surrogate forward pass per member per age per
palaeoelevation surface: 2000 x 109 x 2 = 436,000 passes at 1801 x 3601.  It is not
necessary.  The perturbation produced by ``phanda_emulator`` is a strictly
TWO-PARAMETER family:

    dT(x)            = lambda_T(x) * r  +  z * sigma_T(x)                [K]
    ln R(x)/R_ref(x) = epsilon * ( kappa_P(x) * r  +  z * sigma_lnP(x) )

    r = log2(pCO2 / pCO2_ref)            the CO2 draw, in doublings
    z = the (spatially coherent) standard-normal draw on the sigma field

lambda_T (K per doubling), kappa_P (ln precipitation per doubling) and the sigma
fields come from the emulator fit on the native HadCM3L grid (96 x 73) and are
regridded bilinearly to the 0.1-degree grid.  Everything downstream — the surrogate
erosion field, its area integral, the weathering fluxes — is a smooth function of
(r, z) alone.  So the *expensive* chain is evaluated on a small (r, z) node grid per
age, and any number of Monte Carlo members is obtained by interpolating the
integrated scalars.  With 5 x 3 nodes that is 5 x 3 x 109 x 2 = 3270 passes, and the
member count becomes free.

Interpolation error is controlled by the node count, not by the member count:
``nodes`` prints, for every age, the curvature of the sediment flux in r (its largest
departure from a straight line through the end nodes, relative to its range).

TWO-STAGE USE
-------------
  Stage A (GPU, minutes to hours; resumable)
      python mc_driver.py nodes --erodep-dir ... --temp-dir ... \
          --sensitivity phanda_sensitivity.npz --ckpt best.pt \
          --params-json bufe_2024_params.json --envelope csv \
          --envelope-csv pco2_envelope.csv --out mc_nodes.npz

  Stage B (laptop, seconds)
      python mc_driver.py sample --nodes mc_nodes.npz --n 2000 --seed 0 \
          --out mc_bands --data-dir ../../data

  Self-test (no data needed)
      python mc_driver.py selftest

INPUTS (stage A)
----------------
  --erodep-dir    erodep_<age>Ma.npz from apply_dem.py, 0.1-degree grid (1801 x 3601,
                  lat -90..90, lon -180..180): zpaleo_scotese / zpaleo_zhou (m; the
                  PALEOMAP reference and the rock-chemistry-adjusted surface),
                  ero_scotese / ero_zhou (signed erosion(-)/deposition(+) rate, mm/yr),
                  mask_scotese / mask_zhou (land masks), rain (HadCM3 precipitation,
                  m/yr), lat.
  --temp-dir      <age>Ma.npz with int-packed HadCM3 surface-air temperature
                  (temp_q * scale + offset, degC) on the same grid.
  --sensitivity   phanda_sensitivity.npz written by ``phanda_emulator.py fit``.
  --ckpt          trained NeuroLEM checkpoint (best.pt).
  --params-json   Bufe et al. (2024) regional parameter fits.
  --envelope ...  pCO2 envelope per age (``csv`` from build_pco2_envelope.py,
                  ``factor`` = CO2_ref / f ... CO2_ref * f, or ``records``).

OUTPUTS
-------
  nodes  -> mc_nodes.npz, with leading axes (age, r node, z node[, case]):
            Q      (nA, nr, nz, 3)  net continental export, -sum(erodep * area) over
                                    the land mask, in mm m^2/yr (calibrated to Gt/yr
                                    in stage B, so only its shape matters)
            scf    (nA, nr, nz, 3)  silicate-cation CO2 sink, Gt C/yr
            sf     (nA, nr, nz, 3)  sulfide-sulfate CO2 source, Gt C/yr
            T_land (nA, nr, nz)     mean land temperature, degC (reference land mask)
            q_land (nA, nr, nz)     mean land precipitation, m/yr (reference land mask)
            co2    (nA, nr)         pCO2 at each r node, ppm
            r, z   (nA, nr), (nA, nz)  node coordinates
            plus age, the run settings and reference_node_rmse.
            The three cases are ``reference`` (PALEOMAP surface and land mask),
            ``informed`` (adjusted surface and its own land mask) and
            ``informed_common_cells`` (adjusted surface on the PALEOMAP land mask).
            One checkpoint per age is kept in <out>.ckpt/ so the sweep resumes.
  sample -> <out>/mc_bands.csv        percentiles per age of sediment (Gt/yr),
                                      silicate and net (silicate - sulfide) flux
                                      (Gt C/yr) for the three cases, and of the
                                      informed - reference differences
            <out>/mc_band_widths.csv  median relative band width under both
                                      calibrations
            <out>/mc_correlations.csv per-member Pearson r of the silicate sink
                                      against log10 pCO2 of Foster et al. (2017) and
                                      Judd et al. (2024), its change between the
                                      surfaces and P(delta r <= 0)
            <out>/fig3_sediment_flux_banded.png, <out>/fig7_pco2_banded.png
                                      quick-look plots of the banded series

REFERENCE-NODE CHECK
--------------------
The (r = 0, z = 0) node is by construction the unperturbed forcing, so the
surrogate output there must reproduce the stored ``ero_scotese`` / ``ero_zhou``
fields in the erodep NPZs.  ``nodes`` checks this at every age, prints the RMSE (in
the units of the erosion fields, mm/yr) and aborts if it exceeds
``--reference-tolerance``; a failure means the checkpoint, normalisation or input
fields do not match the ones that produced the archive.

CALIBRATION
-----------
Two amplitudes are fixed on modern observations: the sediment calibration
(present-day net export from the PALEOMAP surface = ``--present-day-flux-gt``,
20 Gt/yr) and the present-day silicate and sulfide fluxes, whose level is set by the
amplitude factors ``--f-sil`` and ``--f-sulf``.  Those are observational anchors that
hold whatever the climate member is, so ``--calibration per-member`` (the default)
re-derives them inside each member: every member's present-day sediment flux is
scaled to the target, and its present-day silicate and sulfide fluxes to those of the
unperturbed reference member.  This removes the part of the climate uncertainty that
simply rescales both histories together, and leaves the part that changes their
shape and their difference, which is what the paper's claim rests on.
``--calibration fixed`` keeps the reference amplitudes and shows the raw sensitivity
instead; both are written to mc_band_widths.csv, because the gap between them is the
amplitude-versus-shape statement.

TIME CORRELATION
----------------
How the draws are correlated across ages changes the band a lot, and the two
parameters are treated differently by default:

  * the CO2 draw ``r`` is proxy uncertainty, largely independent between distant
    stages -> ``--co2-time-correlation independent`` (default): r is drawn
    uniformly (in log2 pCO2) across the node range at each age
  * the ``z`` draw stands for model structural error, which is a coherent bias of
    one model configuration across the whole Phanerozoic ->
    ``--sigma-time-correlation full`` (default): one z per member, used at all ages

Both can be changed; the setting used is printed and should be reported with the
results.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phanda_emulator import (ClimateEmulator, basegrid_axes, load_envelope,  # noqa: E402
                             regrid_bilinear, _clip_scale)

R_EARTH = 6.371e6           # m, mean Earth radius
DEG = np.deg2rad(0.1)       # grid spacing of the 0.1-degree basegrid, rad
LANDPLANT_MA = 420.0        # ages before widespread land plants, shaded in the sediment plot

# Bufe et al. (2024) weathering constants, kept identical to bufe_co2_from_dem.py
R_GAS = 8.3145              # J mol-1 K-1
T_REF_K = 293.15            # K, reference temperature of the Arrhenius term
RHO = 2500.0                # kg m-3; E [mm/yr] * RHO = denudation in t km-2 yr-1
M_SO4 = 96.06               # g mol-1, sulfate
M_CAT = 30.0                # g mol-1, M_cat of the net-flux formula of Bufe et al. (2024)
# Runoff exponent beta: the flux scales as (q / q_ref)**(beta + 1)
BETA = {"silicate": 0.0, "sulfide": -0.75}
DEFAULT_REGIONS = ["Taiwan", "Sichuan", "NewZealand"]

# The three integration cases, in the order of the last axis of Q, scf and sf:
#   reference              PALEOMAP surface, PALEOMAP land mask
#   informed               rock-chemistry-adjusted surface, its own land mask
#   informed_common_cells  adjusted surface integrated over the PALEOMAP land mask
CASES = ("reference", "informed", "informed_common_cells")


# --------------------------------------------------------------------------------------
# Bufe weathering (verbatim port of bufe_co2_from_dem.py, kept in sync deliberately)
# --------------------------------------------------------------------------------------

def median_fit(raw, rxn, regions):
    """Median over ``regions`` of each Bufe et al. (2024) fit parameter of ``rxn``.

    ``raw`` is the parsed parameter JSON ({reaction: {region: {param: value}}}).
    Returns {param: value} with chi_m (-), K_over_sigp1, sigma_plus_1 (-),
    z_t_per_km2 (t km-2 yr-1), Ea_J_per_mol (J/mol) and q_ref_mm_yr (mm/yr).
    """
    ks = ["chi_m", "K_over_sigp1", "sigma_plus_1", "z_t_per_km2", "Ea_J_per_mol", "q_ref_mm_yr"]
    fs = [raw[rxn][r] for r in regions if r in raw[rxn]]
    return {k: float(np.median([f[k] for f in fs])) for k in ks}


def weathering_flux(E_mm, q_mm, T_K, f, beta):
    """Weathering flux W = chi * E * (1 - exp(-a)) * (q / q_ref)**(beta + 1).

    a = K/(sigma+1) * f(T) * (z / E)**(sigma+1), with f(T) an Arrhenius factor about
    T_REF_K.  The runoff factor multiplies the whole flux.

    E_mm : denudation rate, mm/yr (positive); q_mm : runoff, mm/yr; T_K : K.
    f    : parameter dict from ``median_fit``; beta : runoff exponent.
    Returns W in t km-2 yr-1 (numerically equal to g m-2 yr-1), same shape as E_mm.
    """
    # floors keep the power laws finite on non-eroding and dry cells
    E = np.maximum(E_mm, 1e-6) * RHO
    q = np.maximum(q_mm, 1e-3)
    fT = 1.0 if f["Ea_J_per_mol"] == 0 else np.exp(
        f["Ea_J_per_mol"] / R_GAS * (1.0 / T_REF_K - 1.0 / T_K))
    # clipping at 700 avoids underflow in exp(-arg); beyond it the flux is supply-limited
    arg = np.clip(f["K_over_sigp1"] * fT * np.power(f["z_t_per_km2"] / E, f["sigma_plus_1"]),
                  0.0, 700.0)
    W = f["chi_m"] * E * (1.0 - np.exp(-arg))
    return W * np.power(q / f["q_ref_mm_yr"], beta + 1.0)


def bufe_case(ero_mm, mask, q_mm, T_K, area, fits, f_sil, f_sulf):
    """Global silicate CO2 sink and sulfide CO2 source of one surface, Gt C/yr.

    ero_mm : signed erosion(-)/deposition(+) rate, already multiplied by the erosion
             scaling factor, mm/yr (H, W); depositing cells carry no flux.
    mask   : land mask (H, W); only these cells are integrated.
    q_mm   : runoff, mm/yr; T_K : temperature, K; area : cell area, m2 (all H, W).
    fits   : {"silicate": params, "sulfide": params} from ``median_fit``.
    f_sil, f_sulf : amplitude factors applied to the silicate and sulfide fluxes.

    Returns (scf, sf), the two terms of F_net = 0.5 SCF / M_cat - SF / M_SO4
    (Bufe et al., 2024): W [g m-2 yr-1] * area [m2] / M [g/mol] gives mol/yr, times
    0.5, times 12.011 g C/mol and 1e-15 Gt/g.

    The calculation indexes the masked cells first rather than evaluating the full
    1801 x 3601 grid: it runs 3 times per node, 15 nodes per age and 109 ages, and
    land is roughly a quarter of the grid, so this cuts the cost several-fold while
    giving the same sums.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return 0.0, 0.0
    # erosion positive; deposition (positive erodep) gives E = 0
    E = np.maximum(-np.asarray(ero_mm, dtype=np.float64)[m], 0.0)
    qc = np.asarray(q_mm, dtype=np.float64)[m]
    Tc = np.asarray(T_K, dtype=np.float64)[m]
    A = np.asarray(area, dtype=np.float64)[m]
    SCF = f_sil * weathering_flux(E, qc, Tc, fits["silicate"], BETA["silicate"])
    SF = f_sulf * weathering_flux(E, qc, Tc, fits["sulfide"], BETA["sulfide"])
    scf = float(np.nansum(SCF * A)) / M_CAT * 0.5 * 12.011 * 1e-15
    sf = float(np.nansum(SF * A)) / M_SO4 * 12.011 * 1e-15
    return scf, sf


# --------------------------------------------------------------------------------------
# Surrogate
# --------------------------------------------------------------------------------------

class Surrogate:
    """Thin wrapper on the trained NeuroLEM U-Net, matching apply_dem.py exactly.

    Calling it with elevation (m) and precipitation (m/yr), both (H, W) on the
    0.1-degree grid, returns the signed erosion(-)/deposition(+) rate in mm/yr.
    Inputs are standardised with the training means and standard deviations stored
    in the checkpoint; the network predicts the target in a standardised asinh space,
    which is inverted here as s * sinh(t * t_std + t_mean).
    """

    def __init__(self, ckpt: str, device: str | None = None):
        import torch
        self.torch = torch
        GlobalUNet = _load_model_class()
        self.dev = device or _device()
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        self.model = GlobalUNet(in_ch=ck["in_ch"], depth=ck["depth"], base=ck["base"]).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        n = ck["norm"]
        self.in_mean = np.asarray(n["in_mean"], np.float32)
        self.in_std = np.asarray(n["in_std"], np.float32)
        self.s = float(n["asinh_s"])
        self.t_mean = float(n["t_mean"])
        self.t_std = float(n["t_std"])
        # each U-Net level halves the grid, so H and W must be multiples of 2**depth
        self.mult = 2 ** ck["depth"]
        self.val_rmse = ck.get("val_rmse")

    def __call__(self, elev: np.ndarray, rain: np.ndarray) -> np.ndarray:
        x = np.stack([(elev - self.in_mean[0]) / self.in_std[0],
                      (rain - self.in_mean[1]) / self.in_std[1]], 0).astype(np.float32)
        xp, (H, W) = _pad_to(x, self.mult)
        with self.torch.no_grad():
            t = self.model(self.torch.from_numpy(xp)[None].to(self.dev))[0, 0]
            t = t.cpu().numpy()[:H, :W]
        return (self.s * np.sinh(t * self.t_std + self.t_mean)).astype(np.float32)


class StubSurrogate:
    """Plumbing-test stand-in: a cheap, smooth, monotone-in-runoff erosion field.

    NOT physical.  It exists so the driver's integration, calibration, interpolation
    and figure code can be exercised without a GPU; ``nodes --stub-surrogate`` marks
    its output so ``sample`` refuses to treat it as a result.
    """
    val_rmse = None

    def __call__(self, elev, rain):
        # erosion (negative) grows with gradient magnitude and with rain**0.6
        slope = np.gradient(elev.astype(np.float32), axis=0) ** 2 + \
                np.gradient(elev.astype(np.float32), axis=1) ** 2
        e = -1e-6 * np.sqrt(slope) * np.power(np.maximum(rain, 1e-3), 0.6)
        return e.astype(np.float32)


def _load_model_class():
    """Import the GlobalUNet class from paleomap_unet.py, the NeuroLEM model definition in scripts/.

    Uses the module if it is importable, otherwise adds scripts/ (the parent directory) to the path.
    """
    try:
        from paleomap_unet import GlobalUNet
        return GlobalUNet
    except ModuleNotFoundError:
        for c in (HERE.parents[1], HERE):
            if (c / "paleomap_unet.py").exists():
                sys.path.insert(0, str(c))
                from paleomap_unet import GlobalUNet
                return GlobalUNet
        raise SystemExit("cannot locate paleomap_unet.py (scripts/ of this repository)")


def _device():
    """Pick the torch device: CUDA, then Apple MPS, then CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _pad_to(a, mult):
    """Edge-pad the last two axes of ``a`` up to multiples of ``mult``.

    Returns the padded array and the original (H, W) so the output can be cropped.
    """
    H, W = a.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph == 0 and pw == 0:
        return a, (H, W)
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)], mode="edge"), (H, W)


# --------------------------------------------------------------------------------------
# Stage A -- evaluate the expensive chain on the (r, z) node grid
# --------------------------------------------------------------------------------------

def _node_axes(emu, age, c_lo, c_hi, n_r, n_z, z_max):
    """Node coordinates of the (r, z) grid for one age.

    r runs from log2(c_lo / CO2_ref) to log2(c_hi / CO2_ref) (doublings), z from
    -z_max to +z_max (standard deviations).  Both axes have 0 added (the unperturbed
    reference forcing) and are sorted and de-duplicated, so with the default
    n_r = 5, n_z = 3 there are 4 + 1 r nodes and 2 + 1 z nodes.
    Returns (r, z) as 1-D float arrays.
    """
    ref = emu.co2_ref(age)
    r_edges = np.log2(np.array([c_lo, c_hi], dtype=float) / ref)
    r = np.linspace(r_edges[0], r_edges[1], max(n_r - 1, 2))
    r = np.unique(np.concatenate([r, [0.0]]))
    z = np.unique(np.concatenate([np.linspace(-z_max, z_max, max(n_z - 1, 2)), [0.0]]))
    return r, z


def cmd_nodes(a):
    """Stage A: evaluate surrogate and weathering on the (r, z) node grid of every age.

    Writes ``a.out`` (see the module docstring for its arrays) and one checkpoint per
    age; aborts if the unperturbed node does not reproduce the archived erosion fields.
    """
    emu = ClimateEmulator.load(a.sensitivity)
    ed, td = Path(a.erodep_dir), Path(a.temp_dir)
    files = {int(re.search(r"erodep_(\d+)Ma", f.name).group(1)): f
             for f in ed.glob("erodep_*Ma.npz")}
    ages = sorted(int(v) for v in a.ages) if a.ages else sorted(set(files) & set(emu.age.tolist()))
    # stage B calibrates on the present day, so 0 Ma must be among the ages
    if 0 not in ages:
        print("WARNING: 0 Ma is absent, so per-member recalibration will not be possible",
              file=sys.stderr)

    lo, hi = load_envelope(a.envelope, np.array(ages), csv_path=a.envelope_csv,
                           factor=a.envelope_factor, data_dir=a.envelope_data_dir)
    terms = tuple(t for t in a.sigma_terms.split(",") if t) if a.sigma_terms else ()
    clip = (a.runoff_scale_clip[0], a.runoff_scale_clip[1])
    fits = {rxn: median_fit(json.load(open(a.params_json)), rxn, a.regions)
            for rxn in ("silicate", "sulfide")}
    model = StubSurrogate() if a.stub_surrogate else Surrogate(a.ckpt, a.device)
    print(f"[nodes] surrogate={'STUB' if a.stub_surrogate else type(model).__name__} "
          f"val_rmse={model.val_rmse}  ages={len(ages)}  "
          f"sigma_terms={','.join(terms) or 'none'}  epsilon={a.runoff_elasticity}")

    blat, blon = basegrid_axes()        # 0.1-degree target grid of the perturbations
    out = {k: [] for k in ("Q", "scf", "sf", "T_land", "q_land", "co2", "r", "z")}
    ref_rmse = []
    ages_done = []
    # Per-age checkpoints: this is a multi-hour GPU job, so each age is written the
    # moment it completes and reloaded on the next run; --time-budget stops cleanly
    # so the sweep can be split into chunks.
    ckpt = Path(a.checkpoint_dir) if a.checkpoint_dir else Path(str(a.out) + ".ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    n_new = n_cached = 0
    for age in ages:
        cp = ckpt / f"nodes_{int(age)}Ma.npz"
        if cp.exists():
            try:
                with np.load(cp, allow_pickle=False) as zz:
                    for k in ("Q", "scf", "sf", "T_land", "q_land", "co2", "r", "z"):
                        out[k].append(zz[k])
                    if "ref_rmse" in zz and zz["ref_rmse"].size == 2:
                        ref_rmse.append((age, float(zz["ref_rmse"][0]), float(zz["ref_rmse"][1])))
                ages_done.append(age); n_cached += 1
                continue
            except Exception:
                pass                                   # unreadable checkpoint -> refit
        if a.time_budget and time.monotonic() - t_start > a.time_budget:
            print(f"  [{age:>4} Ma] time budget reached -- stopping; re-run to continue",
                  flush=True)
            break
        d = np.load(files[age], allow_pickle=True)
        # all fields below are (1801, 3601), lat -90..90 by lon -180..180
        lat = d["lat"].astype(np.float64)
        rain_ref = d["rain"].astype(np.float32)          # HadCM3 precipitation, m/yr
        zp = {"reference": d["zpaleo_scotese"].astype(np.float32),
              "informed": d["zpaleo_zhou"].astype(np.float32)}
        masks = {"reference": d["mask_scotese"], "informed": d["mask_zhou"],
                 "informed_common_cells": d["mask_scotese"]}
        # spherical cell area, m2: R^2 * dlat * dlon * cos(lat)
        area = np.broadcast_to((R_EARTH * R_EARTH * DEG * DEG)
                               * np.cos(np.deg2rad(lat))[:, None], rain_ref.shape)
        tz = np.load(td / f"{age}Ma.npz", allow_pickle=True)
        # unpack the integer-quantised temperature, degC
        T_ref_C = tz["temp_q"].astype(np.float32) * float(tz["scale"]) + float(tz["offset"])

        c_lo, c_hi = float(lo[ages.index(age)]), float(hi[ages.index(age)])
        # pCO2 envelope (ppm) for this age; the factor envelope is centred on the
        # model's own CO2 so that r = 0 is the reference simulation
        if a.envelope == "factor":
            ref = emu.co2_ref(age)
            c_lo, c_hi = ref / a.envelope_factor, ref * a.envelope_factor
        if not np.isfinite(c_lo):
            print(f"  [{age:>4} Ma] no pCO2 envelope, skipped", file=sys.stderr)
            continue
        r_nodes, z_nodes = _node_axes(emu, age, c_lo, c_hi, a.n_r, a.n_z, a.z_max)

        # per-node integrals: (n_r, n_z, case) for Q, SCF, SF and (n_r, n_z) for TL, QL
        Q = np.zeros((r_nodes.size, z_nodes.size, 3))
        SCF = np.zeros_like(Q)
        SF = np.zeros_like(Q)
        TL = np.zeros((r_nodes.size, z_nodes.size))
        QL = np.zeros_like(TL)
        # indices of the unperturbed node (r = 0, z = 0)
        i0, j0 = int(np.argmin(np.abs(r_nodes))), int(np.argmin(np.abs(z_nodes)))
        for i, r in enumerate(r_nodes):
            for j, z in enumerate(z_nodes):
                # temperature increment (K) and precipitation scale factor (-) on the
                # HadCM3L grid, regridded to 0.1 degree
                dT_c, rs_c = _perturb_coarse(emu, age, r, z, terms, a.runoff_elasticity, clip)
                dT = regrid_bilinear(dT_c, emu.lat, emu.lon, blat, blon)
                rs = regrid_bilinear(rs_c, emu.lat, emu.lon, blat, blon)
                rain = (rain_ref * rs).astype(np.float32)
                T_K = (T_ref_C + dT).astype(np.float64) + 273.15
                # runoff for the weathering law: precipitation (m/yr -> mm/yr) times a
                # uniform runoff coefficient
                q_mm = rain.astype(np.float64) * 1000.0 * a.runoff_coefficient

                # one surrogate pass per surface with the perturbed precipitation; the
                # common-cells case reuses the adjusted-surface field on another mask
                ero = {c: model(zp["reference" if c == "reference" else "informed"], rain)
                       for c in ("reference", "informed")}
                ero["informed_common_cells"] = ero["informed"]
                if i == i0 and j == j0:
                    e = [float(np.sqrt(np.mean((ero["reference"] - d["ero_scotese"]) ** 2))),
                         float(np.sqrt(np.mean((ero["informed"] - d["ero_zhou"]) ** 2)))]
                    ref_rmse.append((age, *e))
                for k, case in enumerate(CASES):
                    m = masks[case]
                    # net export (erosion minus deposition) over the mask, mm m2/yr;
                    # positive when the surface loses material
                    Q[i, j, k] = -float((ero[case].astype(np.float64) * area)[m].sum())
                    # erosion rates scaled by --erate-scale (the sediment-calibrated
                    # factor of the main model) before entering the weathering law
                    SCF[i, j, k], SF[i, j, k] = bufe_case(
                        ero[case].astype(np.float64) * a.erate_scale, m, q_mm, T_K,
                        area, fits, a.f_sil, a.f_sulf)
                # climate diagnostics, plain means over the PALEOMAP land cells
                land = masks["reference"]
                TL[i, j] = float((T_ref_C + dT)[land].mean())
                QL[i, j] = float(rain[land].mean())
        out["Q"].append(Q); out["scf"].append(SCF); out["sf"].append(SF)
        out["T_land"].append(TL); out["q_land"].append(QL)
        out["co2"].append(emu.co2_ref(age) * 2.0 ** r_nodes)
        out["r"].append(r_nodes); out["z"].append(z_nodes)
        ages_done.append(age); n_new += 1
        # write to a temporary file and rename, so an interrupted write cannot leave
        # a truncated checkpoint
        tmp = cp.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, Q=Q, scf=SCF, sf=SF, T_land=TL, q_land=QL,
                            co2=out["co2"][-1], r=r_nodes, z=z_nodes,
                            ref_rmse=np.array(ref_rmse[-1][1:] if ref_rmse and
                                              ref_rmse[-1][0] == age else [np.nan, np.nan]))
        tmp.replace(cp)
        # non-linearity of the reference sediment flux in r at z = 0
        curv = _curvature(Q[:, j0, 0], r_nodes)
        print(f"  [{age:>4} Ma] CO2 {c_lo:6.0f}-{c_hi:6.0f} ppm  "
              f"Q_ref {Q[i0, j0, 0]:.3e} m3/yr  "
              f"net {SCF[i0, j0, 0]-SF[i0, j0, 0]:+.4f} GtC/yr  "
              f"|curvature in r| {curv:.1%}"
              + (f"  refRMSE {ref_rmse[-1][1]:.2e}/{ref_rmse[-1][2]:.2e}" if ref_rmse else ""))

    missing = sorted(set(ages) - set(ages_done))
    print(f"\n{len(ages_done)}/{len(ages)} ages available ({n_new} evaluated now, "
          f"{n_cached} from checkpoints in {ckpt})")
    if missing:
        print(f"  still missing {len(missing)}: {missing[:12]}"
              f"{' ...' if len(missing) > 12 else ''}\n  re-run the same command to continue.")
    if not out["Q"]:
        raise SystemExit("nothing evaluated")
    if len({r.size for r in out["r"]}) > 1 or len({zz.size for zz in out["z"]}) > 1:
        raise SystemExit("node grids differ in size between ages -- cannot stack")
    np.savez_compressed(
        a.out,
        age=np.array(ages_done, dtype=np.int32),
        Q=np.stack(out["Q"]), scf=np.stack(out["scf"]), sf=np.stack(out["sf"]),
        T_land=np.stack(out["T_land"]), q_land=np.stack(out["q_land"]),
        co2=np.stack(out["co2"]), r=np.stack(out["r"]), z=np.stack(out["z"]),
        cases=np.bytes_(",".join(CASES)),
        stub=np.bool_(a.stub_surrogate),
        erate_scale=np.float32(a.erate_scale), f_sil=np.float32(a.f_sil),
        f_sulf=np.float32(a.f_sulf), runoff_coefficient=np.float32(a.runoff_coefficient),
        runoff_elasticity=np.float32(a.runoff_elasticity),
        sigma_terms=np.bytes_(",".join(terms)),
        reference_node_rmse=np.array(ref_rmse, dtype=np.float64),
    )
    if ref_rmse:
        worst = max(max(r[1], r[2]) for r in ref_rmse)
        if a.stub_surrogate:
            print(f"\nreference-node check SKIPPED (stub surrogate); the {worst:.2e} m/yr "
                  f"figure above is meaningless")
        elif worst > a.reference_tolerance:
            raise SystemExit(
                f"\nreference-node check FAILED: worst RMSE {worst:.3e} m/yr against the "
                f"archived ero_scotese / ero_zhou exceeds --reference-tolerance "
                f"{a.reference_tolerance:g}.\nThe (r=0, z=0) node is the unperturbed forcing "
                f"and must reproduce the archive exactly. Check that --ckpt is the checkpoint "
                f"that produced the erodep files and that --erodep-dir matches it.")
        else:
            print(f"\nreference-node check PASSED: worst RMSE vs the archived erosion fields "
                  f"{worst:.3e} m/yr (tolerance {a.reference_tolerance:g})")
    print(f"wrote {a.out}  ({len(ages_done)} ages x {out['r'][0].size} r x {out['z'][0].size} z)")


def _perturb_coarse(emu, age, r, z, terms, eps, clip):
    """Climate perturbation of one (r, z) node on the native HadCM3L grid (73 x 96).

    dT = lambda_T * r + z * sigma_T (K) and runoff scale = exp(eps * (kappa_P * r +
    z * sigma_lnP)), clipped to ``clip``.  ``terms`` selects the sigma components
    (int, fit, struct) added in quadrature; ``eps`` is the runoff elasticity.
    Returns (dT, runoff_scale) as float32 arrays.
    """
    i = emu._i(age)
    sT, sP = emu.sigma_total(age, terms)
    dT = emu.d["lambda_T"][i] * r + z * sT
    dlnP = emu.d["kappa_P"][i] * r + z * sP
    return dT.astype(np.float32), _clip_scale(np.exp(eps * dlnP), clip).astype(np.float32)


def _curvature(y, x):
    """Relative departure of y(x) from a straight line through its endpoints."""
    if x.size < 3:
        return 0.0
    lin = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    span = np.ptp(y) or 1.0
    return float(np.abs(y - lin).max() / abs(span))


# --------------------------------------------------------------------------------------
# Stage B -- sample, calibrate, band
# --------------------------------------------------------------------------------------

def _interp2(node, r_nodes, z_nodes, r, z):
    """Bilinear interpolation of node[(nr, nz, ...)] at scalar (r, z), clamped.

    Values outside the node range are clamped to its edge.  Returns node[0, 0]'s
    trailing shape (here the three cases).
    """
    r = np.clip(r, r_nodes[0], r_nodes[-1])
    z = np.clip(z, z_nodes[0], z_nodes[-1])
    i = np.clip(np.searchsorted(r_nodes, r) - 1, 0, r_nodes.size - 2)
    j = np.clip(np.searchsorted(z_nodes, z) - 1, 0, z_nodes.size - 2)
    tr = (r - r_nodes[i]) / (r_nodes[i + 1] - r_nodes[i])
    tz = (z - z_nodes[j] ) / (z_nodes[j + 1] - z_nodes[j])
    return (node[i, j] * (1 - tr) * (1 - tz) + node[i + 1, j] * tr * (1 - tz)
            + node[i, j + 1] * (1 - tr) * tz + node[i + 1, j + 1] * tr * tz)


def cmd_sample(a):
    """Stage B: draw members, interpolate the node scalars, calibrate and write bands.

    Arrays below are (n members, nA ages[, 3 cases]).
    """
    z = np.load(a.nodes, allow_pickle=False)
    if bool(z["stub"]) and not a.allow_stub:
        raise SystemExit("these nodes were produced with --stub-surrogate and are a "
                         "plumbing test, not a result; pass --allow-stub to override")
    ages = z["age"].astype(int)
    nA = ages.size
    rng = np.random.default_rng(a.seed)
    n = a.n

    # ---- draws -------------------------------------------------------------------
    u = rng.standard_normal(n) if a.co2_time_correlation == "full" else None
    zs = rng.standard_normal(n) if a.sigma_time_correlation == "full" else None
    R = np.empty((n, nA))
    Z = np.empty((n, nA))
    for k in range(nA):
        rn = z["r"][k]
        if a.co2_time_correlation == "full":
            # same quantile of the envelope at every age -- a coherent pCO2 offset.
            # The tanh map takes one standard normal per member into (0, 1); the
            # resulting position is peaked towards the middle of the range, not uniform
            R[:, k] = rn[0] + (0.5 * (1 + np.tanh(u / 1.5))) * (rn[-1] - rn[0])
        else:
            # independent between ages: uniform in log2(pCO2) over the node range
            R[:, k] = rng.uniform(rn[0], rn[-1], n)
        zn = z["z"][k]
        # standard normal, truncated to the z node range (+-z_max) so no member
        # extrapolates beyond the evaluated nodes
        Z[:, k] = np.clip(zs if zs is not None else rng.standard_normal(n), zn[0], zn[-1])

    # ---- interpolate the integrated scalars --------------------------------------
    Q = np.empty((n, nA, 3)); SCF = np.empty_like(Q); SF = np.empty_like(Q)
    for k in range(nA):
        rn, zn = z["r"][k], z["z"][k]
        for m in range(n):
            Q[m, k] = _interp2(z["Q"][k], rn, zn, R[m, k], Z[m, k])
            SCF[m, k] = _interp2(z["scf"][k], rn, zn, R[m, k], Z[m, k])
            SF[m, k] = _interp2(z["sf"][k], rn, zn, R[m, k], Z[m, k])

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ("per-member", "fixed"):
        results[mode] = _calibrate(Q, SCF, SF, ages, mode, a.present_day_flux_gt, z)
    chosen = results[a.calibration]

    pcs = np.array(a.percentiles, dtype=float)
    _write_bands(outd, ages, results, chosen, pcs, a)
    _write_correlations(outd, ages, chosen, a, rng)
    if not a.no_figures:
        _figures(outd, ages, chosen, pcs, a)
    print(f"\nwrote bands, correlations and figures to {outd}  "
          f"(n={n}, calibration={a.calibration}, "
          f"CO2 time-correlation={a.co2_time_correlation}, "
          f"sigma time-correlation={a.sigma_time_correlation})")


def _calibrate(Q, SCF, SF, ages, mode, target_gt, z):
    """Apply the modern anchors.

    Q, SCF, SF : (n, nA, 3) uncalibrated member series (Q in mm m2/yr; SCF and SF in
    Gt C/yr).  ``target_gt`` is the present-day sediment flux, Gt/yr; ``z`` the node
    file (None in the self-test, where the member means stand in for the reference).
    All anchors are taken on the reference case (index 0) at the youngest age.

    Returns a dict of (n, nA, 3) calibrated series -- sediment (Gt/yr), silicate,
    sulfide and net = silicate - sulfide (Gt C/yr) -- and the per-member factors
    k, a_sil and a_sulf.
    """
    i0 = int(np.argmin(np.abs(ages)))
    # the unperturbed reference member: the (r=0, z=0) node at the present day
    if z is not None:
        ir = int(np.argmin(np.abs(z["r"][i0])))
        iz = int(np.argmin(np.abs(z["z"][i0])))
        Q0, SCF0, SF0 = (z["Q"][i0, ir, iz, 0], z["scf"][i0, ir, iz, 0], z["sf"][i0, ir, iz, 0])
    else:                                                            # selftest path
        Q0, SCF0, SF0 = Q[:, i0, 0].mean(), SCF[:, i0, 0].mean(), SF[:, i0, 0].mean()
    if mode == "per-member":
        # the modern anchors are observations and hold in every member, so re-derive
        # them inside each one; this removes the pure-amplitude part of the spread
        # k converts mm m2/yr of net export to Gt/yr of sediment
        k = target_gt / Q[:, i0, 0]                                  # (n,)
        a_sil = SCF0 / SCF[:, i0, 0]
        # the guard keeps a member with zero sulfide flux at factor 1
        a_sulf = np.where(SF[:, i0, 0] > 0, SF0 / SF[:, i0, 0], 1.0)
    else:
        # one calibration, taken from the unperturbed reference member and reused
        k = np.full(Q.shape[0], target_gt / Q0)
        a_sil = np.ones(Q.shape[0])
        a_sulf = np.ones(Q.shape[0])
    sed = Q * k[:, None, None]
    scf = SCF * a_sil[:, None, None]
    sf = SF * a_sulf[:, None, None]
    return dict(sediment=sed, silicate=scf, sulfide=sf, net=scf - sf,
                k=k, a_sil=a_sil, a_sulf=a_sulf)


def _pct(x, pcs, axis=0):
    """Percentiles ``pcs`` of ``x`` across members (axis 0)."""
    return np.percentile(x, pcs, axis=axis)


def _write_bands(outd, ages, results, chosen, pcs, a):
    """Write mc_bands.csv and mc_band_widths.csv and print the band widths.

    mc_bands.csv: one row per age; for sediment (Gt/yr), silicate and net (Gt C/yr)
    and each case, the requested percentiles across members, then the percentiles of
    the informed - reference differences of sediment and net flux.
    mc_band_widths.csv: median over ages of (p_hi - p_lo) / |median| for both
    calibrations, so the amplitude-versus-shape comparison sits in one table.
    """
    rows = []
    for k, age in enumerate(ages):
        row = [int(age)]
        for key in ("sediment", "silicate", "net"):
            for c in range(3):
                row += [f"{v:.6g}" for v in _pct(chosen[key][:, k, c], pcs)]
        # differences are formed member by member before taking percentiles, so the
        # common-mode climate signal cancels
        d = chosen["sediment"][:, k, 1] - chosen["sediment"][:, k, 0]
        row += [f"{v:.6g}" for v in _pct(d, pcs)]
        dn = chosen["net"][:, k, 1] - chosen["net"][:, k, 0]
        row += [f"{v:.6g}" for v in _pct(dn, pcs)]
        rows.append(row)
    hdr = ["age_Ma"]
    for key in ("sediment_GT_yr", "silicate_GtC_yr", "net_GtC_yr"):
        for c in CASES:
            hdr += [f"{key}_{c}_p{p:g}" for p in pcs]
    hdr += [f"delta_sediment_GT_yr_p{p:g}" for p in pcs]
    hdr += [f"delta_net_GtC_yr_p{p:g}" for p in pcs]
    with open(outd / "mc_bands.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(hdr); w.writerows(rows)

    # the amplitude-versus-shape statement, in one small table
    with open(outd / "mc_band_widths.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "calibration", "median_relative_band_width"])
        for mode, res in results.items():
            for key in ("sediment", "net"):
                for c, name in enumerate(CASES):
                    x = res[key][:, :, c]
                    lo, hi = _pct(x, [pcs[0], pcs[-1]])
                    med = np.median(x, axis=0)
                    # band width relative to the median, then the median over ages
                    rel = np.median(np.abs(hi - lo) / np.maximum(np.abs(med), 1e-12))
                    w.writerow([f"{key}_{name}", mode, f"{rel:.4f}"])
            for key in ("sediment", "net"):
                d = res[key][:, :, 1] - res[key][:, :, 0]
                lo, hi = _pct(d, [pcs[0], pcs[-1]])
                rel = np.median(np.abs(hi - lo) / np.maximum(np.abs(np.median(d, axis=0)), 1e-12))
                w.writerow([f"delta_{key}", mode, f"{rel:.4f}"])
    print("\nrelative band width (p{:g}-p{:g} over the median), {} calibration:".format(
        pcs[0], pcs[-1], a.calibration))
    for key in ("sediment", "net"):
        for c, name in enumerate(CASES[:2]):
            x = chosen[key][:, :, c]
            lo, hi = _pct(x, [pcs[0], pcs[-1]])
            rel = np.median(np.abs(hi - lo) / np.maximum(np.abs(np.median(x, 0)), 1e-12))
            print(f"  {key:<9} {name:<10} {rel:6.1%}")
        d = chosen[key][:, :, 1] - chosen[key][:, :, 0]
        lo, hi = _pct(d, [pcs[0], pcs[-1]])
        rel = np.median(np.abs(hi - lo) / np.maximum(np.abs(np.median(d, 0)), 1e-12))
        print(f"  {key:<9} {'DIFFERENCE':<10} {rel:6.1%}   <- the quantity the paper reports")


def _write_correlations(outd, ages, chosen, a, rng):
    """Per-member Pearson r of the silicate sink against each pCO2 record, and dr.

    Reads foster.npy and judd.npy from --data-dir: (n, 2) arrays of age (Ma) and pCO2
    (ppm).  For each record and each interval 0 - amax Ma (--corr-intervals), the
    record's log10 pCO2 is interpolated to the model ages and correlated with every
    member's silicate sink on both surfaces; dr = r(adjusted) - r(reference).
    Writes mc_correlations.csv with the median and 5th/95th percentiles of r and dr
    and the fraction of members with dr <= 0.  ``rng`` is not used.
    """
    dd = Path(a.data_dir) if a.data_dir else HERE
    recs = {}
    for nm in ("foster", "judd"):
        p = dd / f"{nm}.npy"
        if p.exists():
            arr = np.asarray(np.load(p), float)
            recs[nm] = arr[np.argsort(arr[:, 0])]
    if not recs:
        print("  (no foster.npy / judd.npy found -- correlation table skipped)", file=sys.stderr)
        return
    rows = []
    for nm, prod in recs.items():
        for amax in a.corr_intervals:
            # ages within the interval and covered by the record; skip if too few
            m = (ages <= amax) & (ages <= prod[:, 0].max())
            if m.sum() < 5:
                continue
            y = np.interp(ages[m], prod[:, 0], np.log10(prod[:, 1]))
            yc = y - y.mean()
            def rr(x):
                # Pearson r of every member (rows of x) against y at once
                xc = x - x.mean(axis=1, keepdims=True)
                return (xc @ yc) / np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum())
            r_ref = rr(chosen["silicate"][:, m, 0])
            r_inf = rr(chosen["silicate"][:, m, 1])
            dr = r_inf - r_ref
            rows.append([nm, int(amax),
                         f"{np.median(r_ref):.4f}", f"{np.percentile(r_ref, 5):.4f}",
                         f"{np.percentile(r_ref, 95):.4f}",
                         f"{np.median(r_inf):.4f}", f"{np.percentile(r_inf, 5):.4f}",
                         f"{np.percentile(r_inf, 95):.4f}",
                         f"{np.median(dr):.4f}", f"{np.percentile(dr, 5):.4f}",
                         f"{np.percentile(dr, 95):.4f}",
                         f"{float((dr <= 0).mean()):.4f}"])
            print(f"  r({nm}, 0-{amax:.0f} Ma): reference {np.median(r_ref):+.2f} "
                  f"[{np.percentile(r_ref,5):+.2f},{np.percentile(r_ref,95):+.2f}]  "
                  f"informed {np.median(r_inf):+.2f}  dr {np.median(dr):+.3f} "
                  f"[{np.percentile(dr,5):+.3f},{np.percentile(dr,95):+.3f}]  "
                  f"P(dr<=0)={float((dr<=0).mean()):.3f}")
    with open(outd / "mc_correlations.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["record", "interval_Ma", "r_reference_p50", "r_reference_p5", "r_reference_p95",
                    "r_informed_p50", "r_informed_p5", "r_informed_p95",
                    "delta_r_p50", "delta_r_p5", "delta_r_p95", "P_delta_r_le_0"])
        w.writerows(rows)


def _figures(outd, ages, chosen, pcs, a):
    """Quick-look plots of the banded sediment flux and silicate sink.

    fig3_sediment_flux_banded.png: median and p_lo-p_hi band of the calibrated
    sediment flux for both surfaces, with the present-day target marked.
    fig7_pco2_banded.png: banded silicate sink of both surfaces beside each pCO2
    record (log axis).  Skipped quietly if matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable -- CSVs only", file=sys.stderr)
        return
    BLUE, RED, PUR = "#1f4e79", "#c0392b", "#7d5ba6"
    lo, hi = pcs[0], pcs[-1]

    # ---- sediment flux with a band -----------------------------------------------
    sed = chosen["sediment"]
    fig, ax = plt.subplots(figsize=(11, 5))
    if ages.max() > LANDPLANT_MA:
        ax.axvspan(LANDPLANT_MA, ages.max(), color="#8d6e63", alpha=0.09)
        ax.text((LANDPLANT_MA + ages.max()) / 2,
                np.percentile(sed[:, :, 1], hi, 0).max() * 1.06,
                "before widespread\nland plants — caution", fontsize=8.5,
                color="#6d4c41", ha="center", va="top", style="italic")
    for c, col, lab, ls in ((0, BLUE, "Scotese–Wright DEM", "-"),
                            (1, RED, "Geochemically informed DEM", "-"),
                            (2, RED, "Informed DEM, common reference cells", "--")):
        med = np.median(sed[:, :, c], 0)
        ax.plot(ages, med, color=col, lw=2 if ls == "-" else 1.4, ls=ls, label=lab)
        if c < 2:
            ax.fill_between(ages, np.percentile(sed[:, :, c], lo, 0),
                            np.percentile(sed[:, :, c], hi, 0), color=col, alpha=0.18, lw=0)
    ax.scatter([0], [a.present_day_flux_gt], color="k", zorder=5, s=40)
    ax.set_xlabel("Age (Ma)")
    ax.set_ylabel("Continental sediment delivered to oceans (GT yr$^{-1}$)")
    ax.set_title(f"Phanerozoic continental sediment flux with climate-forcing uncertainty "
                 f"(p{lo:g}–p{hi:g}, {a.calibration} calibration)", loc="left", fontsize=11)
    ax.invert_xaxis(); ax.set_ylim(0, np.percentile(sed[:, :, 1], hi, 0).max() * 1.25)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(outd / "fig3_sediment_flux_banded.png", dpi=200)
    plt.close(fig)

    # ---- silicate sink against the pCO2 records, with a band ----------------------
    dd = Path(a.data_dir) if a.data_dir else HERE
    panels = [(nm, dd / f"{nm}.npy") for nm in ("foster", "judd")]
    panels = [(nm, np.load(p)) for nm, p in panels if p.exists()]
    if not panels:
        print("  (fig7 skipped -- no pCO2 records)", file=sys.stderr)
        return
    sil = chosen["silicate"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.3), squeeze=False)
    titles = {"foster": "Foster et al. (2017)\nobservation-based proxy compilation",
              "judd": "Judd et al. (2024) PhanDA\nmodel–data hybrid"}
    for ax, (nm, prod) in zip(axes[0], panels):
        for c, col, lab, ls in ((0, "#2c5f8a", "Scotese silicate sink", "-"),
                                (1, RED, "Informed silicate sink", "-")):
            ax.plot(ages, np.median(sil[:, :, c], 0), color=col, lw=1.7, ls=ls, label=lab)
            ax.fill_between(ages, np.percentile(sil[:, :, c], lo, 0),
                            np.percentile(sil[:, :, c], hi, 0), color=col, alpha=0.18, lw=0)
        ax.set_xlim(prod[:, 0].max(), 0)
        ax.set_xlabel("Age (Ma)")
        ax.set_ylabel("Silicate weathering CO$_2$ sink (Gt C yr$^{-1}$)")
        ax2 = ax.twinx()
        o = np.argsort(prod[:, 0])
        ax2.plot(prod[o, 0], prod[o, 1], color=PUR, lw=1.1, alpha=0.85)
        ax2.set_yscale("log"); ax2.set_ylabel("pCO$_2$ (ppm)", color=PUR)
        ax.set_title(titles.get(nm, nm), fontsize=10, loc="left")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(outd / "fig7_pco2_banded.png", dpi=200)
    plt.close(fig)
    print(f"  figures: fig3_sediment_flux_banded.png, fig7_pco2_banded.png")


# --------------------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------------------

def cmd_selftest(a):
    """Synthetic checks of the interpolator, the curvature diagnostic, the
    calibration and the stub surrogate.  Returns 0 if all pass, else 1."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("1. bilinear (r, z) interpolation of node scalars")
    rn = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    zn = np.array([-2.0, 0.0, 2.0])
    node = (3.0 + 2.0 * rn[:, None] + 0.5 * zn[None, :])[..., None]      # exactly bilinear
    for r, z in [(0.0, 0.0), (0.3, -1.0), (-0.75, 1.5), (1.0, 2.0)]:
        got = float(_interp2(node, rn, zn, r, z)[0])
        check(f"exact on a bilinear function at ({r:+.2f},{z:+.2f})",
              abs(got - (3 + 2 * r + 0.5 * z)) < 1e-12, f"{got:.6f}")
    check("clamps outside the node range",
          abs(float(_interp2(node, rn, zn, 5.0, 9.0)[0]) - (3 + 2 * 1.0 + 0.5 * 2.0)) < 1e-12)

    print("2. curvature diagnostic")
    x = np.linspace(-1, 1, 5)
    check("zero for a straight line", _curvature(2 * x + 1, x) < 1e-12)
    check("non-zero for a quadratic", _curvature(x ** 2, x) > 0.2, f"{_curvature(x**2, x):.3f}")

    print("3. calibration")
    n, nA = 500, 4
    ages = np.array([0, 100, 200, 300])
    rng = np.random.default_rng(0)
    amp = rng.lognormal(0, 0.3, n)                    # a pure amplitude perturbation
    base = np.array([2.0, 3.0, 4.0, 5.0])
    Q = (amp[:, None] * base[None, :])[:, :, None] * np.ones((1, 1, 3))
    SCF = Q * 0.02
    SF = Q * 0.004
    per = _calibrate(Q, SCF, SF, ages, "per-member", 20.0, None)
    fix = _calibrate(Q, SCF, SF, ages, "fixed", 20.0, None)
    spread_per = np.median(np.std(per["sediment"][:, :, 0], 0) / np.mean(per["sediment"][:, :, 0], 0))
    spread_fix = np.median(np.std(fix["sediment"][:, :, 0], 0) / np.mean(fix["sediment"][:, :, 0], 0))
    check("per-member calibration removes a pure amplitude perturbation",
          spread_per < 1e-12, f"CV {spread_per:.2e}")
    check("fixed calibration retains it", spread_fix > 0.2, f"CV {spread_fix:.3f}")
    check("present-day reference hits the target",
          abs(np.median(per["sediment"][:, 0, 0]) - 20.0) < 1e-9)
    check("silicate anchor is preserved per member",
          np.std(per["silicate"][:, 0, 0]) < 1e-12)

    print("4. a shape perturbation survives calibration")
    tilt = rng.normal(0, 0.2, n)
    Q2 = (base[None, :] ** (1 + tilt[:, None]))[:, :, None] * np.ones((1, 1, 3))
    per2 = _calibrate(Q2, Q2 * 0.02, Q2 * 0.004, ages, "per-member", 20.0, None)
    cv = np.std(per2["sediment"][:, -1, 0]) / np.mean(per2["sediment"][:, -1, 0])
    check("shape change is not removed", cv > 0.05, f"CV at 300 Ma {cv:.3f}")

    print("5. stub surrogate is well formed")
    s = StubSurrogate()
    e = s(np.linspace(0, 3000, 64 * 96).reshape(64, 96).astype("f4"),
          np.full((64, 96), 0.5, "f4"))
    check("shape and sign", e.shape == (64, 96) and (e <= 0).all(), str(e.dtype))
    e2 = s(np.linspace(0, 3000, 64 * 96).reshape(64, 96).astype("f4"),
           np.full((64, 96), 1.0, "f4"))
    check("monotone in runoff", (e2 <= e + 1e-12).all())

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------

def main(argv=None):
    """Command-line entry point: sub-commands nodes, sample and selftest."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("nodes", help="stage A: evaluate the surrogate + Bufe on the (r,z) grid")
    q.add_argument("--erodep-dir", required=True)
    q.add_argument("--temp-dir", required=True)
    q.add_argument("--sensitivity", default="phanda_sensitivity.npz")
    q.add_argument("--ckpt", help="NeuroLEM surrogate checkpoint (best.pt)")
    q.add_argument("--params-json", required=True)
    q.add_argument("--out", default="mc_nodes.npz")
    q.add_argument("--ages", nargs="*")
    q.add_argument("--n-r", type=int, default=5)
    q.add_argument("--n-z", type=int, default=3)
    # z nodes span +-z_max standard deviations of the climate-model spread
    q.add_argument("--z-max", type=float, default=2.5)
    q.add_argument("--device")
    q.add_argument("--stub-surrogate", action="store_true",
                   help="plumbing test only: replace the UNet with a cheap analytic stand-in")
    q.add_argument("--checkpoint-dir",
                   help="per-age checkpoints (default: <out>.ckpt); completed ages are "
                        "reloaded, so an interrupted sweep resumes")
    q.add_argument("--time-budget", type=float,
                   help="stop after this many seconds and write what is done")
    q.add_argument("--reference-tolerance", type=float, default=1e-6,
                   help="max RMSE (m/yr) allowed at the (r=0, z=0) node against the archived "
                        "ero_scotese / ero_zhou before the run aborts")
    q.add_argument("--envelope", choices=["csv", "factor", "records"], default="factor")
    q.add_argument("--envelope-csv")
    q.add_argument("--envelope-factor", type=float, default=1.5)
    q.add_argument("--envelope-data-dir")
    # sigma components: internal variability, regression residual, model structure
    q.add_argument("--sigma-terms", default="int,fit,struct")
    q.add_argument("--runoff-scale-clip", type=float, nargs=2, default=[0.2, 5.0])
    # epsilon in R/R_ref = (P'/P'_ref)**epsilon; 1 because the forcing channel is
    # precipitation itself
    q.add_argument("--runoff-elasticity", type=float, default=1.0)
    q.add_argument("--runoff-coefficient", type=float, default=0.4)
    # erosion scaling factor of the main model (sediment-flux calibrated) and the
    # silicate and sulfide amplitude factors of the weathering step
    q.add_argument("--erate-scale", type=float, default=2.0)
    q.add_argument("--f-sil", type=float, default=2.4)
    q.add_argument("--f-sulf", type=float, default=0.1)
    q.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    q.set_defaults(func=cmd_nodes)

    q = sub.add_parser("sample", help="stage B: draw members, calibrate, band, plot")
    q.add_argument("--nodes", default="mc_nodes.npz")
    q.add_argument("--out", required=True)
    q.add_argument("--n", type=int, default=2000)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--calibration", choices=["per-member", "fixed"], default="per-member")
    q.add_argument("--percentiles", type=float, nargs="+", default=[5, 50, 95])
    q.add_argument("--co2-time-correlation", choices=["independent", "full"],
                   default="independent")
    q.add_argument("--sigma-time-correlation", choices=["independent", "full"], default="full")
    q.add_argument("--present-day-flux-gt", type=float, default=20.0)
    q.add_argument("--data-dir", help="dir holding foster.npy / judd.npy")
    q.add_argument("--corr-intervals", type=float, nargs="+", default=[200, 300])
    q.add_argument("--no-figures", action="store_true")
    q.add_argument("--allow-stub", action="store_true")
    q.set_defaults(func=cmd_sample)

    q = sub.add_parser("selftest", help="synthetic checks of interpolation and calibration")
    q.set_defaults(func=cmd_selftest)

    a = p.parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a geochemically constrained palaeoelevation-correction ensemble.

Role in the paper
-----------------
This is the generating half of the 200-member palaeoelevation ensemble
(Methods, "Palaeoelevation" and "Uncertainty and sensitivity", item 5;
Supplementary Information, "Uncertainty ensembles"), which "resamples the
mohometry uncertainty, six isostatic density structures and the sample
weights". It propagates uncertainty in the geochemical observations through
the rock-chemistry assimilation applied to the PALEOMAP (Scotese & Wright)
reconstruction.  It does *not* add pixel noise to the finished DEM and it does
not vary the kinematic prior or the assimilation hyperparameters.  The member
corrections are passed through NeuroLEM and the weathering calculation by
``nature_recast/apply_ensemble_sensitivity.py`` (or ``apply_geochem_ensemble.py``).

What is varied
--------------
Each sample's elevation comes from its crustal thickness Z_c through the
time-dependent Airy relation H = (rho_m(t) - rho_c(t)) / rho_m(t) * (Z_c - Z_0),
with crustal and lithospheric-mantle densities set by the mantle potential
temperature of one of six published thermal histories (Supplementary Methods).
The assimilation's sample table carries, for every history, the elevation (km)
implied by the low, central and high thickness estimates.  For each ensemble
member this program varies only quantities permitted by that table:

1. one of the six published isostatic conversions is selected for the whole
   member (a spatially coherent structural choice; the central product uses
   their mean);
2. every sample is perturbed within its model-specific low/mid/high
   crustal-thickness envelope (the bounds are treated as approximately 2
   standard deviations and draws are truncated at those bounds);
3. the recorded missing-data risk contributes its documented elevation error;
4. an exponential Bayesian-bootstrap multiplier perturbs sample weights,
   representing finite and uneven geochemical sampling.

The random draw attached to a geochemical sample is invariant through time.
Thus a sample used by several adjacent time windows carries the same draw in
all of them and does not create artificial temporal flicker.

The program keeps the Scotese prior, plate reconstruction, temporal kernel,
province grid, masks, spatial length scales, smoothing and correction caps
fixed.  Unsupported cells therefore retain the prior and acquire essentially
no ensemble spread.

The member chain repeats the stages of assimilate_scotese.assimilate_one: declustering (with the
samples that sit on PALEOMAP sea dropped when the assimilation code drops them), province rescaling,
the local-support and province-edge tapers, cross-boundary smoothing, the residual kernel with its
reach taper (RESID_REACH_KM), finalisation and the young-age taper. Each guarded step is applied only
when the loaded assimilation code defines it, so the chain reproduces whichever product the code
builds. Before any member is drawn, the chain run on the unperturbed observations must reproduce the
production grid of --central-dir to PARITY_TOL_M; otherwise nothing is written.

Inputs
------
The assimilation code and data of the pinned paleotopo-data-assimilation
checkout (see below): the 1-degree PALEOMAP grid M_orig of each age (m), the
geochemical sample table with its per-model elevation columns, the plate
model and continent masks.  ``--central-dir`` holds the production grids
<age>Ma_corrected_SW.nc (M_orig, M_corrected, delta in m) of the adopted
surface.

Outputs
-------
Two compact files can be written per age, all fields on the 1-degree grid in
metres:

  <age>Ma_geochem_ensemble_summary.nc
      delta_lower_p05, delta_median_p50, delta_upper_p95, delta_ensemble_mean
      and delta_ensemble_sd of the correction, plus delta_production_central,
      the production correction; int16 at 1 m precision.  Elevation is
      recovered by adding any correction field to M_orig.  Also continent_mask
      and, as attributes, the dependency provenance, member chain and parity.

  <age>Ma_geochem_ensemble_members.nc          (--write-members)
      the temporary member cube used by NeuroLEM: delta_member (member, lat,
      lon), rounded to one metre and packed as compressed int16;
      isostatic_model_index (member); delta_central (float32).  The NeuroLEM
      controller is expected to delete this file after it has processed the
      age.

and one table for all ages processed so far, geochem_elevation_ensemble_summary.csv
in --out-dir: per age, the land-mean and RMS of the central correction, the
ensemble-mean correction, mean standard deviation, mean and 95th-percentile
p05-p95 width, the parity of the member chain and the fraction of continental
cells where the central correction lies outside p05-p95 (all over the
continent mask, m).  Rows of re-run ages are replaced.

Examples
--------
Run a small scientific/technical check::

    PALEOTOPO_PROJECT_ROOT=/path/Paleotopo_data_assimilation \
      python build_geochem_elevation_ensemble.py 220 --members 12

Write the 200-member hand-off cube for NeuroLEM::

    PALEOTOPO_PROJECT_ROOT=/path/Paleotopo_data_assimilation \
      python build_geochem_elevation_ensemble.py 220 --members 200 \
      --write-members --members-out /temporary/220Ma_members.nc

The reported "lower" and "upper" products are p05 and p95.  Literal sample
minima/maxima are deliberately not used because they change with ensemble
size and are not uncertainty bounds with a stable statistical meaning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pinned dependency on the paleotopography assimilation
# ---------------------------------------------------------------------------
# This script belongs to the CO2 paper but reuses the assimilation machinery
# published with the paleotopography paper.  That code is a PINNED external
# dependency, not vendored: one source of truth, at a fixed revision.
#
#   repository  https://github.com/EarthByte/paleotopo-data-assimilation
#   pinned tag  see PALEOTOPO_PIN below
#
# Resolution order for the scripts directory:
#   1. $PALEOTOPO_SCRIPTS            - explicit path, wins if set
#   2. $PALEOTOPO_PROJECT_ROOT/public_repo_scotese/scripts
#   3. $PALEOTOPO_PROJECT_ROOT/scripts_Scotese
#
# The published clone layout is tried before the development layout so that a
# reader who checks out the pinned tag gets byte-identical code.  Data still
# resolves through paths_scotese.PROJECT_ROOT, so PALEOTOPO_PROJECT_ROOT should
# point at the tree that actually holds data/.
PALEOTOPO_PIN = "v1.1.0"            # tag in paleotopo-data-assimilation (reach taper, samples on PALEOMAP sea dropped)


def _resolve_paleotopo_scripts() -> Path:
    """Directory holding assimilate_scotese.py, in the resolution order above."""
    explicit = os.environ.get("PALEOTOPO_SCRIPTS")
    if explicit:
        d = Path(explicit).expanduser().resolve()
        if not (d / "assimilate_scotese.py").exists():
            raise SystemExit(
                f"PALEOTOPO_SCRIPTS={d} does not contain assimilate_scotese.py")
        return d
    root = os.environ.get("PALEOTOPO_PROJECT_ROOT")
    if not root:
        raise SystemExit(
            "Set PALEOTOPO_PROJECT_ROOT to a checkout of "
            "https://github.com/EarthByte/paleotopo-data-assimilation "
            f"at tag {PALEOTOPO_PIN} (or set PALEOTOPO_SCRIPTS directly).")
    root = Path(root).expanduser().resolve()
    for cand in (root / "public_repo_scotese" / "scripts",
                 root / "scripts_Scotese"):
        if (cand / "assimilate_scotese.py").exists():
            return cand
    raise SystemExit(
        f"No assimilate_scotese.py under {root}; expected in "
        "public_repo_scotese/scripts or scripts_Scotese.")


def paleotopo_provenance(scripts_dir: Path) -> str:
    """Describe the dependency actually loaded: tag, commit and cleanliness.

    Recorded in every output file.  Provenance that lives only in a pin
    constant is an assertion; provenance read from the checkout is evidence.
    """
    import subprocess
    def _git(*a):
        try:
            return subprocess.run(("git", "-C", str(scripts_dir)) + a,
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = _git("rev-parse", "--short=12", "HEAD") or "unknown"
    described = _git("describe", "--tags", "--always", "--dirty") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    if described != PALEOTOPO_PIN and commit != "unknown":
        print(f"  [paleotopo] WARNING: checkout is {described}, "
              f"pin is {PALEOTOPO_PIN}", flush=True)
    if dirty:
        print("  [paleotopo] WARNING: dependency checkout has uncommitted "
              "changes; results are not reproducible from the pin", flush=True)
    return (f"paleotopo-data-assimilation pin={PALEOTOPO_PIN} "
            f"checkout={described} commit={commit} "
            f"dirty={'yes' if dirty else 'no'} path={scripts_dir}")


_PALEOTOPO_SCRIPTS = _resolve_paleotopo_scripts()
sys.path.insert(0, str(_PALEOTOPO_SCRIPTS))

import assimilate_scotese as A            # noqa: E402
from paths_scotese import PROJECT_ROOT    # noqa: E402
from plate_model_utils_scotese import cob_mask          # noqa: E402
from sw_io import load_grid as sw_load_grid, nearest_cell_index  # noqa: E402

PALEOTOPO_PROVENANCE = paleotopo_provenance(_PALEOTOPO_SCRIPTS)
# Human-readable list of the assimilation stages the member chain repeats, built from
# the switches of the loaded code and recorded in every output file.
_reach = getattr(A, "RESID_REACH_KM", None)
MEMBER_CHAIN = ("decluster" + ("; samples on PALEOMAP sea dropped" if getattr(A, "DROP_SEA_SITED_SAMPLES", False) else "")
                + "; province rescale; local-support and province-edge tapers; cross-boundary smoothing; residual kernel"
                + (f" with reach taper {_reach[0]:g}-{_reach[1]:g} km" if _reach and _reach[1] > 0 else "")
                + "; finalise; young-age taper")


# Labels of the six isostatic conversions (mantle thermal histories of Herzberg et al.
# for Urey ratios 0.08, 0.23 and 0.38, Brown et al., Davies and Condie et al.;
# Supplementary Methods), in the column order of A.ELEV_MODELS_* below.
MODEL_NAMES = (
    "Herzberg_0.23",
    "Brown_2022",
    "Davis_2009",
    "Condie_2016",
    "Herzberg_0.08",
    "Herzberg_0.38",
)

# Columns are aligned by model: low[i], mid[i], high[i].
MODEL_MID = tuple(A.ELEV_MODELS_MID)
MODEL_LOW = tuple(A.ELEV_MODELS_LOW)
MODEL_HIGH = tuple(A.ELEV_MODELS_HIGH)

DEFAULT_MEMBERS = 200
DEFAULT_SEED = 20260814     # fixed so every age (and every re-run) draws the same sample perturbations
DEFAULT_QUANTILES = (0.05, 0.50, 0.95)


def _closed_continent_mask(t, lat, lon, M):
    """Continental mask (bool, lat x lon) of age ``t``, morphologically closed as in the assimilation."""
    cont = cob_mask(t, lat, lon, M)
    if A.CONT_MASK_CLOSING_ITER and A.CONT_MASK_CLOSING_ITER > 0:
        from scipy.ndimage import binary_closing
        cont = binary_closing(cont, iterations=A.CONT_MASK_CLOSING_ITER)
    return np.asarray(cont, dtype=bool)


def _balanced_model_indices(n_members: int, seed: int) -> np.ndarray:
    """Nearly equal representation of all six structural models.

    Returns an int8 array of length ``n_members``: indices 0..5 repeated and
    cut to length (counts differ by at most one), then shuffled with ``seed``.
    """
    idx = np.tile(np.arange(len(MODEL_NAMES), dtype=np.int8),
                  int(math.ceil(n_members / len(MODEL_NAMES))))[:n_members]
    np.random.default_rng(seed).shuffle(idx)
    return idx


def _draw_matrices(n_members: int, n_samples: int, seed: int):
    """Generate temporally invariant sample-level uncertainty draws.

    The matrices are only about 240 MB for 200 members x ~100,000 samples,
    comfortably within a 64-GB machine.  Recreating them with the
    same seed for each age produces exactly the same sample draws, allowing
    the two-environment, one-age-at-a-time controller to remain stateless.

    Returns three float32 arrays of shape (n_members, n_samples), indexed by
    the stable sample identifier: ``thick_z`` and ``quality_z``, standard
    normal draws truncated at +/-2, for the thickness envelope and the
    missing-data error; ``bootstrap``, Exp(1) Bayesian-bootstrap weight
    multipliers clipped to [1e-4, 10].
    """
    rng = np.random.default_rng(seed)
    shape = (n_members, n_samples)
    thick_z = rng.standard_normal(shape, dtype=np.float32)
    np.clip(thick_z, -2.0, 2.0, out=thick_z)
    quality_z = rng.standard_normal(shape, dtype=np.float32)
    np.clip(quality_z, -2.0, 2.0, out=quality_z)
    bootstrap = rng.exponential(1.0, shape).astype(np.float32)
    # Avoid a vanishing or pathological single multiplier in a finite
    # numerical ensemble.  The interval contains >99.99% of Exp(1) draws.
    np.clip(bootstrap, 1.0e-4, 10.0, out=bootstrap)
    return thick_z, quality_z, bootstrap


PARITY_TOL_M = 0.5      # m; the unperturbed member chain against the production grid of --central-dir


def _decluster(df: pd.DataFrame, lat, lon, M):
    """assimilate_one's declustering: A.decluster, then (when the code drops them) the declustered
    samples whose PALEOMAP cell is at or below sea level (M_orig <= 0) are removed."""
    d = A.decluster(df, lat, lon)
    if getattr(A, "DROP_SEA_SITED_SAMPLES", False) and not d.empty:
        d = d[M[d["iy"].astype(int).values, d["ix"].astype(int).values] > 0].reset_index(drop=True)
    return d


def _prepare_age(age: int):
    """Prepare all age-invariant objects once, before looping over members.

    Returns a dict with the PALEOMAP grid M (m) and its lat/lon, the continent
    mask, the samples restored to this age and on continental cells (df), their
    declustered central set (dec0), the province raster built from dec0 (P_g)
    and the local-support and province-edge taper weights (or None when the
    assimilation disables them).  These structural objects are held fixed for
    all members, so members differ only in the sample values and weights.
    """
    M, lat, lon = sw_load_grid(age)
    M = np.asarray(M, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    cont = _closed_continent_mask(age, lat, lon, M)

    if age == 0:
        return dict(age=age, M=M, lat=lat, lon=lon, cont=cont,
                    df=pd.DataFrame(), dec0=pd.DataFrame(), P_g=None,
                    w_local=None, w_edge=None)

    geochem = A.get_geochem()
    # Stable identifiers are assigned after the one-time oceanic-sample drop.
    # get_geochem() always preserves this row order.
    geochem["_ensemble_sample_id"] = np.arange(len(geochem), dtype=np.int32)
    df = A.prepare_samples(geochem, age)

    if not df.empty:
        # keep only samples whose palaeoposition falls on the continent mask
        iy = nearest_cell_index(lat, df["rlat"].values)
        ix = nearest_cell_index(lon, df["rlon"].values)
        df = df[cont[iy, ix]].reset_index(drop=True)

    PL, PI, _POOLS, GRIDFN, _CLASSIFIER = A._active_classifier()
    if df.empty:
        P_g = np.where(cont, PI.get("Other", 0), -1).astype(np.int8)
        return dict(age=age, M=M, lat=lat, lon=lon, cont=cont, df=df,
                    dec0=pd.DataFrame(), P_g=P_g, w_local=None, w_edge=None)

    dec0 = _decluster(df, lat, lon, M)
    if dec0.empty:                      # every declustered sample sat on the prior's sea: the prior stands
        P_g = np.where(cont, PI.get("Other", 0), -1).astype(np.int8)
        return dict(age=age, M=M, lat=lat, lon=lon, cont=cont, df=pd.DataFrame(),
                    dec0=dec0, P_g=P_g, w_local=None, w_edge=None)
    P_g = GRIDFN(age, lat, lon, cont, dec0, M)
    P_g = A.despeckle_province_raster(P_g, cont)

    if A.LOCAL_SUPPORT_KM and A.LOCAL_SUPPORT_KM > 0:
        # weight decreasing linearly from 1 at a sample to 0 at LOCAL_SUPPORT_KM
        # (400 km in the adopted product)
        dist_km = A.nearest_sample_distance_grid(lat, lon, dec0)
        w_local = np.clip(1.0 - dist_km / A.LOCAL_SUPPORT_KM, 0.0, 1.0)
    else:
        w_local = None

    if A.PROV_TAPER_CELLS and A.PROV_TAPER_CELLS > 0:
        w_edge = A.province_edge_taper(P_g, cont, A.PROV_TAPER_CELLS)
    else:
        w_edge = None

    return dict(age=age, M=M, lat=lat, lon=lon, cont=cont, df=df,
                dec0=dec0, P_g=P_g, w_local=w_local, w_edge=w_edge)


def _finish_member(prep, df_member: pd.DataFrame):
    """Run the member-dependent stages with all structural choices fixed.

    ``df_member`` carries the member's sample elevations (z_obs_m) and weights
    (w_raw).  Returns the member's adjusted elevation grid M_final (m, 1-degree);
    the member correction is M_final - M.  Each step mirrors
    assimilate_scotese.assimilate_one and is applied only if the loaded
    assimilation enables it.
    """
    age = prep["age"]
    M, lat, lon, cont = prep["M"], prep["lat"], prep["lon"], prep["cont"]
    if age == 0 or df_member.empty:
        return M.copy()

    dec = _decluster(df_member, lat, lon, M)
    if dec.empty:
        return M.copy()
    # quantile-by-quantile remapping of the reference hypsometry within each province
    M3_raw = A.province_rescale(M, prep["P_g"], dec, cont,
                                n_min_p=A.N_MIN_P)
    if prep["w_local"] is not None:
        M3 = M + prep["w_local"] * (M3_raw - M)
    else:
        M3 = M3_raw

    if prep["w_edge"] is not None:
        M3 = M + prep["w_edge"] * (M3 - M)

    if A.CROSS_BOUNDARY_SIGMA and A.CROSS_BOUNDARY_SIGMA > 0:
        dz_prov = A._masked_gaussian_smooth(M3 - M, cont,
                                             A.CROSS_BOUNDARY_SIGMA)
        M3 = M + dz_prov

    # residuals at the samples spread with a Gaussian kernel of length RESID_LS_KM
    residual, _n_eff = A.smoothed_residual(
        M3, prep["P_g"], dec, lat, lon, cont, ls_km=A.RESID_LS_KM)
    reach = getattr(A, "RESID_REACH_KM", None)
    if reach and reach[1] > 0:
        # the residual's reach taper, exactly as assimilate_one applies it
        d0, d1 = reach
        dist_r = A.nearest_sample_distance_grid(lat, lon, dec)
        residual = residual * np.clip((d1 - dist_r) / max(d1 - d0, 1e-6), 0.0, 1.0)
    # the assimilation's finalisation step (A.finalise), then, at young ages, the
    # taper of the correction towards zero at the present day
    M_final = A.finalise(M, M3 + residual, cont)
    if A.young_taper_ramp(age) < 1.0:
        M_final = A.apply_young_taper(age, M, M_final, cont)
    return M_final


def _member_observations(base_df: pd.DataFrame, model_idx: int,
                         thick_z: np.ndarray, quality_z: np.ndarray,
                         bootstrap: np.ndarray):
    """Return a member dataframe with geochem-permitted z and weight draws.

    ``model_idx`` selects the isostatic model; ``thick_z``, ``quality_z`` and
    ``bootstrap`` are this member's rows of the draw matrices, indexed by the
    stable sample identifier.  Sets ``z_obs_m`` (sample elevation, m) and
    multiplies the declustering weights ``w_raw`` by the bootstrap multipliers.
    """
    out = base_df.copy()
    sid = out["_ensemble_sample_id"].to_numpy(dtype=np.int64)

    mid_all = out[list(MODEL_MID)].astype(float)
    low_all = out[list(MODEL_LOW)].astype(float)
    high_all = out[list(MODEL_HIGH)].astype(float)
    fallback_mid = mid_all.mean(axis=1, skipna=True).to_numpy()
    fallback_low = low_all.mean(axis=1, skipna=True).to_numpy()
    fallback_high = high_all.mean(axis=1, skipna=True).to_numpy()

    mid = out[MODEL_MID[model_idx]].to_numpy(dtype=float)
    low = out[MODEL_LOW[model_idx]].to_numpy(dtype=float)
    high = out[MODEL_HIGH[model_idx]].to_numpy(dtype=float)
    # where the chosen model has no value for a sample, use the mean of the models that do
    mid = np.where(np.isfinite(mid), mid, fallback_mid)
    low = np.where(np.isfinite(low), low, fallback_low)
    high = np.where(np.isfinite(high), high, fallback_high)

    # Enforce the data-provided ordering without inventing a wider interval.
    low = np.minimum(low, mid)
    high = np.maximum(high, mid)
    # Asymmetric envelope: low and high are taken as -2 and +2 standard deviations,
    # so a negative draw scales by (mid - low)/2 and a positive one by (high - mid)/2.
    zt = thick_z[sid].astype(np.float64)
    scale_km = np.where(zt < 0.0, (mid - low) / 2.0,
                        (high - mid) / 2.0)
    z_km = mid + zt * scale_km

    # Missing-data risk of the sample (km, 0 where absent), converted to an elevation
    # error (m) with the assimilation's KAPPA_QUALITY factor and scaled by a truncated
    # standard normal draw.
    miss = out.get("Missing_Risk_DeltaT (km)",
                   pd.Series(0.0, index=out.index)).fillna(0.0).to_numpy(float)
    z_m = z_km * 1000.0 + quality_z[sid] * A.KAPPA_QUALITY * miss

    # Bayesian bootstrap: Exp(1) multipliers normalised to mean 1 over this age's samples
    wb = bootstrap[sid].astype(np.float64)
    finite_positive = np.isfinite(wb) & (wb > 0)
    if finite_positive.any():
        wb = wb / np.mean(wb[finite_positive])
    else:
        wb = np.ones_like(wb)

    out["z_obs_m"] = z_m
    out["w_raw"] = out["w_raw"].to_numpy(float) * wb
    return out


def _read_production_delta(age: int, M: np.ndarray, central_dir: Path | None):
    """Production correction (float32, m) of ``age`` from ``central_dir``, or None if absent.

    Reads the ``delta`` variable, else M_corrected - M_orig.  ``M`` is unused.
    """
    if central_dir is None:
        return None
    p = central_dir / f"{age}Ma_corrected_SW.nc"
    if not p.exists():
        return None
    with nc.Dataset(p) as ds:
        if "delta" in ds.variables:
            return np.asarray(ds.variables["delta"][:], dtype=np.float32)
        return (np.asarray(ds.variables["M_corrected"][:], dtype=np.float32)
                - np.asarray(ds.variables["M_orig"][:], dtype=np.float32))


def generate_age_ensemble(age: int, n_members: int = DEFAULT_MEMBERS,
                          seed: int = DEFAULT_SEED,
                          central_dir: str | Path | None = None):
    """Generate one age in memory and return member and summary corrections.

    The returned ``delta_members`` array has shape ``(member, lat, lon)`` and
    dtype float32.  It is only ~52 MB for 200 members on the 1-degree grid.

    Before any member is drawn, the chain is run on the unperturbed
    observations and compared with the production grid of ``central_dir``;
    a largest difference above PARITY_TOL_M stops the program.  The result
    dict also holds the p05/p50/p95, mean and standard deviation of the
    member corrections, the production central correction (recomputed from
    the unperturbed chain if no production grid exists), the model index of
    every member, the continent mask, the seed, the parity (m) and the wall
    time (s).
    """
    start = time.time()
    prep = _prepare_age(int(age))
    M = prep["M"]
    model_indices = _balanced_model_indices(n_members, seed)

    # the chain on the unperturbed observations must give the production grid before any member is drawn
    parity_m = 0.0
    cdir = Path(central_dir) if central_dir is not None else None
    if int(age) != 0 and cdir is not None and (cdir / f"{age}Ma_corrected_SW.nc").exists():
        with nc.Dataset(cdir / f"{age}Ma_corrected_SW.nc") as ds:
            M_prod = np.asarray(ds.variables["M_corrected"][:], dtype=np.float64)
        parity_m = float(np.nanmax(np.abs(_finish_member(prep, prep["df"]) - M_prod)))
        print(f"  [{age:>3} Ma] unperturbed member chain against {cdir.name}: {parity_m:.4f} m", flush=True)
        if parity_m > PARITY_TOL_M:
            raise SystemExit(f"[{age} Ma] the member chain does not reproduce {cdir.name} "
                             f"(largest difference {parity_m:.2f} m > {PARITY_TOL_M} m): the assimilation code loaded from "
                             f"{_PALEOTOPO_SCRIPTS} is not the code that built the production grids, or a stage is missing "
                             "from _finish_member. Nothing written.")

    if int(age) == 0 or prep["df"].empty:
        members = np.zeros((n_members,) + M.shape, dtype=np.float32)
    else:
        n_samples = int(prep["df"]["_ensemble_sample_id"].max()) + 1
        # The IDs span the full post-ocean-drop table, not just this age.
        # get_geochem() is cached, so this is cheap and guarantees adequate
        # matrix width for every selected row.
        n_samples = max(n_samples, len(A.get_geochem()))
        thick, quality, boot = _draw_matrices(n_members, n_samples, seed)
        members = np.empty((n_members,) + M.shape, dtype=np.float32)
        for j in range(n_members):
            dfj = _member_observations(prep["df"], int(model_indices[j]),
                                       thick[j], quality[j], boot[j])
            members[j] = (_finish_member(prep, dfj) - M).astype(np.float32)
            if (j + 1) % max(1, min(20, n_members // 5)) == 0 or j + 1 == n_members:
                print(f"  [{age:>3} Ma] ensemble member {j+1}/{n_members}")
        del thick, quality, boot

    q05, q50, q95 = np.quantile(members, DEFAULT_QUANTILES, axis=0)
    dmean = members.mean(axis=0, dtype=np.float64).astype(np.float32)
    dsd = members.std(axis=0, dtype=np.float64, ddof=1 if n_members > 1 else 0).astype(np.float32)

    dcentral = _read_production_delta(int(age), M, cdir)
    if dcentral is None:
        dcentral = (_finish_member(prep, prep["df"]) - M).astype(np.float32)

    result = dict(
        age=int(age), lat=prep["lat"].astype(np.float32),
        lon=prep["lon"].astype(np.float32),
        continent_mask=prep["cont"], model_index=model_indices,
        delta_members=members, delta_p05=q05.astype(np.float32),
        delta_p50=q50.astype(np.float32), delta_p95=q95.astype(np.float32),
        delta_mean=dmean, delta_sd=dsd, delta_central=dcentral,
        random_seed=int(seed), parity_m=parity_m,
        wall_s=time.time() - start,
    )
    return result


def _packed_i2(ds, name, dims, values, long_name, units="m",
               complevel=6, chunksizes=None):
    """Write metre-valued data as shuffled/DEFLATE-compressed int16.

    Values are rounded to whole metres and clipped to +/-32767; -32768 is the
    fill value.  scale_factor = 1 and add_offset = 0 are recorded, so readers
    that apply CF unpacking recover metres directly.
    """
    arr = np.asarray(values)
    packed = np.rint(arr).clip(-32767, 32767).astype(np.int16)
    kwargs = dict(zlib=True, complevel=complevel, shuffle=True)
    if chunksizes is not None:
        kwargs["chunksizes"] = chunksizes
    v = ds.createVariable(name, "i2", dims, fill_value=np.int16(-32768), **kwargs)
    v.set_auto_maskandscale(False)
    v[:] = packed
    v.units = units
    v.long_name = long_name
    v.scale_factor = np.float32(1.0)
    v.add_offset = np.float32(0.0)
    v.packing = "round(value_m); 1 m precision"
    return v


def write_summary(result, path: str | Path):
    """Write the per-age summary netCDF (quantiles, mean, sd and production correction, int16 m)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        nlat, nlon = len(result["lat"]), len(result["lon"])
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)
        ds.createVariable("lat", "f4", ("lat",))[:] = result["lat"]
        ds.createVariable("lon", "f4", ("lon",))[:] = result["lon"]
        chunks = (min(181, nlat), min(361, nlon))
        fields = (
            ("delta_lower_p05", "delta_p05", "5th percentile geochemical elevation correction"),
            ("delta_median_p50", "delta_p50", "median geochemical elevation correction"),
            ("delta_upper_p95", "delta_p95", "95th percentile geochemical elevation correction"),
            ("delta_ensemble_mean", "delta_mean", "ensemble-mean geochemical elevation correction"),
            ("delta_ensemble_sd", "delta_sd", "ensemble standard deviation of elevation correction"),
            ("delta_production_central", "delta_central", "published production central correction"),
        )
        for out_name, key, label in fields:
            _packed_i2(ds, out_name, ("lat", "lon"), result[key], label,
                       chunksizes=chunks)
        vm = ds.createVariable("continent_mask", "i1", ("lat", "lon"),
                               zlib=True, complevel=6, shuffle=True,
                               chunksizes=chunks)
        vm[:] = result["continent_mask"].astype(np.int8)
        ds.target_age_Ma = float(result["age"])
        ds.n_members = int(len(result["model_index"]))
        ds.random_seed = int(result["random_seed"])
        ds.quantiles = "0.05, 0.50, 0.95"
        ds.uncertainty_scope = (
            "geochemical observation uncertainty only: six isostatic models, "
            "sample low/mid/high thickness envelope, missing-data risk, and "
            "Bayesian-bootstrap sample weights; prior and assimilation settings fixed"
        )
        ds.history = "created by scripts/build_geochem_elevation_ensemble.py"
        ds.paleotopo_dependency = PALEOTOPO_PROVENANCE
        ds.member_chain = MEMBER_CHAIN
        ds.member_chain_parity_m = float(result["parity_m"])
    return path


def write_members(result, path: str | Path, seed: int):
    """Write the temporary, compressed 1-degree NeuroLEM hand-off cube."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nm, nlat, nlon = result["delta_members"].shape
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("member", nm)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)
        ds.createVariable("lat", "f4", ("lat",))[:] = result["lat"]
        ds.createVariable("lon", "f4", ("lon",))[:] = result["lon"]
        vi = ds.createVariable("isostatic_model_index", "i1", ("member",),
                               zlib=True, complevel=6, shuffle=True)
        vi[:] = result["model_index"]
        vi.model_names = json.dumps(MODEL_NAMES)
        _packed_i2(ds, "delta_member", ("member", "lat", "lon"),
                   result["delta_members"],
                   "member geochemical elevation correction",
                   chunksizes=(1, min(181, nlat), min(361, nlon)))
        vc = ds.createVariable("delta_central", "f4", ("lat", "lon"), zlib=True, complevel=6, shuffle=True)
        vc[:] = np.asarray(result["delta_central"], np.float32)
        vc.long_name = "production central correction (unpacked), for the consistency check against the archived erosion field"
        vc.units = "m"
        ds.target_age_Ma = float(result["age"])
        ds.n_members = int(nm)
        ds.random_seed = int(seed)
        ds.temporary_handoff = "safe to delete after NeuroLEM processes this age"
        ds.uncertainty_scope = (
            "geochemical observation uncertainty only; no perturbation of the "
            "Scotese prior or assimilation hyperparameters"
        )
        ds.history = "created by scripts/build_geochem_elevation_ensemble.py"
        ds.paleotopo_dependency = PALEOTOPO_PROVENANCE
        ds.member_chain = MEMBER_CHAIN
        ds.member_chain_parity_m = float(result["parity_m"])
    return path


def _summary_row(result):
    """One row of the summary CSV: land statistics (m) over the continent mask of one age."""
    cont = result["continent_mask"]
    width = result["delta_p95"] - result["delta_p05"]
    dc = result["delta_central"]
    return dict(
        age_Ma=result["age"], n_members=len(result["model_index"]),
        central_delta_mean_m=float(np.mean(dc[cont])),
        central_delta_rms_m=float(np.sqrt(np.mean(dc[cont] ** 2))),
        ensemble_delta_mean_m=float(np.mean(result["delta_mean"][cont])),
        ensemble_sd_mean_m=float(np.mean(result["delta_sd"][cont])),
        p05_p95_width_mean_m=float(np.mean(width[cont])),
        p05_p95_width_p95_m=float(np.percentile(width[cont], 95)),
        member_chain_parity_m=float(result["parity_m"]),
        central_below_p05_fraction=float(np.mean(dc[cont] < result["delta_p05"][cont])),
        central_above_p95_fraction=float(np.mean(dc[cont] > result["delta_p95"][cont])),
        wall_s=float(result["wall_s"]),
    )


def main():
    """Build the ensemble for each age on the command line and write its outputs."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ages", nargs="+", type=int,
                   help="one or more ages in Ma (5-Myr S&W cadence)")
    p.add_argument("--members", type=int, default=DEFAULT_MEMBERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out-dir", default=None,
                   help="summary output directory (default: PROJECT_ROOT/data/geochem_elevation_ensemble)")
    p.add_argument("--central-dir", default=None,
                   help="production corrected-grid directory: the exact central member, and the grid the unperturbed "
                        "member chain must reproduce; default: PROJECT_ROOT/data/corrected_scotese_reach500_land")
    p.add_argument("--write-members", action="store_true",
                   help="also write the temporary packed member cube for NeuroLEM")
    p.add_argument("--members-out", default=None,
                   help="member-cube path; only valid for one age. Default is under out-dir/handoff")
    args = p.parse_args()

    if args.members < 6:
        p.error("--members must be at least 6 so every isostatic model is represented")
    if args.members_out and len(args.ages) != 1:
        p.error("--members-out can only be used with one age")

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "data" / "geochem_elevation_ensemble")
    central_dir = Path(args.central_dir) if args.central_dir else (
        PROJECT_ROOT / "data" / "corrected_scotese_reach500_land")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for age in args.ages:
        result = generate_age_ensemble(age, args.members, args.seed, central_dir)
        summary_path = out_dir / f"{age}Ma_geochem_ensemble_summary.nc"
        write_summary(result, summary_path)
        print(f"wrote {summary_path} ({summary_path.stat().st_size / 1e6:.2f} MB)")

        if args.write_members:
            members_path = (Path(args.members_out) if args.members_out else
                            out_dir / "handoff" / f"{age}Ma_geochem_ensemble_members.nc")
            write_members(result, members_path, args.seed)
            print(f"wrote {members_path} ({members_path.stat().st_size / 1e6:.2f} MB)")
        rows.append(_summary_row(result))

    csv_path = out_dir / "geochem_elevation_ensemble_summary.csv"
    new = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        new = pd.concat([old[~old["age_Ma"].isin(new["age_Ma"])], new],
                        ignore_index=True)
    new.sort_values("age_Ma").to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply NeuroLEM and a weathering CO2 calculation to every palaeoelevation ensemble member.

This is the consuming half of the geochemical palaeoelevation-uncertainty
experiment (Methods, "Uncertainty and sensitivity", item 5: a 200-member
ensemble of the elevation adjustment).  It reads one temporary, compressed
1-degree member cube written by ``build_geochem_elevation_ensemble.py``.  Every
member correction is bilinearly transferred to the 0.1-degree goSPL
physiography of the same age (the published goSPL input grids, Salles et al.,
2023), added to it exactly as ``apply_dem.py`` adds the central correction, and
passed independently through NeuroLEM.  For every member the program then
computes continental gross erosion and the silicate, sulfide and net inorganic
weathering CO2 fluxes of the West (2012) law as fitted by Bufe et al. (2024),
using the calculation of ``bufe_co2_from_dem.py`` (median regional fit, one
amplitude factor per phase, runoff = precipitation x a uniform runoff
coefficient).

The routines that load the model, transfer a member to 0.1 degree and run the
inference (``_load_model``, ``_prepare_constant_tensors``, ``_infer_member``,
``_area_m2``, ``_qero_km3yr``) are also imported by
``nature_recast/apply_ensemble_sensitivity.py``, which evaluates the same
members with the weathering formulation and calibration of the main model and
writes the ensemble results reported in the paper.

Inputs
------
--members-nc    <age>Ma_geochem_ensemble_members.nc: delta_member (member, lat,
                lon) in m on the 1-degree grid, isostatic_model_index (member),
                random seed and target age as attributes.
--salles-dir    salles_<age>Ma.npz: goSPL elevation (m) and precipitation (m/yr)
                on the 0.1-degree grid, 1801 x 3601 nodes.
--ckpt          NeuroLEM checkpoint (best.pt).
--temp-dir      <age>Ma.npz temperature fields of the HadCM3L simulations
                (Valdes et al., 2021), read with bufe_co2_from_dem._load_temp_K.
--params-json   regional weathering-law parameters of Bufe et al. (2024).

Outputs (under --out)
---------------------
member_tables/<age>Ma_members.csv
    one row per member: isostatic model; Qero_surface_km3yr (gross erosion over
    the member's own emergent land, km3/yr); Qero_common_reference_cells_km3yr
    (over the emergent land of the unadjusted reference surface);
    land_area_Mkm2 and reference_land_area_Mkm2 (million km2);
    net_co2_surface_GtCyr, scf_surface_GtCyr, sf_surface_GtCyr (net inorganic
    CO2 drawdown, silicate-cation and sulfide-sulfate terms, Gt C/yr, own land);
    net_co2_common_reference_cells_GtCyr (net drawdown on reference land).
age_summaries/<age>Ma_summary.csv
    p05, p50, mean, p95 and standard deviation of every metric across members.
spatial_quantiles/<age>Ma_spatial_quantiles.nc   (--spatial only)
    cellwise p05/p50/p95 of the erosion-deposition rate and of the net CO2
    flux (float32, compressed), the fraction of members in which each cell is
    land, and the reference land mask.

Storage policy
--------------
The full high-resolution member fields are never written.  A small CSV row is
kept for every member.  With ``--spatial`` the program retains the member
fields of this age in RAM, writes compressed float32 p05/p50/p95 maps and
releases the arrays before exit.  The calling controller deletes the 1-degree
handoff cube after this program succeeds.

The controller invokes this program once per age, which gives clean resume
semantics (an age whose member table and summary exist is skipped unless
``--force``) and caps disk usage at one small handoff cube plus the final
tables and maps.

Usage
-----
    python apply_geochem_ensemble.py --members-nc 220Ma_geochem_ensemble_members.nc \
        --salles-dir <data archive>/gospl_salles --ckpt <data archive>/surrogate_model/best.pt \
        --temp-dir <data archive>/valdes_temperature --params-json data/bufe_2024_params.json \
        --out <output directory> [--spatial]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from apply_dem import _device
from bufe_co2_from_dem import (
    BETA, DEFAULT_REGIONS, REACTIONS_USED, T_REF_K, _load_temp_K,
    median_fit, weathering_flux,
)


R_EARTH_M = 6.371e6     # m
GRID_DEG = 0.1          # spacing of the NeuroLEM grid, degrees
M_CAT = 30.0            # g/mol, mean molar mass of silicate cations (Methods, net inorganic flux)
M_SO4 = 96.06           # g/mol, sulfate


def _area_m2(lat, shape):
    """Spherical cell areas (m^2) of the regular 0.1-degree grid, broadcast to ``shape``.

    Area = R^2 * (0.1 deg in rad)^2 * cos(lat), one value per latitude row.
    """
    dg = np.deg2rad(GRID_DEG)
    row = (R_EARTH_M * R_EARTH_M * dg * dg) * np.cos(np.deg2rad(lat))
    return np.broadcast_to(row[:, None], shape)


def _load_model(ckpt_path, device):
    """Load the NeuroLEM checkpoint onto ``device`` in evaluation mode.

    Returns (model, checkpoint dict, norm), where ``norm`` holds the input
    standardisation (in_mean, in_std per channel: elevation, precipitation),
    the target transform (asinh_s, t_mean, t_std) and ``mult`` = 2**depth, the
    factor the padded grid must be divisible by.
    """
    import torch
    from salles_unet import GlobalUNet

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GlobalUNet(in_ch=ck["in_ch"], depth=ck["depth"],
                       base=ck["base"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = ck["norm"]
    return model, ck, dict(
        in_mean=np.asarray(norm["in_mean"], np.float32),
        in_std=np.asarray(norm["in_std"], np.float32),
        asinh_s=float(norm["asinh_s"]),
        t_mean=float(norm["t_mean"]),
        t_std=float(norm["t_std"]),
        mult=2 ** int(ck["depth"]),
    )


def _pad_torch(x, mult):
    """Match apply_dem._pad_to: replicate-pad bottom/right to model multiple."""
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph == 0 and pw == 0:
        return x, (h, w)
    return F.pad(x, (0, pw, 0, ph), mode="replicate"), (h, w)


def _prepare_constant_tensors(z_base, rain, norm, device):
    """Move the member-invariant inputs to ``device`` once per age.

    Returns the reference elevation (m, unnormalised, because every member adds
    its own correction to it) and the standardised precipitation channel.
    """
    import torch
    z = torch.from_numpy(np.asarray(z_base, np.float32)).to(device)
    r = torch.from_numpy(np.asarray(rain, np.float32)).to(device)
    rain_n = (r - float(norm["in_mean"][1])) / float(norm["in_std"][1])
    return z, rain_n


def _infer_member(model, delta_1deg, z_base_t, rain_n_t, target_shape,
                  norm, device):
    """Upsample one correction on-device and run one NeuroLEM inference.

    ``delta_1deg`` is the member correction (m) on the 1-degree grid (181, 361);
    ``target_shape`` is the 0.1-degree shape (1801, 3601). The member elevation
    is the reference goSPL elevation plus the upsampled correction, as in the
    default "delta" mode of apply_dem.py. Returns the erosion-deposition rate
    (float32, mm/yr, negative = erosion) and the member's emergent-land mask
    (elevation > 0), both of ``target_shape``.
    """
    import torch
    import torch.nn.functional as F

    d = torch.from_numpy(np.asarray(delta_1deg, np.float32))[None, None].to(device)
    # Both grids span the same endpoints and have exactly ten target intervals
    # per source interval, so align_corners=True is the same regular-grid
    # bilinear transfer used by apply_dem._bilinear (to rounding precision).
    d_hi = F.interpolate(d, size=target_shape, mode="bilinear",
                         align_corners=True)[0, 0]
    z_member = z_base_t + d_hi
    elev_n = ((z_member - float(norm["in_mean"][0])) /
              float(norm["in_std"][0]))
    x = torch.stack((elev_n, rain_n_t), dim=0)[None]
    xp, (h, w) = _pad_torch(x, norm["mult"])
    # invert the standardised asinh target transform of training (as in apply_dem.infer)
    with torch.inference_mode():
        pred = model(xp)[0, 0, :h, :w]
        erodep = norm["asinh_s"] * torch.sinh(
            pred * norm["t_std"] + norm["t_mean"])
    mask = (z_member > 0).to("cpu").numpy().astype(bool, copy=False)
    erodep = erodep.to("cpu").numpy().astype(np.float32, copy=False)
    return erodep, mask


def _qero_km3yr(erodep, mask, area):
    """Gross erosion (km^3/yr) integrated over the cells of ``mask``.

    Deposition is not netted off: only max(-erodep, 0) is summed. The
    surrogate's erodep field is in mm/yr (equivalently m/Myr), the unit of the
    goSPL erosion-rate fields it was trained on, so an area-weighted sum
    (mm/yr x m^2 = 1e-3 m^3/yr) is divided by 1e12 to give km^3/yr. A direct
    goSPL run of the 460 Ma reference surface gives 1.6 km^3/yr of gross
    erosion, matching this integral.
    """
    return float(np.sum(np.where(mask, np.maximum(-erodep, 0.0), 0.0) * area)
                 / 1.0e12)


def _qnet_km3yr(erodep, mask, area):
    """Signed net continental export (km^3/yr): erosion minus internal deposition.

    erodep is mm/yr; see _qero_km3yr for why the divisor is 1e12.
    """
    return -float(np.sum(np.where(mask, erodep, 0.0) * area) / 1.0e12)


def _weathering_two_masks(erodep_scaled, surface_mask, reference_mask,
                          q_mm, T_K, area, fits, f_sil, f_sulf):
    """Evaluate weathering once, then integrate over two reporting masks.

    This is algebraically equivalent to two calls to bufe_co2_from_dem._one_case
    but avoids repeating the expensive high-resolution exponential/power
    calculations for every ensemble member.

    Arguments: ``erodep_scaled`` is the erosion-deposition rate (mm/yr) after the
    erosion calibration factor; ``surface_mask`` the member's own emergent land;
    ``reference_mask`` the emergent land of the unadjusted reference surface;
    ``q_mm`` runoff (mm/yr); ``T_K`` temperature (K); ``area`` cell areas (m^2);
    ``fits`` the regional parameter fits per phase; ``f_sil``, ``f_sulf`` the
    silicate and sulfide amplitude factors. Depositing cells have zero erosion
    and therefore no weathering flux (Methods, "Depositional land").

    Returns (net CO2 drawdown, silicate-cation term, sulfide term) integrated
    over ``surface_mask`` and the net drawdown over ``reference_mask``, all in
    Gt C/yr, plus the cell field of the net flux (mol C m^-2 yr^-1, zero outside
    ``surface_mask``).
    """
    union = surface_mask | reference_mask
    erosion = np.where(union, np.maximum(-erodep_scaled.astype(np.float64), 0.0), 0.0)
    runoff = np.where(union, q_mm, 0.0)
    # cells outside both masks get the reference temperature so the Arrhenius
    # term stays finite; their flux is zeroed below
    temp = np.where(union, T_K, T_REF_K)
    scf = f_sil * np.where(
        union,
        weathering_flux(erosion, runoff, temp, fits["silicate"], BETA["silicate"]),
        0.0,
    )
    sf = f_sulf * np.where(
        union,
        weathering_flux(erosion, runoff, temp, fits["sulfide"], BETA["sulfide"]),
        0.0,
    )
    # Net inorganic drawdown F_net = 1/2 SCF/M_cat - 1/2 SF/M_SO4 (Methods), in
    # mol C per m^2 and year when the fluxes are in g m^-2 yr^-1.
    fnet = 0.5 * (scf / M_CAT) - (sf / M_SO4)

    carbon = 12.011e-15     # g/mol C x 1e-15 Gt/g: mol C -> Gt C
    def integrate(mask):
        net = float(np.sum(np.where(mask, fnet, 0.0) * area)) * carbon
        sil = float(np.sum(np.where(mask, scf, 0.0) * area)) / M_CAT * 0.5 * carbon
        sulf = float(np.sum(np.where(mask, sf, 0.0) * area)) / M_SO4 * carbon
        return net, sil, sulf

    net_surface, scf_surface, sf_surface = integrate(surface_mask)
    net_reference, _scf_reference, _sf_reference = integrate(reference_mask)
    return (net_surface, scf_surface, sf_surface, net_reference,
            np.where(surface_mask, fnet, 0.0))


def _land_mean_elevation_from_delta(z_base, delta_1deg, lat1, lon1,
                                    lat_t, lon_t):
    """Mean elevation (m) over the emergent land of one member, without a NeuroLEM run.

    ``delta_1deg`` on (lat1, lon1) is interpolated row by row onto the target
    grid and added to ``z_base``; cells above 0 m count as land. Not called by
    main(); kept as a cheap reporting helper.
    """
    from scipy.interpolate import RegularGridInterpolator
    itp = RegularGridInterpolator((lat1, lon1), delta_1deg,
                                  bounds_error=False, fill_value=None)
    # Avoid constructing a second 2-D coordinate mesh: interpolate row by row.
    total = 0.0
    count = 0
    for i, la in enumerate(lat_t):
        pts = np.column_stack((np.full(len(lon_t), la), lon_t))
        zrow = z_base[i] + itp(pts)
        land = zrow > 0
        total += float(zrow[land].sum())
        count += int(land.sum())
    return total / max(count, 1)


def _write_compressed_f4(ds, name, values, dims, long_name, units):
    """Write lossless-with-respect-to-float32 compressed spatial output.

    The quantiles are stored as float32 with DEFLATE compression rather than
    packed integers: packing each quantile into int16 with its own
    data-dependent scale could reverse the p05/p50/p95 order through
    rounding, whereas keeping the float32 values is order safe.
    Non-finite values are rejected.
    """
    a = np.asarray(values, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    chunks = tuple(min(s, c) for s, c in zip(a.shape, (181, 361)))
    v = ds.createVariable(name, "f4", dims,
                          zlib=True, complevel=6, shuffle=True,
                          chunksizes=chunks)
    v[:] = a
    v.units = units
    v.long_name = long_name
    return v


def _write_spatial_quantiles(path, age, lat, lon, erodep_stack, fco2_stack,
                             random_seed, land_fraction=None,
                             reference_land_mask=None,
                             history_source="neurolem/apply_geochem_ensemble.py"):
    """Write cellwise p05/p50/p95 maps of one age to a CF netCDF file.

    ``erodep_stack`` and ``fco2_stack`` have shape (member, lat, lon) and are
    zero outside each member's emergent land. ``land_fraction`` (optional) is
    the fraction of members in which each cell is land; ``reference_land_mask``
    (optional) the land mask of the unadjusted reference surface. The file is
    written to a hidden temporary name, read back and checked (monotonic
    quantiles, float32, valid land fraction and mask), and only then moved onto
    ``path``, so an interrupted run never leaves a partial map in place.
    """
    import netCDF4 as nc

    qs = (0.05, 0.50, 0.95)
    n_members = int(erodep_stack.shape[0])
    # np.quantile may use an in-place partition/copy; do one stack at a time
    # so peak memory stays comfortably below 64 GB.
    er_q = np.quantile(erodep_stack, qs, axis=0).astype(np.float32)
    del erodep_stack
    fc_q = np.quantile(fco2_stack, qs, axis=0).astype(np.float32)
    del fco2_stack

    for name, q in (("erodep", er_q), ("fco2", fc_q)):
        if not (np.all(q[0] <= q[1]) and np.all(q[1] <= q[2])):
            raise RuntimeError(f"{name} quantiles are not monotonically ordered")
    if land_fraction is not None:
        land_fraction = np.asarray(land_fraction, np.float32)
        if land_fraction.shape != er_q.shape[1:]:
            raise ValueError("land_fraction has the wrong shape")
        if (not np.isfinite(land_fraction).all() or
                np.any(land_fraction < 0.0) or np.any(land_fraction > 1.0)):
            raise ValueError("land_fraction must be finite and between zero and one")
    if reference_land_mask is not None:
        reference_land_mask = np.asarray(reference_land_mask, bool)
        if reference_land_mask.shape != er_q.shape[1:]:
            raise ValueError("reference_land_mask has the wrong shape")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.unlink(missing_ok=True)
    with nc.Dataset(tmp, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        vlat = ds.createVariable("lat", "f4", ("lat",))
        vlon = ds.createVariable("lon", "f4", ("lon",))
        vlat[:] = lat
        vlon[:] = lon
        vlat.units = "degrees_north"
        vlat.standard_name = "latitude"
        vlat.axis = "Y"
        vlon.units = "degrees_east"
        vlon.standard_name = "longitude"
        vlon.axis = "X"
        labels = ("p05", "p50", "p95")
        for i, label in enumerate(labels):
            # the erosion-deposition rate of the stack is in mm/yr (see _qero_km3yr)
            vero = _write_compressed_f4(
                ds, f"erodep_{label}", er_q[i], ("lat", "lon"),
                f"{label} NeuroLEM erosion/deposition rate", "mm yr-1")
            vero.sign_convention = (
                "positive is deposition; negative is erosion; zero outside "
                "each member's emergent-land mask before calculating cellwise quantiles")
            vfco2 = _write_compressed_f4(
                ds, f"fco2_{label}", fc_q[i], ("lat", "lon"),
                f"{label} net inorganic CO2 flux", "mol C m-2 yr-1")
            vfco2.sign_convention = (
                "positive is atmospheric CO2 drawdown; negative is net release; "
                "zero outside each member's emergent-land mask before calculating "
                "cellwise quantiles")
        if land_fraction is not None:
            vland = _write_compressed_f4(
                ds, "member_land_fraction", land_fraction, ("lat", "lon"),
                "fraction of ensemble members in which the cell is emergent land",
                "1")
            vland.valid_min = np.float32(0.0)
            vland.valid_max = np.float32(1.0)
        if reference_land_mask is not None:
            chunks = (min(len(lat), 181), min(len(lon), 361))
            vref = ds.createVariable(
                "reference_land_mask", "i1", ("lat", "lon"),
                zlib=True, complevel=6, shuffle=True, chunksizes=chunks)
            vref[:] = reference_land_mask.astype(np.int8)
            vref.long_name = "unmodified Salles/goSPL reference emergent-land mask"
            vref.units = "1"
            vref.flag_values = np.asarray([0, 1], dtype=np.int8)
            vref.flag_meanings = "ocean land"
        ds.target_age_Ma = float(age)
        ds.n_members = n_members
        ds.random_seed = int(random_seed)
        ds.quantiles = "0.05, 0.50, 0.95"
        ds.storage_precision = "float32"
        ds.quantile_order_verified = "true"
        ds.spatial_schema_version = "2.0"
        ds.Conventions = "CF-1.10"
        ds.title = (
            f"Geochemical paleo-elevation ensemble: NeuroLEM and net CO2 "
            f"spatial quantiles at {age} Ma")
        ds.quantile_domain = (
            "cellwise across coherent ensemble members; erosion/deposition and "
            "net CO2 flux are set to zero outside each member's z>0 emergent-land "
            "mask before quantiles are calculated")
        ds.uncertainty_scope = (
            "geochemical reconstruction choices only; paleogeographic prior, "
            "climate, NeuroLEM/goSPL configuration and weathering parameters fixed")
        ds.shoreline_interpretation = (
            "emergent-land masks are reconstruction boundary conditions, not "
            "observational shoreline truth")
        ds.history = f"created by {history_source}"

    # Verify the persisted file before atomically replacing an earlier map.
    with nc.Dataset(tmp) as ds:
        for prefix in ("erodep", "fco2"):
            q = [np.asarray(ds.variables[f"{prefix}_{label}"][:], np.float32)
                 for label in ("p05", "p50", "p95")]
            if not (np.all(q[0] <= q[1]) and np.all(q[1] <= q[2])):
                raise RuntimeError(
                    f"persisted {prefix} quantiles are not monotonically ordered")
            if any(a.dtype != np.dtype("float32") for a in q):
                raise RuntimeError(f"persisted {prefix} quantiles are not float32")
        if land_fraction is not None:
            persisted_land = np.asarray(ds.variables["member_land_fraction"][:],
                                        np.float32)
            if (not np.isfinite(persisted_land).all() or
                    np.any(persisted_land < 0.0) or
                    np.any(persisted_land > 1.0)):
                raise RuntimeError("persisted member_land_fraction is invalid")
        if reference_land_mask is not None:
            persisted_ref = np.asarray(ds.variables["reference_land_mask"][:])
            if not np.isin(persisted_ref, (0, 1)).all():
                raise RuntimeError("persisted reference_land_mask is invalid")
    tmp.replace(path)
    return path


def _write_age_summary(member_csv: Path, summary_csv: Path):
    """Summarise one age's member table: p05, p50, mean, p95 and sd (ddof=1) of every metric."""
    import pandas as pd

    d = pd.read_csv(member_csv)
    metrics = [c for c in d.columns if c not in
               ("age_Ma", "member", "random_seed", "isostatic_model_index",
                "isostatic_model")]
    row = {"age_Ma": int(d["age_Ma"].iloc[0]), "n_members": len(d),
           "random_seed": int(d["random_seed"].iloc[0])}
    for c in metrics:
        x = d[c].to_numpy(float)
        row[f"{c}_p05"] = float(np.quantile(x, 0.05))
        row[f"{c}_p50"] = float(np.quantile(x, 0.50))
        row[f"{c}_mean"] = float(np.mean(x))
        row[f"{c}_p95"] = float(np.quantile(x, 0.95))
        row[f"{c}_sd"] = float(np.std(x, ddof=1))
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    return summary_csv


def main():
    """Process one member cube (one age): NeuroLEM and weathering for every member, then tables and maps."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members-nc", required=True,
                   help="one <age>Ma_geochem_ensemble_members.nc handoff cube")
    p.add_argument("--salles-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--temp-dir", required=True)
    p.add_argument("--params-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    p.add_argument("--runoff-coefficient", type=float, default=0.4)
    p.add_argument("--erate-scale", type=float, default=2.0)
    p.add_argument("--f-sil", type=float, default=2.4)
    p.add_argument("--f-sulf", type=float, default=0.1)
    p.add_argument("--spatial", action="store_true",
                   help="retain this age in RAM and write compressed float32 p05/p50/p95 maps")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    import netCDF4 as nc
    import torch

    out = Path(args.out)
    member_out = out / "member_tables"
    summary_out = out / "age_summaries"
    map_out = out / "spatial_quantiles"
    member_out.mkdir(parents=True, exist_ok=True)
    summary_out.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(args.members_nc) as ds:
        age = int(round(float(ds.target_age_Ma)))
        lat1 = np.asarray(ds.variables["lat"][:], np.float64)
        lon1 = np.asarray(ds.variables["lon"][:], np.float64)
        model_idx = np.asarray(ds.variables["isostatic_model_index"][:], np.int8)
        model_names = json.loads(ds.variables["isostatic_model_index"].model_names)
        delta_members = np.asarray(ds.variables["delta_member"][:], np.float32)
        random_seed = int(ds.random_seed)

    csv_path = member_out / f"{age}Ma_members.csv"
    age_summary_path = summary_out / f"{age}Ma_summary.csv"
    map_path = map_out / f"{age}Ma_spatial_quantiles.nc"
    if csv_path.exists() and age_summary_path.exists() and not args.force:
        print(f"[{age} Ma] already complete: {csv_path}")
        return

    salles_path = Path(args.salles_dir) / f"salles_{age}Ma.npz"
    temp_path = Path(args.temp_dir) / f"{age}Ma.npz"
    if not salles_path.exists():
        raise FileNotFoundError(salles_path)
    if not temp_path.exists():
        raise FileNotFoundError(temp_path)

    sd = np.load(salles_path, allow_pickle=True)
    z_base = sd["inputs"][0].astype(np.float32)        # reference goSPL elevation, m
    rain = sd["inputs"][1].astype(np.float32)          # precipitation, m/yr
    lat = sd["lat"].astype(np.float64)
    lon = sd["lon"].astype(np.float64)
    target_shape = z_base.shape
    area = _area_m2(lat, target_shape)
    ref_mask = z_base > 0                              # emergent land of the reference surface
    ref_area_mkm2 = float(area[ref_mask].sum() / 1e12)  # m^2 -> million km^2

    raw = json.load(open(args.params_json))
    fits = {rxn: median_fit(raw, rxn, args.regions) for rxn in REACTIONS_USED}
    T_K = _load_temp_K(Path(args.temp_dir), age)
    # runoff (mm/yr) = precipitation (m/yr -> mm/yr) x a uniform runoff coefficient
    q_mm = rain.astype(np.float64) * 1000.0 * args.runoff_coefficient

    dev = _device()
    model, ck, norm = _load_model(args.ckpt, dev)
    z_base_t, rain_n_t = _prepare_constant_tensors(z_base, rain, norm, dev)
    print(f"[{age} Ma] device={dev}; members={len(delta_members)}; "
          f"spatial={args.spatial}; checkpoint val_rmse={ck.get('val_rmse')}")

    er_stack = (np.empty((len(delta_members),) + target_shape, dtype=np.float32)
                if args.spatial else None)
    fc_stack = (np.empty((len(delta_members),) + target_shape, dtype=np.float32)
                if args.spatial else None)
    land_count = (np.zeros(target_shape, dtype=np.uint16)
                  if args.spatial else None)

    header = [
        "age_Ma", "member", "random_seed", "isostatic_model_index", "isostatic_model",
        "Qero_surface_km3yr", "Qero_common_reference_cells_km3yr",
        "land_area_Mkm2", "reference_land_area_Mkm2",
        "net_co2_surface_GtCyr", "scf_surface_GtCyr", "sf_surface_GtCyr",
        "net_co2_common_reference_cells_GtCyr",
    ]
    rows = []
    for j, delta in enumerate(delta_members):
        erodep, mask = _infer_member(model, delta, z_base_t, rain_n_t,
                                     target_shape, norm, dev)
        q_surface = _qero_km3yr(erodep, mask, area)
        q_common = _qero_km3yr(erodep, ref_mask, area)
        land_area = float(area[mask].sum() / 1e12)

        # erosion calibration factor: surrogate rate -> total denudation rate (Methods)
        er_scaled = erodep.astype(np.float64) * args.erate_scale
        net, scf, sf, net_common, fnet = _weathering_two_masks(
            er_scaled, mask, ref_mask, q_mm, T_K, area, fits,
            args.f_sil, args.f_sulf)

        rows.append([
            age, j, random_seed, int(model_idx[j]), model_names[int(model_idx[j])],
            q_surface, q_common, land_area, ref_area_mkm2,
            net, scf, sf, net_common,
        ])
        if args.spatial:
            # Ocean predictions are outside the surrogate's valid domain. Set
            # them to zero member by member before calculating spatial
            # quantiles, matching the surface-specific weathering field.
            er_stack[j] = np.where(mask, erodep, 0.0).astype(np.float32)
            fc_stack[j] = fnet.astype(np.float32)
            land_count += mask
        if (j + 1) % max(1, min(20, len(delta_members) // 10)) == 0:
            print(f"  [{age:>3} Ma] NeuroLEM member {j+1}/{len(delta_members)}")

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    _write_age_summary(csv_path, age_summary_path)
    print(f"wrote {csv_path}")
    print(f"wrote {age_summary_path}")

    if args.spatial:
        _write_spatial_quantiles(map_path, age, lat.astype(np.float32),
                                 lon.astype(np.float32), er_stack, fc_stack,
                                 random_seed,
                                 land_fraction=(land_count / len(delta_members)),
                                 reference_land_mask=ref_mask)
        print(f"wrote {map_path} ({map_path.stat().st_size / 1e6:.1f} MB)")

    # Release the model and device tensors promptly (relevant on Apple
    # unified memory) before the controller starts the next age.
    del model, z_base_t, rain_n_t, delta_members
    if dev == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyse the direct goSPL validation experiment: five ages x four cases, 20 runs.

The direct goSPL simulations (Methods, "Direct goSPL tests"; Supplementary
Information) were run on the NCI Gadi supercomputer at 60, 220, 240, 460 and
520 Ma for four surfaces each: the PALEOMAP reference surface, the
rock-chemistry-adjusted surface ("central") and the two ensemble members selected
at the 5th and 95th percentiles of NeuroLEM's net export Qnet ("qnet_p05",
"qnet_p95").  This post-processor reads the verified bundle retrieved from those
runs, computes exact spherical vertex areas of the level-10 icosphere mesh
(10,485,762 nodes), integrates gross erosion and deposition and Bufe et al. (2024)
weathering fluxes (median parameter set, amplitude factors F_SIL and F_SULF, uniform
runoff coefficient), compares the integrals with the matching NeuroLEM predictions,
and writes compact 1-degree maps for Extended Data Fig. 3 (make_figS3_gadi_validation.py).
nature_recast/gospl_validation_sensitivity.py re-uses the geometry cache built here
and evaluates the same runs with the weathering formulation of the main model.

The signed goSPL field supports gross continental erosion and deposition.  It
does not contain routed outlet sediment discharge, so this script never labels
the direct-model integral as ocean sediment delivery or as NeuroLEM's signed
``Qnet`` member-selection metric.

Inputs (under --gadi-root)
  qc/verification_report.json                      must report "passed"
  raw/inputs/nature_validation/icosphere_l10_topology.npz   mesh vertices v, faces c
  raw/inputs/nature_validation/rain_l10_<age>Ma.npz         precipitation r on the mesh, m/yr
  raw/inputs/nature_validation/corrections/selected_corrections_manifest.json
                                                   ensemble member behind each qnet case
  raw/outputs/nature_validation/<age>Ma_<case>_final_fields.npz
                                                   initial_elev, elev (m), erorate (mm/yr,
                                                   negative = erosion)
Other inputs: --temp-dir (HadCM3 temperature, degC, packed), the Zenodo data tree
(surrogate_erodep/ and the ensemble member tables), and the repository tables
figures/age_summary.csv and data/co2_cal.csv holding the NeuroLEM integrals.

Outputs (in --out, default <gadi-root>/derived)
  integrated_metrics.csv    per age and case: land area (Mkm2), mean initial and final
                            land elevation (m), gross erosion, gross deposition and net
                            erosion (km3/yr), silicate drawdown, sulfide release and net
                            CO2 flux (Gt C/yr), each on the case's own land and on the
                            reference land ("common reference cells")
  paired_responses.csv      percentage change of each case against the reference run
  neurolem_comparison.csv   goSPL against NeuroLEM gross erosion and net CO2 flux
  spatial_skill.csv         1-degree correlation, RMSE and bias of the signed
                            erosion-deposition rate, goSPL against NeuroLEM
  validation_maps_1deg.nc   (age, case, 180, 360) maps of elevation, erosion-deposition
                            rate, net inorganic CO2 flux and initial land fraction
  analysis_manifest.json    inputs and weathering settings
  cache/                    vertex_area_m2, lon, lat, bin_index (.npy) of the mesh

Usage
  python3 analyse_gadi_validation.py --gadi-root <bundle> [--out DIR] [--force-geometry]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.ndimage import map_coordinates

from _paths import ERATE_SCALE, F_SIL, F_SULF, PARAMS, TEMP_DIR, zenodo_root
from bufe_co2_from_dem import (
    BETA,
    DEFAULT_REGIONS,
    M_CAT,
    M_SO4,
    REACTIONS_USED,
    median_fit,
    weathering_flux,
)


AGES = (60, 220, 240, 460, 520)
# reference = PALEOMAP surface; central = adjusted surface; qnet_p05 / qnet_p95 =
# ensemble members at the 5th / 95th percentile of NeuroLEM net export
CASES = ("reference", "central", "qnet_p05", "qnet_p95")
EARTH_RADIUS_M = 6_371_000.0
# 1-degree output grid; cell (i, j) spans lat -90+i .. -89+i and lon -180+j .. -179+j
GRID_NLAT = 180
GRID_NLON = 360
GRID_SIZE = GRID_NLAT * GRID_NLON


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def one_age(rows: list[dict[str, str]], age: int) -> dict[str, str]:
    """Return the single row whose ``age`` column equals ``age``; error otherwise."""
    matches = [row for row in rows if int(float(row["age"])) == age]
    if len(matches) != 1:
        raise RuntimeError(f"expected one age {age} row, found {len(matches)}")
    return matches[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write row dicts to CSV, columns in the order of the first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    columns = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_geometry(topology: Path, cache: Path, force: bool) -> dict[str, np.ndarray]:
    """Exact spherical vertex areas and 1-degree bin indices of the goSPL mesh.

    topology : NPZ with vertex coordinates ``v`` (N, 3) and triangle indices ``c``.
    cache    : directory for the .npy results, reused unless ``force``.

    Each spherical triangle's area is shared equally among its three vertices, and
    the areas are checked to sum to the area of a sphere of radius EARTH_RADIUS_M.
    Returns memory-mapped arrays, each (N,): vertex_area_m2 (m2), lon and lat
    (degrees) and bin_index (flat index into the 180 x 360 1-degree grid).
    """
    cache.mkdir(parents=True, exist_ok=True)
    paths = {name: cache / f"{name}.npy" for name in ("vertex_area_m2", "lon", "lat", "bin_index")}
    if not force and all(path.is_file() for path in paths.values()):
        result = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
        # a level-10 icosphere has 10 * 4**10 + 2 vertices; anything else is rebuilt
        if len(result["vertex_area_m2"]) == 10_485_762:
            return result

    print("building exact spherical vertex areas from the level-10 topology", flush=True)
    with np.load(topology, allow_pickle=False) as topo:
        vertices = np.asarray(topo["v"], dtype=np.float64)
        faces = np.asarray(topo["c"], dtype=np.int32)
    # project the vertices onto the unit sphere
    radius = np.linalg.norm(vertices, axis=1)
    unit = vertices / radius[:, None]
    area = np.zeros(len(vertices), dtype=np.float64)
    chunk = 250_000
    for start in range(0, len(faces), chunk):
        stop = min(start + chunk, len(faces))
        f = faces[start:stop]
        a, b, c = unit[f[:, 0]], unit[f[:, 1]], unit[f[:, 2]]
        # spherical excess of the triangle (Van Oosterom & Strackee, 1983):
        # tan(E/2) = |a . (b x c)| / (1 + a.b + b.c + c.a); area = E * R^2
        numerator = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
        denominator = 1.0 + np.einsum("ij,ij->i", a, b)
        denominator += np.einsum("ij,ij->i", b, c)
        denominator += np.einsum("ij,ij->i", c, a)
        share = (2.0 * np.arctan2(numerator, denominator) * EARTH_RADIUS_M**2) / 3.0
        np.add.at(area, f[:, 0], share)
        np.add.at(area, f[:, 1], share)
        np.add.at(area, f[:, 2], share)
        if start % (chunk * 10) == 0:
            print(f"  spherical areas: {stop:,}/{len(faces):,} triangles", flush=True)
    total = float(area.sum())
    sphere = 4.0 * np.pi * EARTH_RADIUS_M**2
    if abs(total / sphere - 1.0) > 1.0e-8:
        raise RuntimeError(f"vertex areas sum to {total:g} m2, not the spherical area {sphere:g} m2")
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0])).astype(np.float32)
    lat = np.degrees(np.arcsin(vertices[:, 2] / radius)).astype(np.float32)
    # 1-degree bin of each vertex (the north pole is folded into the last row)
    ilat = np.clip(np.floor(lat.astype(np.float64) + 90.0).astype(np.int32), 0, GRID_NLAT - 1)
    ilon = np.floor((lon.astype(np.float64) + 180.0) % 360.0).astype(np.int32)
    bin_index = (ilat * GRID_NLON + ilon).astype(np.int32)
    np.save(paths["vertex_area_m2"], area)
    np.save(paths["lon"], lon)
    np.save(paths["lat"], lat)
    np.save(paths["bin_index"], bin_index)
    del faces, vertices, unit, radius
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}


def sample_temperature(temp_path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinearly sample the HadCM3 surface-air temperature at mesh vertices.

    temp_path : NPZ with integer-packed temperature (temp_q * scale + offset, degC)
                on a regular lat/lon grid.
    lat, lon  : vertex coordinates, degrees, (N,).
    Returns temperature in K, (N,) float32.  Longitudes are wrapped into the
    source window; points east of the last source column take its value.
    """
    with np.load(temp_path, allow_pickle=False) as packed:
        values = packed["temp_q"].astype(np.float32) * float(packed["scale"])
        values += float(packed["offset"])
        src_lat = np.asarray(packed["lat"], dtype=np.float64)
        src_lon = np.asarray(packed["lon"], dtype=np.float64)
    # map_coordinates needs ascending axes
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        values = values[::-1]
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        values = values[:, ::-1]
    out = np.empty(len(lat), dtype=np.float32)
    chunk = 1_000_000
    # the source grid is regular, so fractional indices follow from the spacing
    dy = (src_lat[-1] - src_lat[0]) / (len(src_lat) - 1)
    dx = (src_lon[-1] - src_lon[0]) / (len(src_lon) - 1)
    for start in range(0, len(lat), chunk):
        stop = min(start + chunk, len(lat))
        qlon = ((lon[start:stop].astype(np.float64) - src_lon[0]) % 360.0) + src_lon[0]
        qlon = np.minimum(qlon, src_lon[-1])
        iy = (lat[start:stop].astype(np.float64) - src_lat[0]) / dy
        ix = (qlon - src_lon[0]) / dx
        out[start:stop] = map_coordinates(values, (iy, ix), order=1, mode="nearest")
    return out + np.float32(273.15)


def binned_mean(
    values: np.ndarray,
    area: np.ndarray,
    bin_index: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Area-weighted mean of point values in each 1-degree bin.

    values, area, bin_index : (N,) arrays; ``mask`` optionally restricts the points.
    Returns a (180, 360) float32 grid, NaN where a bin holds no valid point.
    """
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    weights = np.where(valid, area, 0.0)
    numerator = np.bincount(
        bin_index, weights=np.where(valid, values, 0.0) * weights, minlength=GRID_SIZE
    )
    denominator = np.bincount(bin_index, weights=weights, minlength=GRID_SIZE)
    out = np.full(GRID_SIZE, np.nan, dtype=np.float32)
    good = denominator > 0
    out[good] = (numerator[good] / denominator[good]).astype(np.float32)
    return out.reshape(GRID_NLAT, GRID_NLON)


def weathering_case(
    erorate: np.ndarray,
    mask: np.ndarray,
    runoff_mm: np.ndarray,
    temp_k: np.ndarray,
    area: np.ndarray,
    fits: dict[str, dict[str, float]],
    erate_scale: float,
    f_sil: float,
    f_sulf: float,
    retain_map: bool,
) -> tuple[dict[str, float], np.ndarray | None]:
    """Global Bufe weathering fluxes of one goSPL field on one land mask.

    erorate   : signed erosion(-)/deposition(+) rate, mm/yr, (N,); depositing
                points carry no flux.
    mask      : points to integrate; runoff_mm (mm/yr), temp_k (K) and area (m2)
                are (N,) on the same mesh.
    fits      : median Bufe parameters per reaction; erate_scale multiplies the
                erosion rate; f_sil and f_sulf are the amplitude factors.
    retain_map: also return the per-vertex net flux.

    Returns ({silicate_drawdown_GtC_yr, sulfide_release_GtC_yr,
    net_drawdown_GtC_yr}, net-flux map in mol C m-2 yr-1 or None).
    The mesh is processed in chunks of 1e6 vertices to bound memory.
    """
    scf_total = sf_total = net_total = 0.0
    flux_map = np.zeros(len(erorate), dtype=np.float32) if retain_map else None
    chunk = 1_000_000
    for start in range(0, len(erorate), chunk):
        stop = min(start + chunk, len(erorate))
        active = mask[start:stop]
        erosion = np.maximum(-erorate[start:stop].astype(np.float64), 0.0) * erate_scale
        runoff = runoff_mm[start:stop].astype(np.float64)
        temperature = temp_k[start:stop].astype(np.float64)
        scf = f_sil * weathering_flux(
            erosion, runoff, temperature, fits["silicate"], BETA["silicate"]
        )
        sf = f_sulf * weathering_flux(
            erosion, runoff, temperature, fits["sulfide"], BETA["sulfide"]
        )
        scf = np.where(active, scf, 0.0)
        sf = np.where(active, sf, 0.0)
        # W is in g m-2 yr-1, so dividing by the molar masses gives mol m-2 yr-1;
        # F_net = 0.5 SCF/M_cat - 0.5 SF/M_SO4 (Bufe et al., 2024), mol C m-2 yr-1
        net = 0.5 * (scf / M_CAT) - (sf / M_SO4)
        local_area = area[start:stop]
        scf_total += float(np.sum(0.5 * scf / M_CAT * local_area))
        sf_total += float(np.sum(sf / M_SO4 * local_area))
        net_total += float(np.sum(net * local_area))
        if flux_map is not None:
            flux_map[start:stop] = net.astype(np.float32)
    carbon = 12.011e-15           # mol C -> Gt C
    return (
        {
            "silicate_drawdown_GtC_yr": scf_total * carbon,
            "sulfide_release_GtC_yr": sf_total * carbon,
            "net_drawdown_GtC_yr": net_total * carbon,
        },
        flux_map,
    )


def neurolem_integrated(
    age: int,
    case: str,
    repo: Path,
    zenodo: Path,
    selected: dict[str, object],
) -> dict[str, object]:
    """NeuroLEM integrals matching one direct goSPL case.

    Reference and central cases come from the main-run tables of the repository
    (figures/age_summary.csv for gross erosion in km3/yr, data/co2_cal.csv for the
    net CO2 flux in Gt C/yr); the qnet cases from the ensemble member tables in the
    Zenodo tree, for the member named in the selection manifest.  For the reference
    case the common-reference-cells values equal the surface values.
    """
    age_row = one_age(read_csv(repo / "figures" / "age_summary.csv"), age)
    co2_row = one_age(read_csv(repo / "data" / "co2_cal.csv"), age)
    if case == "reference":
        return {
            "member": "",
            "selection_Qnet_surface_km3_yr": "",
            "gross_erosion_surface_km3_yr": float(age_row["Qero_reference"]),
            "gross_erosion_common_reference_cells_km3_yr": float(age_row["Qero_reference"]),
            "net_CO2_surface_GtC_yr": float(co2_row["net_reference"]),
            "net_CO2_common_reference_cells_GtC_yr": float(co2_row["net_reference"]),
        }
    if case == "central":
        return {
            "member": "central",
            "selection_Qnet_surface_km3_yr": "",
            "gross_erosion_surface_km3_yr": float(age_row["Qero_informed"]),
            "gross_erosion_common_reference_cells_km3_yr": float(
                age_row["Qero_informed_common_reference_cells"]
            ),
            "net_CO2_surface_GtC_yr": float(co2_row["net_informed"]),
            "net_CO2_common_reference_cells_GtC_yr": float(
                co2_row["net_informed_common_reference_cells"]
            ),
        }
    key = case.replace("qnet_", "qnet_")
    member = int(selected["ages"][str(age)][key]["member"])
    recovery_path = (
        zenodo
        / "geochem_uncertainty_ensemble/neurolem_results/recovery/member_tables"
        / f"{age}Ma_net_export_members.csv"
    )
    co2_path = (
        zenodo
        / "geochem_uncertainty_ensemble/neurolem_results/member_tables"
        / f"{age}Ma_members.csv"
    )
    recovery = [row for row in read_csv(recovery_path) if int(row["member"]) == member]
    co2 = [row for row in read_csv(co2_path) if int(row["member"]) == member]
    if len(recovery) != 1 or len(co2) != 1:
        raise RuntimeError(f"cannot resolve selected member {member} at {age} Ma")
    r, c = recovery[0], co2[0]
    return {
        "member": member,
        "selection_Qnet_surface_km3_yr": float(r["Qnet_surface_km3yr"]),
        "gross_erosion_surface_km3_yr": float(r["Qero_surface_reproduced_km3yr"]),
        "gross_erosion_common_reference_cells_km3_yr": float(
            r["Qero_common_reference_cells_reproduced_km3yr"]
        ),
        "net_CO2_surface_GtC_yr": float(c["net_co2_surface_GtCyr"]),
        "net_CO2_common_reference_cells_GtC_yr": float(
            c["net_co2_common_reference_cells_GtCyr"]
        ),
    }


def regrid_neurolem(path: Path, case: str) -> tuple[np.ndarray, np.ndarray]:
    """Average a NeuroLEM 0.1-degree erosion field onto the 1-degree grid.

    Reads ero_scotese (reference case) or ero_zhou (all other cases) and the
    matching land mask from an erodep NPZ.  Returns (land-only cos(lat)-weighted
    mean of the signed erosion-deposition rate in mm/yr, land fraction), both
    (180, 360).
    """
    key = "ero_scotese" if case == "reference" else "ero_zhou"
    mask_key = "mask_scotese" if case == "reference" else "mask_zhou"
    with np.load(path, allow_pickle=False) as data:
        # drop the last longitude column, which repeats -180 at +180
        values = np.asarray(data[key][:, :-1], dtype=np.float64)
        mask = np.asarray(data[mask_key][:, :-1], dtype=bool)
        lat = np.asarray(data["lat"], dtype=np.float64)
        lon = np.asarray(data["lon"][:-1], dtype=np.float64)
    ilat = np.clip(np.floor(lat + 90.0).astype(np.int32), 0, GRID_NLAT - 1)
    ilon = np.floor((lon + 180.0) % 360.0).astype(np.int32)
    bins = (ilat[:, None] * GRID_NLON + ilon[None, :]).reshape(-1)
    # relative cell area on the regular grid is proportional to cos(lat)
    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], values.shape).reshape(-1)
    grid = binned_mean(values.reshape(-1), area, bins, mask.reshape(-1))
    land_area = np.bincount(bins, weights=area * mask.reshape(-1), minlength=GRID_SIZE)
    all_area = np.bincount(bins, weights=area, minlength=GRID_SIZE)
    fraction = np.divide(land_area, all_area, out=np.zeros_like(land_area), where=all_area > 0)
    return grid, fraction.reshape(GRID_NLAT, GRID_NLON).astype(np.float32)


def write_map_netcdf(path: Path, maps: dict[str, np.ndarray], attrs: dict[str, object]) -> None:
    """Write the (age, case, lat, lon) 1-degree maps to NetCDF4 with units and
    descriptions; written to a temporary file and renamed when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.nc")
    with nc.Dataset(tmp, "w", format="NETCDF4") as ds:
        ds.createDimension("age", len(AGES))
        ds.createDimension("case", len(CASES))
        ds.createDimension("lat", GRID_NLAT)
        ds.createDimension("lon", GRID_NLON)
        age_var = ds.createVariable("age", "i4", ("age",))
        age_var[:] = AGES
        age_var.units = "Ma"
        case_var = ds.createVariable("case", str, ("case",))
        case_var[:] = np.asarray(CASES, dtype=object)
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lat_var[:] = np.arange(-89.5, 90.0, 1.0, dtype=np.float32)
        lon_var[:] = np.arange(-179.5, 180.0, 1.0, dtype=np.float32)
        lat_var.units = lon_var.units = "degrees"
        units = {
            "initial_elevation": "m",
            "final_elevation": "m",
            "erosion_deposition_rate": "mm yr-1",
            "net_inorganic_CO2_flux": "mol C m-2 yr-1",
            "initial_land_fraction": "1",
        }
        descriptions = {
            "initial_elevation": "area-weighted mean of the implemented goSPL initial surface",
            "final_elevation": "area-weighted mean of the final goSPL surface",
            "erosion_deposition_rate": "area-weighted signed final 100-kyr rate; negative is erosion",
            "net_inorganic_CO2_flux": "land-area-weighted Bufe flux; positive is atmospheric drawdown",
            "initial_land_fraction": "fraction of each 1-degree cell with initial elevation above sea level",
        }
        for name, values in maps.items():
            variable = ds.createVariable(
                name, "f4", ("age", "case", "lat", "lon"),
                zlib=True, complevel=5, shuffle=True, fill_value=np.float32(np.nan),
            )
            variable[:] = values
            variable.units = units[name]
            variable.long_name = descriptions[name]
        for key, value in attrs.items():
            ds.setncattr(key, value)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Run the analysis for all ages and cases and write the tables and maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gadi-root", type=Path,
        default=Path("gadi_runs/nature_validation_CO2"),
    )
    parser.add_argument("--zenodo-root", type=Path, default=zenodo_root())
    parser.add_argument("--params-json", type=Path, default=PARAMS)
    parser.add_argument("--temp-dir", type=Path, default=TEMP_DIR)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--erate-scale", type=float, default=ERATE_SCALE)
    parser.add_argument("--runoff-coefficient", type=float, default=0.4)
    parser.add_argument("--f-sil", type=float, default=F_SIL)
    parser.add_argument("--f-sulf", type=float, default=F_SULF)
    parser.add_argument("--force-geometry", action="store_true")
    args = parser.parse_args(argv)
    # repository root (this script sits one level below it)
    repo = Path(__file__).resolve().parents[1]
    gadi = args.gadi_root.resolve()
    raw = gadi / "raw"
    out = args.out.resolve() if args.out else gadi / "derived"
    out.mkdir(parents=True, exist_ok=True)
    verification = json.loads((gadi / "qc" / "verification_report.json").read_text())
    if not verification.get("passed", False):
        raise SystemExit("Gadi retrieval verification has not passed; refusing paper analysis")
    selected_path = (
        raw / "inputs/nature_validation/corrections/selected_corrections_manifest.json"
    )
    selected = json.loads(selected_path.read_text())
    topology = raw / "inputs/nature_validation/icosphere_l10_topology.npz"
    geometry = build_geometry(topology, out / "cache", args.force_geometry)
    area = np.asarray(geometry["vertex_area_m2"])
    lon = np.asarray(geometry["lon"])
    lat = np.asarray(geometry["lat"])
    bin_index = np.asarray(geometry["bin_index"])
    # total mesh area in each 1-degree bin, m2, for the land fractions
    grid_area = np.bincount(bin_index, weights=area, minlength=GRID_SIZE)

    raw_params = json.loads(args.params_json.read_text())
    fits = {
        reaction: median_fit(raw_params, reaction, DEFAULT_REGIONS)
        for reaction in REACTIONS_USED
    }
    shape = (len(AGES), len(CASES), GRID_NLAT, GRID_NLON)
    maps = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in (
            "initial_elevation", "final_elevation", "erosion_deposition_rate",
            "net_inorganic_CO2_flux", "initial_land_fraction",
        )
    }
    metrics: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    spatial: list[dict[str, object]] = []

    for age_index, age in enumerate(AGES):
        print(f"[{age} Ma] loading climate and direct-model fields", flush=True)
        rain_path = raw / "inputs/nature_validation" / f"rain_l10_{age}Ma.npz"
        # runoff = precipitation (m/yr -> mm/yr) times a uniform runoff coefficient
        with np.load(rain_path, allow_pickle=False) as rain_data:
            runoff_mm = np.asarray(rain_data["r"], dtype=np.float32) * np.float32(
                1000.0 * args.runoff_coefficient
            )
        temp_k = sample_temperature(args.temp_dir / f"{age}Ma.npz", lat, lon)
        ref_path = raw / "outputs/nature_validation" / f"{age}Ma_reference_final_fields.npz"
        # land of the reference run defines the common cells on which all cases are
        # also integrated
        with np.load(ref_path, allow_pickle=False) as reference:
            reference_land = np.asarray(reference["initial_elev"] > 0.0)
        reference_row: dict[str, object] | None = None
        gadi_grids: dict[str, np.ndarray] = {}

        for case_index, case in enumerate(CASES):
            path = raw / "outputs/nature_validation" / f"{age}Ma_{case}_final_fields.npz"
            with np.load(path, allow_pickle=False) as data:
                initial = np.asarray(data["initial_elev"], dtype=np.float64)
                final = np.asarray(data["elev"], dtype=np.float64)
                erorate = np.asarray(data["erorate"], dtype=np.float64)
            # each run's own land: initial elevation above sea level
            land = initial > 0.0
            own_area = float(area[land].sum())
            common_area = float(area[reference_land].sum())

            def volume(rate: np.ndarray, mask: np.ndarray, erosion: bool) -> float:
                # gross erosion (or deposition) volume over the mask, km3/yr:
                # mm/yr * m2 = 1e-3 m3/yr and 1 km3 = 1e9 m3
                signed = np.maximum(-rate, 0.0) if erosion else np.maximum(rate, 0.0)
                return float(np.sum(signed[mask] * area[mask])) * 1.0e-12

            own_erosion = volume(erorate, land, True)
            own_deposition = volume(erorate, land, False)
            common_erosion = volume(erorate, reference_land, True)
            common_deposition = volume(erorate, reference_land, False)
            own_co2, flux_map = weathering_case(
                erorate, land, runoff_mm, temp_k, area, fits,
                args.erate_scale, args.f_sil, args.f_sulf, True,
            )
            common_co2, _ = weathering_case(
                erorate, reference_land, runoff_mm, temp_k, area, fits,
                args.erate_scale, args.f_sil, args.f_sulf, False,
            )
            row: dict[str, object] = {
                "age_Ma": age,
                "case": case,
                "initial_land_area_Mkm2": own_area * 1.0e-12,
                "initial_land_mean_elevation_m": float(np.sum(initial[land] * area[land]) / own_area),
                "final_land_mean_elevation_m": float(np.sum(final[land] * area[land]) / own_area),
                "gross_erosion_surface_km3_yr": own_erosion,
                "gross_deposition_surface_km3_yr": own_deposition,
                "net_erosion_surface_km3_yr": own_erosion - own_deposition,
                "gross_erosion_common_reference_cells_km3_yr": common_erosion,
                "gross_deposition_common_reference_cells_km3_yr": common_deposition,
                "net_erosion_common_reference_cells_km3_yr": common_erosion - common_deposition,
                "silicate_drawdown_surface_GtC_yr": own_co2["silicate_drawdown_GtC_yr"],
                "sulfide_release_surface_GtC_yr": own_co2["sulfide_release_GtC_yr"],
                "net_CO2_surface_GtC_yr": own_co2["net_drawdown_GtC_yr"],
                "silicate_drawdown_common_reference_cells_GtC_yr": common_co2["silicate_drawdown_GtC_yr"],
                "sulfide_release_common_reference_cells_GtC_yr": common_co2["sulfide_release_GtC_yr"],
                "net_CO2_common_reference_cells_GtC_yr": common_co2["net_drawdown_GtC_yr"],
            }
            metrics.append(row)
            if case == "reference":
                reference_row = row
            elif reference_row is not None:
                responses.append(
                    {
                        "age_Ma": age,
                        "case": case,
                        "land_area_change_pct": 100.0 * (own_area / (float(reference_row["initial_land_area_Mkm2"]) * 1.0e12) - 1.0),
                        "gross_erosion_surface_change_pct": 100.0 * (own_erosion / float(reference_row["gross_erosion_surface_km3_yr"]) - 1.0),
                        "gross_erosion_common_reference_cells_change_pct": 100.0 * (common_erosion / float(reference_row["gross_erosion_common_reference_cells_km3_yr"]) - 1.0),
                        "net_CO2_surface_change_pct": 100.0 * (own_co2["net_drawdown_GtC_yr"] / float(reference_row["net_CO2_surface_GtC_yr"]) - 1.0),
                        "net_CO2_common_reference_cells_change_pct": 100.0 * (common_co2["net_drawdown_GtC_yr"] / float(reference_row["net_CO2_common_reference_cells_GtC_yr"]) - 1.0),
                    }
                )
            predicted = neurolem_integrated(age, case, repo, args.zenodo_root, selected)
            comparisons.append(
                {
                    "age_Ma": age,
                    "case": case,
                    "selected_member": predicted["member"],
                    "NeuroLEM_selection_Qnet_surface_km3_yr": predicted["selection_Qnet_surface_km3_yr"],
                    "NeuroLEM_gross_erosion_surface_km3_yr": predicted["gross_erosion_surface_km3_yr"],
                    "goSPL_gross_erosion_surface_km3_yr": own_erosion,
                    "goSPL_minus_NeuroLEM_gross_erosion_surface_pct": 100.0 * (own_erosion / float(predicted["gross_erosion_surface_km3_yr"]) - 1.0),
                    "NeuroLEM_gross_erosion_common_reference_cells_km3_yr": predicted["gross_erosion_common_reference_cells_km3_yr"],
                    "goSPL_gross_erosion_common_reference_cells_km3_yr": common_erosion,
                    "goSPL_minus_NeuroLEM_gross_erosion_common_reference_cells_pct": 100.0 * (common_erosion / float(predicted["gross_erosion_common_reference_cells_km3_yr"]) - 1.0),
                    "NeuroLEM_net_CO2_surface_GtC_yr": predicted["net_CO2_surface_GtC_yr"],
                    "goSPL_net_CO2_surface_GtC_yr": own_co2["net_drawdown_GtC_yr"],
                    "goSPL_minus_NeuroLEM_net_CO2_surface_pct": 100.0 * (own_co2["net_drawdown_GtC_yr"] / float(predicted["net_CO2_surface_GtC_yr"]) - 1.0),
                    "NeuroLEM_net_CO2_common_reference_cells_GtC_yr": predicted["net_CO2_common_reference_cells_GtC_yr"],
                    "goSPL_net_CO2_common_reference_cells_GtC_yr": common_co2["net_drawdown_GtC_yr"],
                    "goSPL_minus_NeuroLEM_net_CO2_common_reference_cells_pct": 100.0 * (common_co2["net_drawdown_GtC_yr"] / float(predicted["net_CO2_common_reference_cells_GtC_yr"]) - 1.0),
                }
            )

            # 1-degree maps: elevation and erosion-deposition averaged over all
            # vertices of a bin, the CO2 flux over its land vertices only
            maps["initial_elevation"][age_index, case_index] = binned_mean(initial, area, bin_index)
            maps["final_elevation"][age_index, case_index] = binned_mean(final, area, bin_index)
            maps["erosion_deposition_rate"][age_index, case_index] = binned_mean(erorate, area, bin_index)
            maps["net_inorganic_CO2_flux"][age_index, case_index] = binned_mean(
                np.asarray(flux_map), area, bin_index, land
            )
            land_area_by_bin = np.bincount(
                bin_index, weights=area * land, minlength=GRID_SIZE
            )
            maps["initial_land_fraction"][age_index, case_index] = (
                land_area_by_bin / grid_area
            ).reshape(GRID_NLAT, GRID_NLON).astype(np.float32)
            gadi_grids[case] = maps["erosion_deposition_rate"][age_index, case_index]
            del initial, final, erorate, land, flux_map

        # spatial skill at 1 degree, on bins that are at least half land in NeuroLEM
        for case in ("reference", "central"):
            neuro_grid, neuro_land = regrid_neurolem(
                args.zenodo_root / "surrogate_erodep" / f"erodep_{age}Ma.npz", case
            )
            direct_grid = gadi_grids[case]
            mask = (neuro_land >= 0.5) & np.isfinite(neuro_grid) & np.isfinite(direct_grid)
            n = neuro_grid[mask].astype(np.float64)
            d = direct_grid[mask].astype(np.float64)
            rmse = float(np.sqrt(np.mean((d - n) ** 2)))
            # RMSE is normalised by the root-mean-square of the NeuroLEM field
            scale = float(np.sqrt(np.mean(n**2)))
            spatial.append(
                {
                    "age_Ma": age,
                    "case": case,
                    "one_degree_land_cells": int(mask.sum()),
                    "signed_erodep_correlation": float(np.corrcoef(d, n)[0, 1]),
                    "signed_erodep_RMSE_mm_yr": rmse,
                    "signed_erodep_normalised_RMSE": rmse / scale if scale else np.nan,
                    "signed_erodep_bias_mm_yr": float(np.mean(d - n)),
                    "comparison_mask": "NeuroLEM initial land fraction >=0.5 at 1-degree resolution",
                }
            )
        del runoff_mm, temp_k, reference_land

    write_rows(out / "integrated_metrics.csv", metrics)
    write_rows(out / "paired_responses.csv", responses)
    write_rows(out / "neurolem_comparison.csv", comparisons)
    write_rows(out / "spatial_skill.csv", spatial)
    write_map_netcdf(
        out / "validation_maps_1deg.nc",
        maps,
        {
            "title": "Direct goSPL validation of rock-chemistry-adjusted paleo-elevation perturbations",
            "design": "4 nominal ages x reference, central, coherent Qnet-p05 and coherent Qnet-p95 members",
            "shoreline_treatment": "continuous correction; no inherited binary land mask reapplied",
            "weathering_erate_scale": args.erate_scale,
            "weathering_runoff_coefficient": args.runoff_coefficient,
            "weathering_silicate_amplitude": args.f_sil,
            "weathering_sulfide_amplitude": args.f_sulf,
            "note": "Qnet selected ensemble members; direct compact fields do not contain routed ocean sediment discharge",
        },
    )
    manifest = {
        "ages_Ma": AGES,
        "cases": CASES,
        "inputs": {
            "verified_gadi_bundle": str(gadi),
            "zenodo_data": str(args.zenodo_root.resolve()),
            "selected_members": str(selected_path),
            "temperature": str(args.temp_dir.resolve()),
            "weathering_parameters": str(args.params_json.resolve()),
        },
        "weathering": {
            "erate_scale": args.erate_scale,
            "runoff_coefficient": args.runoff_coefficient,
            "f_sil": args.f_sil,
            "f_sulf": args.f_sulf,
            "temperature_age": "nominal NeuroLEM age",
            "runoff_age": "archived Salles source age retained by the Gadi input builder",
        },
        "outputs": [
            "integrated_metrics.csv", "paired_responses.csv", "neurolem_comparison.csv",
            "spatial_skill.csv", "validation_maps_1deg.nc",
        ],
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"analysis complete: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

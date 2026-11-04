#!/usr/bin/env python3
"""
compress_valdes_temp.py — pack the HadCM3L palaeo-temperature grids into compact NPZs.

The surface-air temperature of the weathering calculation comes from the HadCM3L
simulations of Valdes et al. (2021) for the 109 palaeogeographies (Methods).  The
per-age grids (`<age>Ma.nc`, one per plate age 0–540 Ma) are single-variable
float32 fields already bilinearly regridded onto the 0.1° basegrid
(1801 × 3601, lat −90…90, lon −180…180), stored uncompressed at ~26 MB each
(~2.7 GB for 109 ages).  This script rewrites each as a compressed `<age>Ma.npz` with
int16 packing at 0.01 °C precision — ~3.8 MB each (~7× smaller, ~0.4 GB in total) —
loadable with plain numpy, without a netCDF reader.  These are the
`valdes_temperature/<age>Ma.npz` files read by the weathering calculation, the
climate-state ensemble and `climate_uncertainty/phanda_emulator.py apply`.

Rounding to a 0.01 °C step gives an error of at most 0.005 °C, far below anything
that matters for the temperature dependence of the weathering law; every file is
checked on writing against `--max-err`.  Each output stores:

    temp_q  int16      packed temperature
    scale   float32    0.01  (°C per unit; widened only if the range exceeds 32760 steps)
    offset  float32    per-file minimum (°C)
    lat,lon float32    the basegrid axes read from the input file
    age     int32      plate age (Ma)
    units   'degC'

Reconstruct with:  temp = temp_q.astype('f4') * scale + offset      (°C)
(add 273.15 for Kelvin).  Missing values (NaN) in the input are stored as the
minimum.

The run is resumable: ages whose `.npz` already exists and loads are skipped unless
`--force` is given.  `--retire-nc` moves each original `.nc` into
`<in-dir>/_nc_originals/` after its packed file has been written and verified (or
found valid on resume); nothing is deleted.

Requires numpy and xarray (with a netCDF backend).

USAGE
  python3 compress_valdes_temp.py --in-dir <dir of <age>Ma.nc> [--out-dir valdes_temperature]
  # optional: move the originals aside after a verified pass (moved, not deleted):
  python3 compress_valdes_temp.py --in-dir <dir of <age>Ma.nc> --retire-nc
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np


def _read_temp(nc_path: Path):
    """Read one grid: returns (temperature (nlat, nlon) float32 in °C, lat, lon, variable name).

    Uses the variable ``temp`` if present, otherwise the first data variable.
    """
    import xarray as xr
    ds = xr.open_dataset(nc_path, decode_times=False)
    var = "temp" if "temp" in ds.data_vars else list(ds.data_vars)[0]
    t = np.asarray(ds[var].values, dtype=np.float32)
    # find the lat/lon coords by length (1801 latitudes vs 3601 longitudes on the
    # basegrid, so the match is unambiguous)
    lat = lon = None
    for c in ds.coords:
        v = np.asarray(ds[c].values)
        if v.ndim == 1 and v.size == t.shape[0]:
            lat = v.astype(np.float32)
        elif v.ndim == 1 and v.size == t.shape[1]:
            lon = v.astype(np.float32)
    ds.close()
    return t, lat, lon, var


def _valid_npz(path: Path) -> bool:
    """True only if an already-written .npz loads and has the expected keys.
    Guards resume against a file left half-written by an interrupted run."""
    try:
        with np.load(path, allow_pickle=True) as z:
            return {"temp_q", "scale", "offset"}.issubset(set(z.files)) and z["temp_q"].shape[0] > 0
    except Exception:
        return False


def _pack(t: np.ndarray, scale: float = 0.01):
    """Quantise ``t`` to int16 as (t - offset) / scale with offset = nanmin(t).

    Returns (q int16, scale float32, offset float32).  NaNs are stored as the
    minimum.
    """
    finite = np.isfinite(t)
    offset = float(np.nanmin(t))
    q = np.round((np.nan_to_num(t, nan=offset) - offset) / scale)
    span = q[finite].max() if finite.any() else 0
    if span >= 32760:                       # range too wide for int16 at this step: widen the step
        scale = float((np.nanmax(t) - offset) / 32760.0)
        q = np.round((np.nan_to_num(t, nan=offset) - offset) / scale)
    return q.astype(np.int16), np.float32(scale), np.float32(offset)


def main():
    """Pack every <age>Ma.nc in --in-dir and report sizes and the worst round-trip error."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="dir of <age>Ma.nc")
    ap.add_argument("--out-dir", help="output dir (default: same as --in-dir)")
    ap.add_argument("--retire-nc", action="store_true",
                    help="after a verified write, move each .nc into <in-dir>/_nc_originals/")
    ap.add_argument("--max-err", type=float, default=0.01,
                    help="fail if round-trip abs error exceeds this (°C)")
    ap.add_argument("--force", action="store_true",
                    help="reprocess ages whose .npz already exists (default: skip them, "
                         "so an interrupted run resumes where it left off)")
    a = ap.parse_args()

    ind = Path(a.in_dir); outd = Path(a.out_dir) if a.out_dir else ind
    outd.mkdir(parents=True, exist_ok=True)
    ncs = {int(re.search(r"(\d+)Ma", f.name).group(1)): f
           for f in ind.glob("*Ma.nc")
           if not f.name.startswith(".")}   # skip macOS AppleDouble sidecars
    if not ncs:
        raise SystemExit(f"no <age>Ma.nc in {ind}")
    ages = sorted(ncs)
    print(f"compressing {len(ages)} grids from {ind}")

    tot_in = tot_out = 0
    worst = 0.0
    n_done = n_skip = 0
    retire = (ind / "_nc_originals") if a.retire_nc else None
    if retire:
        retire.mkdir(exist_ok=True)
    for age in ages:
        nc = ncs[age]
        outp = outd / f"{age}Ma.npz"
        if outp.exists() and not a.force and _valid_npz(outp):
            n_skip += 1
            if retire and nc.exists():        # still let --retire-nc move an already-packed original
                nc.replace(retire / nc.name)
            continue
        t, lat, lon, var = _read_temp(nc)
        q, scale, offset = _pack(t)
        # verify the round-trip error before writing (NaN cells are ignored by nanmax)
        err = float(np.nanmax(np.abs(q.astype(np.float32) * scale + offset - t)))
        worst = max(worst, err)
        if err > a.max_err:
            raise SystemExit(f"[{age}Ma] round-trip error {err:.4f} °C exceeds "
                             f"--max-err {a.max_err}; aborting")
        in_bytes = nc.stat().st_size
        np.savez_compressed(outp, temp_q=q, scale=scale, offset=offset,
                            lat=lat, lon=lon, age=np.int32(age), units=np.bytes_("degC"))
        tot_in += in_bytes; tot_out += outp.stat().st_size; n_done += 1
        if retire:
            nc.replace(retire / nc.name)
        print(f"  [{age:>4}Ma] {var}: {t.shape}  err={err:.4f}°C  "
              f"{in_bytes/1e6:.0f}->{outp.stat().st_size/1e6:.1f} MB")

    if n_skip:
        print(f"\nskipped {n_skip} ages already packed (use --force to redo).")
    print(f"Done. {n_done} newly packed, {n_skip} skipped, {len(ages)} total.  "
          f"worst round-trip error {worst:.4f} °C.")
    print(f"  {tot_in/1e9:.2f} GB -> {tot_out/1e9:.2f} GB  ({tot_in/max(tot_out,1):.1f}x smaller)")
    if retire:
        print(f"  originals moved to {retire}  (delete that folder to reclaim space)")
    else:
        print(f"  originals kept; re-run with --retire-nc to move them aside once you're happy.")


if __name__ == "__main__":
    main()

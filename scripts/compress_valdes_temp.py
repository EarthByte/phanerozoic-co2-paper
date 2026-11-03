#!/usr/bin/env python3
"""
compress_valdes_temp.py — compress the Valdes HadCM3 paleo-temperature grids.

Salles' per-age surface-air-temperature grids (`data/Valdes_HadCM_temp/<age>Ma.nc`)
are single-variable float32 fields already bilinear-regridded onto the 0.1°
`basegrid` (1801 × 3601), stored uncompressed at ~26 MB each (~2.7 GB for 109
ages). This rewrites each as a compressed `<age>Ma.npz` using int16 packing at
0.01 °C precision — ~3.8 MB each (~7× smaller, ~0.4 GB total) — which is also
directly loadable by plain numpy (no netCDF reader needed) for the CO2 analysis.

0.01 °C is far finer than the weathering-kinetics temperature sensitivity, so the
packing is lossless for all downstream use. Each output stores:

    temp_q  int16      packed temperature
    scale   float32    0.01  (°C per unit)
    offset  float32    per-file minimum (°C)
    lat,lon float32    the shared basegrid axes
    age     int
    units   'degC'

Reconstruct with:  temp = temp_q.astype('f4') * scale + offset      (°C)
(add 273.15 for Kelvin).

USAGE
  python3 compress_valdes_temp.py --in-dir  data/Valdes_HadCM_temp
  # optional: retire the originals after a verified pass (moved, not deleted):
  python3 compress_valdes_temp.py --in-dir data/Valdes_HadCM_temp --retire-nc
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np


def _read_temp(nc_path: Path):
    import xarray as xr
    ds = xr.open_dataset(nc_path, decode_times=False)
    var = "temp" if "temp" in ds.data_vars else list(ds.data_vars)[0]
    t = np.asarray(ds[var].values, dtype=np.float32)
    # find the lat/lon coords by length
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
    finite = np.isfinite(t)
    offset = float(np.nanmin(t))
    q = np.round((np.nan_to_num(t, nan=offset) - offset) / scale)
    span = q[finite].max() if finite.any() else 0
    if span >= 32760:                       # fall back to coarser step if needed
        scale = float((np.nanmax(t) - offset) / 32760.0)
        q = np.round((np.nan_to_num(t, nan=offset) - offset) / scale)
    return q.astype(np.int16), np.float32(scale), np.float32(offset)


def main():
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
        # verify round-trip before trusting it
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

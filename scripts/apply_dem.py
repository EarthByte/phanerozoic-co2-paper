#!/usr/bin/env python3
"""
apply_dem.py — apply the trained NeuroLEM surrogate to the Scotese-Wright and
geochemically informed paleo-elevation reconstructions (Paper 1).

The question: how does using the data-driven, geochemically informed
paleo-elevation reconstruction instead of the Scotese-Wright reconstruction
change predicted continental erosion/deposition and, downstream,
silicate-weathering CO2 drawdown?

**Why the "delta" method (default).** Salles' `zpaleo` input grids are NOT the raw
1 deg Scotese DEM upsampled. Per his `1-makeInputGrids.ipynb`, they come from his
high-resolution goSPL physiography maps, coarsened 2x and regridded to the 0.1 deg
basegrid with xESMF bilinear (periodic). So `zpaleo` carries ~0.1 deg goSPL-scale
structure. Feeding a smooth 1 deg->0.1 deg upsample would be out of distribution.
Instead we perturb Salles' own in-distribution `zpaleo` by the reconstructed
elevation difference:

    zpaleo_scotese = zpaleo_salles                        (the training input)
    zpaleo_zhou    = zpaleo_salles + upsample(delta)      (delta = M_corrected-M_orig)

`delta` is a smooth, province-scale elevation increment, so a plain bilinear
upsample is faithful (sub-metre differences from xESMF for this field). The
resulting contrast isolates the effect of the alternative paleo-elevation field,
with both cases based on in-distribution inputs and identical runoff.

For a fixed-bathymetry experiment, ``--mode delta-fixed-mask`` applies the
interpolated increment only where the target goSPL physiography is already land.
By default the increment is tapered inland as a smoothstep over the first 100 km
from the target coastline; use ``--coastal-taper-km 0`` for a binary mask. Informed
land elevations are clipped just above sea level if necessary, and the baseline
land mask is used for both experiments. This prevents an increment defined on the
source reconstruction's land mask from creating new land when transferred to the
distinct high-resolution goSPL surface, while avoiding a step at the target
coastline. This mode is a diagnostic and is not the primary manuscript experiment.

An alternative `--mode upsample` runs the surrogate on the full 1 deg DEMs
upsampled to 0.1 deg (both M_orig and M_corrected); use only as a cross-check —
it is out of distribution as noted above.

Runs on CUDA > Apple MPS > CPU (native 0.1 deg OOMs a small CPU box; use a GPU/MPS
machine — e.g. an M-series Mac by default).

Output (per age): <out>/erodep_<age>Ma.npz with
  ero_scotese, ero_zhou    (H,W) erosion(-)/deposition(+) rate [erodeporate], m/yr
  zpaleo_scotese, zpaleo_zhou (H,W) input elevation, m
  delta                    (H,W) upsampled elevation difference, m
  rain                     (H,W) shared runoff forcing, m/yr
  mask_scotese, mask_zhou  (H,W) bool land masks (elevation > 0)
  lat, lon, age

USAGE
  python3 apply_dem.py \
      --zhou-dir  /path/Paleotopo_data_assimilation/data/corrected_scotese_nocraton \
      --salles-dir /path/NeuroLEM/data/salles_training_native \
      --ckpt      /path/runs/salles_unet_v2/best.pt \
      --out       /path/runs/salles_unet_v2/erodep_scotese_vs_zhou
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np


def _load_model_class(explicit=None):
    """Import GlobalUNet from an explicit file, an installed package, or a checkout.

    ``explicit`` is an absolute path to ``salles_unet.py`` and is loaded directly by
    file location, bypassing sys.path entirely -- PYTHONPATH not reaching a detached
    process, or a checkout living somewhere unexpected, then cannot break the run.
    """
    if explicit:
        import importlib.util
        from pathlib import Path as _P
        f = _P(explicit)
        if f.is_dir():
            f = f / "salles_unet.py"
        if not f.exists():
            raise SystemExit(f"--unet-py: no such file: {f}")
        spec = importlib.util.spec_from_file_location("salles_unet", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"[apply_dem] GlobalUNet loaded from {f}")
        return mod.GlobalUNet
    try:
        from salles_unet import GlobalUNet
        return GlobalUNet
    except ModuleNotFoundError:
        import sys
        candidates = [
            Path(__file__).resolve().parents[2] / "neurolem-surrogate" / "neurolem",
            Path(__file__).resolve().parent,
        ]
        for candidate in candidates:
            if (candidate / "salles_unet.py").exists():
                sys.path.insert(0, str(candidate))
                from salles_unet import GlobalUNet
                return GlobalUNet
        raise


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _bilinear(field, lat1, lon1, lat_t, lon_t):
    """Bilinear regrid from a regular (lat1, lon1) grid to (lat_t, lon_t).
    Salles used xESMF bilinear (periodic); for the smooth `delta` this matches to
    sub-metre. Both grids share the -90..90 x -180..180 extent so no wrap needed."""
    if lat1[0] > lat1[-1]:
        lat1 = lat1[::-1]; field = field[::-1, :]
    if lon1[0] > lon1[-1]:
        lon1 = lon1[::-1]; field = field[:, ::-1]
    f = field.astype(np.float64)
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        # numpy-only equivalent: separable bilinear with linear extrapolation at the
        # edges, which is what RegularGridInterpolator(fill_value=None) does.  Agrees
        # with scipy to ~1e-10 m on these fields; scipy is used when present so the
        # archived products remain bit-reproducible.
        def _ax(src, dst):
            i = np.clip(np.searchsorted(src, dst) - 1, 0, src.size - 2)
            return i, (dst - src[i]) / (src[i + 1] - src[i])
        iy, ty = _ax(lat1, np.asarray(lat_t, np.float64))
        ix, tx = _ax(lon1, np.asarray(lon_t, np.float64))
        TY = ty[:, None]; TX = tx[None, :]
        out = (f[np.ix_(iy, ix)] * (1 - TX) * (1 - TY)
               + f[np.ix_(iy, ix + 1)] * TX * (1 - TY)
               + f[np.ix_(iy + 1, ix)] * (1 - TX) * TY
               + f[np.ix_(iy + 1, ix + 1)] * TX * TY)
        return out.astype(np.float32)
    itp = RegularGridInterpolator((lat1, lon1), f, method="linear",
                                  bounds_error=False, fill_value=None)
    LT, LN = np.meshgrid(lat_t, lon_t, indexing="ij")
    return itp((LT, LN)).astype(np.float32)


def _read_dem(path):
    """(lat, lon, M_orig, M_corrected) from a ``*Ma_corrected_SW.nc``.

    Prefers a pre-converted ``.npz`` sidecar (numpy only), then tries xarray,
    netCDF4 and h5py -- netCDF-4 files are HDF5 underneath, so h5py reads them too.  The environment on the Sydney H200 that carries torch has no xarray, and
    the .npz path lets the rebuild run with numpy alone.  All three routes return
    the same four arrays -- nothing about the numerics depends on which is used.
    """
    from pathlib import Path as _P
    p = _P(path)
    npz = p.with_suffix(".npz")
    if npz.exists():                      # preferred: needs numpy only
        with np.load(npz) as z:
            return (z["lat"].astype(np.float64), z["lon"].astype(np.float64),
                    z["M_orig"], z["M_corrected"])
    try:
        import xarray as xr
        with xr.open_dataset(p) as ds:
            return (np.asarray(ds["lat"].values, np.float64),
                    np.asarray(ds["lon"].values, np.float64),
                    np.asarray(ds["M_orig"].values),
                    np.asarray(ds["M_corrected"].values))
    except ImportError:
        pass
    try:
        from netCDF4 import Dataset
        ds = Dataset(str(p))
        try:
            return (np.asarray(ds.variables["lat"][:], np.float64),
                    np.asarray(ds.variables["lon"][:], np.float64),
                    np.asarray(ds.variables["M_orig"][:]),
                    np.asarray(ds.variables["M_corrected"][:]))
        finally:
            ds.close()
    except ImportError:
        pass
    try:
        import h5py                       # netCDF-4 files are HDF5 underneath
        with h5py.File(str(p), "r") as h:
            return (np.asarray(h["lat"][:], np.float64),
                    np.asarray(h["lon"][:], np.float64),
                    np.asarray(h["M_orig"][:]),
                    np.asarray(h["M_corrected"][:]))
    except ImportError:
        pass
    raise SystemExit(
        f"cannot read {p.name}: none of xarray, netCDF4 or h5py is importable and\n"
        f"there is no sidecar {p.with_suffix('.npz').name}.\n"
        "Either install one of those, or place the pre-converted .npz next to the .nc.")


def _pad_to(a, mult):
    H, W = a.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph == 0 and pw == 0:
        return a, (H, W)
    pad = [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(a, pad, mode="edge"), (H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zhou-dir", required=True,
                    help="dir of legacy-named <age>Ma_corrected_SW.nc informed DEM files")
    ap.add_argument("--salles-dir", required=True, help="salles_training_native dir (zpaleo + rain)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["delta", "delta-fixed-mask", "upsample"],
                    default="delta")
    ap.add_argument("--reuse-baseline-dir",
                    help="optional directory containing existing erodep_<age>Ma.npz files; "
                         "reuses ero_scotese after checking the baseline elevation, runoff "
                         "and coordinates, so only the informed case is inferred")
    ap.add_argument("--minimum-land-elevation", type=float, default=0.01,
                    help="minimum informed elevation in metres for reference land cells in "
                         "delta-fixed-mask mode (default: 0.01 m)")
    ap.add_argument("--coastal-taper-km", type=float, default=100.0,
                    help="smoothstep taper distance from the target goSPL coastline for "
                         "delta-fixed-mask mode (default: 100 km; 0 gives a binary mask)")
    ap.add_argument("--unet-py",
                    help="absolute path to salles_unet.py (or its directory); loaded by "
                         "file location, so PYTHONPATH is not involved")
    ap.add_argument("--ages", nargs="*", type=int)
    a = ap.parse_args()

    import torch
    GlobalUNet = _load_model_class(a.unet_py)

    dev = _device()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = GlobalUNet(in_ch=ck["in_ch"], depth=ck["depth"], base=ck["base"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    n = ck["norm"]
    in_mean = np.asarray(n["in_mean"], np.float32); in_std = np.asarray(n["in_std"], np.float32)
    s = float(n["asinh_s"]); t_mean = float(n["t_mean"]); t_std = float(n["t_std"])
    mult = 2 ** ck["depth"]
    print(f"[apply_dem] device={dev}  mode={a.mode}  ckpt val_rmse={ck.get('val_rmse')}")

    zdir = Path(a.zhou_dir); sdir = Path(a.salles_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zf = {int(re.search(r"(\d+)Ma_corrected_SW", f.name).group(1)): f
          for f in zdir.glob("*Ma_corrected_SW.nc")
          if not f.name.startswith(".")}   # skip macOS AppleDouble sidecars
    ages = sorted(a.ages) if a.ages else sorted(zf)

    def infer(elev, rain):
        x = np.stack([(elev - in_mean[0]) / in_std[0], (rain - in_mean[1]) / in_std[1]], 0).astype(np.float32)
        xp, (H, W) = _pad_to(x, mult)
        with torch.no_grad():
            t = model(torch.from_numpy(xp)[None].to(dev))[0, 0].cpu().numpy()[:H, :W]
        return (s * np.sinh(t * t_std + t_mean)).astype(np.float32)

    R = 6.371e6; dgrid = np.deg2rad(0.1)                       # cell-area constant (0.1 deg)
    csv = open(out / "flux_summary.csv", "w")
    csv.write("age,Qero_reference_km3yr,Qero_informed_km3yr,land_frac_reference,land_frac_informed\n")

    for age in ages:
        f = zf.get(age); sp = sdir / f"salles_{age}Ma.npz"
        if f is None or not sp.exists():
            print(f"  [{age}Ma] missing informed DEM or goSPL input — skipped"); continue
        sd = np.load(sp, allow_pickle=True)
        lat_t = sd["lat"].astype(np.float64); lon_t = sd["lon"].astype(np.float64)
        rain = sd["inputs"][1].astype(np.float32)
        lat1, lon1, _m_orig, _m_corr = _read_dem(f)
        delta_raw = _bilinear(_m_corr - _m_orig, lat1, lon1, lat_t, lon_t)
        if a.mode in ("delta", "delta-fixed-mask"):
            z_sco = sd["inputs"][0].astype(np.float32)        # Salles' in-distribution zpaleo
            if a.mode == "delta-fixed-mask":
                from scipy.ndimage import distance_transform_edt
                m_sco = z_sco > 0
                if a.coastal_taper_km > 0:
                    # Approximate physical distance on the 0.1-degree grid. Longitude
                    # convergence is not represented in this first-order taper, but the
                    # taper is used only to remove grid-scale coastal discontinuities.
                    dist_km = distance_transform_edt(m_sco) * 11.119
                    x = np.clip(dist_km / a.coastal_taper_km, 0.0, 1.0)
                    taper = x * x * (3.0 - 2.0 * x)  # C1-continuous smoothstep
                else:
                    taper = m_sco.astype(np.float32)
                z_zho = z_sco.copy()
                z_zho[m_sco] = np.maximum(
                    z_sco[m_sco] + (delta_raw * taper)[m_sco], a.minimum_land_elevation)
                delta = z_zho - z_sco
                m_zho = m_sco.copy()
            else:
                delta = delta_raw
                z_zho = z_sco + delta
                m_sco = z_sco > 0
                m_zho = z_zho > 0
        else:                                                  # cross-check: full upsample
            z_sco = _bilinear(_m_orig, lat1, lon1, lat_t, lon_t)
            z_zho = _bilinear(_m_corr, lat1, lon1, lat_t, lon_t)
            delta = z_zho - z_sco
            m_sco = z_sco > 0
            m_zho = z_zho > 0

        if a.reuse_baseline_dir:
            bp = Path(a.reuse_baseline_dir) / f"erodep_{age}Ma.npz"
            if not bp.exists():
                raise SystemExit(f"missing reusable baseline file: {bp}")
            with np.load(bp, allow_pickle=True) as bd:
                checks = {
                    "zpaleo_scotese": (bd["zpaleo_scotese"], z_sco),
                    "rain": (bd["rain"], rain),
                    "lat": (bd["lat"], lat_t),
                    "lon": (bd["lon"], lon_t),
                }
                for name, (old, new) in checks.items():
                    if old.shape != new.shape or not np.allclose(old, new, rtol=0, atol=1e-5):
                        raise SystemExit(f"baseline reuse check failed for {name} at {age} Ma")
                e_sco = bd["ero_scotese"].astype(np.float32)
        else:
            e_sco = infer(z_sco, rain)
        e_zho = infer(z_zho, rain)
        np.savez_compressed(out / f"erodep_{age}Ma.npz",
                            ero_scotese=e_sco, ero_zhou=e_zho,
                            zpaleo_scotese=z_sco, zpaleo_zhou=z_zho, delta=delta, rain=rain,
                            mask_scotese=m_sco, mask_zhou=m_zho,
                            lat=lat_t.astype(np.float32), lon=lon_t.astype(np.float32), age=age,
                            application_mode=a.mode)

        # Integrated continental erosion flux (km^3/yr): Q = sum_land max(-ero,0)*A.
        # Cell area A = R^2 * dgrid^2 * cos(lat), meridional convergence accounted for.
        area = (R * R * dgrid * dgrid) * np.cos(np.deg2rad(lat_t))[:, None]  # (H,1)->broadcast
        area = np.broadcast_to(area, e_sco.shape)
        q_sco = float((np.maximum(-e_sco, 0.0) * area)[m_sco].sum() / 1e12)
        q_zho = float((np.maximum(-e_zho, 0.0) * area)[m_zho].sum() / 1e12)
        lf_sco = float(m_sco.mean()); lf_zho = float(m_zho.mean())
        csv.write(f"{age},{q_sco:.6g},{q_zho:.6g},{lf_sco:.6g},{lf_zho:.6g}\n"); csv.flush()
        print(f"  [{age}Ma] done  delta[min,max]=[{delta.min():.0f},{delta.max():.0f}] m  "
              f"Qero sco={q_sco:.4g} informed={q_zho:.4g} km3/yr  "
              f"(informed/Scotese={q_zho/q_sco:.3f}; mask_equal={np.array_equal(m_sco, m_zho)})")

    csv.close()
    print(f"\nWrote {len(ages)} ages to {out}  (mode={a.mode}). ero_* = erodeporate (m/yr).")
    print(f"Flux time-series: {out/'flux_summary.csv'}")


if __name__ == "__main__":
    main()

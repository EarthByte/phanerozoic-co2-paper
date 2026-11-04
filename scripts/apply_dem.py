#!/usr/bin/env python3
"""
apply_dem.py -- run the trained NeuroLEM surrogate on the reference (PALEOMAP,
Scotese & Wright 2018) and the rock-chemistry-adjusted palaeoelevation surfaces.

Role in the paper
-----------------
This script produces the two erosion-deposition histories on which every result
is built (Methods, "Palaeoelevation" and "Neural surrogate"): for each of the 109
ages, one NeuroLEM field on the reference physiography and one on the
rock-chemistry-adjusted surface. Its output is the ``surrogate_erodep/`` folder of
the data archive, read by all weathering, sensitivity and figure scripts. The
two surfaces differ only in the long-wavelength increment derived from chemical
mohometry (Zhou et al., 2025), so their contrast isolates the effect of the
palaeoelevation reconstruction on erosion and, downstream, on the weathering
CO2 sink. In variable and option names, ``scotese`` denotes the reference
surface and ``zhou`` the rock-chemistry-adjusted surface; ``salles`` denotes the
published goSPL input and training grids (Salles et al., 2023).

Why the "delta" method (default)
--------------------------------
The goSPL elevation inputs (``zpaleo``) are not the 1-degree PALEOMAP DEM
upsampled. They come from the high-resolution goSPL physiography maps, coarsened
by a factor of two and regridded to the 0.1-degree base grid with periodic
bilinear interpolation (xESMF), so ``zpaleo`` carries goSPL-scale structure at
0.1 degree. A smooth 1-degree to 0.1-degree upsample would lie outside the
distribution NeuroLEM was trained on. Instead, the in-distribution ``zpaleo`` is
perturbed by the reconstructed elevation increment:

    zpaleo_scotese = zpaleo_salles                        (the training input)
    zpaleo_zhou    = zpaleo_salles + upsample(delta)      (delta = M_corrected - M_orig)

``delta`` is a smooth, province-scale increment, so plain bilinear upsampling
reproduces the xESMF result to within a metre. Both cases use in-distribution
inputs and identical precipitation, and both surfaces share the same fine-scale
relief (Methods: "The long-wavelength increment is added to the reference
physiography").

Other modes
-----------
``--mode delta-fixed-mask`` is a fixed-shoreline diagnostic, not the experiment
of the paper. The interpolated increment is applied only where the target goSPL
physiography is already land and, by default, is tapered inland as a smoothstep
over the first 100 km from the target coastline (``--coastal-taper-km 0`` gives a
binary mask). Adjusted land elevations are clipped to just above sea level
(``--minimum-land-elevation``) and the reference land mask is used for both
cases. This keeps an increment defined on the land mask of the 1-degree
reconstruction from creating new land on the distinct high-resolution goSPL
surface, without a step at the coastline.

``--mode upsample`` runs the surrogate on the full 1-degree DEMs (M_orig and
M_corrected) bilinearly upsampled to 0.1 degree. It is out of distribution for
the reason above and serves only as a cross-check.

Inputs
------
--zhou-dir    <age>Ma_corrected_SW.nc, 1-degree palaeoelevation grids of the
              assimilation, with variables lat, lon (degrees), M_orig (PALEOMAP
              elevation, m) and M_corrected (adjusted elevation, m). A
              pre-converted <age>Ma_corrected_SW.npz beside each file is used
              first if present (numpy only).
--salles-dir  salles_<age>Ma.npz with ``inputs`` (2, H, W): channel 0 the goSPL
              elevation zpaleo (m), channel 1 the precipitation ``rain`` (m/yr),
              and lat, lon of the 0.1-degree grid (H, W) = (1801, 3601),
              -90..90 and -180..180 degrees.
--ckpt        NeuroLEM checkpoint (best.pt): network weights, architecture
              (in_ch, depth, base) and the input/target normalisation.

Outputs
-------
<out>/erodep_<age>Ma.npz, per age:
  ero_scotese, ero_zhou       (H,W) float32 signed erosion-deposition rate
                              ("erodeporate"), mm/yr (= m/Myr); negative is
                              erosion, positive is deposition
  zpaleo_scotese, zpaleo_zhou (H,W) float32 input elevation, m
  delta                       (H,W) float32 elevation increment applied, m
  rain                        (H,W) float32 precipitation forcing shared by both
                              cases, m/yr
  mask_scotese, mask_zhou     (H,W) bool emergent-land masks (elevation > 0 m)
  lat, lon                    (H,), (W,) float32 degrees
  age                         Ma
  application_mode            the --mode used
<out>/flux_summary.csv, one row per age: integrated gross erosion over the
  emergent land of each surface (Qero_reference_km3yr, Qero_informed_km3yr,
  km3/yr) and the fraction of grid nodes that are land (land_frac_reference,
  land_frac_informed; node count, not area-weighted).

Hardware
--------
Runs on CUDA, then Apple MPS, then CPU. A full 0.1-degree inference needs a GPU
or a large unified-memory machine; a small CPU-only machine runs out of memory.

Usage
-----
  python3 apply_dem.py \\
      --zhou-dir   <Paleotopo_data_assimilation>/data/corrected_scotese_reach500_land \\
      --salles-dir <data archive>/gospl_salles \\
      --ckpt       <data archive>/surrogate_model/best.pt \\
      --out        <output directory> \\
      [--mode delta] [--reuse-baseline-dir <data archive>/surrogate_erodep] [--ages 0 220]

``--reuse-baseline-dir`` takes ero_scotese from existing files after checking
that the reference elevation, precipitation and coordinates are identical, so
only the adjusted case is inferred (as in nature_recast/rebuild_surface.sh).
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np


def _load_model_class(explicit=None):
    """Import GlobalUNet from an explicit file, an installed package, or a checkout.

    ``explicit`` is an absolute path to ``salles_unet.py`` (or its directory) and is
    loaded directly by file location, bypassing sys.path entirely, so the run does
    not depend on PYTHONPATH reaching a detached process or on where the NeuroLEM
    checkout lives. Without it, an importable ``salles_unet`` is used, then a
    ``neurolem-surrogate/neurolem`` checkout two levels above this file, then this
    file's own directory. Returns the ``GlobalUNet`` class.
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
    """Best available torch device name: 'cuda', then 'mps' (Apple silicon), then 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _bilinear(field, lat1, lon1, lat_t, lon_t):
    """Bilinear regrid from a regular (lat1, lon1) grid to (lat_t, lon_t).

    ``field`` has shape (len(lat1), len(lon1)), rows by latitude; the result is
    float32 with shape (len(lat_t), len(lon_t)). The goSPL inputs were regridded
    with periodic xESMF bilinear interpolation; for the smooth ``delta`` field
    this routine agrees with it to within a metre. Both grids span -90..90 and
    -180..180 degrees, so no longitude wrap is needed. Descending source axes
    are flipped to ascending first.
    """
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
            # index of the left source node and fractional position within its interval
            # for every target coordinate
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

    lat and lon are float64 degrees; M_orig (PALEOMAP elevation) and M_corrected
    (rock-chemistry-adjusted elevation) are (lat, lon) arrays in metres.

    Prefers a pre-converted ``.npz`` sidecar (numpy only), then tries xarray,
    netCDF4 and h5py -- netCDF-4 files are HDF5 underneath, so h5py reads them
    too.  The .npz route lets the script run in a torch environment without any
    netCDF library.  All routes return the same four arrays -- nothing about the
    numerics depends on which is used.
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
    """Edge-pad the last two axes at the bottom and right to multiples of ``mult``.

    The U-Net halves the grid ``depth`` times, so both spatial dimensions must be
    divisible by 2**depth (1801 x 3601 becomes 1808 x 3616 for depth 4). Returns
    the padded array and the original (H, W) for cropping the prediction.
    """
    H, W = a.shape[-2:]
    ph, pw = (-H) % mult, (-W) % mult
    if ph == 0 and pw == 0:
        return a, (H, W)
    pad = [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(a, pad, mode="edge"), (H, W)


def main():
    """Parse the command line, run NeuroLEM on both surfaces at every age and write the outputs."""
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
    # Normalisation stored with the checkpoint: the two input channels (elevation,
    # precipitation) are standardised with the training mean and standard deviation;
    # the network predicts a standardised asinh(erodep / s), inverted in infer() below.
    n = ck["norm"]
    in_mean = np.asarray(n["in_mean"], np.float32); in_std = np.asarray(n["in_std"], np.float32)
    s = float(n["asinh_s"]); t_mean = float(n["t_mean"]); t_std = float(n["t_std"])
    mult = 2 ** ck["depth"]                 # grid size must be divisible by this (see _pad_to)
    print(f"[apply_dem] device={dev}  mode={a.mode}  ckpt val_rmse={ck.get('val_rmse')}")

    zdir = Path(a.zhou_dir); sdir = Path(a.salles_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zf = {int(re.search(r"(\d+)Ma_corrected_SW", f.name).group(1)): f
          for f in zdir.glob("*Ma_corrected_SW.nc")
          if not f.name.startswith(".")}   # skip macOS AppleDouble sidecars
    ages = sorted(a.ages) if a.ages else sorted(zf)

    def infer(elev, rain):
        """NeuroLEM erosion-deposition rate (H, W), mm/yr, from elevation (m) and precipitation (m/yr)."""
        x = np.stack([(elev - in_mean[0]) / in_std[0], (rain - in_mean[1]) / in_std[1]], 0).astype(np.float32)
        xp, (H, W) = _pad_to(x, mult)
        with torch.no_grad():
            t = model(torch.from_numpy(xp)[None].to(dev))[0, 0].cpu().numpy()[:H, :W]
        # undo the target standardisation and the asinh compression of the training target
        return (s * np.sinh(t * t_std + t_mean)).astype(np.float32)

    R = 6.371e6; dgrid = np.deg2rad(0.1)                       # Earth radius (m); 0.1-degree spacing (rad)
    # Reference = unadjusted surface (ero_scotese); informed = rock-chemistry-adjusted surface (ero_zhou).
    csv = open(out / "flux_summary.csv", "w")
    csv.write("age,Qero_reference_km3yr,Qero_informed_km3yr,land_frac_reference,land_frac_informed\n")

    for age in ages:
        f = zf.get(age); sp = sdir / f"salles_{age}Ma.npz"
        if f is None or not sp.exists():
            print(f"  [{age}Ma] missing informed DEM or goSPL input — skipped"); continue
        sd = np.load(sp, allow_pickle=True)
        lat_t = sd["lat"].astype(np.float64); lon_t = sd["lon"].astype(np.float64)
        rain = sd["inputs"][1].astype(np.float32)             # precipitation, m/yr; same for both cases
        lat1, lon1, _m_orig, _m_corr = _read_dem(f)
        # 1-degree increment of the rock-chemistry adjustment (m), interpolated to 0.1 degree
        delta_raw = _bilinear(_m_corr - _m_orig, lat1, lon1, lat_t, lon_t)
        if a.mode in ("delta", "delta-fixed-mask"):
            z_sco = sd["inputs"][0].astype(np.float32)        # goSPL in-distribution zpaleo, m
            if a.mode == "delta-fixed-mask":
                from scipy.ndimage import distance_transform_edt
                m_sco = z_sco > 0
                if a.coastal_taper_km > 0:
                    # Distance of every land node from the nearest ocean node, in grid
                    # cells, converted to km with 11.119 km per 0.1 degree of arc
                    # (2*pi*6371 km / 3600). Longitude convergence is not represented in
                    # this first-order taper, but the taper is used only to remove
                    # grid-scale coastal discontinuities.
                    dist_km = distance_transform_edt(m_sco) * 11.119
                    x = np.clip(dist_km / a.coastal_taper_km, 0.0, 1.0)
                    taper = x * x * (3.0 - 2.0 * x)  # C1-continuous smoothstep
                else:
                    taper = m_sco.astype(np.float32)
                # Only reference land is changed, and it stays land: the tapered
                # increment is added and the result held at or above the minimum.
                z_zho = z_sco.copy()
                z_zho[m_sco] = np.maximum(
                    z_sco[m_sco] + (delta_raw * taper)[m_sco], a.minimum_land_elevation)
                delta = z_zho - z_sco
                m_zho = m_sco.copy()
            else:
                # "delta" (the paper's experiment): the increment is added everywhere,
                # so the emergent-land mask of the adjusted surface may differ from
                # the reference one.
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
            # The reference field depends only on z_sco and rain. It is reused from an
            # existing archive only if those inputs and the grid match to 1e-5.
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

        # Integrated continental gross erosion (km^3/yr): Q = sum_land max(-ero,0)*A,
        # summed over the emergent land of each surface; deposition is not netted off.
        # Cell area A = R^2 * dgrid^2 * cos(lat) (m^2), meridional convergence accounted for.
        # ero is in mm/yr, so mm/yr * m^2 = 1e-3 m^3/yr and dividing by 1e12 gives km^3/yr.
        area =(R * R * dgrid * dgrid) * np.cos(np.deg2rad(lat_t))[:, None]  # (H,1)->broadcast
        area = np.broadcast_to(area, e_sco.shape)
        q_sco = float((np.maximum(-e_sco, 0.0) * area)[m_sco].sum() / 1e12)
        q_zho = float((np.maximum(-e_zho, 0.0) * area)[m_zho].sum() / 1e12)
        lf_sco = float(m_sco.mean()); lf_zho = float(m_zho.mean())
        csv.write(f"{age},{q_sco:.6g},{q_zho:.6g},{lf_sco:.6g},{lf_zho:.6g}\n"); csv.flush()
        print(f"  [{age}Ma] done  delta[min,max]=[{delta.min():.0f},{delta.max():.0f}] m  "
              f"Qero sco={q_sco:.4g} informed={q_zho:.4g} km3/yr  "
              f"(informed/Scotese={q_zho/q_sco:.3f}; mask_equal={np.array_equal(m_sco, m_zho)})")

    csv.close()
    print(f"\nWrote {len(ages)} ages to {out}  (mode={a.mode}). ero_* = erodeporate (mm/yr).")
    print(f"Flux time-series: {out/'flux_summary.csv'}")


if __name__ == "__main__":
    main()

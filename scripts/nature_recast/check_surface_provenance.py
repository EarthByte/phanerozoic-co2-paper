#!/usr/bin/env python3
"""Is every archived surface the adopted rock-chemistry adjustment, and nothing else?

Role in the paper: a provenance guard for every result.  The adjusted surface of the paper (Methods,
"Palaeoelevation") is `Paleotopo_data_assimilation/data/corrected_scotese_reach500_land` (PRODUCT
below: cratonic samples excluded, 400 km support, residual kernel tapered to zero between 250 and
500 km from the nearest sample, samples on PALEOMAP sea dropped, continent masks decided on the
sphere), with `_p05` and `_p95` beside it as its bounds.  Earlier configurations of the adjustment
(`corrected_scotese_nocraton` and others) are not used for any result.  This script measures, at
every age, the `delta` layer (adjusted minus PALEOMAP elevation, m) that the archive carries against
the product:

  surrogate_erodep/erodep_<age>Ma.npz            the 0.1-degree surfaces every table of the paper
                                                 is built on, sampled at the 1-degree nodes
  geochem_uncertainty_ensemble/elevation_summary  the production-central field stored with the
                                                 200-member ensemble, and the ensemble median
  dem_scotese_adjusted/<age>Ma_corrected_SW.nc    the 1-degree grids shipped with the archive

and writes ONE table, data/nature_recast/surface_provenance.csv, with the land-mean of both
fields, their RMSE and the largest difference. The land-mean is the discriminant: a matching
archive agrees to the metre (the ensemble central field is packed to 1 m); a different
configuration differs by tens to hundreds of metres in the land-mean and by up to 2 km at single
cells. The script exits non-zero if any archived field is off by more than its tolerance in the
land-mean (ARCHIVE_TOL, or CENTRAL_TOL for the ensemble central field; the ensemble median is
reported but not tested), or if a reference grid does not carry the product's configuration stamp
(CONFIG_MARKS in its `config` attribute; the present-day grid must instead be the uncorrected bypass).

Columns of the table: age; comparison (archive_vs_product, shipped_vs_product,
ensemble_central_vs_product, ensemble_median_vs_product); n_land (continental cells of the 1-degree
reference grid); mean_archive, mean_reference (land-means, m); rmse, maxabs (m); and, on the
archive_vs_product rows, adjustment_landmean_m and adjustment_maxabs_m, the mean and largest
|delta| of the 0.1-degree archive over the emergent land of the adjusted surface.  "Land" is the
continent mask of the reference grid, intersected with the emergent land of the adjusted surface
for the 0.1-degree archive.

    python3 check_surface_provenance.py [--product-dir DIR] [--ages 220 440] [--out FILE]

The 1-degree reference is read from --product-dir (default: the archive's own copy in
dem_scotese_adjusted/, itself checked against PALEO_ROOT/data/<PRODUCT> when that directory is
present).  Pass --out with --ages so a partial check does not overwrite the full table.
"""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc
import _config as cfg

PRODUCT = 'corrected_scotese_reach500_land'
# Configuration strings that every corrected grid of PRODUCT carries in its netCDF 'config' attribute.
CONFIG_MARKS = ('local_support_km=400', 'resid_reach_km=250-500', 'sea_sited_samples=dropped')   # in the files' config attribute
ARCHIVE_TOL = 0.05      # m, land-mean difference tolerated for the 0.1-degree archive (exact copy expected)
CENTRAL_TOL = 0.6       # m, for the ensemble's stored central field (packed as int16 metres)
PALEO_ROOT = Path(os.environ.get('PALEO_ROOT', cfg.REPO.parent.parent / 'Paleotopo_data_assimilation'))


def read(path, var):
    """(field as float, lat, lon, config attribute or '') of variable ``var`` in a netCDF file."""
    with nc.Dataset(path) as d:
        return np.array(d.variables[var][:], dtype=float), np.array(d.variables['lat'][:]), np.array(d.variables['lon'][:]), d.getncattr('config') if 'config' in d.ncattrs() else ''


def compare(a, b, m):
    """Land-means of ``a`` (archive) and ``b`` (reference), RMSE and largest |a - b| over mask ``m`` (m)."""
    a, b = a[m], b[m]
    return dict(mean_archive=a.mean(), mean_reference=b.mean(), rmse=np.sqrt(np.mean((a - b) ** 2)), maxabs=np.abs(a - b).max())


def main():
    """Compare every archived surface with the product at each age, write the table, exit 1 on a mismatch."""
    p = argparse.ArgumentParser()
    p.add_argument('--product-dir', type=Path, default=cfg.ZENODO / 'dem_scotese_adjusted')
    p.add_argument('--ages', type=int, nargs='*', default=None)
    p.add_argument('--out', type=Path, default=cfg.OUT / 'surface_provenance.csv', help='the table; a partial check (--ages) should not overwrite the full one')
    a = p.parse_args()
    ref_dir = a.product_dir
    ages = a.ages or sorted(int(f.name.split('Ma')[0]) for f in ref_dir.glob('*Ma_corrected_SW.nc'))
    source = PALEO_ROOT / 'data' / PRODUCT
    rows, bad, adjustment = [], [], {}
    for age in ages:
        # reference increment (m) and continent mask on the 1-degree grid
        dn, lat1, lon1, config = read(ref_dir / f'{age}Ma_corrected_SW.nc', 'delta')
        cm = read(ref_dir / f'{age}Ma_corrected_SW.nc', 'continent_mask')[0] > 0.5
        if age == 0:
            # the present day is not corrected (M_corrected = M_orig): the file says so instead of a configuration
            if 'present-day bypass' not in config or np.abs(dn[cm]).max() > 0:
                bad.append(f'{age} Ma: present-day grid is not the uncorrected bypass ({config}; max |delta| {np.abs(dn[cm]).max():.1f} m)')
        elif any(mark not in config for mark in CONFIG_MARKS):
            bad.append(f'{age} Ma: reference grid is not {PRODUCT} ({config})')
        checks = {}
        if source.is_dir() and (source / f'{age}Ma_corrected_SW.nc').exists():
            ds = read(source / f'{age}Ma_corrected_SW.nc', 'delta')[0]
            checks['shipped_vs_product'] = compare(dn, ds, cm)
        ep = cfg.ERODEP_DIR / f'erodep_{age}Ma.npz'
        if ep.exists():
            with np.load(ep) as z:
                d = z['delta'].astype(float); lat = z['lat']; lon = z['lon']; mz = z['mask_zhou']
            # indices of the 1-degree nodes within the 0.1-degree grid (every tenth node)
            ii = np.rint((lat1 - lat[0]) / (lat[1] - lat[0])).astype(int); jj = np.rint((lon1 - lon[0]) / (lon[1] - lon[0])).astype(int)
            checks['archive_vs_product'] = compare(d[np.ix_(ii, jj)], dn, cm & mz[np.ix_(ii, jj)])
            adjustment[age] = (float(d[mz].mean()), float(np.abs(d[mz]).max()))     # the adjustment itself, over the emergent land of the adjusted surface
        es = cfg.ZENODO / 'geochem_uncertainty_ensemble' / 'elevation_summary' / f'{age}Ma_geochem_ensemble_summary.nc'
        if es.exists():
            checks['ensemble_central_vs_product'] = compare(read(es, 'delta_production_central')[0], dn, cm)
            checks['ensemble_median_vs_product'] = compare(read(es, 'delta_median_p50')[0], dn, cm)
        for k, v in checks.items():
            rows.append(dict(age=age, comparison=k, n_land=int(cm.sum()), **{q: round(float(x), 3) for q, x in v.items()}))
            off = abs(v['mean_archive'] - v['mean_reference'])
            tol = {'archive_vs_product': ARCHIVE_TOL, 'shipped_vs_product': ARCHIVE_TOL, 'ensemble_central_vs_product': CENTRAL_TOL}.get(k)
            if tol is not None and off > tol:
                bad.append(f'{age} Ma: {k} land-mean off by {off:.2f} m (RMSE {v["rmse"]:.1f} m)')
        print(f'{age:>3} Ma  ' + '  '.join(f'{k.split("_vs_")[0]} {v["mean_archive"]:+8.2f}/{v["mean_reference"]:+8.2f} m rmse {v["rmse"]:6.2f}' for k, v in checks.items()))
    for r in rows:
        if r['comparison'] == 'archive_vs_product' and r['age'] in adjustment:
            r['adjustment_landmean_m'], r['adjustment_maxabs_m'] = (round(x, 2) for x in adjustment[r['age']])
    out = a.out
    with open(out, 'w', newline='') as f:
        fields = list(dict.fromkeys([k for r in rows for k in r]))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('wrote', out)
    if adjustment:
        m = np.array([v[0] for v in adjustment.values()])
        print(f'adjustment over emergent land: mean over ages {m.mean():+.1f} m (per age {m.min():+.1f} to {m.max():+.1f}); largest |adjustment| {max(v[1] for v in adjustment.values()):.0f} m')
    if bad:
        print(f'\nNOT {PRODUCT}:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print(f'every archived surface carries {PRODUCT}')


if __name__ == '__main__':
    main()

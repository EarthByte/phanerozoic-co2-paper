#!/usr/bin/env python3
"""Is every archived erosion field the output of the published NeuroLEM checkpoint on the archived inputs?

For each age NeuroLEM is run on the archived elevation of both surfaces with the archived precipitation and
compared with the archived erosion field (218 passes). ONE table is written,
data/nature_recast/archive_consistency.csv: RMSE and the relative difference in land-integrated erosion.

    python3 check_archive.py                   # report only
    python3 check_archive.py --ages 280 285    # selected ages
    python3 check_archive.py --repair          # also replace fields that fail

--repair writes erodep_<age>Ma.npz afresh with the failing field replaced by the inference and every other
array carried over unchanged, verifies the new file by reading it back, moves the superseded file to
<data>/_to_delete/ (nothing is deleted) and puts the new file in its place. The input cache is then older
than the archive; `./run_uncertainty_m3.sh main` notices that and rebuilds it.

The check confirms that the archived erosion fields, from which every result is computed, are reproducible
from the archived inputs and the published checkpoint. Columns of the table: age, surface (reference, adjusted), rmse_mm_yr (over the whole grid),
land_erosion_inference_over_archive_minus_1 (area-weighted erosion over the land of the surface), consistent
(yes / NO at --tolerance), repaired (date of the repair, or empty). Rows of earlier runs for other ages are
kept. Needs PyTorch and the NeuroLEM model code (climate_uncertainty/mc_driver.py).
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import _config as cfg

CU = cfg.REPO / 'scripts' / 'climate_uncertainty'
if str(CU) not in sys.path:
    sys.path.insert(0, str(CU))
# (surface label, palaeoelevation, erosion-deposition and land-mask arrays in erodep_<age>Ma.npz)
PAIRS = (('reference', 'zpaleo_scotese', 'ero_scotese', 'mask_scotese'), ('adjusted', 'zpaleo_zhou', 'ero_zhou', 'mask_zhou'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ages', nargs='*', type=int)
    p.add_argument('--ckpt', default=str(cfg.ZENODO / 'surrogate_model' / 'best.pt'))
    p.add_argument('--device')
    p.add_argument('--tolerance', type=float, default=1e-5, help='RMSE in mm/yr')
    p.add_argument('--repair', action='store_true')
    a = p.parse_args()
    import mc_driver as mc
    model = mc.Surrogate(a.ckpt, a.device)
    ages = a.ages or list(range(0, 545, 5))
    out = cfg.OUT / 'archive_consistency.csv'
    old = {}
    if out.exists():
        with open(out, newline='') as fh:
            old = {(int(r['age']), r['surface']): r for r in csv.DictReader(fh)}
    failed = []
    for age in ages:
        f = cfg.ERODEP_DIR / f'erodep_{age}Ma.npz'
        with np.load(f, allow_pickle=True) as d:
            arrays = {k: d[k] for k in d.files}
        # cos(latitude) weights: proportional to cell area on the regular grid
        lat = arrays['lat'].astype(float); w = np.cos(np.deg2rad(lat))[:, None]
        fresh = {}
        for surface, zk, ek, mk in PAIRS:
            t0 = time.time()
            # NeuroLEM erosion-deposition (mm/yr) from the archived elevation and precipitation
            ero = model(arrays[zk].astype(np.float32), arrays['rain'].astype(np.float32))
            m = arrays[mk].astype(bool)
            rmse = float(np.sqrt(np.mean((ero - arrays[ek]) ** 2)))
            # area-weighted erosion (negative part of the field) over the land of the surface
            I = lambda x: float(np.sum((np.maximum(-x, 0.0) * w)[m]))
            rel = I(ero) / I(arrays[ek]) - 1.0
            ok = rmse <= a.tolerance
            old[(age, surface)] = dict(age=age, surface=surface, rmse_mm_yr=f'{rmse:.3e}', land_erosion_inference_over_archive_minus_1=f'{rel:+.3e}',
                                       consistent='yes' if ok else 'NO', repaired='')
            print(f'[{age:>3} Ma] {surface:9s} RMSE {rmse:.2e} mm/yr  land erosion {100 * rel:+.3f}%  {"ok" if ok else "MISMATCH"}  {time.time() - t0:.0f} s', flush=True)
            if not ok:
                fresh[ek] = ero.astype(arrays[ek].dtype); failed.append((age, surface))
        # repair: write the new file beside the old one, read it back and compare every array, then move the old file
        # to <data>/_to_delete/ and put the new one in its place
        if fresh and a.repair:
            new = dict(arrays); new.update(fresh)
            tmp = f.with_name(f'.{f.name}.repaired.npz')
            np.savez_compressed(tmp, **new)
            with np.load(tmp, allow_pickle=True) as chk:
                for k in arrays:
                    want = new[k]
                    if not (np.array_equal(chk[k], want) if want.dtype.kind in 'biufc' and want.ndim else str(chk[k]) == str(want)):
                        raise SystemExit(f'read-back of {tmp} differs in {k}; archive left untouched')
            bin_ = cfg.ZENODO / '_to_delete'; bin_.mkdir(exist_ok=True)
            stale = bin_ / f'erodep_{age}Ma_superseded_{time.strftime("%Y%m%d_%H%M%S")}.npz'
            f.rename(stale); tmp.rename(f)
            for surface, zk, ek, mk in PAIRS:
                if ek in fresh:
                    old[(age, surface)]['repaired'] = time.strftime('%Y-%m-%d')
            print(f'[{age:>3} Ma] repaired {", ".join(fresh)}; superseded file moved to {stale}', flush=True)
    rows = [old[k] for k in sorted(old)]
    tmp = out.with_name(f'.{out.name}.tmp')
    with open(tmp, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); wr.writeheader(); wr.writerows(rows)
    tmp.replace(out)
    print(f'wrote {out}: {len(failed)} mismatch(es) in {2 * len(ages)} fields' + (f' -> {failed}' if failed else ''))
    if failed and not a.repair:
        print('re-run with --repair to replace them')


if __name__ == '__main__':
    main()

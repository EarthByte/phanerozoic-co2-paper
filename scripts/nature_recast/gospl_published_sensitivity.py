#!/usr/bin/env python3
"""Erosion sensitivity from the published goSPL erosion fields of the reference surface, all 109 ages.

The Paleo-Physiography Project fields that NeuroLEM was trained on (gospl_salles/salles_<age>Ma.npz, array
`target`: goSPL erosion-deposition on the 0.1-degree grid) are evaluated with exactly the weathering evaluation
of the main model. The result is the erosion sensitivity, the share of land beyond the optimum and the fluxes as
the physics-based model gives them for the unadjusted surface, independent of the surrogate; set beside the
NeuroLEM values of the main run it measures what the surrogate does to each quantity, age by age. Five of the
ages (60, 160, 260, 360, 460 Ma) were withheld from training. CPU only. The comparison is reported in the
Supplementary Information and Extended Data Fig. 6.

Inputs
  gospl_salles/salles_<age>Ma.npz   inputs[0] elevation (m; land = elevation > 0), inputs[1] precipitation
                                    (m/yr), target (goSPL erosion-deposition, mm/yr, negative = erosion),
                                    lat, lon of the 0.1-degree grid (1801 x 3601, lon -180..180)
  valdes_temperature/<age>Ma.npz    HadCM3 surface-air temperature, packed, degC
  surrogate_erodep/erodep_<age>Ma.npz   ero_scotese, for the cell-by-cell correlation
  --main-run                        prefix of the main run whose calibration, E_opt(runoff) table and
                                    per-age results (<prefix>_ens.csv) are used
Runoff is precipitation times the HadCM3L runoff ratio, as in the main model.

Output: ONE table, data/nature_recast/gospl_published_reference_sensitivity.csv, one row per age:
withheld_from_training, the land correlation of the goSPL and NeuroLEM erosion fields, every metric of
_evaluate.METRIC_COLUMNS for the goSPL field, and <metric>_NeuroLEM for the quantities in COMPARE. Rows
are appended age by age, so an interrupted run resumes (--force recomputes all ages). A summary of
goSPL - NeuroLEM is printed at the end.

    python3 gospl_published_sensitivity.py            # all ages, resumable
    python3 gospl_published_sensitivity.py --ages 60 220
"""
import argparse, csv, time
import numpy as np
import _config as cfg
import _evaluate as ev

OUT = cfg.OUT / 'gospl_published_reference_sensitivity.csv'
COMPARE = ['D_Gt', 'F_sil', 'F_net', 'S_sil', 'S_net', 'S_total_rel_sil', 'beyond_frac_eroding', 'WI']
HEADER = ['age', 'withheld_from_training', 'erosion_correlation_land'] + ev.METRIC_COLUMNS + [f'{k}_NeuroLEM' for k in COMPARE]
WITHHELD = {60, 160, 260, 360, 460}          # ages withheld from NeuroLEM training


def main():
    """Evaluate the published goSPL field of every age and append it to the table."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ages', nargs='*', type=int)
    p.add_argument('--main-run', default=str(cfg.OUT / 'kin_bridge_s2'))
    p.add_argument('--out', default=str(OUT))
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    amps, amp_oc, settings, qgrid, eopt_q = ev.load_calibration(a.main_run)
    main_run = {int(float(r['age'])): r for r in csv.DictReader(open(f'{a.main_run}_ens.csv'))}
    done = set() if a.force else ev.blocks_done(a.out, 'age', 1)
    for age in (a.ages or list(range(0, 545, 5))):
        if age in done:
            continue
        t0 = time.time()
        with np.load(cfg.ZENODO / 'gospl_salles' / f'salles_{age}Ma.npz', allow_pickle=True) as s:
            z = s['inputs'][0].astype(np.float32); rain = s['inputs'][1].astype(float); target = np.asarray(s['target'], np.float32)
            lat = s['lat'].astype(float); lon = s['lon'].astype(float)
        # the target may be stored with a leading channel axis (1, H, W)
        target = target[0] if target.ndim == 3 else target
        land = z > 0
        with np.load(cfg.TEMP_DIR / f'{age}Ma.npz') as t:
            T = (t['temp_q'].astype(np.float32) * float(t['scale']) + float(t['offset'])).astype(float) + 273.15
        # runoff, mm/yr: precipitation (m/yr -> mm/yr) times the HadCM3L runoff ratio
        q = rain * 1000.0 * ev.ratio_grid(age, lat, lon)
        # spherical cell areas, m2; the grid repeats the dateline column (lon -180 and +180),
        # so each of the two edge columns carries half a cell
        A = ((6.371e6 ** 2 * np.deg2rad(0.1) ** 2) * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        A[:, 0] *= 0.5; A[:, -1] *= 0.5
        # own land and reference land are the same mask here (the PALEOMAP surface); NaNs in the
        # target (no data) count as zero erosion
        evaluate = ev.Evaluator(np.flatnonzero(land.ravel()), A.ravel(), (amps, amp_oc, qgrid, eopt_q), settings)
        res = evaluate(np.nan_to_num(target).ravel(), land.ravel(), land.ravel(), q.ravel(), T.ravel())
        with np.load(cfg.ERODEP_DIR / f'erodep_{age}Ma.npz', allow_pickle=True) as d:
            # Pearson correlation of the signed erosion-deposition rates over land
            r = float(np.corrcoef(d['ero_scotese'][land], np.nan_to_num(target)[land])[0, 1])
        row = dict(age=age, withheld_from_training=int(age in WITHHELD), erosion_correlation_land=r); row.update(res)
        row.update({f'{k}_NeuroLEM': float(main_run[age][f'{k}_ref']) for k in COMPARE})
        ev.append_rows(a.out, HEADER, [row], 'age')
        print(f'[{age:>3} Ma] S_sil goSPL {res["S_sil"]:.3f} NeuroLEM {row["S_sil_NeuroLEM"]:.3f} | beyond {res["beyond_frac_eroding"]:.3f} / {row["beyond_frac_eroding_NeuroLEM"]:.3f} '
              f'| D {res["D_Gt"]:.1f} / {row["D_Gt_NeuroLEM"]:.1f} Gt/yr | r {r:.2f} | {time.time() - t0:.0f} s', flush=True)
    rows = list(csv.DictReader(open(a.out)))
    for k in ('S_sil', 'S_net', 'beyond_frac_eroding', 'D_Gt', 'F_sil'):
        g = np.array([float(r[k]) for r in rows]); n = np.array([float(r[f'{k}_NeuroLEM']) for r in rows])
        w = np.array([int(r['withheld_from_training']) for r in rows], bool)
        print(f'  {k:20s} goSPL - NeuroLEM: mean {np.mean(g - n):+.4f}, range {np.min(g - n):+.4f} to {np.max(g - n):+.4f} ({len(rows)} ages)'
              + (f'; withheld ages mean {np.mean((g - n)[w]):+.4f}' if w.any() else ''))


if __name__ == '__main__':
    main()

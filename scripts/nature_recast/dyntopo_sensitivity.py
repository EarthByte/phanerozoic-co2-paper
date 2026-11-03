#!/usr/bin/env python3
"""Sensitivity of the weathering results to dynamic topography, 250 Ma to the present (Supplementary Information).

The term is the change in dynamic topography since the present day predicted by the mantle-flow model gld428 of
Young et al. (2022), carried point by point onto EVERY continental cell of the adjusted PALEOMAP (Scotese-Wright) surface (Paleotopo
scripts_Scotese/dyntopo_term_lagrangian.py: Scotese-Wright plate at t, present-day position, M21 plate, M21 mantle-frame
position at t, DT(t) - DT(0) from the 5-Myr mantle-frame grids low-passed at 600 km), with the level of each continental
block removed (Paleotopo_data_assimilation, outputs_scotese_reach500_land/dt_term_all_crust/dt_term_all_lagr_0-250Ma.nc,
variable term_robust, which here holds the term on all crust; 1 degree, 51 ages, 0 at 0 Ma, NaN off the continents).
Design and decision rule were fixed before any run (2026-10-03).
2026-10-10: replaces the term applied only on the plates where the transfer error was small (dt_term_robust_0-250Ma.nc,
37-90 % of continental area, and 8-18 % of land without a value even before that mask); those outputs are in to_delete.

Two surfaces per age, from the adjusted surface M of the archive (zpaleo_zhou) and the term interpolated bilinearly to
0.1 degree (NaN set to 0 first), the way the long-wavelength rock-chemistry increment is added (apply_dem.py):
  shoreline  (primary)  on land of M: max(M + term, 1 m); sea unchanged, so land and sea stay as reconstructed
  free       (bound)    M + term everywhere, so shorelines move; evaluated on its own land and on the land of both
NeuroLEM runs on both with the archived runoff forcing; the weathering is evaluated with the calibration, runoff placement
and local optimum of the main run, never refitted; climate is unchanged, as for the two palaeoelevation histories.

  run      (GPU; numpy + torch + netCDF4)  per age: the two surfaces; NeuroLEM on M, which must reproduce the archived
           field, and on both surfaces; weathering of all three (the re-evaluated adjusted surface must reproduce the main
           run). ONE csv, appended age by age and resumable (data/nature_recast/dyntopo_gld428_allcrust.csv); the surfaces
           and their erosion fields go to ONE zip per surface in <data archive>/dyntopo_gld428/, stamped
           dyntopo=gld428_allcrust_<surface>. The main archive is only read.
  analyse  (laptop)  changes against the main run, lowland measures, the comparators (climate states, the two
           palaeoelevation histories, the elevation ensemble) and the decision rule; writes
           dyntopo_sensitivity_by_age.csv, dyntopo_sensitivity_table.csv and dyntopo_sensitivity_summary.md.

    python3 dyntopo_sensitivity.py run --ages 0 150       # pilot: checks, one perturbed age
    python3 dyntopo_sensitivity.py run                    # 51 ages; re-run the same command to resume
    python3 dyntopo_sensitivity.py analyse
    python3 dyntopo_sensitivity.py run --term <control>.nc --tag <name> --no-surfaces   # a control term: own table, no surfaces kept
    python3 dyntopo_sensitivity.py analyse --csv ../../data/nature_recast/dyntopo_<name>.csv
    python3 dyntopo_sensitivity.py controls             # the reading of the six control terms (tables dyntopo_control_<name>.csv)

Decision rule (fixed 2026-10-03): the result is WITHIN EXISTING UNCERTAINTY if, for the shoreline surface, for both the
silicate sensitivity and the share of eroding land beyond the optimum, (a) the change in the 5-250 Ma mean is at most half
the width of the climate-state 5-95% range of that mean, and (b) the change is within that age's climate-state 5-95% width
at 45 or more of the 50 ages; otherwise MATERIAL.
"""
from __future__ import annotations

import argparse, csv, io, os, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import _config as cfg
import _evaluate as ev
import run_weathering as rw
import weathering_core as wc

PALEO = Path(os.environ.get('PALEO_ROOT', cfg.REPO.parents[1] / 'Paleotopo_data_assimilation'))
TERM = PALEO / 'outputs_scotese_reach500_land' / 'dt_term_all_crust' / 'dt_term_all_lagr_0-250Ma.nc'
DATA = Path(os.environ.get('NATURE_DYNTOPO', cfg.ZENODO / 'dyntopo_gld428'))
OUT_CSV = cfg.OUT / 'dyntopo_gld428_allcrust.csv'
SURFACES = ('adjusted', 'shoreline', 'free')
STAMP = 'gld428_allcrust_{}'
LOWLAND_M = 1500.0
MIN_LAND_M = 1.0
WINDOW = (5, 250)
CHECK = ['land_Mkm2', 'D_Gt', 'F_sil', 'F_net', 'S_sil', 'S_net', 'beyond_frac_eroding']
LOW = ['lowland_eroding_Mkm2', 'lowland_beyond_frac', 'lowland_x_median', 'lowland_F_sil_share',
       'below_opt_lowland_frac', 'below_opt_lowland_frac_robust']
SURF = ['term_rms_land_m', 'term_mean_land_m', 'dz_rms_land_m', 'robust_frac_land', 'land_to_sea_Mkm2', 'sea_to_land_Mkm2']
HEADER = ['age', 'surface', 'stamp'] + SURF + ev.METRIC_COLUMNS + LOW


# ---------------------------------------------------------------------------------------------------------- inputs
def read_term(path, zero_at_0=True):
    """ages, lat, lon, term (age, lat, lon; NaN off the continents), robust (age, lat, lon)."""
    try:
        from netCDF4 import Dataset
        with Dataset(str(path)) as d:
            d.set_auto_mask(False)
            g = {k: np.asarray(d.variables[k][:]) for k in ('age', 'lat', 'lon', 'term_robust', 'robust')}
    except ImportError:
        import h5py
        with h5py.File(str(path), 'r') as h:
            g = {k: np.asarray(h[k][:]) for k in ('age', 'lat', 'lon', 'term_robust', 'robust')}
    ages = np.rint(g['age']).astype(int)
    if ages.tolist() != list(range(0, 255, 5)):
        raise SystemExit(f'{path}: expected the 51 ages 0-250 Ma, found {ages.tolist()}')
    term = g['term_robust'].astype(np.float64)
    if zero_at_0 and np.nanmax(np.abs(term[0])) != 0:
        raise SystemExit(f'{path}: the term must vanish at 0 Ma')
    return ages, g['lat'].astype(np.float64), g['lon'].astype(np.float64), term, g['robust'].astype(bool)


def surfaces(zM, landM, t01):
    """The shoreline and free surfaces from the adjusted surface and the 0.1-degree term."""
    z_sh = zM.copy()
    z_sh[landM] = np.maximum(zM[landM] + t01[landM], MIN_LAND_M)
    z_fr = (zM + t01).astype(np.float32)
    return z_sh.astype(np.float32), z_fr


# ---------------------------------------------------------------------------------------------------------- fields
def local_state(evaluate, erodep, mask, q, T):
    """Per-cell erosion rate, local net sensitivity and local optimum on the evaluator's cells (main-run calibration)."""
    c = evaluate.cells
    m = np.asarray(mask)[c]; qq = np.asarray(q, float)[c]; TT = np.asarray(T, float)[c]
    E = np.where(m, np.maximum(-np.asarray(erodep)[c].astype(float), 0.0), 0.0) * evaluate.scale
    dsil = dsulf = sil = 0.0
    for r in rw.SETS:
        t = rw.region_terms(E, qq, TT, evaluate.fitsets[r], evaluate.mode)
        dsil = dsil + evaluate.amps[r]['sil'] * t['dsil'] / len(rw.SETS)
        dsulf = dsulf + evaluate.amps[r]['sulf'] * t['dsulf'] / len(rw.SETS)
        sil = sil + evaluate.amps[r]['sil'] * t['sil'] / len(rw.SETS)
    Eopt = rw.eopt_at(qq, qq, evaluate.eopt_q, evaluate.qgrid)
    return m, E, dsil - dsulf, Eopt, sil


def _wmedian(x, w):
    if x.size == 0:
        return float('nan')
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def lowland(evaluate, erodep, mask, q, T, low, robust):
    """Measures on eroding land below 1,500 m of the adjusted surface (low: bool on the evaluator's cells)."""
    m, E, loc, Eopt, sil = local_state(evaluate, erodep, mask, q, T)
    A = evaluate.A
    ero = m & (E > 0); beyond = ero & (loc <= 0); below = ero & ~beyond
    L = ero & low
    aL = float(A[L].sum())
    return dict(lowland_eroding_Mkm2=aL / 1e12,
                lowland_beyond_frac=float(A[L & beyond].sum()) / aL,
                lowland_x_median=_wmedian((E / Eopt)[L], A[L]),
                lowland_F_sil_share=float(np.sum((sil * A)[L])) / float(np.sum((sil * A)[m])),
                below_opt_lowland_frac=float(A[below & low].sum()) / float(A[below].sum()),
                below_opt_lowland_frac_robust=float(A[below & low & robust].sum()) / max(float(A[below & low].sum()), 1e-30))


# ---------------------------------------------------------------------------------------------------------- archive
def zip_path(surface):
    return DATA / f'surfaces_{STAMP.format(surface)}.zip'


def zip_members(surface):
    """Ages held by the surface archive; a damaged archive is moved aside (its ages are recomputed)."""
    p = zip_path(surface)
    if not p.exists():
        return set()
    try:
        with zipfile.ZipFile(p) as z:
            return {int(n.split('_')[1][:3]) for n in z.namelist()}
    except (zipfile.BadZipFile, OSError, ValueError):
        aside = DATA / '_to_delete' / f'{p.stem}_damaged_{time.strftime("%Y%m%d_%H%M%S")}.zip'
        aside.parent.mkdir(parents=True, exist_ok=True); p.rename(aside)
        print(f'  {p.name} could not be read: moved to {aside}; its ages are recomputed')
        return set()


def zip_append(surface, _age, **arrays):
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays)
    with zipfile.ZipFile(zip_path(surface), 'a', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        name = f'{surface}_{int(_age):03d}Ma.npz'
        if name not in z.namelist():
            z.writestr(name, buf.getvalue())


def write_readme():
    p = DATA / 'README.md'
    if p.exists():
        return
    p.write_text(
        '# Dynamic-topography sensitivity surfaces (CO2 paper, Supplementary Information)\n\n'
        'Written by `CO2_paper/scripts/nature_recast/dyntopo_sensitivity.py run`. Not part of the main archive.\n\n'
        'Term: change in gld428 dynamic topography since the present day (Young et al. 2022), carried onto the\n'
        'Scotese-Wright continents point by point on all continental crust, block level removed\n'
        '(Paleotopo_data_assimilation outputs_scotese_reach500_land/dt_term_all_crust/dt_term_all_lagr_0-250Ma.nc,\n'
        'term_robust), bilinear from 1 degree to 0.1 degree, NaN set to 0 first.\n\n'
        f'- `surfaces_{STAMP.format("shoreline")}.zip`: land of the adjusted surface M: max(M + term, {MIN_LAND_M:g} m); sea unchanged.\n'
        f'- `surfaces_{STAMP.format("free")}.zip`: M + term everywhere (shorelines move).\n\n'
        'One member per age (`<surface>_<age>Ma.npz`), on the cells that are land in the surface or in M:\n'
        '`cells` = np.packbits of that mask on the 0.1-degree grid of the archive (row-major, `shape`; unpack with\n'
        'np.unpackbits(cells)[:shape.prod()]), `z` (m) and `erodep` (NeuroLEM, units and sign of ero_zhou in the archive) on\n'
        'those cells in order, `land` = np.packbits of the land mask of the surface on them (first `n_cells` bits), `stamp`, `age`.\n'
        'Runoff forcing and climate are those of the archive.\n')


# ---------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    global STAMP
    if a.tag:                                  # a control term (e.g. shifted in longitude or time): own table, own stamp
        STAMP = a.tag + '_{}'
        if a.out == str(OUT_CSV):
            a.out = str(cfg.OUT / f'dyntopo_{a.tag}.csv')
    keep = not a.no_surfaces
    import climate_sensitivity as cs          # puts the climate_uncertainty folder on the path
    if a.unet_py:
        cs._register_unet(a.unet_py)
    import mc_driver as mc
    from apply_dem import _bilinear
    tages, tlat, tlon, term, robust1 = read_term(a.term, zero_at_0=not a.tag)
    ages = sorted(int(x) for x in a.ages) if a.ages else tages.tolist()
    if not set(ages) <= set(tages.tolist()):
        raise SystemExit(f'ages outside 0-250 Ma in steps of 5: {sorted(set(ages) - set(tages.tolist()))}')
    amps, amp_oc, settings, qgrid, eopt_q = ev.load_calibration(a.main_run)
    cal = (amps, amp_oc, qgrid, eopt_q)
    main = {int(float(r['age'])): r for r in csv.DictReader(open(f'{a.main_run}_ens.csv'))}
    if keep:
        DATA.mkdir(parents=True, exist_ok=True); write_readme()
    held = {s: zip_members(s) for s in ('shoreline', 'free')} if keep else None
    done = ev.blocks_done(a.out, 'age', len(SURFACES))
    if keep:
        done &= held['shoreline'] & held['free']
    model = mc.Surrogate(a.ckpt, a.device)
    print(f'[dyntopo] device={model.dev} ages={len(ages)} term={a.term}\n  out={a.out}\n  surfaces={DATA if keep else "not kept"}', flush=True)
    t_start = time.monotonic(); n_new = 0
    for age in ages:
        if age in done and not a.force:
            continue
        if a.time_budget and time.monotonic() - t_start > a.time_budget:
            print(f'  [{age:>3} Ma] time budget reached; re-run to continue'); break
        t0 = time.time()
        with np.load(cfg.ERODEP_DIR / f'erodep_{age}Ma.npz', allow_pickle=True) as d:
            lat = d['lat'].astype(float); lon = d['lon'].astype(float)
            rain = d['rain'].astype(np.float32)
            zM = d['zpaleo_zhou'].astype(np.float32)
            landM = d['mask_zhou'].astype(bool); mref = d['mask_scotese'].astype(bool)
            archived = d['ero_zhou'].astype(np.float32)
        if not np.array_equal(landM, zM > 0):
            raise SystemExit(f'[{age} Ma] mask_zhou of the archive is not zpaleo_zhou > 0')
        k = int(np.flatnonzero(tages == age)[0])
        t1 = np.nan_to_num(term[k], nan=0.0)
        t01 = _bilinear(t1, tlat, tlon, lat, lon).astype(np.float32)
        r01 = _bilinear(robust1[k].astype(np.float64), tlat, tlon, lat, lon) >= 0.5
        if age == 0 and not a.tag and np.abs(t01).max() != 0:
            raise SystemExit('the interpolated term does not vanish at 0 Ma')
        z_sh, z_fr = surfaces(zM, landM, t01)
        m_sh = z_sh > 0; m_fr = z_fr > 0
        if not np.array_equal(m_sh, landM):
            raise SystemExit(f'[{age} Ma] the shoreline surface changed the land mask')
        ero_M = model(zM, rain)
        rmse = float(np.sqrt(np.mean((ero_M - archived) ** 2)))
        if rmse > a.reference_tolerance:
            raise SystemExit(f'[{age} Ma] NeuroLEM on the adjusted surface does not reproduce the archived field: RMSE {rmse:.3e} > '
                             f'{a.reference_tolerance:g}. Check the checkpoint and the archive (./run_uncertainty_m3.sh archive-check).')
        ero_sh = model(z_sh, rain)
        ero_fr = ero_M.copy() if np.abs(t01).max() == 0 else model(z_fr, rain)      # with no term the free surface is M itself
        with np.load(cfg.TEMP_DIR / f'{age}Ma.npz') as t:
            T_K = ((t['temp_q'].astype(np.float64) * float(t['scale']) + float(t['offset'])) + 273.15).ravel()
        ratio = ev.ratio_grid(age, lat, lon)
        q = (rain.astype(np.float64) * 1000.0 * ratio).ravel()
        A = ((6.371e6 ** 2 * np.deg2rad(0.1) ** 2) * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        A[:, 0] *= 0.5; A[:, -1] *= 0.5
        cells = np.flatnonzero((mref | landM | m_fr).ravel())
        evaluate = ev.Evaluator(cells, A.ravel(), cal, settings)
        lowc = (zM.ravel()[cells] < LOWLAND_M)
        robc = r01.ravel()[cells]
        Ac = A.ravel()[cells]; lm = landM.ravel()[cells]

        def surfstats(z, m):
            zc = z.ravel()[cells]; tc = t01.ravel()[cells]; zMc = zM.ravel()[cells]; mc_ = m.ravel()[cells]
            w = Ac[lm]
            return dict(term_rms_land_m=float(np.sqrt(np.sum(tc[lm] ** 2 * w) / w.sum())),
                        term_mean_land_m=float(np.sum(tc[lm] * w) / w.sum()),
                        dz_rms_land_m=float(np.sqrt(np.sum((zc - zMc)[lm] ** 2 * w) / w.sum())),
                        robust_frac_land=float(np.sum(w[robc[lm]]) / w.sum()),
                        land_to_sea_Mkm2=float(Ac[lm & ~mc_].sum()) / 1e12, sea_to_land_Mkm2=float(Ac[~lm & mc_].sum()) / 1e12)

        jobs = (('adjusted', ero_M, zM, landM, landM, lm), ('shoreline', ero_sh, z_sh, m_sh, landM, lm),
                ('free', ero_fr, z_fr, m_fr, landM & m_fr, lm & m_fr.ravel()[cells]))
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            fut = {s: (pool.submit(evaluate, e.ravel(), m.ravel(), common.ravel(), q, T_K),
                       pool.submit(lowland, evaluate, e.ravel(), common.ravel(), q, T_K, lowc & lmask, robc))
                   for s, e, z, m, common, lmask in jobs}
            rows = []
            for s, e, z, m, common, lmask in jobs:
                row = dict(age=age, surface=s, stamp='adjusted' if s == 'adjusted' else f'dyntopo={STAMP.format(s)}')
                row.update(surfstats(z, m)); row.update(fut[s][0].result()); row.update(fut[s][1].result())
                rows.append(row)
        if age in main:
            for kk in CHECK:
                want = float(main[age][f'{kk}_adj'])
                if abs(rows[0][kk] - want) > 1e-3 * abs(want) + 1e-5:
                    raise SystemExit(f'[{age} Ma] re-evaluated adjusted {kk} = {rows[0][kk]:.8g}, main run has {want:.8g}')
        for s, e, z, m in ((('shoreline', ero_sh, z_sh, m_sh), ('free', ero_fr, z_fr, m_fr)) if keep else ()):
            if age not in held[s] or a.force:
                u = (m | landM).ravel()
                zip_append(s, age, cells=np.packbits(u), shape=np.array(zM.shape), z=z.ravel()[u], erodep=e.ravel()[u],
                           land=np.packbits(m.ravel()[u]), n_cells=int(u.sum()), stamp=f'dyntopo={STAMP.format(s)}', age=age)
                held[s].add(age)
        ev.append_rows(a.out, HEADER, rows, 'age')
        n_new += 1
        b = {r['surface']: r for r in rows}
        print(f'  [{age:>3} Ma] term rms {b["shoreline"]["term_rms_land_m"]:5.0f} m  '
              f'S_sil {b["adjusted"]["S_sil"]:.3f} -> {b["shoreline"]["S_sil"]:.3f} (free {b["free"]["S_sil"]:.3f})  '
              f'beyond {b["adjusted"]["beyond_frac_eroding"]:.3f} -> {b["shoreline"]["beyond_frac_eroding"]:.3f} (free {b["free"]["beyond_frac_eroding"]:.3f})  '
              f'F_sil {100 * (b["shoreline"]["F_sil"] / b["adjusted"]["F_sil"] - 1):+.1f}%  archive reproduced  {time.time() - t0:.0f} s', flush=True)
    complete = ev.blocks_done(a.out, 'age', len(SURFACES))
    if keep:
        complete &= zip_members('shoreline') & zip_members('free')
    missing = sorted(set(ages) - complete)
    print(f'{n_new} ages computed now; {len(set(ages) & complete)}/{len(ages)} complete in {a.out}'
          + (f'; missing {missing[:15]}: re-run the same command' if missing else ''))


# ---------------------------------------------------------------------------------------------------------- analyse
def _climate_members(nodes):
    """Climate-state members of the main analysis (same nodes, n, seed and correlations as climate_sensitivity.py sample)."""
    import climate_sensitivity as cs
    ns = argparse.Namespace(nodes=str(nodes), n=2000, seed=0, co2_time_correlation='independent', sigma_time_correlation='full')
    return cs.sample_series(ns)


def cmd_analyse(a):
    import pandas as pd
    stem = 'dyntopo_sensitivity' if Path(a.csv).resolve() == OUT_CSV.resolve() else f'{Path(a.csv).stem}_sensitivity'
    d = pd.read_csv(a.csv)
    counts = d.groupby('age').size()
    ages = sorted(int(x) for x in counts.index[counts == len(SURFACES)])
    win = [x for x in ages if WINDOW[0] <= x <= WINDOW[1]]
    expected = list(range(WINDOW[0], WINDOW[1] + 1, 5))
    partial = win != expected
    if partial and not a.allow_partial:
        raise SystemExit(f'{len(win)} of {len(expected)} ages of {WINDOW[0]}-{WINDOW[1]} Ma complete: finish the run '
                         '(or --allow-partial for a provisional look; no decision is drawn from it)')
    S = {s: d[d.surface == s].set_index('age').loc[ages] for s in SURFACES}
    adj = S['adjusted']
    main = pd.read_csv(f'{a.main_run}_ens.csv').set_index('age')
    bands = pd.read_csv(cfg.OUT / 'climate_sensitivity_bands.csv').set_index('age')
    ens = pd.read_csv(cfg.OUT / 'ensemble_sensitivity_age_summary.csv').set_index('age')

    # per-age changes
    rows = []
    for x in ages:
        r = dict(age=x)
        for s in ('shoreline', 'free'):
            b = S[s].loc[x]; m0 = adj.loc[x]
            r[f'F_sil_pct_{s}'] = 100 * (b.F_sil / m0.F_sil - 1)
            r[f'F_net_pct_{s}'] = 100 * (b.F_net / m0.F_net - 1)
            r[f'S_sil_{s}'] = b.S_sil - m0.S_sil
            r[f'S_net_{s}'] = b.S_net - m0.S_net
            r[f'beyond_pp_{s}'] = 100 * (b.beyond_frac_eroding - m0.beyond_frac_eroding)
            r[f'lowland_beyond_pp_{s}'] = 100 * (b.lowland_beyond_frac - m0.lowland_beyond_frac)
            r[f'lowland_x_median_ratio_{s}'] = b.lowland_x_median / m0.lowland_x_median
            r[f'D_pct_{s}'] = 100 * (b.D_Gt / m0.D_Gt - 1)
            if s == 'free':
                r['F_sil_pct_free_common'] = 100 * (b.F_sil_common / m0.F_sil - 1)
                r['S_sil_free_common'] = b.S_sil_common - m0.S_sil
                r['beyond_pp_free_common'] = 100 * (b.beyond_frac_eroding_common - m0.beyond_frac_eroding)
        r['lowland_x_median_adjusted'] = adj.loc[x].lowland_x_median
        r['lowland_beyond_frac_adjusted'] = adj.loc[x].lowland_beyond_frac
        r['below_opt_lowland_frac_adjusted'] = adj.loc[x].below_opt_lowland_frac
        r['term_rms_land_m'] = S['shoreline'].loc[x].term_rms_land_m
        r['clim_width_S_sil'] = bands.loc[x, 'S_sil_adj_p95'] - bands.loc[x, 'S_sil_adj_p05']
        r['clim_width_beyond_pp'] = 100 * (bands.loc[x, 'beyond_frac_eroding_adj_p95'] - bands.loc[x, 'beyond_frac_eroding_adj_p05'])
        r['clim_width_F_sil_pct'] = 100 * (bands.loc[x, 'F_sil_adj_p95'] - bands.loc[x, 'F_sil_adj_p05']) / bands.loc[x, 'F_sil_adj_central']
        r['ens_width_S_sil'] = ens.loc[x, 'S_sil_p95'] - ens.loc[x, 'S_sil_p05']
        r['ens_width_beyond_pp'] = 100 * (ens.loc[x, 'beyond_frac_eroding_p95'] - ens.loc[x, 'beyond_frac_eroding_p05'])
        r['adj_minus_ref_S_sil'] = main.loc[x, 'S_sil_adj'] - main.loc[x, 'S_sil_ref']
        r['adj_minus_ref_beyond_pp'] = 100 * (main.loc[x, 'beyond_frac_eroding_adj'] - main.loc[x, 'beyond_frac_eroding_ref'])
        r['adj_minus_ref_F_sil_pct'] = 100 * (main.loc[x, 'F_sil_adj'] / main.loc[x, 'F_sil_ref'] - 1)
        rows.append(r)
    by = pd.DataFrame(rows).set_index('age')
    W = by.loc[win]

    # window-mean comparators: the climate-state members and the elevation-ensemble members over the same ages
    nd, cages, series = _climate_members(cfg.OUT / 'climate_sensitivity_nodes.csv')
    ci = [list(cages).index(x) for x in win]
    for k in ('S_sil', 'beyond_frac_eroding'):            # the members must reproduce the published bands
        s_ = series[('adj', k)]
        p05 = np.quantile(s_, .05, axis=0); p95 = np.quantile(s_, .95, axis=0)
        want05 = bands.loc[list(cages), f'{k}_adj_p05'].to_numpy(); want95 = bands.loc[list(cages), f'{k}_adj_p95'].to_numpy()
        if not (np.allclose(p05, want05, rtol=1e-9, atol=1e-12) and np.allclose(p95, want95, rtol=1e-9, atol=1e-12)):
            raise SystemExit(f'climate members do not reproduce climate_sensitivity_bands.csv for {k}: re-run '
                             './run_uncertainty_m3.sh finalise (the nodes changed after the bands were drawn)')
    cm = {k: series[('adj', k)][:, ci].mean(axis=1) for k in ('S_sil', 'beyond_frac_eroding', 'F_sil')}
    clim_mean_width = {'S_sil': float(np.quantile(cm['S_sil'], .95) - np.quantile(cm['S_sil'], .05)),
                       'beyond_pp': 100 * float(np.quantile(cm['beyond_frac_eroding'], .95) - np.quantile(cm['beyond_frac_eroding'], .05))}
    em = pd.read_csv(cfg.ZENODO / 'geochem_uncertainty_ensemble' / 'nature_recast' / 'ensemble_sensitivity_members.csv')
    em = em[(em.member >= 0) & em.age.isin(win)].groupby('member')[['S_sil', 'beyond_frac_eroding']].mean()
    ens_mean_width = {'S_sil': float(em.S_sil.quantile(.95) - em.S_sil.quantile(.05)),
                      'beyond_pp': 100 * float(em.beyond_frac_eroding.quantile(.95) - em.beyond_frac_eroding.quantile(.05))}

    # decision rule (shoreline surface)
    rule = {}
    for key, col, wcol in (('S_sil', 'S_sil_shoreline', 'clim_width_S_sil'), ('beyond_pp', 'beyond_pp_shoreline', 'clim_width_beyond_pp')):
        dm = float(W[col].mean())
        ok_a = abs(dm) <= 0.5 * clim_mean_width[key]
        n_in = int((W[col].abs() <= W[wcol]).sum())
        ok_b = n_in >= 45
        rule[key] = dict(change_of_mean=dm, half_climate_width_of_mean=0.5 * clim_mean_width[key], a=ok_a,
                         ages_within=n_in, n_ages=len(win), b=ok_b)
    within = all(v['a'] and v['b'] for v in rule.values())
    outcome = 'WITHIN EXISTING UNCERTAINTY' if within else 'MATERIAL'
    if partial:
        outcome = f'PROVISIONAL ({len(win)} of {len(expected)} ages): no decision'

    # Supplementary Table Sx
    def stat(col):
        v = W[col]; return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    T = []
    for label, col in (('Silicate sink (%)', 'F_sil_pct'), ('Net inorganic sink (%)', 'F_net_pct'), ('Silicate sensitivity S_sil', 'S_sil'),
                       ('Net sensitivity S_net', 'S_net'), ('Share of eroding land beyond the optimum (pp)', 'beyond_pp'),
                       ('Lowland share beyond the optimum (pp)', 'lowland_beyond_pp'),
                       ('Lowland median E/E_opt (ratio)', 'lowland_x_median_ratio'), ('Erosion flux (%)', 'D_pct')):
        row = dict(quantity=label)
        for s in ('shoreline', 'free'):
            st = stat(f'{col}_{s}'); row.update({f'{s}_{k}': v for k, v in st.items()})
        if col in ('F_sil_pct', 'S_sil', 'beyond_pp'):
            fc = {'F_sil_pct': 'F_sil_pct_free_common', 'S_sil': 'S_sil_free_common', 'beyond_pp': 'beyond_pp_free_common'}[col]
            st = stat(fc); row.update({f'free_common_land_{k}': v for k, v in st.items()})
        if col == 'S_sil':
            row.update(climate_width_median=float(W.clim_width_S_sil.median()), climate_width_of_mean=clim_mean_width['S_sil'],
                       ensemble_width_median=float(W.ens_width_S_sil.median()), ensemble_width_of_mean=ens_mean_width['S_sil'],
                       adjusted_minus_reference_mean=float(W.adj_minus_ref_S_sil.mean()))
        if col == 'beyond_pp':
            row.update(climate_width_median=float(W.clim_width_beyond_pp.median()), climate_width_of_mean=clim_mean_width['beyond_pp'],
                       ensemble_width_median=float(W.ens_width_beyond_pp.median()), ensemble_width_of_mean=ens_mean_width['beyond_pp'],
                       adjusted_minus_reference_mean=float(W.adj_minus_ref_beyond_pp.mean()))
        if col == 'F_sil_pct':
            row.update(climate_width_median=float(W.clim_width_F_sil_pct.median()), adjusted_minus_reference_mean=float(W.adj_minus_ref_F_sil_pct.mean()))
        T.append(row)
    T = pd.DataFrame(T)
    by.to_csv(cfg.OUT / f'{stem}_by_age.csv', float_format='%.6g')
    T.to_csv(cfg.OUT / f'{stem}_table.csv', index=False, float_format='%.6g')

    f = lambda v, n=3: f'{v:.{n}f}'
    L = [f'# Dynamic-topography sensitivity (gld428, all continental crust), {WINDOW[0]}-{WINDOW[1]} Ma', '',
         f'Written by dyntopo_sensitivity.py analyse from {Path(a.csv).name} ({len(ages)} ages complete).', '',
         f'## Outcome of the decision rule: {outcome}', '']
    for key, v in rule.items():
        L.append(f'- {key}: change of the {WINDOW[0]}-{WINDOW[1]} Ma mean {v["change_of_mean"]:+.4f} against half the climate-state 5-95% '
                 f'width of that mean {v["half_climate_width_of_mean"]:.4f} -> (a) {"holds" if v["a"] else "fails"}; within the climate-state '
                 f'width at {v["ages_within"]} of {v["n_ages"]} ages -> (b) {"holds" if v["b"] else "fails"} (needs 45)')
    sh, fr = 'shoreline', 'free'
    L += ['', '## Numbers for supplement Sx.4 and Sx.5', '',
          f'- Share of eroding land below the optimum that lies below 1,500 m (adjusted surface): mean {100 * W.below_opt_lowland_frac_adjusted.mean():.0f}% '
          f'({100 * W.below_opt_lowland_frac_adjusted.min():.0f}-{100 * W.below_opt_lowland_frac_adjusted.max():.0f}% at single ages).',
          f'- Shoreline surface: silicate sink {W[f"F_sil_pct_{sh}"].mean():+.2f}% ({W[f"F_sil_pct_{sh}"].min():+.2f} to {W[f"F_sil_pct_{sh}"].max():+.2f}%), '
          f'S_sil {W[f"S_sil_{sh}"].mean():+.4f} ({W[f"S_sil_{sh}"].min():+.4f} to {W[f"S_sil_{sh}"].max():+.4f}), '
          f'S_net {W[f"S_net_{sh}"].mean():+.4f} ({W[f"S_net_{sh}"].min():+.4f} to {W[f"S_net_{sh}"].max():+.4f}), '
          f'share beyond {W[f"beyond_pp_{sh}"].mean():+.2f} pp ({W[f"beyond_pp_{sh}"].min():+.2f} to {W[f"beyond_pp_{sh}"].max():+.2f}).',
          f'- Free surface, own land: silicate sink {W[f"F_sil_pct_{fr}"].mean():+.2f}%, S_sil {W[f"S_sil_{fr}"].mean():+.4f}, '
          f'S_net {W[f"S_net_{fr}"].mean():+.4f}, share beyond {W[f"beyond_pp_{fr}"].mean():+.2f} pp; on land of both surfaces: silicate sink '
          f'{W.F_sil_pct_free_common.mean():+.2f}%, S_sil {W.S_sil_free_common.mean():+.4f}, share beyond {W.beyond_pp_free_common.mean():+.2f} pp.',
          f'- Lowland (< 1,500 m) eroding land: share beyond {100 * W.lowland_beyond_frac_adjusted.mean():.1f}% on the adjusted surface, change '
          f'{W[f"lowland_beyond_pp_{sh}"].mean():+.2f} pp shoreline ({W[f"lowland_beyond_pp_{sh}"].min():+.2f} to {W[f"lowland_beyond_pp_{sh}"].max():+.2f}), '
          f'{W[f"lowland_beyond_pp_{fr}"].mean():+.2f} pp free; median E/E_opt {W.lowland_x_median_adjusted.mean():.3f} on the adjusted surface, '
          f'ratio {W[f"lowland_x_median_ratio_{sh}"].mean():.3f} shoreline ({W[f"lowland_x_median_ratio_{sh}"].min():.3f}-{W[f"lowland_x_median_ratio_{sh}"].max():.3f}), '
          f'{W[f"lowland_x_median_ratio_{fr}"].mean():.3f} free.',
          f'- Comparators, S_sil: climate-state 5-95% width median {W.clim_width_S_sil.median():.4f} per age, {clim_mean_width["S_sil"]:.4f} of the mean; '
          f'elevation ensemble {W.ens_width_S_sil.median():.4f} per age, {ens_mean_width["S_sil"]:.4f} of the mean; adjusted minus reference '
          f'{W.adj_minus_ref_S_sil.mean():+.4f} (mean), |.| {W.adj_minus_ref_S_sil.abs().mean():.4f}.',
          f'- Comparators, share beyond (pp): climate {W.clim_width_beyond_pp.median():.2f} per age, {clim_mean_width["beyond_pp"]:.2f} of the mean; '
          f'ensemble {W.ens_width_beyond_pp.median():.2f} per age, {ens_mean_width["beyond_pp"]:.2f} of the mean; adjusted minus reference '
          f'{W.adj_minus_ref_beyond_pp.mean():+.2f} (mean), |.| {W.adj_minus_ref_beyond_pp.abs().mean():.2f}.',
          f'- Ratio of the shoreline change to the adjusted-minus-reference difference (mean |.| over ages): S_sil '
          f'{W[f"S_sil_{sh}"].abs().mean() / W.adj_minus_ref_S_sil.abs().mean():.2f}, share beyond '
          f'{W[f"beyond_pp_{sh}"].abs().mean() / W.adj_minus_ref_beyond_pp.abs().mean():.2f}.',
          f'- Term on land of the adjusted surface (0.1 degree): rms {W.term_rms_land_m.median():.0f} m median over ages.',
          '', f'Per age: {stem}_by_age.csv. Supplementary Table Sx: {stem}_table.csv.']
    (cfg.OUT / f'{stem}_summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------------------------------------- controls
CONTROLS = ('tsh075', 'tsh150', 'tsh225', 'lon060', 'lon120', 'lon180')
CONTROL_DIR = PALEO / 'outputs_scotese_reach500_land' / 'dt_term_all_crust'


def _changes(path):
    """Per-age changes of the shoreline surface against the adjusted surface in one run table."""
    import pandas as pd
    d = pd.read_csv(path)
    n = d.groupby('age').size(); ages = sorted(int(x) for x in n.index[n == len(SURFACES)])
    a = d[d.surface == 'adjusted'].set_index('age').loc[ages]; s = d[d.surface == 'shoreline'].set_index('age').loc[ages]
    return pd.DataFrame({'S_sil': s.S_sil - a.S_sil, 'beyond_pp': 100 * (s.beyond_frac_eroding - a.beyond_frac_eroding),
                         'F_sil_pct': 100 * (s.F_sil / a.F_sil - 1), 'S_net': s.S_net - a.S_net})


def cmd_controls(a):
    """The reading of the control terms fixed by Paleotopo before the runs (controls.json): with at least three admitted
    controls, per metric (change of the 5-250 Ma mean of S_sil and of the share beyond the optimum, shoreline surface),
    GEOGRAPHY if |real| exceeds |control| for every admitted control, otherwise GENERIC; fewer than three: INCONCLUSIVE."""
    import json
    import pandas as pd
    adm = json.loads(Path(a.admission).read_text())
    win = list(range(WINDOW[0], WINDOW[1] + 1, 5))
    real = _changes(OUT_CSV)
    if list(real.index[real.index.isin(win)]) != win:
        raise SystemExit('the real-term run is incomplete')
    cols = {'real': real}
    for c in a.names:
        f = cfg.OUT / f'dyntopo_control_{c}.csv'
        if not f.exists():
            print(f'  {c}: no table ({f.name}); not run'); continue
        ch = _changes(f)
        if list(ch.index[ch.index.isin(win)]) != win:
            print(f'  {c}: incomplete ({ch.index.isin(win).sum()} of {len(win)} ages); left out'); continue
        cols[c] = ch
    admitted = [c for c in cols if c != 'real' and adm['controls'].get(c, {}).get('admitted')]
    by = pd.concat({k: v.loc[win] for k, v in cols.items()}, axis=1)
    by.columns = [f'{m}_{k}' for k, m in by.columns]
    by.to_csv(cfg.OUT / 'dyntopo_controls_by_age.csv', float_format='%.6g')
    mean = {k: v.loc[win].mean() for k, v in cols.items()}
    verdict = {}
    for m in ('S_sil', 'beyond_pp'):
        if len(admitted) < 3:
            verdict[m] = 'INCONCLUSIVE'
        else:
            verdict[m] = 'GEOGRAPHY' if all(abs(mean['real'][m]) > abs(mean[c][m]) for c in admitted) else 'GENERIC'
    L = [f'# Dynamic-topography control terms, {WINDOW[0]}-{WINDOW[1]} Ma (shoreline surface)', '',
         'Reading fixed before the runs (Paleotopo controls.json): ' + adm.get('reading', ''), '',
         '| term | admitted | tilt ratio | S_sil change of mean | share beyond, pp | silicate sink, % | S_net | ages S_sil lower | ages share higher |',
         '|---|---|---|---|---|---|---|---|---|']
    for k, v in cols.items():
        info = adm['controls'].get(k, {})
        tilt = adm['real_tilt_ratio_lowland_median_5_250'] if k == 'real' else info.get('tilt_ratio_lowland_median', float('nan'))
        w = v.loc[win]
        L.append(f'| {k} | {"(real term)" if k == "real" else ("yes" if info.get("admitted") else "no")} | {tilt:.3f} | {w.S_sil.mean():+.4f} | '
                 f'{w.beyond_pp.mean():+.2f} | {w.F_sil_pct.mean():+.2f} | {w.S_net.mean():+.4f} | {(w.S_sil < 0).sum()} | {(w.beyond_pp > 0).sum()} |')
    L += ['', f'Admitted controls: {", ".join(admitted) or "none"}.', '']
    for m in verdict:
        L.append(f'- {m}: {verdict[m]} (real {mean["real"][m]:+.4f}; admitted ' +
                 ', '.join(f'{c} {mean[c][m]:+.4f}' for c in admitted) + ')')
    L += ['', 'Per age: dyntopo_controls_by_age.csv.']
    (cfg.OUT / 'dyntopo_controls_summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('run')
    q.add_argument('--ages', nargs='*')
    q.add_argument('--term', default=str(TERM))
    q.add_argument('--out', default=str(OUT_CSV))
    q.add_argument('--main-run', default=str(cfg.OUT / 'kin_bridge_s2'))
    q.add_argument('--ckpt', default=str(cfg.ZENODO / 'surrogate_model' / 'best.pt'))
    q.add_argument('--unet-py', help='salles_unet.py to load by file location')
    q.add_argument('--device')
    q.add_argument('--reference-tolerance', type=float, default=1e-5, help='RMSE allowed against the archived erosion field (archive units)')
    q.add_argument('--workers', type=int, default=3, help='CPU threads for the weathering step')
    q.add_argument('--time-budget', type=float, help='seconds; stop cleanly after the age in progress')
    q.add_argument('--force', action='store_true', help='recompute ages already complete')
    q.add_argument('--tag', help='name of a control term (e.g. gld428_lonshift_p30): table dyntopo_<tag>.csv, surfaces stamped <tag>_<surface>; '
                                 'the term need not vanish at 0 Ma')
    q.add_argument('--no-surfaces', action='store_true', help='do not keep the surfaces and erosion fields (controls)')
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser('controls')
    q.add_argument('--names', nargs='*', default=list(CONTROLS))
    q.add_argument('--admission', default=str(CONTROL_DIR / 'controls.json'))
    q.set_defaults(fn=cmd_controls)
    q = sub.add_parser('analyse')
    q.add_argument('--csv', default=str(OUT_CSV))
    q.add_argument('--main-run', default=str(cfg.OUT / 'kin_bridge_s2'))
    q.add_argument('--allow-partial', action='store_true')
    q.set_defaults(fn=cmd_analyse)
    a = p.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Erosion sensitivity of the carbon sink for every member of the palaeoelevation ensemble.

The 200-member elevation ensemble of the paper (Methods, "Uncertainty and sensitivity", item 5; Extended
Data Fig. 6; Supplementary Information, "Uncertainty ensembles"); run age by age by
run_ensemble_sensitivity_m3.sh and summarised by collate_ensemble_sensitivity.py.

Consuming half of the geochemical palaeoelevation-uncertainty experiment under the weathering
formulation of the Nature-format manuscript. It reads one temporary 1-degree member cube written by
``../build_geochem_elevation_ensemble.py``, transfers every member correction to the 0.1-degree
physiography, runs NeuroLEM once per member (the routines of ``../apply_geochem_ensemble.py``), and
evaluates the same quantities as ``run_weathering.py`` for the main model: silicate, sulfide and
net inorganic fluxes and the two organic pathways, erosion sensitivities, the share of eroding land at or beyond
its local optimum, kinetically limited share, weathering intensity and riverine d7Li.

Consistency with the main model
  * runoff = rain x HadCM3L runoff ratio of the same age (``hadcm3l_runoff_scotese_spinupa.npz``)
  * runoff dependence inside the kinetic term; three regional parameter sets kept intact
  * calibration amplitudes and the E_opt(runoff) table are READ from the main run
    (``<main>_amplitudes.txt``, ``<main>_hist.npz``); nothing is recalibrated per member
  * cell areas with the duplicate dateline column halved

Rows written per age: member -2 is the unadjusted reference surface (zero correction), member -1 the
mean correction of the ensemble, members 0..N-1 the ensemble. Each member is integrated over its own
emergent land and, with suffix ``_common``, over the emergent land of the reference surface.

Verification: the cube carries the production central correction (``delta_central``, the grid
passed to the generator with --central-dir). Before the members are evaluated, NeuroLEM run on that
correction must reproduce the archived adjusted erosion field of the age (``surrogate_erodep/erodep_<age>Ma.npz``) and the
evaluation must reproduce the main-run table; otherwise the age is rejected, because the ensemble
would then not be bracketing the main model.

Storage: ONE csv for the whole experiment. The rows of an age are appended in a single write after
the age has finished; rows of an interrupted age are never written. Nothing else is kept.

Columns (HEADER): age, member, random_seed, isostatic_model (density structure of the member, or 'reference' /
'ensemble_mean'), Qero_surface_km3yr and land_area_old_Mkm2 (erosion volume flux and land area as computed by
../apply_geochem_ensemble.py), then the metrics of run_weathering.summarise on own land (fluxes Gt C/yr,
sensitivities and shares dimensionless), the _common metrics on the land of the reference surface, and the
silicate sensitivity and sink of each parameter set.

Inputs: the member cube (--members-nc), the goSPL physiography and precipitation of the age
(<data>/gospl_salles/salles_<age>Ma.npz), HadCM3L temperature (valdes_temperature), the HadCM3L runoff table,
the NeuroLEM checkpoint, the main run and the archived erosion field (for the check).

    python apply_ensemble_sensitivity.py --members-nc <cube.nc> --out-csv <members.csv> [--main-run PREFIX]
                                         [--workers N] [--no-verify] [--force]
    python apply_ensemble_sensitivity.py --list-done N --out-csv <members.csv>    # ages complete for N members
"""
from __future__ import annotations

import argparse, ast, csv, io, json, os, sys, time, re
from pathlib import Path

import numpy as np

import _config as cfg                       # adds ../ (apply_geochem_ensemble, apply_dem) to the path
_SURROGATE = Path(os.environ.get('NEUROLEM_SURROGATE', cfg.REPO.parent / 'neurolem-surrogate' / 'neurolem'))
if str(_SURROGATE) not in sys.path:
    sys.path.append(str(_SURROGATE))        # optional NeuroLEM checkout; paleomap_unet.py itself is in scripts/

import weathering_core as wc
import run_weathering as rw
from build_cache import runoff_ratio_field, R_EARTH

# columns of run_weathering.summarise kept per member on its own land and (suffix _common) on the reference land
KEEP = ['land_Mkm2', 'eroding_Mkm2', 'E_mean_eroding_mm', 'q_mean_mm', 'D_Gt',
        'F_sil', 'F_sulf', 'F_net', 'F_ocp', 'F_ocb', 'S_sil', 'S_sulf', 'S_net',
        'S_net_oc_rel_sil', 'S_total_rel_sil', 'Gamma_sil', 'WI', 'd7Li_riv',
        'beyond_frac_eroding', 'beyond_oc_frac_eroding', 'silflux_share_beyond', 'Eopt_mean_mm']
KEEP_COMMON = ['D_Gt', 'F_sil', 'F_net', 'S_sil', 'S_net', 'beyond_frac_eroding']
ID = ['age', 'member', 'random_seed', 'isostatic_model', 'Qero_surface_km3yr', 'land_area_old_Mkm2']
HEADER = (ID + KEEP + [f'{k}_common' for k in KEEP_COMMON] + [f'S_sil_{r}' for r in rw.SETS]
          + [f'F_sil_{r}' for r in rw.SETS])


def load_calibration(main_prefix):
    """Amplitudes and E_opt(runoff) of the main run; the run settings are returned for cross-checking."""
    lines = Path(f'{main_prefix}_amplitudes.txt').read_text().strip().splitlines()
    amps, amp_oc = ast.literal_eval(lines[0]), ast.literal_eval(lines[1])
    settings = dict(re.findall(r"(\w+)=(\{[^}]*\}|\S+)", lines[2]))     # values may be dict reprs with spaces (driver={...})
    # every surface is evaluated with the weathering form of the paired main run. The sulfide runoff exponent is carried over
    # by setting weathering_core.BETA for the rest of the process (the main run uses the West form, 0; a file without the
    # field means the published -0.75). The other forms a run can take (reference-runoff scaling, precipitation-driven sets,
    # log-stretch) are not implemented in the evaluators, so a main run in one of those forms is refused rather than
    # silently paired with them
    wc.BETA['sulfide'] = float(settings.get('beta_sulf', '-0.75'))
    if (settings.get('qref_scale', 'none') != 'none' or "'precip'" in settings.get('driver', '')
            or float(settings.get('stretch', 1)) != 1.0):
        raise SystemExit(f'{main_prefix}: the main run uses a weathering form the experiments do not carry yet '
                         f'(beta_sulf={settings.get("beta_sulf")}, qref_scale={settings.get("qref_scale")}, driver={settings.get("driver")}, stretch={settings.get("stretch", 1)}); '
                         'generalise _evaluate.Evaluator and apply_ensemble_sensitivity.Evaluator before running them')
    with np.load(f'{main_prefix}_hist.npz') as h:
        qgrid, eopt_q = h['qgrid'].astype(float), h['eopt_q'].astype(float)
    return amps, amp_oc, settings, qgrid, eopt_q


def area_corrected(lat, nlon):
    """Spherical cell areas (m2) of the 0.1-degree grid, len(lat) x nlon, the duplicate dateline columns halved
    (as in build_cache.py)."""
    dg = np.deg2rad(0.1)
    a = (R_EARTH ** 2 * dg * dg * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, nlon))
    a[:, 0] *= 0.5; a[:, -1] *= 0.5
    return a


class Evaluator:
    """Weathering of one age on the cells that are land in ANY surface considered.

    cells  flat indices of those cells; q runoff (mm/yr), T temperature (K) and A cell area (m2) on the full flattened
    grid, fixed for the age (climate is the same for every member); cal = (amps, amp_oc, qgrid, eopt_q) of the main
    run; mode, oc_form and scale (erosion calibration factor) as in the main run.
    """

    def __init__(self, cells, q, T, A, cal, mode, oc_form, scale):
        self.cells, self.q, self.T, self.A = cells, q[cells], T[cells], A[cells]
        self.amps, self.amp_oc, self.qgrid, self.eopt_q = cal
        self.mode, self.oc_form, self.scale = mode, oc_form, scale
        self.fitsets = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}

    def __call__(self, erodep, mask, ref_mask):
        """Metrics of one member: erodep (mm/yr, negative = erosion), its land mask and the reference land mask,
        all on the full flattened grid. Returns a dict with the metric columns of HEADER."""
        c = self.cells
        m = mask[c]; mref = ref_mask[c]
        E = np.where(m | mref, np.maximum(-erodep[c].astype(float), 0.0), 0.0) * self.scale
        terms = {r: rw.region_terms(E, self.q, self.T, self.fitsets[r], self.mode) for r in rw.SETS}
        cal = {r: {k: self.amps[r]['sil' if 'sil' in k else 'sulf'] * terms[r][k] for k in ('sil', 'dsil', 'sulf', 'dsulf')}
               for r in rw.SETS}
        ens = {k: sum(cal[r][k] for r in rw.SETS) / len(rw.SETS) for k in ('sil', 'dsil', 'sulf', 'dsulf')}
        g = sum(terms[r]['g'] * cal[r]['sil'] for r in rw.SETS) / np.maximum(sum(cal[r]['sil'] for r in rw.SETS), 1e-300)
        Jp, dJp = wc.oc_petro(E, self.oc_form); Jb, dJb = wc.oc_bio_burial(E)
        ocp, docp = self.amp_oc['ocp'] * Jp / 12.011, self.amp_oc['ocp'] * dJp / 12.011
        ocb, docb = self.amp_oc['ocb'] * Jb / 12.011, self.amp_oc['ocb'] * dJb / 12.011
        args = (ens['sil'], ens['dsil'], ens['sulf'], ens['dsulf'], g, ocp, docp, ocb, docb, self.eopt_q, self.qgrid)
        own = rw.summarise(E, self.q, m, self.A, *args)
        com = rw.summarise(E, self.q, mref, self.A, *args)
        out = {k: own[k] for k in KEEP}
        out.update({f'{k}_common': com[k] for k in KEEP_COMMON})
        Am = self.A[m]
        for r in rw.SETS:
            Fs = float(np.sum(cal[r]['sil'][m] * Am)); dFs = float(np.sum(cal[r]['dsil'][m] * Am))
            out[f'S_sil_{r}'] = dFs / Fs; out[f'F_sil_{r}'] = Fs * wc.GTC
        return out


def ages_done(path, n_rows):
    """{age: row count} for the ages of the member table that hold exactly n_rows rows (complete ages)."""
    if not path.exists():
        return {}
    import pandas as pd
    d = pd.read_csv(path, usecols=['age', 'member'])
    return {int(a): int(n) for a, n in d.groupby('age').size().items() if int(n) == n_rows}


def append_age(path, rows):
    """One write per age. A file that holds rows of an interrupted or superseded age is rebuilt beside
    the original and swapped in atomically, so a crash can never damage completed ages."""
    import pandas as pd
    age = rows[0]['age']
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HEADER, lineterminator='\n')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        w.writeheader(); w.writerows(rows)
        tmp = path.with_name(f'.{path.name}.tmp'); tmp.write_text(buf.getvalue()); tmp.replace(path)
        return
    old = pd.read_csv(path, usecols=['age'])
    w.writerows(rows)
    if (old.age == age).any():
        full = pd.read_csv(path)
        tmp = path.with_name(f'.{path.name}.tmp')
        with open(tmp, 'w') as fh:
            full[full.age != age].to_csv(fh, index=False, lineterminator='\n'); fh.write(buf.getvalue())
        tmp.replace(path)
    else:
        with open(path, 'a') as fh:
            fh.write(buf.getvalue()); fh.flush(); os.fsync(fh.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--members-nc')
    p.add_argument('--list-done', type=int, metavar='N', help='print the ages already complete for N members and exit')
    p.add_argument('--out-csv', required=True, help='the ONE member table of the experiment')
    p.add_argument('--salles-dir', default=str(cfg.ZENODO / 'gospl_salles'))
    p.add_argument('--ckpt', default=str(cfg.ZENODO / 'surrogate_model' / 'best.pt'))
    p.add_argument('--main-run', default=str(cfg.OUT / 'kin_bridge_s2'), help='prefix of the main run whose calibration is used')
    p.add_argument('--reference-tolerance', type=float, default=1e-4, help='RMSE (mm/yr) allowed between NeuroLEM on the central correction and the archived adjusted erosion field')
    p.add_argument('--workers', type=int, default=6, help='CPU threads for the weathering step')
    p.add_argument('--no-verify', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    if a.list_done is not None:
        print(' '.join(str(x) for x in sorted(ages_done(Path(a.out_csv), a.list_done + 2)))); return
    if not a.members_nc:
        p.error('--members-nc is required')

    import netCDF4 as nc
    import torch
    from apply_dem import _device
    from apply_geochem_ensemble import _area_m2, _infer_member, _load_model, _prepare_constant_tensors, _qero_km3yr

    with nc.Dataset(a.members_nc) as ds:
        age = int(round(float(ds.target_age_Ma)))
        model_idx = np.asarray(ds.variables['isostatic_model_index'][:], int)
        model_names = json.loads(ds.variables['isostatic_model_index'].model_names)
        deltas = np.asarray(ds.variables['delta_member'][:], np.float32)
        seed = int(ds.random_seed)
        delta_central = np.asarray(ds.variables['delta_central'][:], np.float32) if 'delta_central' in ds.variables else None
    n = len(deltas)
    out_csv = Path(a.out_csv)
    if age in ages_done(out_csv, n + 2) and not a.force:
        print(f'[{age} Ma] already complete in {out_csv}'); return

    amps, amp_oc, settings, qgrid, eopt_q = load_calibration(a.main_run)
    mode, oc_form, scale = settings['mode'], settings['oc_form'], float(settings['scale'])
    if settings['qsrc'] != 'bridge':
        raise SystemExit('the main run must use the HadCM3L runoff ratio (--qsrc bridge)')

    # reference physiography (m) and precipitation forcing (m/yr) on the 0.1-degree grid: the inputs of NeuroLEM
    with np.load(Path(a.salles_dir) / f'salles_{age}Ma.npz', allow_pickle=True) as sd:
        z_base = sd['inputs'][0].astype(np.float32); rain = sd['inputs'][1].astype(np.float32)
        lat = sd['lat'].astype(float); lon = sd['lon'].astype(float)
    shape = z_base.shape
    with np.load(cfg.TEMP_DIR / f'{age}Ma.npz') as t:
        T = (t['temp_q'].astype(np.float32) * float(t['scale']) + float(t['offset'])).astype(float) + 273.15
    # runoff (mm/yr) = precipitation x HadCM3L runoff ratio, as in the main model
    ratio, puma = runoff_ratio_field(age, lat, lon)
    q = rain.astype(float) * 1000.0 * ratio
    # A_new: areas of the weathering evaluation; A_old: areas of apply_geochem_ensemble.py, used only for the
    # Qero_surface_km3yr and land_area_old_Mkm2 columns
    A_new = area_corrected(lat, lon.size); A_old = _area_m2(lat, shape)
    # land of the reference (unadjusted) surface
    ref_mask = z_base > 0

    verify = not a.no_verify
    if verify and delta_central is None:
        raise SystemExit(f'[{age} Ma] the member cube carries no delta_central: rebuild it with the current build_geochem_elevation_ensemble.py')
    archived = main_row = None
    if verify:
        with np.load(cfg.ERODEP_DIR / f'erodep_{age}Ma.npz', allow_pickle=True) as d:
            archived = d['ero_zhou'].astype(np.float32)
        main_rows = {int(round(float(r['age']))): r for r in csv.DictReader(open(f'{a.main_run}_ens.csv'))}
        main_row = main_rows.get(age)

    dev = _device()
    model, ck, norm = _load_model(a.ckpt, dev)
    z_t, rain_t = _prepare_constant_tensors(z_base, rain, norm, dev)
    print(f'[{age} Ma] device={dev} members={n} runoff={puma} mode={mode} scale={scale} oc={oc_form} '
          f'verify={"yes" if verify else "no"}', flush=True)

    # land in any surface: bound it by the largest correction so the weathering arrays stay small
    dmax = np.max(deltas, axis=0)
    e0, m_up = _infer_member(model, dmax, z_t, rain_t, shape, norm, dev)
    cells = np.flatnonzero((m_up | ref_mask).ravel())
    ev = Evaluator(cells, q.ravel(), T.ravel(), A_new.ravel(), (amps, amp_oc, qgrid, eopt_q), mode, oc_form, scale)
    refm = ref_mask.ravel()

    if verify:
        # the main model must sit inside this experiment: NeuroLEM on the production correction gives the archived field
        e_c, m_c = _infer_member(model, delta_central, z_t, rain_t, shape, norm, dev)
        rmse = float(np.sqrt(np.mean((e_c - archived) ** 2)))
        if rmse > a.reference_tolerance:
            raise SystemExit(f'[{age} Ma] NeuroLEM on the central correction does not reproduce the archived adjusted erosion field: '
                             f'RMSE {rmse:.3e} mm/yr > {a.reference_tolerance:g}. The archive and --central-dir are not the same surface: '
                             'rebuild the archive on the production grids (run_uncertainty_m3.sh rebuild-surface) before running the ensemble.')
        rc = ev(e_c.ravel(), m_c.ravel(), refm)
        if main_row is not None:
            for k in ('S_sil', 'S_net', 'beyond_frac_eroding', 'F_sil', 'F_net'):
                want = float(main_row[f'{k}_adj'])
                if abs(rc[k] - want) > 1e-3 * abs(want) + 1e-5:
                    raise SystemExit(f'[{age} Ma] central member {k} = {rc[k]:.8g}, main run has {want:.8g}: the ensemble is not evaluated as the main model')
        print(f'[{age} Ma] central correction reproduces the archived erosion field (RMSE {rmse:.2e} mm/yr) and the main run', flush=True)

    # member -2: zero correction (reference surface); member -1: mean correction of the ensemble; then every member
    specials = [(-2, 'reference', np.zeros_like(deltas[0])), (-1, 'ensemble_mean', deltas.mean(axis=0))]
    todo = specials + [(j, model_names[model_idx[j]], deltas[j]) for j in range(n)]
    # inference runs on the accelerator in this thread; the weathering of finished members runs
    # beside it on CPU threads (numpy releases the GIL), a few members in flight at most
    from concurrent.futures import ThreadPoolExecutor
    pending, rows, t0 = [], [], time.time()

    def collect(block):
        while pending and (block or pending[0][1].done() or len(pending) > 2 * a.workers):
            r, fut = pending.pop(0); r.update(fut.result()); rows.append(r)
            j = r['member']
            if j >= 0 and (j + 1) % max(1, n // 10) == 0:
                print(f'  [{age:>3} Ma] member {j + 1}/{n}  S_sil {r["S_sil"]:.3f}  beyond {r["beyond_frac_eroding"]:.3f}  '
                      f'{(time.time() - t0) / len(rows):.1f} s/member', flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for j, name, delta in todo:
            erodep, mask = _infer_member(model, delta, z_t, rain_t, shape, norm, dev)
            if np.any(mask & ~(m_up | ref_mask)):
                raise RuntimeError('member land outside the bounding land mask')
            qero = _qero_km3yr(erodep, mask, A_old); land_old = float(A_old[mask].sum() / 1e12)
            r = dict(age=age, member=j, random_seed=seed, isostatic_model=name, Qero_surface_km3yr=qero, land_area_old_Mkm2=land_old)
            pending.append((r, pool.submit(ev, erodep.ravel(), mask.ravel(), refm)))
            collect(False)
        collect(True)
    append_age(out_csv, rows)
    # rows keep the order of submission, so rows[0] is the reference surface and rows[2:] the members
    S = np.array([r['S_sil'] for r in rows[2:]]); B = np.array([r['beyond_frac_eroding'] for r in rows[2:]])
    print(f'[{age} Ma] S_sil reference {rows[0]["S_sil"]:.3f}, ensemble p05/p50/p95 {np.quantile(S, .05):.3f}/{np.quantile(S, .5):.3f}/{np.quantile(S, .95):.3f}; '
          f'beyond {np.quantile(B, .05):.3f}/{np.quantile(B, .5):.3f}/{np.quantile(B, .95):.3f}', flush=True)
    del model, z_t, rain_t
    if dev == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()

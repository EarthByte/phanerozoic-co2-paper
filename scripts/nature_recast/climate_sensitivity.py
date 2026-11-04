#!/usr/bin/env python3
"""Climate-forcing uncertainty of the erosion sensitivity (27 climate states per age).

The climate-state experiment of the paper (Methods, "Uncertainty and sensitivity", item 6; Extended Data
Fig. 6; Supplementary Information, "Uncertainty ensembles"). Same design as
../climate_uncertainty/mc_driver.py, under the weathering formulation of the Nature-format manuscript. The climate perturbation is a two-parameter family: r = log2(pCO2 / pCO2_ref), spanning the
proxy envelope of reconstructed CO2, and z, a spatially coherent draw of the structural, fit and
internal-variability spread of the climate model. Stage A evaluates the expensive chain on a 9 x 3 grid
of (r, z) states at every age; stage B interpolates any number of Monte Carlo members from it.

  nodes   (GPU; numpy + torch only)  for every age and state: perturb precipitation and temperature with
          the PhanDA pattern-scaling emulator, run NeuroLEM on BOTH surfaces with the perturbed
          precipitation, and evaluate fluxes, erosion sensitivities and the share of land beyond the
          optimum with the calibration of the main run. Runoff = perturbed precipitation x the runoff
          ratio of the reference simulation. ONE csv, appended age by age; resumable.
          Checks at the unperturbed state (r = 0, z = 0): the surrogate reproduces the archived erosion
          fields, and the evaluation reproduces the main-run table.
  sample  (CPU, seconds)  draws members, interpolates, writes the age bands and the headline table.

Inputs: archived elevation, precipitation and erosion fields (surrogate_erodep/erodep_<age>Ma.npz), HadCM3L
temperature (valdes_temperature), the emulator (phanda_sensitivity.npz), the pCO2 envelope
(data/pco2_envelope.csv: age, co2_lo, co2_hi in ppm), the filled runoff-ratio table (fill_runoff_ratio.py),
the NeuroLEM checkpoint (surrogate_model/best.pt) and the main run (kin_bridge_s2_amplitudes.txt, _hist.npz,
_ens.csv).

Outputs (data/nature_recast/):
  climate_sensitivity_nodes.csv      one row per age, state and surface: node indices i_r, i_z; r (CO2 doublings from
                                     the forcing simulation); z (standard deviations of the model spread); co2_ppm;
                                     surface (ref, adj); T_land_C and rain_land_m (unweighted means over the reference
                                     land, deg C and m/yr); the metrics of _evaluate.METRIC_COLUMNS
  climate_sensitivity_bands.csv      per age: central (r = 0, z = 0) value and p05 / p50 / p95 across members of the
                                     metrics in BAND for both surfaces, and of the adjusted-to-reference ratios
  climate_sensitivity_headline.csv   per metric and surface: p05 / p50 / p95 across members of the time-mean, minimum
                                     and maximum; median over ages of the p05-p95 width; the number of members with a
                                     higher share beyond the optimum today than at 480 Ma

    python3 climate_sensitivity.py nodes --ages 240            # pilot, one age
    python3 climate_sensitivity.py nodes                       # all ages; re-run to resume
    python3 climate_sensitivity.py sample --n 2000
"""
from __future__ import annotations

import argparse, csv, importlib.util, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import _config as cfg
import _evaluate as ev

CU = cfg.REPO / 'scripts' / 'climate_uncertainty'
if str(CU) not in sys.path:
    sys.path.insert(0, str(CU))

NODES_CSV = cfg.OUT / 'climate_sensitivity_nodes.csv'
# identifying columns of the node table; HEADER adds the metrics of the shared evaluator
ID = ['age', 'i_r', 'i_z', 'r', 'z', 'co2_ppm', 'surface', 'T_land_C', 'rain_land_m']
HEADER = ID + ev.METRIC_COLUMNS
# metrics that must match the main-run table at the unperturbed state
CHECK = ['land_Mkm2', 'D_Gt', 'F_sil', 'F_net', 'S_sil', 'S_net', 'beyond_frac_eroding']
# metrics carried into the Monte Carlo bands
BAND = ['D_Gt', 'F_sil', 'F_net', 'F_ocp', 'S_sil', 'S_net', 'S_total_rel_sil', 'beyond_frac_eroding', 'WI', 'd7Li_riv',
        'S_sil_common', 'beyond_frac_eroding_common']


def _register_unet(path):
    """Load the model file by location (--unet-py), so that a chosen network definition is used whatever else is
    importable. A directory must hold paleomap_unet.py, or salles_unet.py as the file is named in the NeuroLEM package."""
    f = Path(path)
    if f.is_dir():
        f = f / 'paleomap_unet.py' if (f / 'paleomap_unet.py').exists() else f / 'salles_unet.py'
    if not f.exists():
        raise SystemExit(f'--unet-py: no such file: {f}')
    spec = importlib.util.spec_from_file_location('paleomap_unet', f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.modules['paleomap_unet'] = mod
    print(f'GlobalUNet from {f}')


def cmd_nodes(a):
    """Stage A: evaluate the (r, z) node grid at every age and append one block of rows per age (resumable)."""
    from phanda_emulator import ClimateEmulator, basegrid_axes, load_envelope, regrid_bilinear
    import mc_driver as mc
    if a.unet_py:
        _register_unet(a.unet_py)
    emu = ClimateEmulator.load(a.sensitivity)
    ages = sorted(int(x) for x in a.ages) if a.ages else list(range(0, 545, 5))
    # pCO2 envelope (ppm) at each age; the r nodes span it in log2 about the CO2 of the forcing simulation
    lo, hi = load_envelope('csv', np.array(ages), csv_path=a.envelope_csv)
    # components of the climate-model spread combined into z: internal variability, fit residual, structural
    terms = tuple(t for t in a.sigma_terms.split(',') if t)
    clip = tuple(a.runoff_scale_clip)
    amps, amp_oc, settings, qgrid, eopt_q = ev.load_calibration(a.main_run)
    main = {int(float(r['age'])): r for r in csv.DictReader(open(f'{a.main_run}_ens.csv'))}
    n_states = None
    done = set()
    model = mc.Surrogate(a.ckpt, a.device)
    print(f'[nodes] device={model.dev} ages={len(ages)} grid={a.n_r} r x {a.n_z} z  sigma={",".join(terms)}  '
          f'mode={settings["mode"]} scale={settings["scale"]} out={a.out}', flush=True)
    blat, blon = basegrid_axes()
    t_start = time.monotonic()
    for ia, age in enumerate(ages):
        c_lo, c_hi = float(lo[ia]), float(hi[ia])
        if not np.isfinite(c_lo):
            print(f'  [{age:>3} Ma] no pCO2 envelope; skipped'); continue
        # node axes, each including 0 (the unperturbed state): r across the envelope, z in [-z_max, z_max]
        r_nodes, z_nodes = mc._node_axes(emu, age, c_lo, c_hi, a.n_r, a.n_z, a.z_max)
        # rows per complete age: every state on both surfaces
        n_states = 2 * r_nodes.size * z_nodes.size
        if not a.force and age in ev.blocks_done(a.out, 'age', n_states):
            done.add(age); continue
        if a.time_budget and time.monotonic() - t_start > a.time_budget:
            print(f'  [{age:>3} Ma] time budget reached; re-run to continue'); break
        with np.load(cfg.ERODEP_DIR / f'erodep_{age}Ma.npz', allow_pickle=True) as d:
            lat = d['lat'].astype(float); lon = d['lon'].astype(float)
            rain_ref = d['rain'].astype(np.float32)
            zp = {'ref': d['zpaleo_scotese'].astype(np.float32), 'adj': d['zpaleo_zhou'].astype(np.float32)}
            mask = {'ref': d['mask_scotese'].astype(bool), 'adj': d['mask_zhou'].astype(bool)}
            archived = {'ref': d['ero_scotese'].astype(np.float32), 'adj': d['ero_zhou'].astype(np.float32)}
        if not (np.allclose(lat, blat) and np.allclose(lon, blon)):
            raise SystemExit('base grid of the emulator differs from the erosion archive')
        with np.load(cfg.TEMP_DIR / f'{age}Ma.npz') as t:
            T_ref_C = t['temp_q'].astype(np.float32) * float(t['scale']) + float(t['offset'])
        ratio = ev.ratio_grid(age, lat, lon)
        # spherical cell areas (m2) of the 0.1-degree grid, duplicate dateline columns halved, as in build_cache.py
        A = ((6.371e6 ** 2 * np.deg2rad(0.1) ** 2) * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
        A[:, 0] *= 0.5; A[:, -1] *= 0.5
        cells = np.flatnonzero((mask['ref'] | mask['adj']).ravel())
        evaluate = ev.Evaluator(cells, A.ravel(), (amps, amp_oc, qgrid, eopt_q), settings)
        mref = mask['ref'].ravel(); land = mask['ref']
        # indices of the unperturbed node (r = 0, z = 0)
        i0, j0 = int(np.argmin(np.abs(r_nodes))), int(np.argmin(np.abs(z_nodes)))
        t0 = time.time(); pending = []
        with ThreadPoolExecutor(max_workers=a.workers) as pool:      # weathering on CPU threads beside the inference
            for i, r in enumerate(r_nodes):
                for j, z in enumerate(z_nodes):
                    # perturbation on the emulator grid: dT = lambda_T r + z sigma_T (K) and a multiplicative
                    # precipitation factor exp(kappa_P r + z sigma_lnP), limited to --runoff-scale-clip; then
                    # interpolated to the 0.1-degree grid
                    dT_c, rs_c = mc._perturb_coarse(emu, age, r, z, terms, 1.0, clip)
                    dT = regrid_bilinear(dT_c, emu.lat, emu.lon, blat, blon)
                    rs = regrid_bilinear(rs_c, emu.lat, emu.lon, blat, blon)
                    # perturbed precipitation (m/yr) drives NeuroLEM; runoff (mm/yr) = precipitation x the runoff ratio
                    # of the reference simulation
                    rain = (rain_ref * rs).astype(np.float32)
                    T_K = ((T_ref_C + dT).astype(np.float64) + 273.15).ravel()
                    q = (rain.astype(np.float64) * 1000.0 * ratio).ravel()
                    for s in ('ref', 'adj'):
                        # NeuroLEM erosion-deposition field (mm/yr) of this surface under the perturbed precipitation
                        ero = model(zp[s], rain)
                        if i == i0 and j == j0:
                            rmse = float(np.sqrt(np.mean((ero - archived[s]) ** 2)))
                            if rmse > a.reference_tolerance:
                                raise SystemExit(f'[{age} Ma] unperturbed state does not reproduce the archived {s} erosion field: '
                                                 f'RMSE {rmse:.3e} mm/yr > {a.reference_tolerance:g}.\nThe archived field is not what this checkpoint gives on the archived inputs: '
                                                 f'run ./run_uncertainty_m3.sh archive-repair, then main, then climate again (or: resume).')
                        row = dict(age=age, i_r=i, i_z=j, r=float(r), z=float(z), co2_ppm=float(emu.co2_ref(age) * 2.0 ** r), surface=s,
                                   T_land_C=float((T_ref_C + dT)[land].mean()), rain_land_m=float(rain[land].mean()))
                        pending.append((row, pool.submit(evaluate, ero.ravel(), mask[s].ravel(), mref, q, T_K)))
                        # keep at most about 2 x workers evaluations queued, to bound memory
                        while len(pending) > 2 * a.workers and not pending[-2 * a.workers][1].done():
                            time.sleep(0.05)
            rows = []
            for row, fut in pending:
                row.update(fut.result()); rows.append(row)
        # second check: the unperturbed state reproduces the main-run table (relative 1e-3)
        for row in rows:
            if row['i_r'] == i0 and row['i_z'] == j0 and age in main:
                for k in CHECK:
                    want = float(main[age][f'{k}_{row["surface"]}'])
                    if abs(row[k] - want) > 1e-3 * abs(want) + 1e-5:
                        raise SystemExit(f'[{age} Ma] unperturbed {row["surface"]} {k} = {row[k]:.8g}, main run has {want:.8g}')
        ev.append_rows(a.out, HEADER, rows, 'age')
        S = [x['S_sil'] for x in rows if x['surface'] == 'adj']; B = [x['beyond_frac_eroding'] for x in rows if x['surface'] == 'adj']
        print(f'  [{age:>3} Ma] CO2 {c_lo:5.0f}-{c_hi:5.0f} ppm  S_sil {min(S):.3f}-{max(S):.3f}  beyond {min(B):.3f}-{max(B):.3f}  '
              f'unperturbed state reproduces archive and main run  {time.time() - t0:.0f} s', flush=True)
        done.add(age)
    missing = sorted(set(ages) - done)
    print(f'{len(done)}/{len(ages)} ages complete in {a.out}' + (f'; missing {missing[:15]}: re-run the same command' if missing else ''))


def _interp(node, rn, zn, r, z):
    """Bilinear in (r, z), clamped; node has shape (nr, nz), r and z are arrays."""
    r = np.clip(r, rn[0], rn[-1]); z = np.clip(z, zn[0], zn[-1])
    i = np.clip(np.searchsorted(rn, r) - 1, 0, rn.size - 2); j = np.clip(np.searchsorted(zn, z) - 1, 0, zn.size - 2)
    tr = (r - rn[i]) / (rn[i + 1] - rn[i]); tz = (z - zn[j]) / (zn[j + 1] - zn[j])
    return (node[i, j] * (1 - tr) * (1 - tz) + node[i + 1, j] * tr * (1 - tz) + node[i, j + 1] * (1 - tr) * tz + node[i + 1, j + 1] * tr * tz)


def sample_series(a):
    """Monte Carlo members interpolated from the node table: (node table, ages, {(surface, metric): members x ages}).
    Deterministic for given nodes, n, seed and correlation choices; used by cmd_sample and by experiments that need the
    member series over a window of ages (dyntopo_sensitivity.py).

    CO2: with --co2-time-correlation independent (default) each member draws r uniformly across the node range at every
    age; with 'full' one standard-normal draw per member is mapped through tanh onto the range at all ages. Model
    spread: with --sigma-time-correlation full (default) one standard-normal z per member is used at all ages, clipped
    to the node range; with 'independent' z is drawn afresh at each age.
    """
    import pandas as pd
    # ages with fewer rows than the most complete age (an interrupted or pilot run) are left out
    d = pd.read_csv(a.nodes)
    size = d.groupby('age').size(); full = int(size.max())
    bad = sorted(int(x) for x in size.index[size != full])
    if bad:
        print('incomplete ages ignored:', bad)
    d = d[~d.age.isin(bad)]
    ages = np.array(sorted(d.age.unique()), int)
    rng = np.random.default_rng(a.seed); n = a.n
    u = rng.standard_normal(n) if a.co2_time_correlation == 'full' else None
    zs = rng.standard_normal(n) if a.sigma_time_correlation == 'full' else None
    series = {(s, k): np.empty((n, ages.size)) for s in ('ref', 'adj') for k in BAND}
    for ia, age in enumerate(ages):
        g = d[d.age == age]
        rn = np.sort(g.r.unique()); zn = np.sort(g.z.unique())
        R = (rn[0] + 0.5 * (1 + np.tanh(u / 1.5)) * (rn[-1] - rn[0])) if u is not None else rng.uniform(rn[0], rn[-1], n)
        Z = np.clip(zs if zs is not None else rng.standard_normal(n), zn[0], zn[-1])
        for s in ('ref', 'adj'):
            gs = g[g.surface == s].sort_values(['i_r', 'i_z'])
            for k in BAND:
                series[(s, k)][:, ia] = _interp(gs[k].to_numpy(float).reshape(rn.size, zn.size), rn, zn, R, Z)
    return d, ages, series


def cmd_sample(a):
    """Stage B: write climate_sensitivity_bands.csv and climate_sensitivity_headline.csv from the node table."""
    import pandas as pd
    d, ages, series = sample_series(a)
    n = a.n
    central = d[(d.r == 0) & (d.z == 0)]
    rows = []
    for ia, age in enumerate(ages):
        row = dict(age=int(age), n_members=n)
        for s in ('ref', 'adj'):
            c = central[(central.age == age) & (central.surface == s)].iloc[0]
            for k in BAND:
                x = series[(s, k)][:, ia]
                row[f'{k}_{s}_central'] = float(c[k])
                row.update({f'{k}_{s}_p05': float(np.quantile(x, .05)), f'{k}_{s}_p50': float(np.quantile(x, .5)), f'{k}_{s}_p95': float(np.quantile(x, .95))})
        for k in ('D_Gt', 'F_sil', 'F_net'):
            x = series[('adj', k)][:, ia] / series[('ref', k)][:, ia]
            row.update({f'ratio_{k}_p05': float(np.quantile(x, .05)), f'ratio_{k}_p50': float(np.quantile(x, .5)), f'ratio_{k}_p95': float(np.quantile(x, .95))})
        rows.append(row)
    out1 = cfg.OUT / 'climate_sensitivity_bands.csv'; pd.DataFrame(rows).to_csv(out1, index=False)
    head = []
    for s in ('ref', 'adj'):
        for k in BAND:
            x = series[(s, k)]
            for stat, v in (('time_mean', x.mean(axis=1)), ('time_min', x.min(axis=1)), ('time_max', x.max(axis=1))):
                head.append(dict(metric=k, surface=s, statistic=stat, p05=float(np.quantile(v, .05)), p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95))))
            w = np.quantile(x, .95, axis=0) - np.quantile(x, .05, axis=0)
            head.append(dict(metric=k, surface=s, statistic='p05_p95_width_median_over_ages', p50=float(np.median(w))))
    for k in ('D_Gt', 'F_sil', 'F_net'):
        x = series[('adj', k)] / series[('ref', k)]
        w = np.quantile(x, .95, axis=0) - np.quantile(x, .05, axis=0)
        head.append(dict(metric=f'ratio_{k}', surface='adj/ref', statistic='p05_p95_width_median_over_ages', p50=float(np.median(w))))
        v = x.mean(axis=1); head.append(dict(metric=f'ratio_{k}', surface='adj/ref', statistic='time_mean', p05=float(np.quantile(v, .05)), p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95))))
    if 0 in ages and 480 in ages:
        b = series[('adj', 'beyond_frac_eroding')]
        head.append(dict(metric='members_with_beyond_share_higher_today_than_at_480Ma', surface='adj', statistic='count',
                         p50=float((b[:, list(ages).index(0)] > b[:, list(ages).index(480)]).sum())))
    out2 = cfg.OUT / 'climate_sensitivity_headline.csv'; pd.DataFrame(head).to_csv(out2, index=False)
    print(f'{ages.size} ages, {n} members (CO2 {a.co2_time_correlation}, model spread {a.sigma_time_correlation})\nwrote {out1}\nwrote {out2}')
    h = pd.DataFrame(head)
    for k in ('S_sil', 'S_net', 'beyond_frac_eroding'):
        x = h[(h.metric == k) & (h.surface == 'adj') & (h.statistic == 'time_mean')].iloc[0]
        print(f'  time-mean {k:22s} p05 {x.p05:.3f}  p50 {x.p50:.3f}  p95 {x.p95:.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('nodes')
    q.add_argument('--ages', nargs='*')
    q.add_argument('--out', default=str(NODES_CSV))
    q.add_argument('--main-run', default=str(cfg.OUT / 'kin_bridge_s2'))
    q.add_argument('--sensitivity', default=str(cfg.SENS))
    q.add_argument('--envelope-csv', default=str(cfg.REPO / 'data' / 'pco2_envelope.csv'))
    q.add_argument('--ckpt', default=str(cfg.ZENODO / 'surrogate_model' / 'best.pt'))
    q.add_argument('--unet-py', help='model file (paleomap_unet.py) to load by file location; default: the one in scripts/')
    q.add_argument('--device')
    q.add_argument('--n-r', type=int, default=9); q.add_argument('--n-z', type=int, default=3); q.add_argument('--z-max', type=float, default=2.5)
    q.add_argument('--sigma-terms', default='int,fit,struct')
    q.add_argument('--runoff-scale-clip', type=float, nargs=2, default=[0.2, 5.0])
    q.add_argument('--reference-tolerance', type=float, default=1e-5, help='RMSE (mm/yr) allowed against the archived erosion fields')
    q.add_argument('--time-budget', type=float, help='seconds; stop cleanly after the age in progress')
    q.add_argument('--workers', type=int, default=4, help='CPU threads for the weathering step')
    q.add_argument('--force', action='store_true')
    q.set_defaults(fn=cmd_nodes)
    q = sub.add_parser('sample')
    q.add_argument('--nodes', default=str(NODES_CSV))
    q.add_argument('--n', type=int, default=2000); q.add_argument('--seed', type=int, default=0)
    q.add_argument('--co2-time-correlation', choices=['independent', 'full'], default='independent')
    q.add_argument('--sigma-time-correlation', choices=['independent', 'full'], default='full')
    q.set_defaults(fn=cmd_sample)
    a = p.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()

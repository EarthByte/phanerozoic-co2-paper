#!/usr/bin/env python3
"""Erosion sensitivity from the direct goSPL simulations, against NeuroLEM.

The 20 direct goSPL runs (reference surface, adjusted surface and two coherent ensemble members at
60, 220, 240, 460 and 520 Ma) are evaluated on their own 10.5-million-node mesh with exactly the
weathering evaluation of the main model: same parameter sets, calibration, runoff ratio and local
optimum, with exact spherical vertex areas as weights. Each case is set beside its NeuroLEM
counterpart: the main-run table for the reference and adjusted surfaces, the ensemble member table
for the two members (when the ensemble run holds that age). The sulfide runoff exponent is that of the
paired main run (_evaluate.load_calibration), which for the runs on the goSPL surfaces is the published
value of Bufe et al. (2024); it leaves the silicate fluxes and their sensitivity unchanged.

The goSPL runs use an adjusted surface built with a variant of the correction (corrected_scotese_nocraton:
cratonic samples excluded, 400 km support, no distance taper, samples on flooded regions retained), which
differs from the adjusted surface used throughout the paper (corrected_scotese_reach500_land). The NeuroLEM
counterparts are therefore read from the main run and ensemble table computed on the goSPL surfaces,
data/nature_recast/gospl_surfaces/ (with the same weathering calibration as this evaluation), not from the
main run of the paper: the comparison is surrogate against goSPL on identical inputs, and it is that pairing
that tests the surrogate. The reference-surface runs use the unadjusted PALEOMAP physiography, which is the
paper's reference surface unchanged. The results enter the Supplementary Information and Extended Data Fig. 3.

Inputs
  --gadi-root    the verified bundle of ../analyse_gadi_validation.py: raw/inputs/nature_validation/
                 (rain_l10_<age>Ma.npz, precipitation r on the mesh in m/yr; the mesh topology; the
                 manifest of selected ensemble members), raw/outputs/nature_validation/
                 <age>Ma_<case>_final_fields.npz (erorate in mm/yr, negative = erosion; initial_elev in
                 m) and derived/cache/ (vertex areas and coordinates, built if missing)
  --main-run     prefix of the paired NeuroLEM main run: <prefix>_ens.csv, <prefix>_amplitudes.txt and
                 <prefix>_hist.npz (calibration amplitudes and the E_opt(runoff) table)
  --ensemble-csv NeuroLEM ensemble member table on the same surfaces
  HadCM3 temperature (valdes_temperature/) and the HadCM3L runoff ratio (via _evaluate.ratio_at)

Output: one long-format table, data/nature_recast/gospl_validation_sensitivity.csv, one row per age,
case and metric, with columns goSPL, NeuroLEM, difference (goSPL - NeuroLEM) and relative_difference;
the metrics are listed in METRICS (fluxes in Gt C/yr, D_Gt in Gt/yr, sensitivities S and shares
dimensionless; suffix _common = integrated over the reference run's land). A summary of the differences
is printed.

    python3 gospl_validation_sensitivity.py [--gadi-root DIR] [--ages 60 220 ...] [--cases reference central ...]
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import numpy as np

import _config as cfg
import _evaluate as ev

# Quantities compared (names as in run_weathering.summarise): land area, sediment flux D, mean erosion
# rate of eroding land, silicate, sulfide, net and petrogenic-oxidation fluxes, erosion sensitivities S,
# shares of eroding land beyond the optimum, kinetic share Gamma, weathering intensity WI, and the same
# on the reference land (_common)
METRICS = ['land_Mkm2', 'D_Gt', 'E_mean_eroding_mm', 'F_sil', 'F_sulf', 'F_net', 'F_ocp', 'S_sil', 'S_net', 'S_total_rel_sil',
           'beyond_frac_eroding', 'beyond_oc_frac_eroding', 'Gamma_sil', 'WI', 'S_sil_common', 'S_net_common',
           'beyond_frac_eroding_common', 'F_sil_common', 'F_net_common', 'D_Gt_common']
PAIRED = cfg.OUT / 'gospl_surfaces'          # main run and ensemble table of the surfaces the goSPL runs used


def geometry(gadi, force):
    """Vertex areas (m2), lon and lat (degrees) of the goSPL mesh, from the cache of
    analyse_gadi_validation.py, built there if missing or if ``force``."""
    cache = gadi / 'derived' / 'cache'
    names = ('vertex_area_m2', 'lon', 'lat')
    if not force and all((cache / f'{n}.npy').is_file() for n in names):
        return {n: np.load(cache / f'{n}.npy', mmap_mode='r') for n in names}
    from analyse_gadi_validation import build_geometry
    return build_geometry(gadi / 'raw/inputs/nature_validation/icosphere_l10_topology.npz', cache, force)


def main():
    """Evaluate every direct run, pair it with NeuroLEM and write the table."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gadi-root', type=Path, default=Path('gadi_runs/nature_validation_CO2'))
    p.add_argument('--main-run', default=str(PAIRED / 'kin_bridge_s2'), help='NeuroLEM run on the surfaces of the goSPL runs')
    p.add_argument('--ensemble-csv', default=str(PAIRED / 'ensemble_sensitivity_members.csv'))
    p.add_argument('--ages', nargs='*', type=int, default=[60, 220, 240, 460, 520])
    p.add_argument('--cases', nargs='*', default=['reference', 'central', 'qnet_p05', 'qnet_p95'])
    p.add_argument('--out', default=str(cfg.OUT / 'gospl_validation_sensitivity.csv'))
    p.add_argument('--force-geometry', action='store_true')
    a = p.parse_args()
    from analyse_gadi_validation import sample_temperature

    gadi = a.gadi_root.resolve(); raw = gadi / 'raw'
    if not raw.is_dir():
        raise SystemExit(f'{raw} not found: is the direct goSPL run bundle present? (or pass --gadi-root)')
    report = gadi / 'qc' / 'verification_report.json'
    if report.exists() and not json.loads(report.read_text()).get('passed', False):
        raise SystemExit('the retrieval verification of the Gadi bundle has not passed')
    sel_path = raw / 'inputs/nature_validation/corrections/selected_corrections_manifest.json'
    selected = json.loads(sel_path.read_text()) if sel_path.exists() else {'ages': {}}
    g = geometry(gadi, a.force_geometry)
    area = np.asarray(g['vertex_area_m2'], float); lat = np.asarray(g['lat'], float); lon = np.asarray(g['lon'], float)

    if not Path(f'{a.main_run}_ens.csv').exists():
        raise SystemExit(f'{a.main_run}_ens.csv not found: the NeuroLEM run on the surfaces of the goSPL runs is required '
                         '(run_uncertainty_m3.sh rebuild-surface preserves it under data/nature_recast/gospl_surfaces/)')
    # calibration amplitudes and local-optimum table of the paired main run, never refitted
    amps, amp_oc, settings, qgrid, eopt_q = ev.load_calibration(a.main_run)
    main_run = {int(float(r['age'])): r for r in csv.DictReader(open(f'{a.main_run}_ens.csv'))}
    members = {}
    if Path(a.ensemble_csv).exists():
        for r in csv.DictReader(open(a.ensemble_csv)):
            members[(int(float(r['age'])), int(float(r['member'])))] = r

    rows = []
    for age in a.ages:
        with np.load(raw / 'inputs/nature_validation' / f'rain_l10_{age}Ma.npz', allow_pickle=False) as rd:
            rain = np.asarray(rd['r'], float)
        # runoff, mm/yr: precipitation (m/yr -> mm/yr) times the HadCM3L runoff ratio at each vertex
        q = rain * 1000.0 * ev.ratio_at(age, lat, lon)
        T = sample_temperature(cfg.TEMP_DIR / f'{age}Ma.npz', lat, lon).astype(float)
        fields = {}
        for case in a.cases:
            f = raw / 'outputs/nature_validation' / f'{age}Ma_{case}_final_fields.npz'
            if not f.exists():
                print(f'[{age} Ma] {case}: no direct run'); continue
            with np.load(f, allow_pickle=False) as d:
                fields[case] = (np.asarray(d['erorate'], np.float32), np.asarray(d['initial_elev']) > 0.0)
        if 'reference' not in fields:
            raise SystemExit(f'[{age} Ma] the reference run is needed for the common-land integrals')
        # land of each run = initial elevation above sea level; the evaluator works on the union of all
        # runs' land at this age so that one set of cells serves every case
        ref_land = fields['reference'][1]
        any_land = np.zeros_like(ref_land)
        for _, land in fields.values():
            any_land |= land
        evaluate = ev.Evaluator(np.flatnonzero(any_land), area, (amps, amp_oc, qgrid, eopt_q), settings)
        for case, (erorate, land) in fields.items():
            # metrics on the run's own land and, suffix _common, on the reference run's land
            res = evaluate(erorate, land, ref_land, q, T)
            # NeuroLEM counterpart: main-run columns _ref / _adj / _adjc (adjusted surface on
            # reference land), or the ensemble member named in the selection manifest
            if case == 'reference':
                neuro = {k: main_run[age].get(f'{k}_ref') for k in METRICS}; label = 'main run, reference surface'
                for k in METRICS:                                   # reference common = own
                    if k.endswith('_common'):
                        neuro[k] = main_run[age].get(f'{k[:-7]}_ref')
            elif case == 'central':
                neuro = {k: main_run[age].get(f'{k}_adj') for k in METRICS}; label = 'main run, adjusted surface'
                for k in METRICS:
                    if k.endswith('_common'):
                        neuro[k] = main_run[age].get(f'{k[:-7]}_adjc')
            else:
                m = selected['ages'].get(str(age), {}).get(case, {}).get('member')
                rec = members.get((age, int(m))) if m is not None else None
                neuro = {k: (rec.get(k) if rec else None) for k in METRICS}
                label = f'ensemble member {m}' + ('' if rec else ' (not in the ensemble table yet)')
            for k in METRICS:
                nv = float(neuro[k]) if neuro.get(k) not in (None, '') else np.nan
                rows.append(dict(age=age, case=case, neurolem_source=label, paired_run=str(a.main_run), metric=k, goSPL=res[k], NeuroLEM=nv,
                                 difference=res[k] - nv, relative_difference=(res[k] / nv - 1.0) if nv else np.nan))
            print(f'[{age:>3} Ma] {case:9s} S_sil goSPL {res["S_sil"]:.3f} NeuroLEM {float(neuro["S_sil"]) if neuro.get("S_sil") else float("nan"):.3f} | '
                  f'beyond goSPL {res["beyond_frac_eroding"]:.3f} NeuroLEM {float(neuro["beyond_frac_eroding"]) if neuro.get("beyond_frac_eroding") else float("nan"):.3f}', flush=True)

    # write beside the target and rename, so an interrupted run leaves the previous table intact
    out = Path(a.out); tmp = out.with_name(f'.{out.name}.tmp')
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    tmp.replace(out)
    print('wrote', out)
    for k in ('S_sil', 'S_net', 'beyond_frac_eroding', 'F_sil', 'D_Gt'):
        d = np.array([r['difference'] for r in rows if r['metric'] == k and np.isfinite(r['difference'])])
        rel = np.array([r['relative_difference'] for r in rows if r['metric'] == k and np.isfinite(r['relative_difference'])])
        if d.size:
            print(f'  {k:22s} goSPL - NeuroLEM: median {np.median(d):+.4f}, range {d.min():+.4f} to {d.max():+.4f} '
                  f'(relative {100 * rel.min():+.1f}% to {100 * rel.max():+.1f}%), n = {d.size}')


if __name__ == '__main__':
    main()

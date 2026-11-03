#!/usr/bin/env python3
"""How much the results depend on the width of the simulated erosion-rate distribution.

The OCTOPUS comparison (octopus_comparison.py) shows the simulated distribution of erosion rates across the
basins with cosmogenic-nuclide rates to be narrower than the measured one: a standard deviation of log10 rate of 0.47 against
0.79 (ratio 0.59; 0.57–0.61 for basins of one cell and larger). This script widens the distribution of
every age in log space about its area-weighted median, E' = c·med·(E/med)^k, with c holding the erosion flux
of the age (the calibrated quantity) fixed, for k = 1.3, 1.5 and 1.8 (1/0.59 = 1.7), re-runs the weathering
evaluation of the main model on each (run_weathering.py --log-stretch k; calibration repeated on the
stretched present-day field as in the main model) and sets the results beside the main model in ONE table,
data/nature_recast/distribution_tests.csv: Phanerozoic-mean, minimum and maximum S_sil, mean S_net, mean
share of eroding land beyond the optimum and its values at 480 and 0 Ma, mean silicate sink.

    python3 distribution_tests.py [--k 1.3 1.5 1.8] [--force]
"""
import argparse, csv, subprocess, sys
from pathlib import Path
import numpy as np
import _config as cfg


def table(prefix):
    rows = list(csv.DictReader(open(f'{prefix}_ens.csv')))
    return {k: np.array([float(r[k]) for r in rows]) for k in ('age', 'S_sil_adj', 'S_net_adj', 'beyond_frac_eroding_adj', 'Gamma_sil_adj', 'F_sil_adj', 'D_Gt_adj')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', default='kin_bridge_s2')
    p.add_argument('--k', type=float, nargs='*', default=[1.3, 1.5, 1.8])
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    main_t = table(cfg.OUT / a.main)
    out_rows = []
    for k in [1.0] + list(a.k):
        if k == 1.0:
            t = main_t; label = 'as simulated (main model)'
        else:
            prefix = cfg.OUT / f'dst_k{k:g}'
            if a.force or not Path(f'{prefix}_ens.csv').exists():
                cmd = [sys.executable, str(here / 'run_weathering.py'), '--mode', 'kinetic', '--qsrc', 'bridge', '--log-stretch', f'{k:g}', '--out', str(prefix)]
                print('running', ' '.join(cmd[1:]), flush=True)
                with open(cfg.OUT / f'log_dst_k{k:g}.txt', 'w') as log:
                    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            t = table(prefix); label = f'log-stretch k = {k:g}'
        i480 = int(np.argmin(np.abs(t['age'] - 480))); i0 = int(np.argmin(np.abs(t['age'])))
        r = dict(configuration=label, k=k, S_sil_mean=round(float(t['S_sil_adj'].mean()), 4), S_sil_min=round(float(t['S_sil_adj'].min()), 4), S_sil_max=round(float(t['S_sil_adj'].max()), 4),
                 S_net_mean=round(float(t['S_net_adj'].mean()), 4), beyond_mean=round(float(t['beyond_frac_eroding_adj'].mean()), 4),
                 beyond_480=round(float(t['beyond_frac_eroding_adj'][i480]), 4), beyond_0=round(float(t['beyond_frac_eroding_adj'][i0]), 4),
                 Gamma_mean=round(float(t['Gamma_sil_adj'].mean()), 4), F_sil_mean=round(float(t['F_sil_adj'].mean()), 4), D_Gt_mean=round(float(t['D_Gt_adj'].mean()), 3))
        out_rows.append(r)
        print(f"{label:28s} S_sil {r['S_sil_mean']:.3f} ({r['S_sil_min']:.3f}–{r['S_sil_max']:.3f})  S_net {r['S_net_mean']:+.3f}  beyond {r['beyond_mean']:.3f} ({r['beyond_480']:.3f} -> {r['beyond_0']:.3f})  D {r['D_Gt_mean']:.1f} Gt/yr")
    out = cfg.OUT / 'distribution_tests.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print('wrote', out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""The two open forks of the runoff treatment, quantified against the main model.

Runs the weathering evaluation of the main model (kin_bridge_s2: runoff in the rate constant, HadCM3L
runoff ratio, erosion scale 2) under four alternative forms and sets them beside it:

  sulfide runoff exponent as published (-0.75)          --beta-sulf -0.75
  Sichuan reference converted to runoff (x0.62)         --qref-scale Sichuan=0.62
  Sichuan set driven by precipitation                   --sichuan-precip
  Sichuan driven by precipitation + published exponent --sichuan-precip --beta-sulf -0.75

Each variant is a full run of run_weathering.py from the input cache (CPU, minutes each), written to
data/nature_recast/rft_<name>_*. ONE table comes out, data/nature_recast/runoff_form_tests.csv:
Phanerozoic-mean, minimum and maximum silicate sensitivity, mean net sensitivity, mean share of eroding
land beyond the optimum and its values at 480 and 0 Ma, mean kinetically limited share, mean silicate
sink, and the largest absolute difference from the main model in S_sil and S_net at any age. The main
model's own row is read from its table, never re-run here.

    python3 runoff_form_tests.py [--main kin_bridge_s2] [--only NAME ...]
"""
import argparse, csv, subprocess, sys
from pathlib import Path
import numpy as np
import _config as cfg

VARIANTS = [
    ('pure West form (main model)', None, None),
    ('sulfide runoff exponent as published (-0.75)', 'rft_betapub', ['--beta-sulf', '-0.75']),
    ('Sichuan reference converted to runoff (x0.62)', 'rft_qref062', ['--qref-scale', 'Sichuan=0.62']),
    ('Sichuan set driven by precipitation', 'rft_precip', ['--sichuan-precip']),
    ('Sichuan driven by precipitation + published sulfide exponent', 'rft_precip_betapub', ['--sichuan-precip', '--beta-sulf', '-0.75']),
]


def table(prefix):
    rows = list(csv.DictReader(open(f'{prefix}_ens.csv')))
    return {k: np.array([float(r[k]) for r in rows]) for k in ('age', 'S_sil_adj', 'S_net_adj', 'beyond_frac_eroding_adj', 'Gamma_sil_adj', 'F_sil_adj')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', default='kin_bridge_s2')
    p.add_argument('--only', nargs='*', default=None, help='variant output names to (re)run; default: those without a table')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    main_t = table(cfg.OUT / a.main)
    out_rows = []
    for label, name, flags in VARIANTS:
        if name is None:
            t = main_t
        else:
            prefix = cfg.OUT / name
            if a.force or (a.only and name in a.only) or not Path(f'{prefix}_ens.csv').exists():
                cmd = [sys.executable, str(here / 'run_weathering.py'), '--mode', 'kinetic', '--qsrc', 'bridge', *flags, '--out', str(prefix)]
                print('running', ' '.join(cmd[1:]), flush=True)
                with open(cfg.OUT / f'log_{name}.txt', 'w') as log:
                    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            t = table(prefix)
        if not np.array_equal(t['age'], main_t['age']):
            raise SystemExit(f'{label}: ages differ from the main run')
        i480 = int(np.argmin(np.abs(t['age'] - 480))); i0 = int(np.argmin(np.abs(t['age'])))
        out_rows.append(dict(configuration=label,
                             S_sil_mean=round(float(t['S_sil_adj'].mean()), 4), S_sil_min=round(float(t['S_sil_adj'].min()), 4), S_sil_max=round(float(t['S_sil_adj'].max()), 4),
                             S_net_mean=round(float(t['S_net_adj'].mean()), 4),
                             beyond_mean=round(float(t['beyond_frac_eroding_adj'].mean()), 4), beyond_480=round(float(t['beyond_frac_eroding_adj'][i480]), 4),
                             beyond_0=round(float(t['beyond_frac_eroding_adj'][i0]), 4),
                             Gamma_mean=round(float(t['Gamma_sil_adj'].mean()), 4), F_sil_mean=round(float(t['F_sil_adj'].mean()), 4),
                             max_abs_dS_sil_vs_main=round(float(np.abs(t['S_sil_adj'] - main_t['S_sil_adj']).max()), 4),
                             max_abs_dS_net_vs_main=round(float(np.abs(t['S_net_adj'] - main_t['S_net_adj']).max()), 4)))
        r = out_rows[-1]
        print(f"{label:50s} S_sil {r['S_sil_mean']:.3f}  beyond {r['beyond_mean']:.3f} ({r['beyond_480']:.3f} -> {r['beyond_0']:.3f})  "
              f"max|dS_sil| {r['max_abs_dS_sil_vs_main']:.4f}  max|dS_net| {r['max_abs_dS_net_vs_main']:.4f}")
    out = cfg.OUT / 'runoff_form_tests.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print('wrote', out)


if __name__ == '__main__':
    main()

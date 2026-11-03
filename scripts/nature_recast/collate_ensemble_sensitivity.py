#!/usr/bin/env python3
"""Collate the palaeoelevation-ensemble member table into the two small tables the paper uses.

    python3 collate_ensemble_sensitivity.py [member_table.csv]

Reads the ONE member table written by apply_ensemble_sensitivity.py and writes, into data/nature_recast/:

  ensemble_sensitivity_age_summary.csv   per age: reference surface, mean-correction surface, main-model value,
                                         and p05 / p50 / mean / p95 across members for every metric, including
                                         member-to-reference ratios on own land and on common land
  ensemble_sensitivity_headline.csv      per metric: p05 / p50 / mean / p95 across members of the time-mean,
                                         minimum and maximum over all ages present (member j carries the same
                                         geochemical draws at every age, so its time series is one coherent history)

Ages with an incomplete set of rows are ignored and listed.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import _config as cfg

DEFAULT = cfg.ZENODO / 'geochem_uncertainty_ensemble' / 'nature_recast' / 'ensemble_sensitivity_members.csv'
METRICS = ['D_Gt', 'E_mean_eroding_mm', 'F_sil', 'F_sulf', 'F_net', 'F_ocp', 'F_ocb', 'S_sil', 'S_net', 'S_net_oc_rel_sil',
           'S_total_rel_sil', 'beyond_frac_eroding', 'beyond_oc_frac_eroding', 'Gamma_sil', 'WI', 'd7Li_riv',
           'S_sil_Taiwan', 'S_sil_Sichuan', 'S_sil_NewZealand', 'S_sil_common', 'S_net_common', 'beyond_frac_eroding_common']
RATIOS = ['D_Gt', 'F_sil', 'F_net', 'D_Gt_common', 'F_sil_common', 'F_net_common']
Q = (('p05', 0.05), ('p50', 0.50), ('p95', 0.95))


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    d = pd.read_csv(src)
    n = int(d[d.member >= 0].groupby('age').size().max())
    size = d.groupby('age').size()
    bad = sorted(int(a) for a in size.index[size != n + 2])
    if bad:
        print('incomplete ages ignored:', bad)
    d = d[~d.age.isin(bad)]
    ages = sorted(int(a) for a in d.age.unique())
    main_run = pd.read_csv(cfg.OUT / 'kin_bridge_s2_ens.csv').set_index('age')
    ref = d[d.member == -2].set_index('age'); mean_surface = d[d.member == -1].set_index('age')
    mem = d[d.member >= 0].copy()
    for k in RATIOS:
        mem[f'ratio_{k}'] = mem[k].to_numpy() / ref[k.replace('_common', '')].reindex(mem.age).to_numpy()

    rows = []
    for age, g in mem.groupby('age'):
        r = dict(age=int(age), n_members=len(g))
        for k in METRICS + [f'ratio_{x}' for x in RATIOS]:
            x = g[k].to_numpy(float)
            if not k.startswith('ratio_'):
                r[f'{k}_reference'] = float(ref.at[age, k]); r[f'{k}_mean_surface'] = float(mean_surface.at[age, k])
                if f'{k}_adj' in main_run.columns:
                    r[f'{k}_main_model'] = float(main_run.at[age, f'{k}_adj'])
            for lab, q in Q:
                r[f'{k}_{lab}'] = float(np.quantile(x, q))
            r[f'{k}_mean'] = float(x.mean())
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values('age')
    out1 = cfg.OUT / 'ensemble_sensitivity_age_summary.csv'; summary.to_csv(out1, index=False)

    head = []
    for k in METRICS + [f'ratio_{x}' for x in RATIOS]:
        piv = mem.pivot(index='member', columns='age', values=k)
        for stat, v in (('time_mean', piv.mean(axis=1)), ('time_min', piv.min(axis=1)), ('time_max', piv.max(axis=1))):
            h = dict(metric=k, statistic=stat, n_members=len(v), n_ages=len(ages))
            h.update({lab: float(np.quantile(v, q)) for lab, q in Q}); h['mean'] = float(v.mean())
            h['min_member'] = float(v.min()); h['max_member'] = float(v.max())
            head.append(h)
    # statements the text makes about every member
    b = mem.pivot(index='member', columns='age', values='beyond_frac_eroding')
    s = mem.pivot(index='member', columns='age', values='S_sil')
    extra = {'members_with_S_sil_below_0.5_at_every_age': int((s.max(axis=1) < 0.5).sum()),
             'members_with_beyond_share_above_0.5_at_every_age_since_300Ma': int((b[[a for a in ages if a <= 300]].min(axis=1) > 0.5).sum()) if any(a <= 300 for a in ages) else -1}
    if 480 in ages and 0 in ages:
        extra['members_with_beyond_share_higher_today_than_at_480Ma'] = int((b[0] > b[480]).sum())
    for k, v in extra.items():
        head.append(dict(metric=k, statistic='count', n_members=len(b), n_ages=len(ages), mean=float(v)))
    out2 = cfg.OUT / 'ensemble_sensitivity_headline.csv'; pd.DataFrame(head).to_csv(out2, index=False)

    inside = ((summary.S_sil_main_model >= summary.S_sil_p05) & (summary.S_sil_main_model <= summary.S_sil_p95)).sum()
    print(f'{len(ages)} ages x {n} members from {src}')
    print(f'wrote {out1}\nwrote {out2}')
    hm = pd.DataFrame(head).set_index(['metric', 'statistic'])
    for k in ('S_sil', 'S_net', 'beyond_frac_eroding', 'ratio_D_Gt', 'ratio_F_sil', 'ratio_F_net'):
        x = hm.loc[(k, 'time_mean')]
        print(f'  time-mean {k:22s} p05 {x.p05:.3f}  p50 {x.p50:.3f}  p95 {x.p95:.3f}')
    print(f'  main-model S_sil lies inside the ensemble p05-p95 at {int(inside)} of {len(ages)} ages '
          '(the main model uses the full-resolution adjusted surface, the members a 1-degree correction)')


if __name__ == '__main__':
    main()

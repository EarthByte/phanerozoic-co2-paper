#!/usr/bin/env python3
"""Mountain terrain (above 1,500 m) through time: position relative to the local optimum and carbon fluxes."""
import csv, numpy as np
import cellfields as cf
from _load import OUT
rows = []
for age in range(0, 545, 5):
    f, d = cf.compute(age, full=False)
    A = f['area']; m = f['mask'] & (f['E'] > 0); hi = m & (f['z'] > 1500); lo = m & ~hi
    Ah = A[hi].sum()
    r = dict(age=age, high_Mkm2=Ah / 1e12, high_share_eroding=Ah / A[m].sum(),
             high_beyond=A[hi & (f['dneti'] <= 0)].sum() / Ah, low_beyond=A[lo & (f['dneti'] <= 0)].sum() / A[lo].sum(),
             high_source=A[hi & (f['net'] < 0)].sum() / Ah,
             high_x_median=float(np.median(f['x'][hi])), high_E=float(np.average(f['E'][hi], weights=A[hi])), high_q=float(np.average(f['q'][hi], weights=A[hi])),
             low_q=float(np.average(f['q'][lo], weights=A[lo])),
             high_net_density=float(np.sum((f['net'] * A)[hi]) / Ah) * 12.011, low_net_density=float(np.sum((f['net'] * A)[lo]) / A[lo].sum()) * 12.011,
             high_share_net=float(np.sum((f['net'] * A)[hi]) / np.sum((f['net'] * A)[m])), high_share_sil=float(np.sum((f['sil'] * A)[hi]) / np.sum((f['sil'] * A)[m])),
             high_share_neti=float(np.sum((f['neti'] * A)[hi]) / np.sum((f['neti'] * A)[m])),
             high_share_denudation=float(np.sum((f['E'] * A)[hi]) / np.sum((f['E'] * A)[m])),
             high_S_net=float(np.sum((f['dnet'] * A)[hi]) / np.sum((f['net'] * A)[hi])), low_S_net=float(np.sum((f['dnet'] * A)[lo]) / np.sum((f['net'] * A)[lo])),
             high_S_sil=float(np.sum((f['dsil'] * A)[hi]) / np.sum((f['sil'] * A)[hi])), low_S_sil=float(np.sum((f['dsil'] * A)[lo]) / np.sum((f['sil'] * A)[lo])))
    rows.append(r); print(age, {k: round(v, 3) for k, v in r.items() if k != 'age'}, flush=True)
with open(OUT / 'highland_stats.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator='\n'); w.writeheader(); w.writerows(rows)

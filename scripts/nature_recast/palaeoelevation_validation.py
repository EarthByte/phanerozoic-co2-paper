#!/usr/bin/env python3
"""Independent tests of the rock-chemistry-adjusted palaeoelevation against the unadjusted PALEOMAP surface.

(1) Held-out rock chemistry. The adjustment is rebuilt five times, each time without one fold of the declustered
mohometry samples (folds are 6-degree blocks, so a held-out sample has no neighbour in its own block), at every age
from 5 to 540 Ma; at each held-out sample the elevation implied by its crustal thickness (z) is compared with the
PALEOMAP elevation (prior) and with the adjusted surface built without it (pred). Scored on samples on PALEOMAP land,
all and within 150 km of the nearest training sample on land, where the correction acts.
(2) Sedimentary basins. Cells that hold more than 2 km of sediment today (GST-1, Bird & Mooney 2026) should have stood
low; the area under the ROC curve of the negative window-median elevation separates them from the rest, and the share
of such ground held above 500 m measures what a correction that lifts basins would do (windows 540-5 and 250-5 Ma).

Inputs are the outputs of the palaeoelevation workflow (Paleotopo_data_assimilation, reach_decision_basin):
reach_heldout_samples.csv.gz and reach_basin_summary.csv, candidate 'reach500_land' (the adjusted surface of this study)
and surface 'input' (PALEOMAP). Writes ONE table, data/nature_recast/palaeoelevation_validation.csv.

    python3 palaeoelevation_validation.py
"""
import os
from pathlib import Path
import numpy as np, pandas as pd
import _config as cfg

PALEO = Path(os.environ.get('PALEO_ROOT', cfg.REPO.parent.parent / 'Paleotopo_data_assimilation'))
SRC = PALEO / 'outputs_scotese_reach500_land' / 'reach_decision_basin'
OUT = cfg.OUT / 'palaeoelevation_validation.csv'

d = pd.read_csv(SRC / 'reach_heldout_samples.csv.gz'); d = d[d.candidate == 'reach500_land']
rows = []
def score(name, x):
    rp = float(np.sqrt(np.mean((x.prior - x.z) ** 2))); ra = float(np.sqrt(np.mean((x.pred - x.z) ** 2)))
    rows.append(dict(test='held-out rock chemistry', subset=name, n=len(x), metric='RMS (m)', paleomap=round(rp), adjusted=round(ra), change_pct=round(100 * (ra / rp - 1), 1)))
    rows.append(dict(test='held-out rock chemistry', subset=name, n=len(x), metric='bias (m)', paleomap=round(float(np.mean(x.prior - x.z))), adjusted=round(float(np.mean(x.pred - x.z))), change_pct=np.nan))
    rows.append(dict(test='held-out rock chemistry', subset=name, n=len(x), metric='correlation', paleomap=round(float(np.corrcoef(x.prior, x.z)[0, 1]), 2), adjusted=round(float(np.corrcoef(x.pred, x.z)[0, 1]), 2), change_pct=np.nan))
land = d[d.land_sited]
score('land, within 150 km of a training sample', land[land.dist_land_km <= 150])
score('land, all distances', land)
for a0, a1 in ((5, 95), (100, 245), (250, 540)):
    score(f'land, all distances, {a0}-{a1} Ma', land[(land.age >= a0) & (land.age <= a1)])
b = pd.read_csv(SRC / 'reach_basin_summary.csv')
for w in ('540-5', '250-5'):
    p = b[(b.window == w) & (b.region == 'world') & (b.surface == 'input')].iloc[0]
    q = b[(b.window == w) & (b.region == 'world') & (b.surface == 'reach500_land')].iloc[0]
    rows.append(dict(test='sedimentary basins (GST-1)', subset=f'world, {w} Ma', n=int(p.n_thick), metric='AUC, > 2 km of sediment', paleomap=round(float(p.auc_2km), 3), adjusted=round(float(q.auc_2km), 3), change_pct=np.nan))
    rows.append(dict(test='sedimentary basins (GST-1)', subset=f'world, {w} Ma', n=int(p.n_thick), metric='share of > 2 km ground above 500 m (%)', paleomap=round(100 * float(p.thick_share_high), 1), adjusted=round(100 * float(q.thick_share_high), 1), change_pct=np.nan))
T = pd.DataFrame(rows); tmp = OUT.with_suffix('.partial.csv'); T.to_csv(tmp, index=False); os.replace(tmp, OUT)
print(T.to_string(index=False)); print('wrote', OUT)

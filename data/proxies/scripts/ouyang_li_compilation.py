#!/usr/bin/env python3
"""Tidy CSVs from the Ouyang et al. (2025) d7Li compilation (LiIsoNEW.mat) and their Gaussian smoothing."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import loadmat
from scipy.stats import norm
src = Path(sys.argv[1] if len(sys.argv) > 1 else '../../../../data/Quyang_Li_2025/LiIsoNEW.mat')
out = Path(__file__).resolve().parents[1]
d = loadmat(src)['LiIsoNEW']; age = d[:, 0] * 1000.0; v = d[:, 1]
pd.DataFrame(dict(age_Ma=np.round(age, 3), d7Li_carbonate_permil=v)).sort_values('age_Ma').to_csv(out / 'd7Li_carbonate_compilation_OuyangLi2025_0-3000Ma.csv', index=False)
rows = []
for t in range(0, 3001, 10):
    s = 20 + t / 10; w = norm.pdf(age, t, s); m = np.sum(w * v) / np.sum(w)
    sd = np.sqrt(np.sum(w * (v - m) ** 2) / np.sum(w)); se = sd / np.sqrt(np.sum(w / norm.pdf(0, 0, s)))
    rows.append((t, m, sd, se))
sm = pd.DataFrame(rows, columns=['age_Ma', 'd7Li_smoothed_carbonate_permil', 'sd_permil', 'se_permil'])
off = 31.0 - sm.d7Li_smoothed_carbonate_permil.iloc[0]
sm['d7Li_seawater_permil'] = sm.d7Li_smoothed_carbonate_permil + off; sm['offset_to_seawater_permil'] = off
sm.round(4).to_csv(out / 'd7Li_seawater_smoothed_OuyangLi2025_0-3000Ma.csv', index=False)
print('wrote', len(d), 'values; offset', round(off, 2))

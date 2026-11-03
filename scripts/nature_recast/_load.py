"""Shared loaders for the Nature-recast figures."""
from pathlib import Path
import numpy as np, pandas as pd

import _config as cfg
OUT, FIG, PROX = cfg.OUT, cfg.FIG, cfg.PROX
SETS = ['Taiwan', 'Sichuan', 'NewZealand']


def series(cfg='kin_bridge_s2', which='ens'):
    return pd.read_csv(OUT / f'{cfg}_{which}.csv').set_index('age')


def envelope(col, cfgs=('kin_bridge_s2',), sets=SETS + ['ens']):
    """min/max of a column across parameter sets and configurations."""
    a = np.array([series(c, s)[col].values for c in cfgs for s in sets])
    return a.min(0), a.max(0)


def degassing():
    """Gross atmospheric carbon influx of Muller et al. (2024), preferred model, Mt C yr-1 at 1-Myr steps."""
    return pd.read_csv(cfg.DEGASSING, comment='#').set_index('age')

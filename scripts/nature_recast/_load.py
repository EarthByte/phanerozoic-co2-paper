"""Shared loaders for the Nature-recast figures.

Result tables are those written by run_weathering.py into data/nature_recast/, named
`<run>_<set>.csv` with <set> one of the parameter sets or 'ens' (the three-set mean, the central
estimate). Fluxes in these tables are in Gt C/yr.
"""
from pathlib import Path
import numpy as np, pandas as pd

import _config as cfg
OUT, FIG, PROX = cfg.OUT, cfg.FIG, cfg.PROX
# the three regional parameter sets of Bufe et al. (2024)
SETS = ['Taiwan', 'Sichuan', 'NewZealand']


def series(cfg='kin_bridge_s2', which='ens'):
    """Result table of one run and parameter set, indexed by age (Ma); default: the main model, three-set mean."""
    return pd.read_csv(OUT / f'{cfg}_{which}.csv').set_index('age')


def envelope(col, cfgs=('kin_bridge_s2',), sets=SETS + ['ens']):
    """min/max of a column across parameter sets and configurations."""
    a = np.array([series(c, s)[col].values for c in cfgs for s in sets])
    return a.min(0), a.max(0)


def degassing():
    """Gross atmospheric carbon influx of Muller et al. (2024), preferred model, Mt C yr-1 at 1-Myr steps.

    Columns min, mean, max; index age (Ma). Written by extract_degassing.py.
    """
    return pd.read_csv(cfg.DEGASSING, comment='#').set_index('age')

"""Paths for the Nature-recast analysis.

Defaults follow the layout of the CO2-paper repository and its companion data archive
(see ../_paths.py); every location can be overridden with an environment variable.

  NEUROLEM_CO2_REPO   repository root (default: two levels above this file)
  NEUROLEM_CO2_DATA   companion data archive (default: <repo>/../neurolem_CO2_zenodo_data)
  NATURE_CACHE        land-only cache written by build_cache.py (one zip archive)
  NATURE_RUNOFF       HadCM3L precipitation and runoff table written by fetch_bridge_runoff.py
  NATURE_OUT          small result tables (git-tracked)
  NATURE_FIG          figures
  NATURE_PROX         weathering-proxy data
"""
import os, sys
from pathlib import Path

_env = os.environ.get
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.append(str(HERE.parent))        # _epsl_style, _wintri_map and _paths live one level up
REPO = Path(_env('NEUROLEM_CO2_REPO', HERE.parents[1]))
ZENODO = Path(_env('NEUROLEM_CO2_DATA', REPO.parent / 'neurolem_CO2_zenodo_data'))
CACHE = Path(_env('NATURE_CACHE', ZENODO / 'nature_recast_cache.zip'))
OUT = Path(_env('NATURE_OUT', REPO / 'data' / 'nature_recast'))
RUNOFF = Path(_env('NATURE_RUNOFF', OUT / 'hadcm3l_runoff_scotese_spinupa.npz'))
FIG = Path(_env('NATURE_FIG', REPO / 'figures' / 'nature'))
PROX = Path(_env('NATURE_PROX', REPO / 'data' / 'proxies'))
if str(REPO / 'scripts') not in sys.path:
    sys.path.append(str(REPO / 'scripts'))   # the same folder unless NEUROLEM_CO2_REPO points elsewhere
PARAMS = REPO / 'data' / 'bufe_2024_params.json'
ERODEP_DIR = ZENODO / 'surrogate_erodep'
TEMP_DIR = ZENODO / 'valdes_temperature'
SENS = ZENODO / 'climate_uncertainty' / 'phanda_sensitivity.npz'
EXPERIMENTS = REPO / 'scripts' / 'climate_uncertainty' / 'phanda_experiments.csv'
DEGASSING = OUT / 'muller2024_gross_degassing.csv'
PCO2 = {'foster': REPO / 'data' / 'foster.npy', 'judd': REPO / 'data' / 'judd.npy'}
for _d in (OUT, FIG):
    _d.mkdir(parents=True, exist_ok=True)

#!/usr/bin/env python3
"""Where the uncertainty experiments stand, and the next command to run.   python3 status.py"""
import csv
from pathlib import Path
import _config as cfg

ENS = cfg.ZENODO / 'geochem_uncertainty_ensemble' / 'nature_recast' / 'ensemble_sensitivity_members.csv'
CLI = cfg.OUT / 'climate_sensitivity_nodes.csv'


def blocks(path):
    n = {}
    if path.exists():
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                n[int(float(r['age']))] = n.get(int(float(r['age'])), 0) + 1
    return n


def line(ok, text):
    print(('  done   ' if ok else '  TO DO  ') + text)


ages = list(range(0, 545, 5)); nxt = []
print('Inputs')
for p, what in ((cfg.ERODEP_DIR, 'archived erosion fields'), (cfg.ZENODO / 'gospl_salles', 'goSPL physiography'),
                (cfg.ZENODO / 'surrogate_model' / 'best.pt', 'NeuroLEM checkpoint'), (cfg.OUT / 'kin_bridge_s2_amplitudes.txt', 'main-run calibration'),
                (cfg.OUT / 'hadcm3l_runoff_ratio_filled.npz', 'filled runoff-ratio table'), (cfg.SENS, 'climate emulator')):
    line(p.exists(), f'{what}: {p}')
sp = cfg.OUT / 'surface_provenance.csv'
if sp.exists():
    rows = list(csv.DictReader(open(sp, newline='')))
    arc = [r for r in rows if r['comparison'] == 'archive_vs_product']
    off = max((abs(float(r['mean_archive']) - float(r['mean_reference'])) for r in arc), default=1e9)
    line(len(arc) == 109 and off <= 0.05, f'surface provenance: {len(arc)}/109 archived surfaces measured against the product; largest land-mean difference {off:.3f} m')
    if off > 0.05:
        nxt.append('./run_uncertainty_m3.sh rebuild-surface   # the archive is not the adopted surface')
    elif len(arc) < 109:
        nxt.append('./run_uncertainty_m3.sh provenance   # CPU, one table; only a partial check has been run')
else:
    line(False, 'surface provenance: not measured'); nxt.append('./run_uncertainty_m3.sh provenance   # CPU, one table')
rft = cfg.OUT / 'runoff_form_tests.csv'
line(rft.exists() and rft.stat().st_mtime > (cfg.OUT / 'kin_bridge_s2_ens.csv').stat().st_mtime if (cfg.OUT / 'kin_bridge_s2_ens.csv').exists() else False,
     'runoff-form tests newer than the main run')
if not rft.exists() or ((cfg.OUT / 'kin_bridge_s2_ens.csv').exists() and rft.stat().st_mtime < (cfg.OUT / 'kin_bridge_s2_ens.csv').stat().st_mtime):
    nxt.append('./run_uncertainty_m3.sh runoff-tests   # CPU, four runs')
main_t = (cfg.OUT / 'kin_bridge_s2_ens.csv').stat().st_mtime if (cfg.OUT / 'kin_bridge_s2_ens.csv').exists() else None
for f, what, cmd in ((cfg.OUT / 'distribution_tests.csv', 'distribution-width tests newer than the main run', './run_uncertainty_m3.sh distribution-tests   # CPU, three runs'),
                     (cfg.OUT / 'octopus_summary.md', 'OCTOPUS comparison newer than the main run', './run_uncertainty_m3.sh octopus   # CPU; needs geopandas'),
                     (cfg.OUT / 'martin_lamb_summary.md', 'Martin & Lamb comparison newer than the main run', './run_uncertainty_m3.sh martin-lamb   # CPU'),
                     (cfg.OUT / 'orogen_tracks_before_after.csv', 'mountain belts followed through time newer than the main run', './run_uncertainty_m3.sh orogen-tracks   # CPU')):
    ok = f.exists() and main_t is not None and f.stat().st_mtime > main_t
    line(ok, what)
    if not ok: nxt.append(cmd)
ac = cfg.OUT / 'archive_consistency.csv'
if ac.exists():
    rows = list(csv.DictReader(open(ac, newline='')))
    bad = [(r['age'], r['surface']) for r in rows if r['consistent'] != 'yes' and not r['repaired']]
    line(len(rows) == 218 and not bad, f'archive consistency: {len(rows)}/218 fields checked' + (f'; NOT consistent and not repaired: {bad}' if bad else ''))
    if bad:
        nxt.append('./run_uncertainty_m3.sh archive-repair')
else:
    line(False, 'archive consistency: not checked (a freshly rebuilt archive is NeuroLEM output by construction; the climate run checks it age by age)')
newest = max((f.stat().st_mtime for f in cfg.ERODEP_DIR.glob('erodep_*Ma.npz')), default=0)
mt = cfg.OUT / 'kin_bridge_s2_ens.csv'
line(mt.exists() and mt.stat().st_mtime > newest, 'main-model tables are newer than the erosion archive')
if not (mt.exists() and mt.stat().st_mtime > newest):
    nxt.append('./run_uncertainty_m3.sh main         # tables, figures, audit; rebuilds the cache when the archive changed')
print('Runs')
e = blocks(ENS); full = [a for a in ages if e.get(a) == 202]; other = sorted(a for a in e if e[a] != 202)
line(len(full) == 109, f'palaeoelevation ensemble: {len(full)}/109 ages with 200 members' + (f'; ages with another member count (smoke test or interrupted): {other}' if other else ''))
if len(full) < 109:
    nxt.append('caffeinate -i ./run_uncertainty_m3.sh ensemble 2>&1 | tee -a ~/uncertainty_ensemble.log')
c = blocks(CLI); cfull = [a for a in ages if c.get(a) == 54]
line(len(cfull) == 109, f'climate states: {len(cfull)}/109 ages with 27 states x 2 surfaces')
if len(cfull) < 109:
    nxt.append('caffeinate -i ./run_uncertainty_m3.sh climate 2>&1 | tee -a ~/uncertainty_climate.log' if cfull else './run_uncertainty_m3.sh climate-pilot')
dt = blocks(cfg.OUT / 'dyntopo_gld428_robust.csv'); dfull = [a for a in range(0, 255, 5) if dt.get(a) == 3]
line(len(dfull) == 51, f'dynamic topography (gld428, retained plates, 0-250 Ma): {len(dfull)}/51 ages')
if len(dfull) < 51:
    nxt.append('caffeinate -i ./run_uncertainty_m3.sh dyntopo 2>&1 | tee -a ~/uncertainty_dyntopo.log' if dfull else './run_uncertainty_m3.sh dyntopo-pilot')
cdone = [c for c in ('tsh075', 'tsh150', 'tsh225', 'lon060', 'lon120', 'lon180')
         if len([a for a, n in blocks(cfg.OUT / f'dyntopo_control_{c}.csv').items() if n == 3]) == 51]
line(len(cdone) == 6, f'dynamic-topography control terms: {len(cdone)}/6 complete')
if len(dfull) == 51 and len(cdone) < 6:
    nxt.append('caffeinate -i ./run_uncertainty_m3.sh dyntopo-controls 2>&1 | tee -a ~/uncertainty_dyntopo.log')
g = cfg.OUT / 'gospl_validation_sensitivity.csv'
line(g.exists(), f'direct goSPL evaluation: {g.name}')
if not g.exists():
    nxt.append('./run_uncertainty_m3.sh gospl        # needs the direct goSPL run bundle')
gp = cfg.OUT / 'gospl_published_reference_sensitivity.csv'
ngp = len(blocks(gp))
line(ngp == 109, f'published goSPL fields, reference surface: {ngp}/109 ages')
if ngp < 109:
    nxt.append('./run_uncertainty_m3.sh gospl-published   # CPU only')
rs = cfg.OUT / 'reactivity_summary.csv'
gaps = -1
if rs.exists():
    rows_ = list(csv.DictReader(open(rs, newline='')))
    gaps = sum(1 for r in rows_ if r.get('k_fixforcing', '') in ('', 'nan'))
line(gaps == 0, 'reactivity summary complete at every age' + (f' ({gaps} ages without a value)' if gaps > 0 else ''))
if gaps != 0:
    nxt.append('./run_uncertainty_m3.sh reactivity   # CPU only')
print('Products')
for n in ('ensemble_sensitivity_age_summary.csv', 'ensemble_sensitivity_headline.csv', 'climate_sensitivity_bands.csv',
          'climate_sensitivity_headline.csv', 'uncertainty_statements.md', 'dyntopo_sensitivity_summary.md', 'dyntopo_controls_summary.md'):
    line((cfg.OUT / n).exists(), n)
line((cfg.FIG / 'ed_uncertainty.png').exists(), 'figures/nature/ed_uncertainty.png')
line(cfg.CACHE.exists(), f'input cache for Figs 1 and 4 and the reactivity runs: {cfg.CACHE.name}')
if not cfg.CACHE.exists():
    nxt.append('./run_uncertainty_m3.sh cache        # three minutes; independent of everything else')
nxt.append('./run_uncertainty_m3.sh finalise     # any time; uses whatever is complete')
print('Next\n  ' + '\n  '.join(nxt))

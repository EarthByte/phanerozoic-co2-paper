#!/usr/bin/env python3
"""Gross solid-Earth degassing of Muller et al. (2024), preferred model, as a small CSV.

Reads sheet 'Atmospheric Influx Comparisons' of the supplementary tables (columns 8-10:
bias-corrected rift lengths before 440 Ma, continental-arc outgassing from thermodynamic
modelling) and writes data/nature_recast/muller2024_gross_degassing.csv (Mt C/yr, 1-Myr steps).

The table (columns age in Ma, and min, mean, max of the gross atmospheric carbon influx, 0-545 Ma) is the
solid-Earth degassing of the Methods and Fig. 5a; it ships with the repository, so this script only needs to
run to rebuild it. Without an argument the workbook is looked for in <repo>/data and <repo>/../data.

    python3 extract_degassing.py [path/to/Muller_etal_solid_Earth_carbon_cycle_1Ga-present_SI_tables.xlsx]
"""
import sys
import pandas as pd
import _config as cfg

NAME = 'Muller_etal_solid_Earth_carbon_cycle_1Ga-present_SI_tables.xlsx'
src = sys.argv[1] if len(sys.argv) > 1 else next(p for p in (cfg.REPO / 'data' / NAME, cfg.REPO.parent / 'data' / NAME) if p.exists())
x = pd.read_excel(src, sheet_name='Atmospheric Influx Comparisons', header=None)
# the header row identifies the preferred-model columns; data start on row 6 (columns 1 and 8-10: age, min, mean, max)
assert 'preferred model' in str(x.iloc[3, 7])
d = x.iloc[5:, [0, 7, 8, 9]].apply(pd.to_numeric, errors='coerce').dropna()
d.columns = ['age', 'min', 'mean', 'max']; d['age'] = d['age'].astype(int); d = d[d.age <= 545]
with open(cfg.DEGASSING, 'w') as fh:
    fh.write("# Gross atmospheric carbon influx (solid-Earth degassing), Mt C yr-1: bias-corrected rift lengths before 440 Ma, continental-arc outgassing from\n"
             "# thermodynamic modelling (preferred model). Source: Muller et al. (2024) Geochem. Geophys. Geosyst. 25, e2024GC011713, supplementary tables,\n"
             "# sheet 'Atmospheric Influx Comparisons', columns 8-10.\n")
    d.to_csv(fh, index=False, float_format='%.4f')
print('wrote', cfg.DEGASSING, len(d), 'rows')

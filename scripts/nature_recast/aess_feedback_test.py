#!/usr/bin/env python3
"""How much stronger the weathering feedback becomes if the climate's response to CO2 is the apparent Earth-system
sensitivity of PhanDA (Judd et al. 2024, Science 385, eadk3705) rather than the HadCM3L model's own.

Supplementary Information, "The weathering feedback at a larger climate sensitivity".

WHY. The intrinsic strength of the weathering feedback in the paper, n = d ln F_sil / d ln CO2 (Extended Data Fig. 9b;
run_reactivity.py and reactivity_post.py), comes from the pattern-scaling emulator fitted to the HadCM3L prior ensemble
of PhanDA (climate_uncertainty/phanda_emulator.py): per grid cell and age,
    T(x) = T_ref(x) + lambda_T(x) log2(C/C_ref),   P(x) = (P_ref(x) + P0) exp(kappa_P(x) log2(C/C_ref)) - P0.
The area-weighted global mean of lambda_T, the emulator's warming per doubling, is 3.1-4.6 C across the 109 ages
(phanda_sensitivity.npz, ecs_area_weighted). PhanDA's global temperature regressed on proxy CO2 gives an apparent
Earth-system sensitivity (AESS) of 7.7 +- 0.3 C per doubling, the same in the Cenozoic and the Palaeozoic. If the real
long-term response to CO2 is closer to the AESS, the emulator understates how much a doubling changes land
temperature and runoff, and n is a lower bound. This test measures by how much.

DESIGN (fixed before the run). At each age the emulator's per-doubling fields are scaled by s(t) = AESS / ECS(t), with
ECS(t) the emulator's area-weighted global warming per doubling at that age, so that the global mean warming per
doubling equals the AESS while the spatial pattern of the model is kept. Two variants:
  T      lambda_T scaled, kappa_P as fitted (the hydrological response per doubling stays the model's)
  TP     lambda_T and kappa_P both scaled by s(t) (the hydrological response scales with the warming)
TP is the upper bracket: precipitation change per degree of warming is held at the model's value. The silicate sink
of the adjusted surface (main run, kin_bridge_s2 calibration and amplitudes, three-set mean) is evaluated at 280, 560
and 1,120 ppm exactly as in run_reactivity.py; n = ln(F_1120 / F_280) / ln 4. A third column, model, reproduces
run_reactivity.py and is checked against reactivity_fixed_co2.csv (relative difference <= 1e-6 at every age run), so
the only difference between the columns is the scaling. Land-mean temperature and runoff per doubling are reported
for each variant. AESS is set by --aess (default 7.7; the 1-sigma range 7.4-8.0 changes s by +-4%).

INPUTS: the input cache (build_cache.py), the emulator (phanda_sensitivity.npz), the main-run amplitudes
(kin_bridge_s2_amplitudes.txt) and reactivity_fixed_co2.csv (run_reactivity.py) for the regression check.

OUTPUTS (one table and one summary, nothing else): data/nature_recast/aess_feedback_test.csv (per age: ECS, s,
F_sil at 280/560/1120 ppm, n, land dT and dq per doubling, for model, T and TP) and aess_feedback_test_summary.md
(Phanerozoic mean and range of n per variant; the sink change per doubling implied by n, 2^n - 1; land warming
and runoff change per doubling).
Columns: co2_ref (ppm), ecs_model (deg C per doubling), scale (s), F_sil_<C>_<variant> (Gt C/yr), n_<variant> (-),
dT_land_per_doubling_<variant> (deg C, area-weighted land mean), dq_land_pct_per_doubling_<variant> (%).
A full run appends each age to aess_feedback_test.partial.csv as it finishes; once all ages are present the
complete table is written and the partial file is moved to data/nature_recast/_to_delete/.

    python3 aess_feedback_test.py 0 300            # pilot: two ages, including the regression check
    python3 aess_feedback_test.py                  # all 109 ages (resumable: re-run the same command), then the summary
"""
import ast, csv, sys
import numpy as np
import weathering_core as wc
import run_weathering as rw
import _cache
import _config as cfg

# apparent Earth-system sensitivity, deg C per doubling (--aess); the remaining arguments are ages
AESS = float(sys.argv[sys.argv.index('--aess') + 1]) if '--aess' in sys.argv else 7.7
args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith('--') and sys.argv[i - 1] != '--aess']
SENS = np.load(cfg.SENS)
OUT = cfg.OUT
wc.check_form(OUT / 'kin_bridge_s2_amplitudes.txt')
AMPS = ast.literal_eval(open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[0])
CO2 = [280.0, 560.0, 1120.0]
fitsets = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}
slat = SENS['lat'].astype(float); slon = SENS['lon'].astype(float); sage = [int(a) for a in SENS['age']]
# emulator's area-weighted global warming per doubling at each age, deg C
ECS = SENS['ecs_area_weighted'].astype(float)
# the run_reactivity.py table, against which the unscaled variant is checked
REF = {int(float(r['age'])): r for r in csv.DictReader(open(OUT / 'reactivity_fixed_co2.csv'))}
VARIANTS = ('model', 'T', 'TP')


def sink(E, q, T, A, m):
    """Global silicate sink (Gt C/yr), three-set mean with the main-run amplitudes, over the land mask m.
    E erosion rate (mm/yr), q runoff (mm/yr), T temperature (K), A cell area (m2)."""
    sil = 0.0
    for r in rw.SETS:
        t = rw.region_terms(E, q, T, fitsets[r], 'kinetic')
        sil += AMPS[r]['sil'] * float(np.sum((t['sil'] * A)[m])) * wc.GTC / 3
    return sil


def one_age(age):
    """All three variants at one age; returns the table row. Exits if the unscaled variant does not reproduce
    reactivity_fixed_co2.csv to a relative 1e-6."""
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    lat = d['lat'].astype(float); lon = d['lon'].astype(float)
    ilat = np.abs(slat[:, None] - lat[None, :]).argmin(0)
    ilon = (np.abs(((slon[:, None] - lon[None, :]) + 180) % 360 - 180)).argmin(0)
    r_i, c_i = np.unravel_index(idx, shp)
    k = sage.index(age)
    lam0 = SENS['lambda_T'][k][ilat[r_i], ilon[c_i]].astype(float)
    kap0 = SENS['kappa_P'][k][ilat[r_i], ilon[c_i]].astype(float)
    cref = float(SENS['co2_ref'][k]); s = AESS / ECS[k]
    A = d['area'].astype(float); T0 = d['T'].astype(float) + 273.15; P0 = d['rain'].astype(float) * 1000.0
    ratio = d['ratio'].astype(float)
    # adjusted surface, erosion rate (mm/yr) at the main-model calibration factor
    m = d['mz']; E = np.where(m, np.maximum(-d['ero_z'].astype(float), 0.0), 0.0) * 2.0
    Am = A[m].sum()
    row = dict(age=age, co2_ref=cref, ecs_model=round(float(ECS[k]), 4), scale=round(float(s), 4))
    for v in VARIANTS:
        # per-doubling temperature and precipitation slopes of the variant
        lam = lam0 * (s if v in ('T', 'TP') else 1.0)
        kap = kap0 * (s if v == 'TP' else 1.0)
        F = {}; Tl = {}; ql = {}
        for C in CO2:
            dbl = np.log2(C / cref)
            T = T0 + lam * dbl
            P = np.maximum((P0 + 10.0) * np.exp(kap * dbl) - 10.0, 0.0)
            q = P * ratio
            F[C] = sink(E, q, T, A, m)
            Tl[C] = float(np.sum((T * A)[m]) / Am); ql[C] = float(np.sum((q * A)[m]) / Am)
        for C in CO2:
            row[f'F_sil_{int(C)}_{v}'] = F[C]
        row[f'n_{v}'] = float(np.log(F[1120.0] / F[280.0]) / np.log(4.0))
        # 280 -> 1120 ppm is two doublings: half the temperature change, square root of the runoff ratio
        row[f'dT_land_per_doubling_{v}'] = (Tl[1120.0] - Tl[280.0]) / 2.0
        row[f'dq_land_pct_per_doubling_{v}'] = 100.0 * ((ql[1120.0] / ql[280.0]) ** 0.5 - 1.0)
    ref = REF.get(age)
    if ref is None:
        raise SystemExit(f'[{age} Ma] not in reactivity_fixed_co2.csv: run run_reactivity.py first')
    for C in CO2:
        want = float(ref[f'F_sil_{int(C)}_adj']); got = row[f'F_sil_{int(C)}_model']
        if abs(got / want - 1.0) > 1e-6:
            raise SystemExit(f'[{age} Ma] model variant does not reproduce reactivity_fixed_co2.csv at {int(C)} ppm: {got:.8g} vs {want:.8g}')
    return row


ages = [int(a) for a in args] or sage
out = OUT / 'aess_feedback_test.csv'; part = OUT / 'aess_feedback_test.partial.csv'
done = {}
if not args and part.exists():                       # resume a full run: ages already in the partial table are kept
    done = {int(float(r['age'])): r for r in csv.DictReader(open(part))}
rows = []
for age in ages:
    if age in done:
        continue
    r = one_age(age); rows.append(r)
    print(f"{age:4d} Ma: ECS {r['ecs_model']:.2f} C, scale {r['scale']:.2f}; n model {r['n_model']:.3f}, T {r['n_T']:.3f}, TP {r['n_TP']:.3f}; "
          f"land dT/doubling {r['dT_land_per_doubling_model']:.1f} -> {r['dT_land_per_doubling_T']:.1f} C; "
          f"runoff/doubling {r['dq_land_pct_per_doubling_model']:+.1f}% -> {r['dq_land_pct_per_doubling_TP']:+.1f}% (TP)", flush=True)
    if not args:                                     # full run: append each age as it is done, so an interrupted run resumes
        new_file = not part.exists()
        with open(part, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(r), lineterminator='\n')
            if new_file:
                w.writeheader()
            w.writerow(r)
if args:
    print(f'pilot: {len(ages)} ages, regression check passed; nothing written (run without ages for the full table)')
    sys.exit(0)
# full run complete: assemble the table and write the summary
import pandas as pd
t = pd.read_csv(part).drop_duplicates('age', keep='last').set_index('age').sort_index()
if sorted(t.index.tolist()) != sorted(sage):
    raise SystemExit(f'{len(t)} of {len(sage)} ages in {part.name}; run again to continue')
t.reset_index().to_csv(out, index=False)
(OUT / '_to_delete').mkdir(exist_ok=True); part.rename(OUT / '_to_delete' / 'aess_feedback_test.partial.csv')   # the complete table is aess_feedback_test.csv
L = [f'# Weathering feedback with the climate response scaled to the PhanDA apparent Earth-system sensitivity ({AESS} C per doubling)', '',
     'Written by aess_feedback_test.py; per age in aess_feedback_test.csv. n = d ln F_sil / d ln CO2 between 280 and 1,120 ppm, adjusted surface, main calibration.', '',
     f'Emulator warming per doubling (ECS): {t.ecs_model.min():.2f}-{t.ecs_model.max():.2f} C (mean {t.ecs_model.mean():.2f}); scale factor {t.scale.min():.2f}-{t.scale.max():.2f}.', '',
     '| variant | n mean | n range | sink change per doubling, mean | land dT per doubling, mean | land runoff change per doubling, mean |',
     '|---|---|---|---|---|---|']
# per variant: mean and range of n, and the sink change per doubling, 2^n - 1
for v, name in (('model', 'HadCM3L emulator as fitted (Extended Data Fig. 9b)'), ('T', 'temperature scaled to the AESS'), ('TP', 'temperature and precipitation scaled')):
    n = t[f'n_{v}']; per = 100 * (2 ** n - 1)
    L.append(f'| {name} | {n.mean():.3f} | {n.min():.3f} to {n.max():.3f} | {per.mean():+.1f}% (max {per.max():+.1f}%) | '
             f'{t[f"dT_land_per_doubling_{v}"].mean():.1f} C | {t[f"dq_land_pct_per_doubling_{v}"].mean():+.1f}% |')
pos = t[t.n_model > 0.01]
L += ['', f'Ratio n_T / n_model: median {(pos.n_T / pos.n_model).median():.2f}; n_TP / n_model: median {(pos.n_TP / pos.n_model).median():.2f} ({len(pos)} ages with n_model > 0.01).']
(OUT / 'aess_feedback_test_summary.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))

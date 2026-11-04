"""Per-cell fields on the 0.1 degree grid for one age: elevation, denudation, fluxes, local regime.

Library used by the map and mountain-belt figures (make_fig2_maps.py, make_fig4_endmembers.py,
make_fig1_optimum.py, make_fig3_erosion_sensitivity.py) and by highland_stats.py and
beyond_vs_co2.py. It evaluates the main model (kinetic runoff placement, HadCM3L runoff ratio,
erosion scale 2) cell by cell from the input cache, with the calibration amplitudes and the
E_opt(runoff) table of the main run (data/nature_recast/kin_bridge_s2_amplitudes.txt), never refitted.

Fluxes are the three-set mean of the calibrated predictions, in mol C m-2 yr-1:
  sil, dsil    silicate CO2 sink and its derivative d/dlnE
  sulf         sulfide-oxidation CO2 source
  ocp, ocb     rock organic-carbon oxidation (source) and biospheric organic-carbon burial (sink, ages <= 400 Ma)
  neti/dneti   the inorganic budget, silicate - sulfide, on which the drawdown optimum (Eopt, x) is defined
  net/dnet     what the figures call the net sink: by default (NET_BUDGET=inorg, the budget of the paper) the
               inorganic budget; with NET_BUDGET=full the full budget, silicate - sulfide - rock OC oxidation
               + biospheric OC burial from 400 Ma (the organic pathways of Extended Data Fig. 7 added)

Environment: NET_BUDGET = 'inorg' (default) or 'full'.
"""
import ast, os, numpy as np
import weathering_core as wc
import run_weathering as rw
import _cache
from _load import OUT

# calibration amplitudes and E_opt table, read once on first use
_AMPS = None; _EOPT = None
# runoff grid of the E_opt table, 1 to 10,000 mm/yr, as in run_weathering.run
QGRID = np.logspace(0, 4, 81)
# parameters of each regional set, kept intact
FITS = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}


def amps():
    """Silicate and sulfide amplitudes of the main run, {set: {'sil': .., 'sulf': ..}} (line 1 of the amplitudes file)."""
    global _AMPS
    if _AMPS is None:
        wc.check_form(OUT / 'kin_bridge_s2_amplitudes.txt')
        _AMPS = ast.literal_eval(open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[0])
    return _AMPS


BUDGET = os.environ.get('NET_BUDGET', 'inorg')     # 'inorg' (the paper) or 'full' (organic pathways added): what net/dnet mean
PLANTS_MA = 400                  # biospheric organic-carbon burial from 400 Ma (as kin_bridge columns F_ocb, Fig. 5)


def ocamps():
    """Organic-carbon amplitudes of the main run, {'ocp': .., 'ocb': ..} (line 2 of the amplitudes file)."""
    return ast.literal_eval(open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[1])


def eopt():
    """Optimum erosion rate (mm/yr) of the net inorganic drawdown at each runoff of QGRID (kinetic placement)."""
    global _EOPT
    if _EOPT is None:
        _EOPT = rw.eopt_table(FITS, amps(), 'kinetic', QGRID)
    return _EOPT


def compute(age, surface='adj', scale=2.0, full=True):
    """Fields of one age on one palaeoelevation surface.

    surface  'adj' (rock-chemistry-adjusted surface, main model) or 'ref' (reference PALEOMAP surface)
    scale    erosion calibration factor (2.0 in the main model)
    full     True: return 2-D grids (1801 x 3601, NaN off the land of the surface) plus 'lat' and 'lon' axes;
             False: return 1-D arrays on the cached cells (land on either surface), with 'mask' marking the
             land of this surface

    Returns (fields, cache dict). Fields: z (palaeoelevation, m), E (erosion rate, mm/yr, zero where depositing
    or not land), q (runoff, mm/yr), T (deg C), the fluxes of the module docstring (mol C m-2 yr-1), x = E/Eopt
    (NaN where E = 0), Eopt (mm/yr), area (m2), mask, ero (signed erosion-deposition rate times scale, mm/yr,
    negative = erosion).
    """
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    ek, mk, zk = ('ero_z', 'mz', 'z_z') if surface == 'adj' else ('ero_s', 'ms', 'z_s')
    m = d[mk]; A = d['area'].astype(float); T = d['T'].astype(float) + 273.15
    # runoff (mm/yr) = precipitation (m/yr -> mm/yr) x HadCM3L runoff ratio
    q = d['rain'].astype(float) * 1000.0 * d['ratio'].astype(float)
    # erosion rate: the negative part of the erosion-deposition field, on land only, times the calibration factor
    E = np.where(m, np.maximum(-d[ek].astype(float), 0.0), 0.0) * scale
    a = amps(); sil = dsil = sulf = dsulf = 0.0
    # central estimate: mean of the three calibrated parameter sets
    for r in rw.SETS:
        t = rw.region_terms(E, q, T, FITS[r], 'kinetic')
        sil = sil + a[r]['sil'] * t['sil'] / 3; dsil = dsil + a[r]['sil'] * t['dsil'] / 3
        sulf = sulf + a[r]['sulf'] * t['sulf'] / 3; dsulf = dsulf + a[r]['sulf'] * t['dsulf'] / 3
    # local optimum, interpolated in log runoff (runoff below 1 mm/yr takes the value at 1 mm/yr)
    Eo = np.interp(np.log(np.maximum(q, QGRID[0])), np.log(QGRID), eopt())
    # organic pathways: rock organic-carbon oxidation, and burial of biospheric organic carbon from 400 Ma, once land
    # plants supply it; amplitudes of the main run (line 2 of the amplitudes file); /12.011 as in run_weathering.run
    ao = ocamps(); Jp, dJp = wc.oc_petro(E, 'declining'); Jb, dJb = wc.oc_bio_burial(E)
    pl = 1.0 if age <= PLANTS_MA else 0.0
    ocp, docp = ao['ocp'] * Jp / 12.011, ao['ocp'] * dJp / 12.011
    ocb, docb = pl * ao['ocb'] * Jb / 12.011, pl * ao['ocb'] * dJb / 12.011
    f = dict(z=d[zk].astype(float), E=E, q=q, T=T - 273.15, sil=sil, sulf=sulf, ocp=ocp, ocb=ocb,
             neti=sil - sulf, dneti=dsil - dsulf, net=sil - sulf - ocp + ocb, dnet=dsil - dsulf - docp + docb,
             dsil=dsil, x=np.where(E > 0, E / Eo, np.nan), Eopt=Eo, area=A, mask=m, ero=d[ek].astype(float) * scale)
    if BUDGET == 'inorg':
        f['net'], f['dnet'] = f['neti'], f['dneti']
    if not full:
        return f, d
    # scatter the cached cells back onto the full grid; cells off the land of this surface become NaN
    g = {}
    for k, v in f.items():
        arr = np.full(shp[0] * shp[1], np.nan); arr[idx] = np.where(m, v, np.nan) if k != 'mask' else m
        g[k] = arr.reshape(shp)
    g['lat'] = d['lat'].astype(float); g['lon'] = d['lon'].astype(float)
    return g, d


def net_series(a, case='adj'):
    """Global net sink (Gt C/yr) and its erosion sensitivity S (-) from a kin_bridge table indexed by age.

    With NET_BUDGET=inorg (default) these are the table's inorganic F_net and S_net; with NET_BUDGET=full the
    full budget, silicate - sulfide - rock OC oxidation + biospheric burial from PLANTS_MA. `case` is a column
    suffix of the table ('adj', 'ref' or 'adjc').
    """
    if BUDGET == 'inorg':
        return a[f'F_net_{case}'], a[f'S_net_{case}']
    pl = (a.index.values <= PLANTS_MA).astype(float)
    F = a[f'F_net_{case}'] - a[f'F_ocp_{case}'] + pl * a[f'F_ocb_{case}']
    # dF/dlnE of the full budget, from S_net = dF_net / F_net and the organic derivatives
    dF = a[f'F_net_{case}'] * a[f'S_net_{case}'] - a[f'dF_ocp_{case}'] + pl * a[f'dF_ocb_{case}']
    return F, dF / F

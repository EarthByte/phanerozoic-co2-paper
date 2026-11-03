"""Per-cell fields on the 0.1 degree grid for one age: elevation, denudation, fluxes, local regime.
net/dnet: the full weathering budget (silicate - sulfide - rock OC oxidation + biospheric OC burial from 400 Ma; v16);
neti/dneti: the inorganic budget, on which the drawdown optimum (Eopt, x) is defined."""
import ast, os, numpy as np
import weathering_core as wc
import run_weathering as rw
import _cache
from _load import OUT

_AMPS = None; _EOPT = None
QGRID = np.logspace(0, 4, 81)
FITS = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}


def amps():
    global _AMPS
    if _AMPS is None:
        wc.check_form(OUT / 'kin_bridge_s2_amplitudes.txt')
        _AMPS = ast.literal_eval(open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[0])
    return _AMPS


BUDGET = os.environ.get('NET_BUDGET', 'inorg')     # 'inorg' (paper) or 'full' (organic pathways added; the v16 test): what net/dnet mean
PLANTS_MA = 400                  # biospheric organic-carbon burial from 400 Ma (as kin_bridge columns F_ocb, Fig. 5)


def ocamps():
    return ast.literal_eval(open(OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[1])


def eopt():
    global _EOPT
    if _EOPT is None:
        _EOPT = rw.eopt_table(FITS, amps(), 'kinetic', QGRID)
    return _EOPT


def compute(age, surface='adj', scale=2.0, full=True):
    d = _cache.load(age); shp = tuple(d['shape']); idx = d['idx']
    ek, mk, zk = ('ero_z', 'mz', 'z_z') if surface == 'adj' else ('ero_s', 'ms', 'z_s')
    m = d[mk]; A = d['area'].astype(float); T = d['T'].astype(float) + 273.15
    q = d['rain'].astype(float) * 1000.0 * d['ratio'].astype(float)
    E = np.where(m, np.maximum(-d[ek].astype(float), 0.0), 0.0) * scale
    a = amps(); sil = dsil = sulf = dsulf = 0.0
    for r in rw.SETS:
        t = rw.region_terms(E, q, T, FITS[r], 'kinetic')
        sil = sil + a[r]['sil'] * t['sil'] / 3; dsil = dsil + a[r]['sil'] * t['dsil'] / 3
        sulf = sulf + a[r]['sulf'] * t['sulf'] / 3; dsulf = dsulf + a[r]['sulf'] * t['dsulf'] / 3
    Eo = np.interp(np.log(np.maximum(q, QGRID[0])), np.log(QGRID), eopt())
    # organic pathways (v16: part of the net budget): rock organic-carbon oxidation, and burial of biospheric organic carbon
    # from 400 Ma, once land plants supply it; amplitudes of the main run (line 2 of the amplitudes file)
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
    g = {}
    for k, v in f.items():
        arr = np.full(shp[0] * shp[1], np.nan); arr[idx] = np.where(m, v, np.nan) if k != 'mask' else m
        g[k] = arr.reshape(shp)
    g['lat'] = d['lat'].astype(float); g['lon'] = d['lon'].astype(float)
    return g, d


def net_series(a, case='adj'):
    """Global net sink and its erosion sensitivity from a kin_bridge table (Gt C/yr): the full budget (silicate - sulfide -
    rock OC oxidation + biospheric burial from PLANTS_MA) or, with NET_BUDGET=inorg, the inorganic budget."""
    if BUDGET == 'inorg':
        return a[f'F_net_{case}'], a[f'S_net_{case}']
    pl = (a.index.values <= PLANTS_MA).astype(float)
    F = a[f'F_net_{case}'] - a[f'F_ocp_{case}'] + pl * a[f'F_ocb_{case}']
    dF = a[f'F_net_{case}'] * a[f'S_net_{case}'] - a[f'dF_ocp_{case}'] + pl * a[f'dF_ocb_{case}']
    return F, dF / F

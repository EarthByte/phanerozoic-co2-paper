#!/usr/bin/env python3
"""Global weathering, erosion sensitivity and buffering through the Phanerozoic.

This is the main weathering calculation of the paper (Methods, "Weathering laws" to "Erosion
sensitivity, local optimum and kinetic share"). Its tables feed Figs 1-5, Extended Data Figs 4-9
and the number audit (verify_numbers.py); other scripts import its functions (region_terms,
eopt_table, summarise) so that every experiment is evaluated exactly as the main model.

Reads the per-age cache (build_cache.py) for the 109 ages 0-540 Ma. Each of the three regional
parameter sets of Bufe et al. (2024) -- Taiwan, Sichuan, New Zealand -- is kept intact (its own
chi, K, sigma, z, Ea and reference runoff), calibrated separately to the modern global silicate
and sulfide fluxes, and evaluated in every land cell. The central estimate is the mean of the three
calibrated predictions ("ens"); the three sets give the parameter range.

Inputs per cell (cache): erosion-deposition rate of both palaeoelevation surfaces (mm/yr), emergent
masks, precipitation (m/yr), HadCM3L runoff ratio, surface-air temperature (deg C), cell area (m2).
Erosion rate E = max(-erosion-deposition, 0) x scale on land, so depositing cells carry no
weathering; runoff q = precipitation x runoff ratio (--qsrc bridge) or x 0.4 (--qsrc coef).

Each age is evaluated for three cases, which become column suffixes:
  ref    reference (PALEOMAP) surface on its own land
  adj    rock-chemistry-adjusted surface on its own land (main model)
  adjc   erosion of the adjusted surface integrated over the land of the reference surface (common land)

Outputs, per configuration (<out> is the --out prefix):
  <out>_<set>.csv        one row per age, for set = ens, Taiwan, Sichuan, NewZealand; columns <quantity>_<case>:
                         land_Mkm2, eroding_Mkm2 (10^6 km2); E_mean_mm, E_mean_eroding_mm, q_mean_mm (mm/yr,
                         area-weighted); D_Gt (denudation flux, Gt/yr); F_sil, F_sulf, F_net (= F_sil - F_sulf),
                         F_ocp, F_ocb (Gt C/yr; silicate sink, sulfide source, net inorganic sink, rock organic-carbon
                         oxidation, biospheric organic-carbon burial); dF_* (dF/dln(lambda), Gt C/yr); global erosion
                         sensitivities S = dlnF/dln(lambda) for a uniform change E -> lambda*E (S_sil, S_sulf, S_net;
                         S_net_oc_rel_sil and S_total_rel_sil, the derivative of the net budget with rock OC oxidation,
                         and with biospheric burial too, divided by the silicate sink); Gamma_sil (flux-weighted
                         kinetically limited share of the silicate flux); WI (weathering intensity); d7Li_riv (riverine
                         d7Li, permil); beyond_frac_land, beyond_frac_eroding, beyond_oc_frac_eroding (area share at or
                         beyond the local drawdown optimum); supplylim_frac_eroding (share with gamma < 0.5);
                         silflux_share_beyond (share of the silicate flux from land beyond the optimum); Eopt_mean_mm;
                         ens only: area-weighted percentiles E_pNN of E (mm/yr) and x_pNN of E/E_opt over eroding land
  <out>_hist.npz         area-weighted histograms over eroding land, ensemble, per age and case: E_<case> (log10 E),
                         x_<case> (log10 E/E_opt), netx_<case> (log10 E/E_opt weighted by net inorganic drawdown);
                         bin edges ebins, xbins; ages; the E_opt table (qgrid, eopt_q, rgrid)
  <out>_amplitudes.txt   line 1: silicate and sulfide amplitudes of each set; line 2: organic-carbon amplitudes;
                         line 3: the run settings (read back by the experiments, which never recalibrate)

    python3 run_weathering.py --mode kinetic --qsrc bridge --out ../../data/nature_recast/kin_bridge_s2   # main model
    python3 run_weathering.py --mode kinetic --qsrc bridge --ages 0 300 480 --out /tmp/pilot           # pilot

Options: --mode kinetic|outer (runoff in the rate constant or on the whole flux), --qsrc bridge|coef
(simulated runoff ratio or a uniform 0.4), --scale (erosion calibration factor, 2), --oc-form
declining|linear, --beta-sulf, --qref-scale, --sichuan-precip (runoff-form tests), --log-stretch
(widened erosion-rate distribution), --ages. The calibration is made at the first age given, so a
run should include 0 Ma first.
"""
import argparse, csv
from pathlib import Path
import numpy as np
import weathering_core as wc

import _config as cfg
import _cache
CACHE = cfg.CACHE
PARAMS = str(cfg.PARAMS)
SOLIDS_PER_CATION = 2.1   # silicate dissolved solids per unit silicate cation mass (550 Mt yr-1 per 4350 Gmol C yr-1; Caves Rugenstein et al., 2019)
# Modern calibration targets, Gt C yr-1, all on the long-term footing of the net-flux formulation (after carbonate precipitation):
#   sil  0.072 = half of the river-based silicate CO2 consumption of Moon et al. (2014), 10-14 x 10^12 mol yr-1 (central 12), the other
#        half returning to the atmosphere when carbonate precipitates in the ocean (Coogan & Caves Rugenstein, Fig. 13: rivers 6.2 Tmol yr-1)
#   sulf 0.0145 = 0.020 x 72/99: the sulfide target stands to the silicate target as 0.020 to 0.099, and the optimum, the shares
#        beyond it and every sensitivity depend only on that ratio; the Methods quote it as 14.5 Mt C yr-1, 1.2 x 10^12 mol of
#        pyrite-derived sulfate per year, against the pyrite-oxidation flux of Burke et al. (2018), 1.3 +- 0.2 x 10^12 mol S yr-1
#   ocp  0.068, modern rock organic-carbon oxidation; ocb 0.060, modern biospheric organic-carbon burial (Methods)
# Only the absolute fluxes depend on these targets.
TARGET = dict(sil=0.072, sulf=0.020 * 0.072 / 0.099, ocp=0.068, ocb=0.060)
SETS = ['Taiwan', 'Sichuan', 'NewZealand']
# the water variable that drives each set. The main model drives all three with runoff. The sets were calibrated on runoff
# estimated as precipitation minus evapotranspiration (Taiwan), on gauged runoff (New Zealand) and on TRMM precipitation
# (Sichuan; Bufe et al. 2024 supplement); --sichuan-precip drives the Sichuan set with precipitation, a runoff-form test
DRIVER = {'Taiwan': 'runoff', 'Sichuan': 'runoff', 'NewZealand': 'runoff'}
RGRID = np.linspace(0.05, 1.0, 20)     # runoff / precipitation, for the E_opt table when a set is precipitation-driven
# (column suffix, erosion field in the cache, land mask in the cache); see the module docstring
CASES = (('ref', 'ero_s', 'ms'), ('adj', 'ero_z', 'mz'), ('adjc', 'ero_z', 'ms'))
# percentiles reported for E and E/E_opt over eroding land
PCT = (5, 10, 25, 50, 75, 90, 95)
EBINS = np.linspace(-4, 1, 101)      # log10 E (mm/yr)
XBINS = np.linspace(-3, 3, 121)      # log10 E/E_opt


def wpercentile(x, w, p):
    """Weighted percentiles p (0-100) of x with weights w (here cell areas), by interpolating the cumulative weight."""
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp(np.asarray(p) / 100.0, c, x[o])


def region_terms(E, q, T, fits, mode):
    """CO2 fluxes of one parameter set in every cell, before calibration.

    E erosion rate (mm/yr), q runoff (mm/yr), T temperature (K), fits one entry of weathering_core.load_fits.
    Returns sil and sulf (mol C m-2 yr-1, silicate sink and sulfide source), dsil and dsulf (their derivatives
    d/dlnE, same units) and g (gamma, the kinetically limited share of the silicate flux, -).
    """
    Ws, dWs, gs = wc.flux_and_sens(E, q, T, fits['silicate'], wc.BETA['silicate'], mode)
    Wf, dWf, _ = wc.flux_and_sens(E, q, T, fits['sulfide'], wc.BETA['sulfide'], mode)
    return dict(sil=wc.co2_sil(Ws), dsil=wc.co2_sil(dWs), g=gs, sulf=wc.co2_sulf(Wf), dsulf=wc.co2_sulf(dWf))


def eopt_table(fitsets, amps, mode, qgrid):
    """Erosion rate (mm/yr) maximising the ensemble net inorganic drawdown, as a function of runoff
    (1-D) or, when a set is precipitation-driven, of runoff and the runoff/precipitation ratio (2-D)."""
    # erosion rates searched, 10^-4 to 10^1.5 mm/yr; the optimum is taken at 15 deg C (288.15 K), since the small
    # activation energies shift it by only about 4% between -5 and 30 deg C
    E = np.logspace(-4, 1.5, 1200)
    two_d = any(DRIVER[r] == 'precip' for r in SETS)
    rg = RGRID if two_d else np.array([1.0])
    out = np.empty((qgrid.size, rg.size))
    for i, q in enumerate(qgrid):
        for j, ratio in enumerate(rg):
            net = 0.0
            for r in SETS:
                w = q / ratio if DRIVER[r] == 'precip' else q
                t = region_terms(E, np.full_like(E, w), np.full_like(E, 288.15), fitsets[r], mode)
                net = net + amps[r]['sil'] * t['sil'] - amps[r]['sulf'] * t['sulf']
            out[i, j] = E[np.argmax(net)]
    return out if two_d else out[:, 0]


def eopt_at(q, P, eopt_q, qgrid):
    """Local optimum (mm/yr) of every cell from the table: interpolated in log runoff (1-D table), or bilinearly in
    log runoff and the runoff ratio q/P (2-D table, when a set is precipitation-driven). q and P in mm/yr."""
    lq = np.log(np.maximum(q, qgrid[0]))
    if eopt_q.ndim == 1:
        return np.interp(lq, np.log(qgrid), eopt_q)
    ratio = np.clip(np.where(P > 0, q / np.maximum(P, 1e-9), RGRID[0]), RGRID[0], RGRID[-1])
    i = np.clip(np.searchsorted(np.log(qgrid), lq) - 1, 0, qgrid.size - 2); j = np.clip(np.searchsorted(RGRID, ratio) - 1, 0, RGRID.size - 2)
    ti = (lq - np.log(qgrid[i])) / (np.log(qgrid[i + 1]) - np.log(qgrid[i])); tj = (ratio - RGRID[j]) / (RGRID[j + 1] - RGRID[j])
    return (eopt_q[i, j] * (1 - ti) * (1 - tj) + eopt_q[i + 1, j] * ti * (1 - tj) + eopt_q[i, j + 1] * (1 - ti) * tj + eopt_q[i + 1, j + 1] * ti * tj)


def summarise(E, q, m, A, sil, dsil, sulf, dsulf, g, ocp, docp, ocb, docb, eopt_q, qgrid, hist=None, P=None):
    """Global quantities of one age, one case and one parameter set (or the three-set mean).

    E, q   erosion rate and runoff per cell, mm/yr
    m      land mask over which everything is integrated
    A      cell area, m2
    sil .. docb   calibrated fluxes and their derivatives d/dlnE, mol C m-2 yr-1
    g      kinetically limited share gamma of the silicate flux per cell
    eopt_q, qgrid  E_opt table (eopt_table) and its runoff axis
    hist   if a dict is given, area-weighted histograms and the percentiles of E and E/E_opt are added (ens only)
    P      precipitation (mm/yr), needed only for a 2-D E_opt table

    Returns a dict of the columns listed in the module docstring.
    """
    ero = m & (E > 0)
    Am = A[m]
    # area integral over land: mol C m-2 yr-1 x m2 -> mol C yr-1 -> Gt C yr-1
    I = lambda f: float(np.sum(f[m] * Am)) * wc.GTC
    Fs, dFs, Ff, dFf = I(sil), I(dsil), I(sulf), I(dsulf)
    Fp, dFp, Fb, dFb = I(ocp), I(docp), I(ocb), I(docb)
    net, dnet = Fs - Ff, dFs - dFf
    Aland = float(Am.sum()); Aero = float(A[ero].sum())
    # a cell is at or beyond its local optimum where the net inorganic drawdown no longer rises with erosion
    loc = dsil - dsulf
    beyond = ero & (loc <= 0)
    beyond_oc = ero & ((loc - docp) <= 0)
    # weathering intensity and riverine d7Li (Caves Rugenstein et al., 2019, eqs 21-25 and code)
    # sil x 2 x M_cat inverts co2_sil back to the cation flux (g m-2 = t km-2); x 2.1 converts cations to dissolved solids
    Wsol = SOLIDS_PER_CATION * sil * 2.0 * wc.M_CAT          # t km-2 yr-1 of dissolved silicate solids
    # total denudation, t km-2 yr-1; weathering intensity WI = dissolved silicate flux / denudation, kept inside (0, 1)
    Dm = np.maximum(E * wc.RHO, 1e-9)
    WIc = np.clip(Wsol / Dm, 1e-6, 0.999)
    # riverine d7Li (permil) of each cell from its own weathering intensity
    d7 = ((1 - WIc) / WIc) * 4.5 * np.exp(-0.0575 / WIc) + 1.5
    # dissolved lithium fraction from Rayleigh distillation, clay-water fractionation -17 permil about the 1.5 permil source
    fLi = np.exp((d7 - 1.5) / -17.0)
    # the global riverine value is weighted by the dissolved lithium flux of eroding cells
    wLi = Wsol * fLi * A * ero
    Eopt = eopt_at(q, P if P is not None else q, eopt_q, qgrid)
    x = np.where(ero, E / Eopt, np.nan)
    r = dict(land_Mkm2=Aland / 1e12, eroding_Mkm2=Aero / 1e12,
             E_mean_mm=float(np.sum(E[m] * Am) / Aland), E_mean_eroding_mm=float(np.sum(E[ero] * A[ero]) / Aero),
             q_mean_mm=float(np.sum(q[m] * Am) / Aland),
             # mm/yr x RHO = t km-2 yr-1; x m2 x 1e-6 -> t yr-1; x 1e-9 -> Gt yr-1
             D_Gt=float(np.sum(E[m] * wc.RHO * Am)) * 1e-6 * 1e-9,
             F_sil=Fs, F_sulf=Ff, F_net=net, F_ocp=Fp, F_ocb=Fb,
             dF_sil=dFs, dF_sulf=dFf, dF_ocp=dFp, dF_ocb=dFb,
             S_sil=dFs / Fs, S_sulf=dFf / Ff, S_net=dnet / net,
             S_net_oc_rel_sil=(dnet - dFp) / Fs, S_total_rel_sil=(dnet - dFp + dFb) / Fs,
             Gamma_sil=float(np.sum((g * sil)[m] * Am) / np.sum(sil[m] * Am)),
             WI=float(np.sum(Wsol[m] * Am) / np.sum(Dm[m] * Am)),
             d7Li_riv=float(np.sum(wLi * d7) / np.sum(wLi)),
             beyond_frac_land=float(A[beyond].sum()) / Aland, beyond_frac_eroding=float(A[beyond].sum()) / Aero,
             beyond_oc_frac_eroding=float(A[beyond_oc].sum()) / Aero,
             supplylim_frac_eroding=float(A[ero & (g < 0.5)].sum()) / Aero,
             silflux_share_beyond=float(np.sum((sil * A)[beyond]) / np.sum((sil * A)[m])),
             Eopt_mean_mm=float(np.sum(Eopt[ero] * A[ero]) / Aero))
    if hist is not None:
        pe = wpercentile(E[ero], A[ero], PCT); px = wpercentile(x[ero], A[ero], PCT)
        for p, a, b in zip(PCT, pe, px):
            r[f'E_p{p:02d}'] = float(a); r[f'x_p{p:02d}'] = float(b)
        hist['E'] = np.histogram(np.log10(E[ero]), bins=EBINS, weights=A[ero])[0]
        hist['x'] = np.histogram(np.log10(x[ero]), bins=XBINS, weights=A[ero])[0]
        hist['netx'] = np.histogram(np.log10(x[ero]), bins=XBINS, weights=((sil - sulf) * A)[ero])[0]
    return r


def _wmedian(x, w):
    """Weighted median of x with weights w."""
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def run(mode, qsrc, scale, oc_form, out, ages=None, beta_sulf=None, qref_scale='', sichuan_precip=False, log_stretch=1.0):
    """Evaluate all ages and write the tables, histograms and amplitudes under the prefix `out` (see the module docstring).

    The options of the runoff-form tests change module state (DRIVER, weathering_core.BETA, the q_ref of a set) for the
    rest of the process; they are recorded on line 3 of the amplitudes file.
    """
    fitsets = {r: wc.load_fits(PARAMS, [r]) for r in SETS}
    if sichuan_precip:
        DRIVER['Sichuan'] = 'precip'
    if beta_sulf is not None:
        wc.BETA['sulfide'] = beta_sulf
    for kv in filter(None, qref_scale.split(',')):
        r_, f_ = kv.split('='); fitsets[r_]['silicate']['q_ref_mm_yr'] *= float(f_); fitsets[r_]['sulfide']['q_ref_mm_yr'] *= float(f_)
    # the 109 ages 0, 5, ..., 540 Ma unless given
    ages = ages or list(range(0, 545, 5))
    amps = None; amp_oc = None; eopt_q = None
    qgrid = np.logspace(0, 4, 81)
    rows = {k: [] for k in ['ens'] + SETS}
    H = {c[0]: dict(E=[], x=[], netx=[]) for c in CASES}
    for age in ages:
        d = _cache.load(age)
        A = d['area'].astype(float); T = d['T'].astype(float) + 273.15
        # precipitation, m/yr -> mm/yr; runoff from the HadCM3L runoff ratio or a uniform coefficient of 0.4
        P = d['rain'].astype(float) * 1000.0
        q = P * (0.4 if qsrc == 'coef' else d['ratio'].astype(float))
        water = {r: (P if DRIVER[r] == 'precip' else q) for r in SETS}
        row = {k: dict(age=age) for k in rows}
        for case, ek, mk in CASES:
            m = d[mk]
            # erosion rate (mm/yr): the eroding part of the signed field, on land, times the calibration factor
            E = np.where(m, np.maximum(-d[ek].astype(float), 0.0), 0.0) * scale
            if log_stretch != 1.0:
                # widen (k > 1) or narrow the distribution of erosion rates across eroding land, E' = c med (E/med)^k,
                # with c chosen so that the erosion flux of the age (the calibrated quantity) is unchanged: the test of
                # a simulated distribution that is too narrow (OCTOPUS comparison, octopus_comparison.py)
                er = m & (E > 0); med = _wmedian(E[er], A[er])
                Es = np.where(er, med * np.power(np.maximum(E, 1e-12) / med, log_stretch), E)
                E = Es * (float(np.sum(E[er] * A[er])) / float(np.sum(Es[er] * A[er])))
            terms = {r: region_terms(E, water[r], T, fitsets[r], mode) for r in SETS}
            Jp, dJp = wc.oc_petro(E, oc_form); Jb, dJb = wc.oc_bio_burial(E)
            # organic-carbon terms are proportional to the flux; /12.011 puts them on a mol C footing before the amplitude
            Jp, dJp, Jb, dJb = Jp / 12.011, dJp / 12.011, Jb / 12.011, dJb / 12.011
            # calibrate every set once, at the first age (0 Ma) and the first case (the reference surface, which the
            # adjustment leaves unchanged at the present day): amplitude = target / modern global flux of the set.
            # The E_opt table is built from the calibrated sets, so it is fixed for all ages and cases.
            if amps is None:       # calibrate every set once, on the 0 Ma surface
                I = lambda f: float(np.sum(f[m] * A[m])) * wc.GTC
                amps = {r: dict(sil=TARGET['sil'] / I(terms[r]['sil']), sulf=TARGET['sulf'] / I(terms[r]['sulf'])) for r in SETS}
                amp_oc = dict(ocp=TARGET['ocp'] / I(Jp), ocb=TARGET['ocb'] / I(Jb))
                eopt_q = eopt_table(fitsets, amps, mode, qgrid)
            ocp, docp, ocb, docb = amp_oc['ocp'] * Jp, amp_oc['ocp'] * dJp, amp_oc['ocb'] * Jb, amp_oc['ocb'] * dJb
            # calibrated fluxes of each set, their three-set mean (central estimate), and gamma weighted by silicate flux
            cal = {r: {k: amps[r]['sil' if 'sil' in k else 'sulf'] * terms[r][k] for k in ('sil', 'dsil', 'sulf', 'dsulf')} for r in SETS}
            ens = {k: sum(cal[r][k] for r in SETS) / len(SETS) for k in ('sil', 'dsil', 'sulf', 'dsulf')}
            gsum = sum(terms[r]['g'] * cal[r]['sil'] for r in SETS) / np.maximum(sum(cal[r]['sil'] for r in SETS), 1e-300)
            h = {}
            res = summarise(E, q, m, A, ens['sil'], ens['dsil'], ens['sulf'], ens['dsulf'], gsum, ocp, docp, ocb, docb, eopt_q, qgrid, h, P=P)
            for k in H[case]:
                H[case][k].append(h[k])
            row['ens'].update({f'{k}_{case}': v for k, v in res.items()})
            for r in SETS:
                res = summarise(E, q, m, A, cal[r]['sil'], cal[r]['dsil'], cal[r]['sulf'], cal[r]['dsulf'], terms[r]['g'],
                                ocp, docp, ocb, docb, eopt_q, qgrid, P=P)
                row[r].update({f'{k}_{case}': v for k, v in res.items()})
        for k in rows:
            rows[k].append(row[k])
        e = row['ens']
        print(age, f"Fsil {e['F_sil_adj']:.4f} Fnet {e['F_net_adj']:.4f} S_sil {e['S_sil_adj']:.3f} S_net {e['S_net_adj']:.3f} "
                   f"beyond {e['beyond_frac_eroding_adj']:.3f} WI {e['WI_adj']:.3f}", flush=True)
    for k in rows:
        with open(f'{out}_{k}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[k][0].keys()), lineterminator='\n')
            w.writeheader(); w.writerows(rows[k])
    np.savez_compressed(f'{out}_hist.npz', ages=np.array(ages), ebins=EBINS, xbins=XBINS, qgrid=qgrid, eopt_q=eopt_q, rgrid=RGRID,
                        **{f'{k}_{c}': np.array(H[c][k]) for c in H for k in H[c]})
    with open(f'{out}_amplitudes.txt', 'w') as fh:
        fh.write(repr(amps) + '\n' + repr(amp_oc) + f"\nmode={mode} qsrc={qsrc} scale={scale} oc_form={oc_form} beta_sulf={wc.BETA['sulfide']} qref_scale={qref_scale or 'none'} driver={DRIVER} stretch={log_stretch}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='kinetic', choices=['kinetic', 'outer'])
    ap.add_argument('--qsrc', default='bridge', choices=['bridge', 'coef'])
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--oc-form', default='declining')
    ap.add_argument('--beta-sulf', type=float, default=wc.BETA['sulfide'], help='runoff exponent of the sulfide term (default 0, the pure West form; -0.75 = Bufe et al. 2024 as published)')
    ap.add_argument('--qref-scale', default='', help='scale the reference runoff of a set, e.g. Sichuan=0.5')
    ap.add_argument('--sichuan-precip', action='store_true', help='drive the Sichuan set with precipitation, the variable it was calibrated on')
    ap.add_argument('--log-stretch', type=float, default=1.0, help='exponent k applied to E/median over eroding land (1 = as simulated; 1.8 matches the cosmogenic spread)')
    ap.add_argument('--ages', nargs='*', type=int)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run(a.mode, a.qsrc, a.scale, a.oc_form, a.out, a.ages, a.beta_sulf, a.qref_scale, a.sichuan_precip, a.log_stretch)

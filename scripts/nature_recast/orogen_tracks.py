#!/usr/bin/env python3
"""Mountain-building events followed through time: the same crust before, during and after uplift.

The crust of a mountain belt (land above 1,500 m that erodes, in a window about the belt at the age of the belt) is
tracked through the PALEOMAP plate model (Scotese & Wright 2023 rotations and plate polygons, the frame of the
palaeoelevation surfaces) to every other age of the reconstruction within the tracking span. At each age the
simulated fields of the main model are sampled where that crust lies, and its erosion rates relative to the local
optimum, its runoff, elevation and net CO2 drawdown are summarised. The change from the age before uplift to the
age of the belt is split into the part due to erosion and the part due to climate by evaluating the weathering
laws with the erosion rates of one age and the runoff and temperature of the other.

    python3 orogen_tracks.py select     # belt crust at the belt ages (needs the input cache) -> orogen_belt_cells.csv
    python3 orogen_tracks.py track      # positions at every age (needs pygplates)            -> orogen_tracks.npz
    python3 orogen_tracks.py evaluate   # fields along the tracks (needs the input cache)    -> orogen_tracks_points.npz
    python3 orogen_tracks.py analyse    # trajectories, before/after, decomposition           -> orogen_tracks_summary.csv,
                                        #                                                         orogen_tracks_before_after.csv/.md
    python3 orogen_tracks.py figure     # trajectories of the four belts (Supplementary Fig. 1)  -> figures/nature/supp1_orogen_tracks
    python3 orogen_tracks.py sensitivity  # event-scale erosion sensitivity and the split of the change in net drawdown
                                          # into exposure (erosion, emergence) and climate      -> orogen_sensitivity.csv/.md
    OROGEN_SET=events python3 orogen_tracks.py select|track|evaluate|analyse|sensitivity|events-figure
                                        # the same for the catalogue of orogen_catalogue.py (every file name gets _events);
                                        # events-figure draws the event-scale sensitivities on their own -> figures/nature/orogen_sensitivity
                                        # (the same panels are Fig. 4e,f, make_fig4_endmembers.py)
Every step writes one file (two for analyse) in data/nature_recast/. The plate model is read from
Paleotopo_data_assimilation/data/Scotese_Wright_plate_model_revised/ beside NeuroLEM (override with PALEOMAP_PLATE_DIR).
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import _config as cfg

# name, age (Ma), window centre lat, lon (palaeo-coordinates at that age), half-widths (deg), label
BELTS = [('caledonian', 420, -10.6, -13.2, 16, 30, 'Caledonian belt, 420 Ma'),
         ('central_pangaean', 300, -5.9, -2.3, 16, 30, 'Central Pangaean belt, 300 Ma'),
         ('himalaya_tibet', 0, 32.0, 88.0, 10, 20, 'Himalaya and Tibet, present'),
         ('central_andes', 0, -18.0, -67.0, 12, 8, 'Central Andes, present')]
SET = os.environ.get('OROGEN_SET', 'paper')       # 'paper': the four belts of Fig. 4; 'events': the catalogue of orogen_catalogue.py
SUFFIX = '' if SET == 'paper' else '_' + SET
RADIUS = {}                                        # events located from a reference point: belt crust within this arc (deg) of it
if SET == 'events':
    _paper = {b[0]: b for b in BELTS}; _ev = []
    for r in csv.DictReader(open(cfg.OUT / 'orogen_catalogue.csv')):
        if r['used'] != '1':
            continue
        if r['key'] in _paper:
            _ev.append(_paper[r['key']]); continue
        _ev.append((r['key'], int(r['age']), float(r['lat']), float(r['lon']), 0, 0, f"{r['label']} orogen, {int(r['age'])} Ma"))
        RADIUS[r['key']] = float(r['radius'])
    BELTS = sorted(_ev, key=lambda b: -b[1])
ZMIN = 1500.0
SPAN_BEFORE, SPAN_AFTER = 120, 60          # Myr tracked before and after the belt age
LEAD = 60                                  # Myr before (and after) the belt age searched for the low stand / reported after
SCALE = 2.0
PLATE_DIR = Path(os.environ.get('PALEOMAP_PLATE_DIR', cfg.REPO.parent.parent / 'Paleotopo_data_assimilation' / 'data' / 'Scotese_Wright_plate_model_revised'))
ROT = PLATE_DIR / 'Scotese_Wright_PlateModel_2023.rot'
POLY = PLATE_DIR / 'Scotese_Wright_PlatePolygons_2023.gpml'
CELLS = cfg.OUT / f'orogen_belt_cells{SUFFIX}.csv'
TRACKS = cfg.OUT / f'orogen_tracks{SUFFIX}.npz'
POINTS = cfg.OUT / f'orogen_tracks_points{SUFFIX}.npz'
SUMMARY = cfg.OUT / f'orogen_tracks_summary{SUFFIX}.csv'
BA_CSV = cfg.OUT / f'orogen_tracks_before_after{SUFFIX}.csv'
BA_MD = cfg.OUT / f'orogen_tracks_before_after{SUFFIX}.md'
SENS_CSV = cfg.OUT / f'orogen_sensitivity{SUFFIX}.csv'
SENS_MD = cfg.OUT / f'orogen_sensitivity{SUFFIX}.md'
GTC = 12.011       # cellfields fluxes x 12.011 = t C km-2 yr-1 (as in Fig. 4)
MIN_RATIO = 1.5    # events-figure: smallest change in the erosion flux of the tracked crust for an event-scale sensitivity


def _grid(d):
    nlat, nlon = (int(v) for v in d['shape'])
    inv = np.full(nlat * nlon, -1, np.int64); inv[d['idx'].astype(np.int64)] = np.arange(d['idx'].size)
    return nlat, nlon, inv


def _cell(d, lat, lon):
    """Index into the compact arrays of the cache for points at lat/lon (-1 where the cache holds no cell)."""
    nlat, nlon, inv = _grid(d)
    la, lo = d['lat'].astype(float), d['lon'].astype(float)
    i = np.clip(np.rint((lat - la[0]) / (la[1] - la[0])).astype(np.int64), 0, nlat - 1)
    j = np.rint(((lon - lo[0]) % 360.0) / (lo[1] - lo[0])).astype(np.int64) % (nlon - 1)
    return inv[i * nlon + j]


def _arc(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))


def select():
    import _cache
    rows = []
    for name, age, lat0, lon0, dla, dlo, _ in BELTS:
        d = _cache.load(age); nlat, nlon, _inv = _grid(d)
        idx = d['idx'].astype(np.int64); la = d['lat'].astype(float)[idx // nlon]; lo = d['lon'].astype(float)[idx % nlon]
        E = np.where(d['mz'], np.maximum(-d['ero_z'].astype(float), 0.0), 0.0) * SCALE
        dlon = (lo - lon0 + 180.0) % 360.0 - 180.0
        if name in RADIUS:
            inside = _arc(la, lo, lat0, lon0) <= RADIUS[name]
        else:
            inside = (np.abs(la - lat0) <= dla) & (np.abs(dlon) <= dlo)
        sel = d['mz'] & (E > 0) & (d['z_z'] > ZMIN) & inside & (lo < 180.0)
        for a, b, A in zip(la[sel], lo[sel], d['area'][sel].astype(float) / 1e6):
            rows.append((name, age, round(float(a), 2), round(float(b), 2), round(float(A), 3)))
        print(f'{name}: {int(sel.sum())} cells, {d["area"][sel].astype(float).sum() / 1e12:.2f} M km2 above {ZMIN:.0f} m', flush=True)
    with open(CELLS, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['belt', 'age', 'lat', 'lon', 'area_km2']); w.writerows(rows)
    print('wrote', CELLS)


def track():
    import pygplates
    rot = pygplates.RotationModel(str(ROT)); polys = pygplates.FeatureCollection(str(POLY))
    cells = list(csv.DictReader(open(CELLS)))
    out = {}
    for name, age, *_ in BELTS:
        c = [r for r in cells if r['belt'] == name]
        lat = np.array([float(r['lat']) for r in c]); lon = np.array([float(r['lon']) for r in c]); A = np.array([float(r['area_km2']) for r in c])
        part = pygplates.PlatePartitioner(polys, rot, reconstruction_time=float(age))
        pid = np.zeros(lat.size, np.int64)
        for k, (a, b) in enumerate(zip(lat, lon)):
            pp = part.partition_point(pygplates.PointOnSphere(float(a), float(b)))
            pid[k] = pp.get_feature().get_reconstruction_plate_id() if pp else 0
        ages = np.arange(max(0, age - SPAN_AFTER), min(540, age + SPAN_BEFORE) + 1, 5)
        LAT = np.full((ages.size, lat.size), np.nan); LON = np.full_like(LAT, np.nan)
        for p in np.unique(pid):
            k = np.where(pid == p)[0]
            pts = [pygplates.PointOnSphere(float(lat[i]), float(lon[i])) for i in k]
            for ia, t in enumerate(ages):
                R = rot.get_rotation(float(t), int(p), float(age))          # from the belt age to age t, same plate
                for i, pt in zip(k, pts):
                    LAT[ia, i], LON[ia, i] = (R * pt).to_lat_lon()
        unassigned = float(A[pid == 0].sum() / A.sum())
        if (pid == 0).any() and (pid > 0).any():         # cells in gaps between the reconstructed polygons take the plate of the nearest assigned cell
            v = lambda la, lo: np.c_[np.cos(np.radians(la)) * np.cos(np.radians(lo)), np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))]
            a_, u_ = np.where(pid > 0)[0], np.where(pid == 0)[0]
            Va = v(lat[a_], lon[a_])
            for c0 in range(0, u_.size, 2000):
                cu = u_[c0:c0 + 2000]
                pid[cu] = pid[a_[np.argmax(v(lat[cu], lon[cu]) @ Va.T, axis=1)]]
        out.update({f'{name}_ages': ages, f'{name}_lat': LAT.astype(np.float32), f'{name}_lon': LON.astype(np.float32),
                    f'{name}_area': A.astype(np.float32), f'{name}_plate': pid})
        print(f'{name}: {lat.size} points on {np.unique(pid).size} plates ({100 * unassigned:.1f}% of area in gaps between plate polygons, given the plate of the nearest cell), '
              f'{ages[0]}-{ages[-1]} Ma', flush=True)
    np.savez_compressed(TRACKS, **out, rotation_file=str(ROT.name), polygon_file=str(POLY.name))
    print('wrote', TRACKS)


def _inputs(d, k):
    """Erosion, runoff, temperature, elevation and land flag at compact indices k (k = -1: no land)."""
    ok = k >= 0; kk = np.where(ok, k, 0)
    land = ok & d['mz'][kk]
    E = np.where(land, np.maximum(-d['ero_z'][kk].astype(float), 0.0), 0.0) * SCALE
    q = np.where(ok, d['rain'][kk].astype(float) * 1000.0 * d['ratio'][kk].astype(float), np.nan)
    T = np.where(ok, d['T'][kk].astype(float) + 273.15, np.nan)
    z = np.where(ok, d['z_z'][kk].astype(float), np.nan)
    return E, q, T, z, land


def weather(E, q, T, age=None):
    """Central-estimate silicate and sulfide terms and their erosion derivatives (as cellfields.compute). With an age (v16),
    net and dnet are the full budget, adding rock organic-carbon oxidation and biospheric burial (from 400 Ma); neti and
    dneti are always the inorganic budget, on which the optimum is defined."""
    import cellfields as cf, run_weathering as rw
    a = cf.amps(); sil = dsil = sulf = dsulf = 0.0
    qq = np.nan_to_num(q, nan=1.0); TT = np.nan_to_num(T, nan=288.15)
    for r in rw.SETS:
        t = rw.region_terms(E, qq, TT, cf.FITS[r], 'kinetic')
        sil = sil + a[r]['sil'] * t['sil'] / 3; dsil = dsil + a[r]['sil'] * t['dsil'] / 3
        sulf = sulf + a[r]['sulf'] * t['sulf'] / 3; dsulf = dsulf + a[r]['sulf'] * t['dsulf'] / 3
    Eo = np.interp(np.log(np.maximum(qq, cf.QGRID[0])), np.log(cf.QGRID), cf.eopt())
    w = dict(sil=sil, dsil=dsil, net=sil - sulf, dnet=dsil - dsulf, neti=sil - sulf, dneti=dsil - dsulf, Eopt=Eo)
    if age is not None:
        ocp, ocb, docp, docb = _oc(E, age, deriv=True)
        w['net'] = w['neti'] - ocp + ocb; w['dnet'] = w['dneti'] - docp + docb
    return w


def evaluate():
    import _cache
    tr = np.load(TRACKS)
    need = sorted({int(t) for name, *_ in BELTS for t in tr[f'{name}_ages']})
    store = {}
    for t in need:
        d = _cache.load(t)
        for name, *_ in BELTS:
            ages = tr[f'{name}_ages']
            if t not in ages:
                continue
            ia = int(np.where(ages == t)[0][0])
            k = _cell(d, tr[f'{name}_lat'][ia].astype(float), tr[f'{name}_lon'][ia].astype(float))
            E, q, T, z, land = _inputs(d, k)
            for key, v in (('E', E), ('q', q), ('T', T), ('z', z), ('land', land)):
                store.setdefault(f'{name}_{key}', [None] * ages.size)[ia] = v.astype(np.float32 if key != 'land' else bool)
        print(t, 'Ma', flush=True)
    out = {k: np.stack(v) for k, v in store.items()}
    for name, *_ in BELTS:
        out[f'{name}_ages'] = tr[f'{name}_ages']; out[f'{name}_area'] = tr[f'{name}_area']
    np.savez_compressed(POINTS, **out)
    print('wrote', POINTS)


def _stats(E, q, T, z, land, A, age=None):
    import cellfields as cf
    w = weather(E, q, T, age if cf.BUDGET == 'full' else None)
    ero = land & (E > 0); Ae = A * ero; Al = A * land
    x = np.where(ero, E / w['Eopt'], np.nan)
    def wq(v, p):
        o = np.argsort(v[ero]); c = np.cumsum(Ae[ero][o]); return float(v[ero][o][np.searchsorted(c / c[-1], p)]) if ero.any() else np.nan
    F = lambda v: float(np.sum(v * Ae))
    return dict(land_share=float(Al.sum() / A.sum()), eroding_share=float(Ae.sum() / A.sum()),
                z_mean=float(np.sum(np.nan_to_num(z) * Al) / max(Al.sum(), 1e-9)),
                E_mean=float(np.sum(E * Ae) / max(Ae.sum(), 1e-9)), q_mean=float(np.sum(np.nan_to_num(q) * Ae) / max(Ae.sum(), 1e-9)),
                x_p25=wq(x, 0.25), x_p50=wq(x, 0.5), x_p75=wq(x, 0.75),
                beyond=float(np.sum(Ae * (w['dneti'] <= 0)) / max(Ae.sum(), 1e-9)),
                sil_density=GTC * F(w['sil']) / A.sum(), net_density=GTC * F(w['net']) / A.sum(),
                S_sil=F(w['dsil']) / F(w['sil']) if F(w['sil']) > 0 else np.nan, S_net=F(w['dnet']) / F(w['net']) if F(w['net']) != 0 else np.nan,
                net_Mt=GTC * F(w['net']) * 1e-6, x=x)


def analyse():
    P = np.load(POINTS)
    rows = []; ba = []
    L = ['# Mountain belts followed through time on the main model', '',
         'Crust above 1,500 m at the age of each belt, tracked through the PALEOMAP plate model; fluxes per km² of that crust '
         '(t C km⁻² yr⁻¹), net = silicate sink minus sulfide source. "Before" is the age, within the 60 Myr before the belt, at '
         'which the land of the tracked crust stood lowest (its mean elevation); "after" is 60 Myr after the belt age.', '']
    for name, age, *_, label in BELTS:
        ages = P[f'{name}_ages']; A = P[f'{name}_area'].astype(float)
        S = {}
        for ia, t in enumerate(ages):
            s = _stats(P[f'{name}_E'][ia].astype(float), P[f'{name}_q'][ia].astype(float), P[f'{name}_T'][ia].astype(float),
                       P[f'{name}_z'][ia].astype(float), P[f'{name}_land'][ia].astype(bool), A, int(t))
            S[int(t)] = s
            rows.append(dict(belt=name, age=int(t), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items() if k != 'x'}))
        # before: the low stand of the tracked land in the LEAD Myr before the belt age (lowest mean elevation)
        older = [t for t in sorted(S) if age < t <= age + LEAD and S[t]['land_share'] > 0.05]
        t0 = min(older, key=lambda t: S[t]['z_mean']) if older else max(S)
        t2 = age - LEAD if (age - LEAD) in S else None
        i0 = int(np.where(ages == t0)[0][0]); i1 = int(np.where(ages == age)[0][0])
        g = lambda key, i: P[f'{name}_{key}'][i].astype(float)
        E0, q0, T0, z0, l0 = g('E', i0), g('q', i0), g('T', i0), g('z', i0), P[f'{name}_land'][i0].astype(bool)
        E1, q1, T1, z1, l1 = g('E', i1), g('q', i1), g('T', i1), g('z', i1), P[f'{name}_land'][i1].astype(bool)
        both = l0 & l1
        # decomposition on crust that is land at both ages: erosion of the belt age with the climate of the age before, and vice versa
        def net(E, q, T, m):
            import cellfields as cf; w = weather(E, q, T, age if cf.BUDGET == 'full' else None); return GTC * float(np.sum(w['net'] * A * (m & (E > 0)))) / A.sum()   # land plants of the belt age
        n00, n11 = net(E0, q0, T0, both), net(E1, q1, T1, both)
        n10, n01 = net(E1, q0, T0, both), net(E0, q1, T1, both)
        d_ero = 0.5 * ((n10 - n00) + (n11 - n01)); d_clim = 0.5 * ((n01 - n00) + (n11 - n10))
        d_emerge = net(E1, q1, T1, l1 & ~l0) - net(E0, q0, T0, l0 & ~l1)       # crust that rose from (or sank below) sea level
        s0, s1 = S[t0], S[age]
        s2 = S[t2] if t2 is not None else None
        r = dict(belt=name, label=label, age_before=t0, age_belt=age, age_after=t2 if t2 is not None else '', land_share_before=s0['land_share'], land_share_belt=s1['land_share'],
                 z_before=s0['z_mean'], z_belt=s1['z_mean'], E_before=s0['E_mean'], E_belt=s1['E_mean'], q_before=s0['q_mean'], q_belt=s1['q_mean'],
                 x_p50_before=s0['x_p50'], x_p50_belt=s1['x_p50'], beyond_before=s0['beyond'], beyond_belt=s1['beyond'],
                 net_before=s0['net_density'], net_belt=s1['net_density'], sil_before=s0['sil_density'], sil_belt=s1['sil_density'],
                 S_net_before=s0['S_net'], S_net_belt=s1['S_net'],
                 common_land_share=float(A[both].sum() / A.sum()), dnet_common=n11 - n00, dnet_from_erosion=d_ero, dnet_from_climate=d_clim, dnet_from_emergence=d_emerge,
                 x_p50_after=s2['x_p50'] if s2 else '', beyond_after=s2['beyond'] if s2 else '', net_after=s2['net_density'] if s2 else '',
                 S_net_after=s2['S_net'] if s2 else '')
        ba.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        L.append(f"## {label}")
        L.append('')
        L.append(f"Before ({t0} Ma) → belt ({age} Ma). Land share of the tracked crust {s0['land_share']:.2f} → {s1['land_share']:.2f}; mean elevation {s0['z_mean']:.0f} → {s1['z_mean']:.0f} m; "
                 f"mean erosion rate {s0['E_mean']:.3f} → {s1['E_mean']:.3f} mm/yr; runoff {s0['q_mean']:.0f} → {s1['q_mean']:.0f} mm/yr; "
                 f"median E/E_opt {s0['x_p50']:.2f} → {s1['x_p50']:.2f} (quartiles {s0['x_p25']:.2f}–{s0['x_p75']:.2f} → {s1['x_p25']:.2f}–{s1['x_p75']:.2f}); "
                 f"share beyond the optimum {s0['beyond']:.2f} → {s1['beyond']:.2f}; net drawdown {s0['net_density']:.3f} → {s1['net_density']:.3f} t C km⁻² yr⁻¹ "
                 f"(silicate {s0['sil_density']:.3f} → {s1['sil_density']:.3f}); S_net {s0['S_net']:+.2f} → {s1['S_net']:+.2f}.")
        L.append(f"Change in net drawdown per km² of tracked crust {s1['net_density'] - s0['net_density']:+.3f} t C km⁻² yr⁻¹: {d_ero:+.3f} from the change in erosion and "
                 f"{d_clim:+.3f} from the change in runoff and temperature on the crust that is land at both ages ({100 * r['common_land_share']:.0f}% of it), "
                 f"{d_emerge:+.3f} from crust that emerged from (or sank below) sea level.")
        if s2:
            L.append(f"After ({t2} Ma): median E/E_opt {s2['x_p50']:.2f}, share beyond {s2['beyond']:.2f}, net drawdown {s2['net_density']:.3f} t C km⁻² yr⁻¹, S_net {s2['S_net']:+.2f}.")
        L.append('')
    keys = list(rows[0])
    with open(SUMMARY, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(BA_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ba[0])); w.writeheader(); w.writerows(ba)
    BA_MD.write_text('\n'.join(L) + '\n'); print('\n'.join(L)); print('wrote', SUMMARY, BA_CSV, BA_MD)


def figure():
    import pandas as pd
    import _nature_style as ns
    import matplotlib.pyplot as plt
    S = pd.read_csv(SUMMARY); B = pd.read_csv(BA_CSV).set_index('belt')
    fig, axs = plt.subplots(2, 4, figsize=(ns.DOUBLE, 3.3), gridspec_kw=dict(left=0.095, right=0.93, top=0.93, bottom=0.12, wspace=0.62, hspace=0.18))
    for k, (name, age, *_, label) in enumerate(BELTS):
        g = S[S.belt == name].sort_values('age'); t = g.age.values; t0 = int(B.loc[name, 'age_before'])
        ok = g.eroding_share.values > 0.05
        ax = axs[0, k]
        ax.fill_between(t[ok], g.x_p25.values[ok], g.x_p75.values[ok], color=ns.C['adj'], alpha=0.25, lw=0)
        ax.plot(t[ok], g.x_p50.values[ok], color=ns.C['adj'], lw=1.1)
        ax.axhline(1, color='k', lw=0.5); ax.set_yscale('log'); ax.set_ylim(0.05, 40)
        ax.set_xlim(t.max(), t.min()); ax.tick_params(labelbottom=False)
        ax.text(0.5, 1.03, label, transform=ax.transAxes, ha='center', va='bottom', fontsize=ns.SMALL)
        if k == 0:
            ax.set_ylabel('$E/E_{opt}$ of the\ntracked land')
        ax2 = ax.twinx(); ax2.spines['right'].set_visible(True)
        ax2.plot(t, g.land_share.values, color=ns.C['grey'], lw=0.8, ls=(0, (3, 1.5))); ax2.set_ylim(0, 1.05)
        if k == 3:
            ax2.set_ylabel('Share of the\ncrust above sea', color=ns.C['grey'])
        ax2.tick_params(axis='y', colors=ns.C['grey'])
        ax = axs[1, k]
        ax.plot(t, g.net_density.values, color='k', lw=1.1)
        ax.axhline(0, color='k', lw=0.4); ax.set_xlim(t.max(), t.min())
        ax.set_xlabel('Age (Ma)')
        if k == 0:
            ax.set_ylabel('Net drawdown\n(t C km$^{-2}$ yr$^{-1}$)')
        for a in axs[:, k]:
            a.axvline(age, color=ns.C['ref'], lw=0.7); a.axvline(t0, color=ns.C['ref'], lw=0.7, ls=':')
        ns.panel_letter(axs[0, k], 'abcd'[k], x=-0.36, y=0.93)
    fig.savefig(cfg.FIG / 'supp1_orogen_tracks.png', dpi=400); fig.savefig(cfg.FIG / 'supp1_orogen_tracks.pdf')
    print('wrote', cfg.FIG / 'supp1_orogen_tracks.png')


def _oc(E, age, deriv=False):
    """Rock organic-carbon oxidation and biospheric organic-carbon burial per cell, in the units of weather() (main-run
    amplitudes); burial only from 400 Ma, once land plants supply biospheric carbon (v16; before, burial was applied at all ages)."""
    import ast, weathering_core as wc
    amp = ast.literal_eval(open(cfg.OUT / 'kin_bridge_s2_amplitudes.txt').read().split('\n')[1])
    Jp, dJp = wc.oc_petro(E, 'declining'); Jb, dJb = wc.oc_bio_burial(E)
    pl = 1.0 if age <= 400 else 0.0
    out = (amp['ocp'] * Jp / 12.011, pl * amp['ocb'] * Jb / 12.011)
    return out + (amp['ocp'] * dJp / 12.011, pl * amp['ocb'] * dJb / 12.011) if deriv else out


def sensitivity():
    """Event-scale erosion sensitivity of each belt, from the age before uplift to the age of the belt.

    Fluxes are integrated over the tracked crust (cells that are land and erode; submerged crust has no erosion and no
    weathering, so crust that emerges counts with its whole flux). The realised sensitivity of flux X is
    a_X = ln(X_belt / X_before) / ln(erosion flux_belt / erosion flux_before), the slope of a power law through the two
    states, as the exponent a of Bufe et al. (2024). With climate held fixed, the erosion rates of the two ages are evaluated
    under the runoff and temperature of one age and then of the other, and the two log ratios are averaged; crust that is
    sea at the age whose climate is used takes the climate of the other age. The change in net drawdown per km2 of
    tracked crust is split exactly into exposure (erosion and emergence, at fixed climate) and climate (runoff and
    temperature, at fixed erosion), each the mean of both orders."""
    P = np.load(POINTS); BA = {r['belt']: r for r in csv.DictReader(open(BA_CSV))}
    rows = []
    L = ['# Erosion sensitivity of mountain-building events on the main model', '',
         'Tracked crust as in orogen_tracks_before_after' + SUFFIX + '.md. a = ln(flux ratio)/ln(erosion-flux ratio) from the age before uplift '
         'to the age of the belt: "realised" with the simulated climate of both ages, "fixed climate" with the climate of one age '
         'applied to the erosion rates of both (mean of both orders). Net = silicate sink minus sulfide source; net with OC adds '
         'biospheric burial and subtracts rock organic-carbon oxidation. Changes in net drawdown in t C km⁻² yr⁻¹ of tracked crust.', '',
         '| Belt | before → belt (Ma) | erosion flux (Mt/yr) | a_sil realised / fixed | a_sulf realised / fixed | a_ocp realised / fixed | Δnet: exposure + climate = total | Δnet with OC: exposure + climate |',
         '|---|---|---|---|---|---|---|---|']
    for name, age, *_, label in BELTS:
        ages = P[f'{name}_ages']; A = P[f'{name}_area'].astype(float); At = A.sum()
        t0 = int(BA[name]['age_before']); i0 = int(np.where(ages == t0)[0][0]); i1 = int(np.where(ages == age)[0][0])
        g = lambda key, i: P[f'{name}_{key}'][i].astype(float)
        E0, E1 = g('E', i0), g('E', i1)
        q0, q1, T0, T1 = g('q', i0), g('q', i1), g('T', i0), g('T', i1)
        q0, q1 = np.where(np.isfinite(q0), q0, q1), np.where(np.isfinite(q1), q1, q0)
        T0, T1 = np.where(np.isfinite(T0), T0, T1), np.where(np.isfinite(T1), T1, T0)

        def F(E, q, T):
            ero = E > 0; w = weather(E, q, T); ocp, ocb = _oc(E, age)     # the land plants of the belt age at both ages
            I = lambda v: GTC * float(np.sum(np.nan_to_num(v) * A * ero))          # t C yr-1 over the tracked crust
            r = dict(sil=I(w['sil']), sulf=I(w['sil'] - w['net']), net=I(w['net']), ocp=I(ocp), ocb=I(ocb),
                     ero_Mt=2.5e-3 * float(np.sum(E * A * ero)))                 # mm yr-1 km2 x 1e3 m3 x 2.5 t m-3 = Mt yr-1
            r['net_oc'] = r['net'] - r['ocp'] + r['ocb']
            return r
        f00, f11, f10, f01 = F(E0, q0, T0), F(E1, q1, T1), F(E1, q0, T0), F(E0, q1, T1)
        dlnE = np.log(f11['ero_Mt'] / f00['ero_Mt']) if f00['ero_Mt'] > 0 else np.nan
        lr = lambda a, b: np.log(a / b) if a > 0 and b > 0 else np.nan
        r = dict(belt=name, label=label, age_before=t0, age_belt=age, ero_before_Mt=f00['ero_Mt'], ero_belt_Mt=f11['ero_Mt'],
                 land_share_before=float(BA[name]['land_share_before']), land_share_belt=float(BA[name]['land_share_belt']))
        for x in ('sil', 'sulf', 'ocp', 'ocb'):
            r[f'{x}_before'] = f00[x] / At; r[f'{x}_belt'] = f11[x] / At
            r[f'a_{x}'] = lr(f11[x], f00[x]) / dlnE
            r[f'a_{x}_fixed'] = 0.5 * (lr(f10[x], f00[x]) + lr(f11[x], f01[x])) / dlnE
        for x in ('net', 'net_oc'):
            r[f'{x}_before'] = f00[x] / At; r[f'{x}_belt'] = f11[x] / At
            r[f'd{x}_exposure'] = 0.5 * ((f10[x] - f00[x]) + (f11[x] - f01[x])) / At
            r[f'd{x}_climate'] = 0.5 * ((f01[x] - f00[x]) + (f11[x] - f10[x])) / At
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        L.append(f"| {label} | {t0} → {age} | {f00['ero_Mt']:.0f} → {f11['ero_Mt']:.0f} | {r['a_sil']:.2f} / {r['a_sil_fixed']:.2f} | "
                 f"{r['a_sulf']:.2f} / {r['a_sulf_fixed']:.2f} | {r['a_ocp']:.2f} / {r['a_ocp_fixed']:.2f} | "
                 f"{r['dnet_exposure']:+.3f} {r['dnet_climate']:+.3f} = {r['net_belt'] - r['net_before']:+.3f} | "
                 f"{r['dnet_oc_exposure']:+.3f} {r['dnet_oc_climate']:+.3f} |")
    with open(SENS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    SENS_MD.write_text('\n'.join(L) + '\n'); print('\n'.join(L)); print('wrote', SENS_CSV, SENS_MD)


def event_table():
    """The events with a defined event sensitivity, ordered by the age of the belt (oldest first)."""
    import pandas as pd
    D = pd.read_csv(SENS_CSV).sort_values('age_belt', ascending=False).reset_index(drop=True)
    # an event-scale sensitivity needs a change in erosion to divide by: crust whose erosion flux changed by less than
    # MIN_RATIO between the two ages (the Central Asian belt, whose erosion flux fell) is left out
    keep = (D['ero_belt_Mt'] / D['ero_before_Mt']) >= MIN_RATIO
    print('left out (erosion flux changed by less than a factor of %.1f):' % MIN_RATIO, ', '.join(D.loc[~keep, 'label']))
    return D[keep].reset_index(drop=True)


def event_panels(axa, axb, compact=False):
    """Draw the event-scale sensitivities (axa) and the split of the change in net drawdown (axb).
    compact=True is the version in Fig. 4: one-line labels (name and age of the belt, * for crust that was
    largely below sea level before uplift) and short legends."""
    import pandas as pd
    import _nature_style as ns
    from matplotlib.lines import Line2D
    D = event_table()
    G = pd.read_csv(cfg.OUT / 'kin_bridge_s2_ens.csv')
    s_lo, s_hi = float(G['S_sil_adj'].min()), float(G['S_sil_adj'].max())
    x = np.arange(len(D)); ms = 3.4 if compact else 4.0
    ax = axa
    ax.axhspan(s_lo, s_hi, color=ns.C['light'], alpha=0.5, lw=0)
    ax.axhline(1.0, color='k', lw=0.5, ls=(0, (4, 2))); ax.axhline(0.0, color='k', lw=0.4)
    for k, (col, key) in enumerate(((ns.C['sil'], 'sil'), (ns.C['sulf'], 'sulf'), (ns.C['oc'], 'ocp'))):
        dx = (k - 1) * 0.22
        ax.plot(x + dx - 0.05, D[f'a_{key}'], 'o', ms=ms, color=col, mec=col, ls='none')
        ax.plot(x + dx + 0.05, D[f'a_{key}_fixed'], 'o', ms=ms, mfc='white', mec=col, mew=0.9, ls='none')
    ax.set_ylabel('Event erosion\nsensitivity, $a$')
    ax.text(len(D) - 0.45, 0.5 * (s_lo + s_hi), 'global $S_{sil}$' if compact else 'global $S_{sil}$,\n540–0 Ma', ha='right', va='center',
            fontsize=ns.SMALL, color=ns.C['grey'])
    hs = [Line2D([], [], marker='o', ls='none', color=ns.C['sil'], ms=ms, label='Silicate weathering'),
          Line2D([], [], marker='o', ls='none', color=ns.C['sulf'], ms=ms, label='Sulfide oxidation'),
          Line2D([], [], marker='o', ls='none', color=ns.C['oc'], ms=ms, label='Rock organic-carbon oxidation' if not compact else 'Rock OC oxidation'),
          Line2D([], [], marker='o', ls='none', mfc='white', mec='k', ms=ms, label='Climate held fixed')]
    ax.set_ylim(-0.3, 1.62 if compact else 1.42)
    ax.legend(handles=hs, loc='upper left', ncol=2 if compact else 4, fontsize=ns.SMALL, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0))
    ax = axb
    w = 0.36
    ax.bar(x - w / 2, D['dnet_exposure'], w, color=ns.C['adj'], label='Exposure' if compact else 'Exposure (erosion and emergence, climate fixed)')
    ax.bar(x + w / 2, D['dnet_climate'], w, color=ns.C['ref'], label='Climate' if compact else 'Climate (runoff and temperature, erosion fixed)')
    ax.plot(x, D['net_belt'] - D['net_before'], 'D', ms=3.2 if compact else 3.6, color='k', ls='none', label='Total')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Change in net drawdown\n(t C km$^{-2}$ yr$^{-1}$)')
    ax.legend(loc='upper right', fontsize=ns.SMALL, ncol=1, handletextpad=0.3, borderaxespad=0.1)
    short = lambda l: l.split(',')[0].replace(' belt', '').replace(' orogen', '').replace('Himalaya and Tibet', 'Himalaya–Tibet')
    for a in (axa, axb):
        a.set_xticks(x); a.set_xlim(-0.6, len(D) - 0.4)
    if compact:
        lab = [f"{short(l)} {int(a)}{'*' if f < 0.5 else ''}" for l, a, f in zip(D['label'], D['age_belt'], D['land_share_before'])]
    else:
        lab = [f"{short(l)}, {int(b)}–{int(a)} Ma\n{100 * f:.0f}% land before"
               for l, b, a, f in zip(D['label'], D['age_before'], D['age_belt'], D['land_share_before'])]
    for a, show in ((axa, False), (axb, True)) if not compact else ((axa, True), (axb, True)):
        if show:
            a.set_xticklabels(lab, rotation=90, ha='center', va='top', fontsize=ns.SMALL, linespacing=1.0)
        else:
            a.tick_params(labelbottom=False)
    return D


def events_figure():
    """Event-scale erosion sensitivities of the catalogued orogens (a) and the split of their change in net drawdown (b),
    as one stand-alone figure (the same panels are Fig. 4e,f, drawn there by make_fig4_endmembers.py)."""
    import _nature_style as ns
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(ns.DOUBLE, 5.6), sharex=True,
                            gridspec_kw=dict(left=0.1, right=0.985, top=0.975, bottom=0.27, hspace=0.1))
    event_panels(axs[0], axs[1])
    ns.panel_letter(axs[0], 'a', x=-0.09, y=0.95); ns.panel_letter(axs[1], 'b', x=-0.09, y=0.95)
    fig.savefig(cfg.FIG / 'orogen_sensitivity.png', dpi=400); fig.savefig(cfg.FIG / 'orogen_sensitivity.pdf')
    print('wrote', cfg.FIG / 'orogen_sensitivity.png')


if __name__ == '__main__':
    {'select': select, 'track': track, 'evaluate': evaluate, 'analyse': analyse, 'figure': figure,
     'sensitivity': sensitivity, 'events-figure': events_figure}[sys.argv[1] if len(sys.argv) > 1 else 'analyse']()

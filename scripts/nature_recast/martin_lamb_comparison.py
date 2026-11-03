#!/usr/bin/env python3
"""Present-day eroding and depositing land against the global geomorphic process-domain map of Martin & Lamb (2026).

Martin & Lamb (2026, Science Advances 12, eaeb5187) classify Earth's non-Antarctic land at 250 m into eight process
domains: glacial, lacustrine, alluvial river corridor, bedrock river corridor, aeolian, bedrock hillslope (>30°),
soil-mantled hillslope (2.5–30°) and flatlands (≤2.5°). From 50 random flatland points they find 40% of flatlands
depositional (infilled basins, delta and fan plains), 38% tablelands and 18% shields. Their map is a census of what
the surface does over 10²–10⁵ yr at 250 m; our depositing land is the set of 0.1° cells in which the calibrated
steady-state erosion–deposition field is net depositional. This script sets the two side by side on the present-day
surface and measures what the difference does to the weathering results.

    python3 martin_lamb_comparison.py fetch       # once: the domain map (figshare 10.6084/m9.figshare.29902013, 269 MB) to data/martin_lamb/
    python3 martin_lamb_comparison.py aggregate   # 250-m domains -> area of each domain in every 0.1° cell, ONE file:
                                                  #   data/martin_lamb/martin_lamb_domains_0p1deg.npz (reads the tiles from the zip)
    python3 martin_lamb_comparison.py             # comparison: data/nature_recast/martin_lamb_comparison.csv and martin_lamb_summary.md

The comparison uses the land common to both (cells where the PALEOMAP present-day surface is land and at least half
the cell is mapped land in Martin & Lamb). It reports (1) the domain shares and the share of each domain lying in
depositing cells; (2) the mean simulated erosion rate and the share of the erosion flux on each domain (each domain
in a cell takes the cell's rate); (3) the simulated state at Martin & Lamb's evaluation points (data S1, the 50
flatland points with their type; data S2, the 667 error-estimation points with the corrected class); (4) the
present-day silicate sink, its erosion sensitivity S_sil, S_net and the share of eroding land beyond the optimum
when, in addition to the cells the simulation deposits in, weathering is withheld from the alluvial river-corridor and lake
area of every cell, and further from 40% or all of its flatland area. Requires rasterio and pyproj (aggregate only)
and the input cache of the main run (cellfields.py).
"""
import sys, csv, io, zipfile, urllib.request
from pathlib import Path
import numpy as np
import _config as cfg

URL = 'https://ndownloader.figshare.com/files/57221645'
DATA = cfg.REPO / 'data' / 'martin_lamb'
ZIP = DATA / 'mask_geomorph_EE_241118.zip'
AGG = DATA / 'martin_lamb_domains_0p1deg.npz'
SUPP = cfg.REPO.parent / 'Papers' / 'Martin-Lamb-Earth_is_mostly_diffusive_SciAdv2026_S1-S5.zip'
NAMES = ['no data', 'glacial', 'lacustrine', 'alluvial river corridor', 'bedrock river corridor', 'aeolian',
         'bedrock hillslope', 'soil-mantled hillslope', 'flatlands']
PIX_KM2 = 0.0625
NLAT, NLON = 1801, 3601          # node-registered 0.1° grid of the archive (-90..90, -180..180)
FLAT_DEPOSITIONAL = 0.40         # share of flatlands Martin & Lamb class as depositional (20 of their 50 points)


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    if ZIP.exists():
        print('present:', ZIP); return
    tmp = ZIP.with_suffix('.part')
    print('downloading', URL); urllib.request.urlretrieve(URL, tmp); tmp.rename(ZIP); print('wrote', ZIP)


def aggregate():
    import rasterio
    from pyproj import Transformer
    lut = np.full(256, -1, np.int64)                       # pixel value -> class index (ocean -1, skipped)
    lut[10] = 0; lut[1:9] = np.arange(1, 9); lut[12] = 2
    counts = np.zeros(NLAT * 3600 * 9, np.int64)
    inv = Transformer.from_crs(8857, 4326, always_xy=True)
    with zipfile.ZipFile(ZIP) as z:
        tiles = sorted(n for n in z.namelist() if n.endswith('.tif'))
    for k, name in enumerate(tiles):
        with rasterio.open(f'/vsizip/{ZIP}/{name}') as ds:
            T = ds.transform
            for _, win in ds.block_windows(1):
                v = ds.read(1, window=win)
                cls = lut[v]
                r, c = np.nonzero(cls >= 0)
                if r.size == 0:
                    continue
                y = T.f + T.e * (win.row_off + np.arange(win.height) + 0.5)
                lon1, lat = inv.transform(np.full(y.size, 1.0e5), y)          # Equal Earth: latitude depends on y only,
                kx = np.asarray(lon1) / 1.0e5                                  # longitude is linear in x along a row
                x = T.c + T.a * (win.col_off + c + 0.5)
                lon = x * kx[r]; la = np.asarray(lat)[r]
                ok = np.abs(lon) <= 180.0
                il = np.clip(np.rint((la[ok] + 90.0) / 0.1).astype(np.int64), 0, NLAT - 1)
                jl = np.rint((lon[ok] + 180.0) / 0.1).astype(np.int64) % 3600
                key = (il * 3600 + jl) * 9 + cls[r[ok], c[ok]]
                u, n = np.unique(key, return_counts=True)
                counts[u] += n
        print(f'{k + 1}/{len(tiles)} {name}', flush=True)
    a = counts.reshape(NLAT, 3600, 9)
    a = np.concatenate([a, a[:, :1]], axis=1)              # column 3600 (180°) repeats column 0
    km2 = (a * PIX_KM2).astype(np.float32)
    np.savez_compressed(AGG, area_km2=km2, names=np.array(NAMES), lat=np.linspace(-90, 90, NLAT), lon=np.linspace(-180, 180, NLON),
                        source='Martin & Lamb (2026) mask_geomorph_EE_241118, figshare 10.6084/m9.figshare.29902013 (CC BY 4.0)')
    tot = km2[..., 1:].sum()
    print('wrote', AGG, f'; mapped land {tot / 1e6:.1f} M km2;',
          ', '.join(f'{NAMES[i]} {100 * km2[..., i].sum() / tot:.1f}%' for i in range(1, 9)))


def _points():
    """Martin & Lamb data S1 (flatland points with type) and S2 (error-estimation points with corrected class)."""
    out = {}
    with zipfile.ZipFile(SUPP) as z:
        for key, member in (('S1', 'aeb5187_data_s1.csv'), ('S2', 'aeb5187_data_s2.csv')):
            txt = z.read(member).decode('latin-1')
            out[key] = list(csv.reader(io.StringIO(txt)))
    deg = lambda s: float(s.replace('\xb0', '').replace('�', '').strip())
    s1 = [dict(kind=r[1].strip(), lat=deg(r[2]), lon=deg(r[3])) for r in out['S1'][1:] if len(r) > 3 and r[2].strip()]
    s2 = []
    for r in out['S2'][1:]:
        if len(r) < 5 or not r[2].strip():
            continue
        pred = int(float(r[4])); true = int(float(r[5])) if r[5].strip() else pred
        s2.append(dict(lat=float(r[2]), lon=float(r[3]), pred=pred, true=2 if true == 12 else true))
    return s1, s2


def compare():
    import cellfields
    g, _ = cellfields.compute(0)
    with np.load(AGG) as d:
        dom = d['area_km2'].astype(float)
    mapped = dom[..., 1:].sum(-1)
    land = np.isfinite(g['E']) & g['mask'].astype(bool)
    A = g['area'] / 1e6                                     # km2
    common = land & (mapped >= 0.5 * A)
    E = np.nan_to_num(g['E'])
    eroding = common & (E > 0); depo = common & ~(E > 0)
    f = np.where(common[..., None], dom[..., 1:] / np.maximum(mapped, 1e-9)[..., None], 0.0)   # domain fractions of mapped land
    Ac = np.where(common, A, 0.0)
    rows = []; L = ['# Simulated present-day erosion and deposition against the Martin & Lamb (2026) process-domain map', '']
    Atot = Ac.sum()
    L.append(f'Common land: {Atot / 1e6:.1f} M km² ({int(common.sum())} cells of 0.1°; PALEOMAP present-day land with at least half of the cell '
             f'mapped by Martin & Lamb). Net-depositing cells: {100 * Ac[depo].sum() / Atot:.1f}% of it.')
    L.append('')
    L.append('| domain | share of land (%) | share in depositing cells (%) | mean simulated erosion rate (mm/yr) | share of erosion flux (%) |')
    L.append('|---|---|---|---|---|')
    flux = np.sum(Ac * E)
    for i, nm in enumerate(NAMES[1:]):
        ai = Ac * f[..., i]
        sh = ai.sum() / Atot; dep = ai[depo].sum() / max(ai.sum(), 1e-9)
        em = np.sum(ai * E) / max(ai.sum(), 1e-9); fx = np.sum(ai * E) / flux
        rows.append(dict(item='domain', name=nm, share_land=round(sh, 4), share_in_depositing=round(dep, 4), mean_E_mm=round(em, 4), share_erosion_flux=round(fx, 4)))
        L.append(f'| {nm} | {100 * sh:.1f} | {100 * dep:.1f} | {em:.3f} | {100 * fx:.1f} |')
    i_al, i_lac, i_flat = 2, 1, 7
    nonero_ml = (f[..., i_al] + f[..., i_lac] + FLAT_DEPOSITIONAL * f[..., i_flat])
    L.append('')
    L.append(f'Martin & Lamb non-erosional land (alluvial river corridors + lakes + {FLAT_DEPOSITIONAL:.0%} of flatlands): '
             f'{100 * np.sum(Ac * nonero_ml) / Atot:.1f}% of the common land; in the simulation\'s depositing cells this fraction averages '
             f'{100 * np.sum(Ac[depo] * nonero_ml[depo]) / max(Ac[depo].sum(), 1e-9):.1f}%, in its eroding cells '
             f'{100 * np.sum(Ac[eroding] * nonero_ml[eroding]) / Ac[eroding].sum():.1f}%; flatland fraction '
             f'{100 * np.sum(Ac[depo] * f[..., i_flat][depo]) / max(Ac[depo].sum(), 1e-9):.1f}% against {100 * np.sum(Ac[eroding] * f[..., i_flat][eroding]) / Ac[eroding].sum():.1f}%.')
    # evaluation points
    s1, s2 = _points()
    lat, lon = g['lat'], g['lon']
    def at(p):
        i = int(np.argmin(np.abs(lat - p['lat']))); j = int(np.argmin(np.abs(lon - p['lon'])))
        return (bool(land[i, j]), float(E[i, j]) if land[i, j] else np.nan)
    L.append(''); L.append('## Martin & Lamb evaluation points (simulation at the containing 0.1° cell)'); L.append('')
    L.append('| set | class | n on simulated land | share depositing (%) | median simulated erosion rate (mm/yr) |'); L.append('|---|---|---|---|---|')
    for kind in sorted({p['kind'] for p in s1}):
        v = [at(p) for p in s1 if p['kind'] == kind]; v = [e for ok, e in v if ok]
        if v:
            L.append(f'| S1 flatlands | {kind} | {len(v)} | {100 * np.mean(np.array(v) <= 0):.0f} | {np.median(v):.3f} |')
            rows.append(dict(item='S1', name=kind, n=len(v), share_depositing=round(float(np.mean(np.array(v) <= 0)), 3), median_E_mm=round(float(np.median(v)), 4)))
    code = {1: 'glacial', 2: 'lacustrine', 3: 'alluvial river corridor', 4: 'bedrock river corridor', 5: 'aeolian', 6: 'bedrock hillslope',
            7: 'soil-mantled hillslope', 8: 'flatlands', 12: 'lacustrine'}
    for c in (3, 2, 8, 5, 7, 6, 4, 1):
        v = [at(p) for p in s2 if p['true'] == c]; v = [e for ok, e in v if ok]
        if v:
            L.append(f'| S2 corrected class | {code[c]} | {len(v)} | {100 * np.mean(np.array(v) <= 0):.0f} | {np.median(v):.3f} |')
            rows.append(dict(item='S2', name=code[c], n=len(v), share_depositing=round(float(np.mean(np.array(v) <= 0)), 3), median_E_mm=round(float(np.median(v)), 4)))
    # weathering with additional land withheld
    sil, dsil, net, dnet = (np.nan_to_num(g[k]) for k in ('sil', 'dsil', 'neti', 'dneti'))   # inorganic budget, as when this comparison was made
    Aall = np.where(land, g['area'], 0.0)
    L.append(''); L.append('## Present-day weathering with additional land withheld'); L.append('')
    L.append('Weathering is withheld, in every cell, from the area fraction named, in addition to the cells the simulation deposits in '
             '(outside the common land nothing changes). Fluxes before recalibration, relative to the main model.')
    L.append(''); L.append('| withheld | land withheld (%) | silicate sink (rel.) | S_sil | S_net | share of eroding land beyond the optimum |'); L.append('|---|---|---|---|---|---|')
    eroall = land & (E > 0); beyond = eroall & (dnet <= 0)
    base = None
    for label, w in (('net-depositing cells only (main model)', 0.0),
                     ('+ alluvial river corridors and lakes', f[..., i_al] + f[..., i_lac]),
                     (f'+ {FLAT_DEPOSITIONAL:.0%} of flatlands', f[..., i_al] + f[..., i_lac] + FLAT_DEPOSITIONAL * f[..., i_flat]),
                     ('+ all flatlands', f[..., i_al] + f[..., i_lac] + f[..., i_flat])):
        W = Aall * (1.0 - np.where(common, w, 0.0))
        Fs = np.sum(W * sil); S_s = np.sum(W * dsil) / Fs; S_n = np.sum(W * dnet) / np.sum(W * net)
        sh = np.sum(W[beyond]) / np.sum(W[eroall])
        wd = (np.sum(Aall[land & ~(E > 0)]) + np.sum((Aall - W)[eroall])) / np.sum(Aall[land])    # share of all land without weathering
        base = base or Fs
        L.append(f'| {label} | {100 * wd:.1f} | {Fs / base:.3f} | {S_s:.3f} | {S_n:+.3f} | {sh:.3f} |')
        rows.append(dict(item='withheld', name=label, land_withheld=round(wd, 4), F_sil_rel=round(Fs / base, 4), S_sil=round(S_s, 4), S_net=round(S_n, 4), beyond=round(sh, 4)))
    out_csv = cfg.OUT / 'martin_lamb_comparison.csv'
    keys = ['item', 'name'] + sorted({k for r in rows for k in r} - {'item', 'name'})
    with open(out_csv, 'w', newline='') as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys); wtr.writeheader(); wtr.writerows(rows)
    out_md = cfg.OUT / 'martin_lamb_summary.md'; out_md.write_text('\n'.join(L) + '\n')
    print('\n'.join(L)); print('wrote', out_csv, out_md)


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    {'fetch': fetch, 'aggregate': aggregate, 'compare': compare}[step]()

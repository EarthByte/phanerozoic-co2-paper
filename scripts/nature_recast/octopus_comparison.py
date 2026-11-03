#!/usr/bin/env python3
"""The present-day simulated erosion field against the OCTOPUS cosmogenic compilation, basin by basin.

Answers the question whether the global erosion factor (2.0) hides a spatial pattern: at every basin of the
OCTOPUS v2 catchment-averaged ¹⁰Be denudation collections (Codilean et al. 2022; the global and Australian
collections with their CAIRN-recalculated rates, and the 2024 large-basin collection with its published rates;
4,500 basins; the compilation of Harel et al. 2016 is contained in them), the
scaled simulated erosion rate of the present day is averaged over the basin polygon (area-weighted over the
0.1° cells whose centres fall inside; a basin smaller than a cell takes the cell of its outlet) and compared
with the measured rate. The ratio is reported by measured rate, by simulated rate, by basin area, by region
and by mean slope, with the spread of the two distributions and a total-least-squares slope in log space.

    python3 octopus_comparison.py fetch        download the three basin shapefiles (95 MB) into data/octopus/
    python3 octopus_comparison.py              compare; writes ONE table, one summary, one figure

Writes data/nature_recast/octopus_comparison.csv (one row per basin), data/nature_recast/octopus_summary.md
(every number the text quotes) and figures/nature/ed10_octopus.png/.pdf. Needs geopandas and shapely 2.
"""
import sys, subprocess
from pathlib import Path
import numpy as np
import _config as cfg

DATA = cfg.REPO / 'data' / 'octopus'
SOURCES = {   # Zenodo records of the OCTOPUS v2 CRN denudation collections (basin outlines with rates)
    'INT': ('https://zenodo.org/api/records/13270793/files/crn_INTv2_shapes.zip/content', 'crn_INTv2_shapes.zip', 'crn_int_basins.shp'),
    'AUS': ('https://zenodo.org/api/records/13272695/files/crn_AUSv2_shapes.zip/content', 'crn_AUSv2_shapes.zip', 'crn_aus_basins.shp'),
    'XXL': ('https://zenodo.org/api/records/13369360/files/crn_xxl_basins.zip/content', 'crn_xxl_basins.zip', 'crn_xxl_basins.shp'),
}
SCALE = 2.0
BINS_E = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 100]


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for key, (url, zname, _) in SOURCES.items():
        z = DATA / zname
        if not z.exists():
            print('downloading', zname); subprocess.run(['curl', '-sSL', '-o', str(z), url], check=True)
        subprocess.run(['unzip', '-o', '-q', str(z), '-d', str(DATA / key)], check=True)
    print('shapefiles in', DATA)


def shapefile(key):
    hits = [p for p in (DATA / key).rglob(SOURCES[key][2]) if '__MACOSX' not in str(p)]
    if not hits:
        raise SystemExit(f'{SOURCES[key][2]} not found under {DATA / key}: run "octopus_comparison.py fetch"')
    return hits[0]


def present_day_field():
    with np.load(cfg.ERODEP_DIR / 'erodep_0Ma.npz', allow_pickle=True) as d:
        E = np.maximum(-d['ero_zhou'].astype(np.float32), 0.0) * SCALE     # mm/yr, erosion positive, as the main model
        land = d['mask_zhou'].astype(bool); lat = d['lat'].astype(float); lon = d['lon'].astype(float)
    dg = np.deg2rad(0.1); A = ((6.371e6 ** 2 * dg * dg) * np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
    return E, land, A.astype(np.float32), lat, lon


def _deg(slp_m_per_km):
    """OCTOPUS SLP_AVE is the mean basin slope in m/km (-9999.99 where not computed); convert to degrees."""
    return float(np.degrees(np.arctan(slp_m_per_km / 1000.0))) if slp_m_per_km > -9000 else np.nan


def compare():
    import geopandas as gpd, pandas as pd, shapely
    E, land, A, lat, lon = present_day_field()
    frames = []
    for key in SOURCES:
        g = gpd.read_file(shapefile(key)).to_crs(4326); g['collection'] = key; frames.append(g)
    g = pd.concat(frames, ignore_index=True)
    # the global and Australian collections carry CAIRN-recalculated rates (EBE_MMKYR); the large-basin collection
    # carries the published rates only (BE10EP, mm/kyr), which are used where no recalculated rate exists
    g['rate_mmkyr'] = np.where(g.EBE_MMKYR > 0, g.EBE_MMKYR, np.where(g.BE10EP > 0, g.BE10EP, np.nan))
    g['rate_err_mmkyr'] = np.where(g.EBE_MMKYR > 0, g.EBE_ERR, np.where(g.BE10EP > 0, g.BE10EP_ERR, np.nan))
    g['rate_source'] = np.where(g.EBE_MMKYR > 0, 'recalculated', 'published')
    g = g[g.rate_mmkyr > 0].copy()
    rows = []
    for r in g.itertuples():
        geom = r.geometry; minx, miny, maxx, maxy = geom.bounds
        i0, i1 = np.searchsorted(lat, [miny - 0.1, maxy + 0.1]); j0, j1 = np.searchsorted(lon, [minx - 0.1, maxx + 0.1])
        LA, LO = np.meshgrid(lat[i0:i1], lon[j0:j1], indexing='ij')
        inside = shapely.contains_xy(geom, LO.ravel(), LA.ravel()).reshape(LA.shape)
        sE, sA, sL = E[i0:i1, j0:j1], A[i0:i1, j0:j1], land[i0:i1, j0:j1]
        sel = inside & sL & np.isfinite(sE)
        if sel.sum():
            e_sim = float(np.average(sE[sel], weights=sA[sel])); n = int(sel.sum())
        else:
            i = int(np.argmin(np.abs(lat - r.Y_WGS84))); j = int(np.argmin(np.abs(lon - r.X_WGS84)))
            if not (land[i, j] and np.isfinite(E[i, j])):
                continue
            e_sim = float(E[i, j]); n = 0
        rows.append(dict(collection=r.collection, obsid=r.OBSID1, study=r.STUDYID, region=r.REGION_INT, country=r.CNTRY, lat=r.Y_WGS84, lon=r.X_WGS84,
                         area_km2=r.AREA, slope_deg=_deg(r.SLP_AVE), elev_m=(r.ELEV_AVE if r.ELEV_AVE > -9000 else np.nan), e_obs=r.rate_mmkyr / 1000.0, e_obs_err=r.rate_err_mmkyr / 1000.0, rate_source=r.rate_source,
                         e_sim=e_sim, n_cells=n))
    df = pd.DataFrame(rows)
    df['log_ratio'] = np.log10(df.e_sim.where(df.e_sim > 0) / df.e_obs)
    out = cfg.OUT / 'octopus_comparison.csv'; df.to_csv(out, index=False); print('wrote', out, len(df), 'basins')
    return df


def summary(df):
    import pandas as pd
    d = df[df.e_sim > 0].copy()
    L = ['# Simulated present-day erosion rates against OCTOPUS cosmogenic-nuclide denudation rates', '',
         f'{len(df)} basins with ¹⁰Be rates ({", ".join(f"{k} {n}" for k, n in df.collection.value_counts().items())}); '
         f'{int((df.n_cells == 0).sum())} smaller than a 0.1° cell (outlet cell used); {len(d)} where the simulation erodes.', '']

    def block(name, x):
        lx, ly = np.log10(x.e_obs.values), np.log10(x.e_sim.values)
        cov = np.cov(lx, ly); sxx, syy, sxy = cov[0, 0], cov[1, 1], cov[0, 1]
        tls = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
        return (f'- {name}: n {len(x)}; median rate measured {x.e_obs.median():.3f} mm/yr, simulated {x.e_sim.median():.3f}; '
                f'median ratio simulated/measured {10 ** x.log_ratio.median():.2f} (16–84%: {10 ** x.log_ratio.quantile(.16):.2f}–{10 ** x.log_ratio.quantile(.84):.2f}); '
                f's.d. of log10 rate measured {np.sqrt(sxx):.2f}, simulated {np.sqrt(syy):.2f} (ratio {np.sqrt(syy / sxx):.2f}); '
                f'r of logs {sxy / np.sqrt(sxx * syy):.2f}; total-least-squares slope of log simulated on log measured {tls:.2f}; '
                f'share above 1 mm/yr measured {(x.e_obs > 1).mean():.3f}, simulated {(x.e_sim > 1).mean():.3f}; '
                f'share below 0.02 mm/yr measured {(x.e_obs < 0.02).mean():.3f}, simulated {(x.e_sim < 0.02).mean():.3f}')
    L += ['## All basins, and basins at least one cell in size', '', block('all basins', d), block('basins ≥ 100 km²', d[d.area_km2 >= 100]), block('basins ≥ 1,000 km²', d[d.area_km2 >= 1000]), '']
    q = [5, 25, 50, 75, 95]
    L += ['## Quantiles of the two distributions over the same basins (mm/yr)', '',
          '| quantile | measured | simulated |', '|---|---|---|'] + [f'| {p}% | {np.percentile(d.e_obs, p):.3f} | {np.percentile(d.e_sim, p):.3f} |' for p in q] + ['']

    def table(title, groups):
        L.extend(['## ' + title, '', '| bin | n | measured median | simulated median | ratio median | 16% | 84% |', '|---|---|---|---|---|---|---|'])
        for key, x in groups:
            if len(x) < 5:
                continue
            L.append(f'| {key} | {len(x)} | {x.e_obs.median():.3f} | {x.e_sim.median():.3f} | {10 ** x.log_ratio.median():.2f} | {10 ** x.log_ratio.quantile(.16):.2f} | {10 ** x.log_ratio.quantile(.84):.2f} |')
        L.append('')
    table('By measured rate (mm/yr)', d.groupby(pd.cut(d.e_obs, BINS_E), observed=True))
    table('By simulated rate (mm/yr)', d.groupby(pd.cut(d.e_sim, BINS_E), observed=True))
    table('By basin area (km²)', d.groupby(pd.cut(d.area_km2, [0, 10, 100, 1000, 10000, 1e8]), observed=True))
    table('By mean basin slope (degrees; basins with a slope in OCTOPUS)', d.dropna(subset=['slope_deg']).groupby(pd.cut(d.dropna(subset=['slope_deg']).slope_deg, [0, 5, 10, 15, 20, 25, 30, 90]), observed=True))
    reg = [(k, x) for k, x in d.groupby('region') if len(x) >= 40]
    table('By region (regions with at least 40 basins)', sorted(reg, key=lambda kv: -len(kv[1])))
    out = cfg.OUT / 'octopus_summary.md'; out.write_text('\n'.join(L)); print('\n'.join(L)); print('wrote', out)


def figure(df):
    import pandas as pd
    import _nature_style as ns
    import matplotlib.pyplot as plt
    d = df[df.e_sim > 0]
    fig, axs = plt.subplots(1, 3, figsize=(ns.DOUBLE, 2.25), gridspec_kw=dict(left=0.075, right=0.985, top=0.9, bottom=0.2, wspace=0.42))     # v16: height 2.5 -> 2.25 in, empty strip below the panels removed
    ax = axs[0]
    ax.scatter(d.e_obs, d.e_sim, s=2, c=ns.C['grey'], alpha=0.35, lw=0, rasterized=True)
    g = d.groupby(pd.cut(d.e_obs, np.logspace(-3, 1.3, 24), include_lowest=True), observed=True)
    mid = np.array([float(iv.mid) for iv in g.e_obs.median().index]); med = g.e_sim.median().values; n = g.size().values
    ok = n >= 10
    ax.plot(g.e_obs.median().values[ok], med[ok], color=ns.C['adj'], lw=1.2, label='median of simulated')
    ax.plot([1e-3, 20], [1e-3, 20], color='k', lw=0.6, ls=':', label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-3, 20); ax.set_ylim(1e-3, 20)
    ax.set_xlabel('Measured erosion rate (mm yr⁻¹)'); ax.set_ylabel('Simulated rate (mm yr⁻¹)')
    ax.legend(loc='upper left', handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    ns.panel_letter(ax, 'a', x=-0.27, y=1.04)
    ax = axs[1]
    g = d.groupby(pd.cut(d.e_obs, BINS_E), observed=True)
    edges = np.array(BINS_E, float); edges[0] = 0.003; edges[-1] = 20.0
    x = np.sqrt(edges[:-1] * edges[1:])[:len(g)]
    r50 = 10 ** g.log_ratio.median().values; r16 = 10 ** g.log_ratio.quantile(.16).values; r84 = 10 ** g.log_ratio.quantile(.84).values
    ax.fill_between(x, r16, r84, color=ns.C['adj'], alpha=0.25, lw=0); ax.plot(x, r50, color=ns.C['adj'], lw=1.2, marker='o', ms=3)
    ax.axhline(1, color='k', lw=0.6, ls=':'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.003, 20)
    ax.set_xlabel('Measured erosion rate (mm yr⁻¹)'); ax.set_ylabel('Simulated / measured'); ax.set_ylim(0.05, 150)
    for xi, ni in zip(x, g.size().values):
        ax.text(xi, 110, f'{ni}', ha='center', va='top', fontsize=ns.SMALL, color=ns.C['grey'])
    ns.panel_letter(ax, 'b', x=-0.27, y=1.04)
    ax = axs[2]
    q = np.linspace(1, 99, 99)
    ax.plot(q, np.percentile(d.e_obs, q), color='k', lw=1.2, label='measured')
    ax.plot(q, np.percentile(d.e_sim, q), color=ns.C['adj'], lw=1.2, label='simulated')
    ax.set_yscale('log'); ax.set_xlabel('Percentile of basins'); ax.set_ylabel('Erosion rate (mm yr⁻¹)'); ax.set_ylim(1e-3, 20)
    ax.legend(loc='upper left', handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)
    ns.panel_letter(ax, 'c', x=-0.27, y=1.04)
    ns.equal_gaps(fig, list(axs))
    fig.savefig(cfg.FIG / 'ed10_octopus.png', dpi=400); fig.savefig(cfg.FIG / 'ed10_octopus.pdf')
    print('wrote', cfg.FIG / 'ed10_octopus.png')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fetch':
        fetch()
    elif len(sys.argv) > 1 and sys.argv[1] == 'figure':
        # redraw Extended Data Fig. 10 from the per-basin table written by the full run (no geopandas needed)
        import pandas as pd
        figure(pd.read_csv(cfg.OUT / 'octopus_comparison.csv'))
    else:
        df = compare(); summary(df); figure(df)

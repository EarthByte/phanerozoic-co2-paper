#!/usr/bin/env python3
"""Catalogue of Phanerozoic mountain-building events for the event-scale erosion sensitivity.

Each event is a named orogen with a present-day reference point on crust that took part in it and an age range from
the literature. The reference point is rotated with the PALEOMAP plate model to every age in that range; the age at
which the most eroding land above 1,500 m stands within RADIUS degrees of the reconstructed point on the main-model
surface is the age of the belt, and the cap of that radius about the reconstructed point at that age holds the belt
crust that orogen_tracks.py then follows through time (OROGEN_SET=events). The four belts of Fig. 4 keep the windows
and ages they have there, so that the numbers quoted for them agree. Events with less than MIN_AREA of such land at
every age of their range (orogens that the reconstruction does not raise above 1,500 m) or without an earlier
reconstruction to compare with are listed but not used.

    python3 orogen_catalogue.py scan     # 1-degree summary of land above 1,500 m at all 109 ages (needs the input cache) -> orogen_scan.npz
    python3 orogen_catalogue.py locate   # age and window of every event (needs pygplates and orogen_scan.npz)        -> orogen_catalogue.csv
Each step writes one file in data/nature_recast/. The plate model is read as in orogen_tracks.py.
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import _config as cfg

SCAN = cfg.OUT / 'orogen_scan.npz'
CATALOGUE = cfg.OUT / 'orogen_catalogue.csv'
ZMIN = 1500.0
SCALE = 2.0
RADIUS = 10.0          # degrees of arc about the reconstructed reference point
MIN_AREA = 0.3         # M km2 of eroding land above ZMIN within RADIUS, at the age of the belt
PLATE_DIR = Path(os.environ.get('PALEOMAP_PLATE_DIR', cfg.REPO.parent.parent / 'Paleotopo_data_assimilation' / 'data' / 'Scotese_Wright_plate_model_revised'))

# key, label, present-day reference point (lat, lon), age range searched (Ma, old, young)
EVENTS = [('pan_african', 'East African (Pan-African)', -7.0, 36.0, 540, 500),
          ('taconic', 'Taconic', 44.0, -73.5, 480, 440),
          ('lachlan', 'Lachlan', -35.0, 148.0, 450, 380),
          ('acadian', 'Acadian', 46.0, -68.0, 410, 370),
          ('variscan', 'Variscan', 50.0, 13.0, 345, 295),
          ('alleghanian', 'Alleghanian', 37.0, -81.0, 325, 280),
          ('gondwanide', 'Gondwanide', -33.5, 20.5, 285, 230),
          ('uralian', 'Uralian', 60.0, 59.0, 300, 250),
          ('caob', 'Central Asian', 43.0, 85.0, 320, 250),
          ('indosinian', 'Indosinian', 33.5, 108.0, 250, 195),
          ('mongol_okhotsk', 'Mongol-Okhotsk', 52.0, 118.0, 185, 140),
          ('sevier', 'Sevier', 40.0, -114.0, 125, 70),
          ('alpine', 'Alpine-Pyrenean', 46.5, 10.0, 40, 0)]
# the four belts of Fig. 4, with the windows and ages of orogen_tracks.BELTS
PAPER = ['caledonian', 'central_pangaean', 'himalaya_tibet', 'central_andes']


def scan():
    import _cache
    ages = sorted(_cache.ages_present())
    HI = np.zeros((len(ages), 180, 360), np.float32); LAND = np.zeros_like(HI); ZMAX = np.full_like(HI, -1e4)
    for ia, t in enumerate(ages):
        d = _cache.load(t)
        nlat, nlon = (int(v) for v in d['shape'])
        idx = d['idx'].astype(np.int64); i = idx // nlon; j = idx % nlon
        la = d['lat'].astype(float)[i]; lo = d['lon'].astype(float)[j]
        I = np.clip(np.floor(la + 90.0).astype(int), 0, 179); J = np.floor((lo + 180.0) % 360.0).astype(int) % 360
        keep = lo < 180.0                                   # the dateline column is repeated in the 0.1-degree grid
        E = np.where(d['mz'], np.maximum(-d['ero_z'].astype(float), 0.0), 0.0) * SCALE
        A = d['area'].astype(float) / 1e6
        land = d['mz'] & keep; hi = land & (E > 0) & (d['z_z'] > ZMIN)
        np.add.at(LAND[ia], (I[land], J[land]), A[land]); np.add.at(HI[ia], (I[hi], J[hi]), A[hi])
        np.maximum.at(ZMAX[ia], (I[land], J[land]), d['z_z'][land].astype(np.float32))
        print(t, 'Ma', f'{HI[ia].sum() / 1e6:.2f} M km2 above {ZMIN:.0f} m', flush=True)
    tmp = SCAN.with_name(SCAN.stem + '.partial.npz')
    np.savez_compressed(tmp, ages=np.array(ages), hi_km2=HI, land_km2=LAND, zmax=ZMAX)
    os.replace(tmp, SCAN); print('wrote', SCAN)


def _arc(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))


def locate():
    import pygplates
    S = np.load(SCAN); ages = S['ages']; HI = S['hi_km2']
    rot = pygplates.RotationModel(str(PLATE_DIR / 'Scotese_Wright_PlateModel_2023.rot'))
    part = pygplates.PlatePartitioner(pygplates.FeatureCollection(str(PLATE_DIR / 'Scotese_Wright_PlatePolygons_2023.gpml')), rot, reconstruction_time=0.0)
    LO, LA = np.meshgrid(np.arange(360) - 179.5, np.arange(180) - 89.5)
    rows = []
    for key, label, la, lo, a_old, a_young in EVENTS:
        pp = part.partition_point(pygplates.PointOnSphere(la, lo))
        pid = pp.get_feature().get_reconstruction_plate_id() if pp else 0
        cand = []
        for t in ages[(ages <= a_old) & (ages >= a_young)]:
            p = (rot.get_rotation(float(t), int(pid)) * pygplates.PointOnSphere(la, lo)).to_lat_lon()
            h = float(HI[int(np.where(ages == t)[0][0])][_arc(LA, LO, p[0], p[1]) <= RADIUS].sum() / 1e6)
            cand.append((h, int(t), p[0], p[1]))
        h, t, plat, plon = max(cand)
        if h < MIN_AREA:
            use, note = 0, f'less than {MIN_AREA} M km2 of eroding land above {ZMIN:.0f} m at every age of its range'
        elif t >= int(ages.max()):
            use, note = 0, 'highest at the oldest reconstruction: no earlier surface to compare with'
        else:
            use, note = 1, ''
        rows.append(dict(key=key, label=label, ref_lat=la, ref_lon=lo, plate=pid, age_old=a_old, age_young=a_young, age=t,
                         lat=round(plat, 2), lon=round(plon, 2), radius=RADIUS, area_hi_Mkm2=round(h, 3), used=use, note=note))
        print(f'{key:15s} {t:3d} Ma ({plat:6.1f}, {plon:7.1f}) {h:5.2f} M km2  {"used" if use else "not used: " + note}', flush=True)
    for key in PAPER:
        rows.append(dict(key=key, label='', ref_lat='', ref_lon='', plate='', age_old='', age_young='', age='', lat='', lon='', radius='',
                         area_hi_Mkm2='', used=1, note='window and age of Fig. 4 (orogen_tracks.BELTS)'))
    with open(CATALOGUE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote', CATALOGUE)


if __name__ == '__main__':
    {'scan': scan, 'locate': locate}[sys.argv[1]]()

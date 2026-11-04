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

Role in the paper: the catalogue of orogens behind the event-scale erosion sensitivities of Fig. 4e,f (Methods and
Supplementary Information, "Mountain belts followed through time"). Reference points and literature age ranges are
those of EVENTS.

Inputs
    nature_recast_cache.zip (build_cache.py): land-only fields of the main model at every age (scan)
    Scotese_Wright_PlateModel_2023.rot, Scotese_Wright_PlatePolygons_2023.gpml: PALEOMAP rotations and plate polygons (locate)

Outputs (data/nature_recast/)
    orogen_scan.npz       ages (Ma); hi_km2, land_km2 (ages x 180 x 360): area (km2) of eroding land above ZMIN and of
                          all land of the adjusted surface in each 1-degree bin; zmax (m): highest land elevation per bin
                          (-1e4 where the bin holds no land). Bin (i, j) is centred on lat i - 89.5, lon j - 179.5.
    orogen_catalogue.csv  one row per event: key, label, ref_lat/ref_lon (present-day reference point), plate (PALEOMAP
                          plate ID), age_old/age_young (range searched, Ma), age (age of the belt, Ma), lat/lon
                          (reconstructed reference point at that age), radius (degrees of arc), area_hi_Mkm2 (eroding
                          land above ZMIN in the cap, million km2), used (1/0), note; then one row for each of the four
                          belts of Fig. 4, whose windows and ages come from orogen_tracks.BELTS

    python3 orogen_catalogue.py scan     # 1-degree summary of land above 1,500 m at all 109 ages (needs the input cache) -> orogen_scan.npz
    python3 orogen_catalogue.py locate   # age and window of every event (needs pygplates and orogen_scan.npz)        -> orogen_catalogue.csv
Each step writes one file in data/nature_recast/. The plate model is read as in orogen_tracks.py. The catalogue is then
followed through time with OROGEN_SET=events python3 orogen_tracks.py select|track|evaluate|analyse|sensitivity.
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import _config as cfg

SCAN = cfg.OUT / 'orogen_scan.npz'
CATALOGUE = cfg.OUT / 'orogen_catalogue.csv'
ZMIN = 1500.0          # m; elevation above which eroding land counts as mountain-belt crust (as orogen_tracks.py)
SCALE = 2.0            # erosion calibration factor of the main model (as orogen_tracks.py)
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
    """Bin the eroding land above ZMIN, all land and the highest land elevation of the adjusted surface into 1-degree
    cells at every age of the cache, so that locate can search many ages without the cache. Writes SCAN."""
    import _cache
    ages = sorted(_cache.ages_present())
    HI = np.zeros((len(ages), 180, 360), np.float32); LAND = np.zeros_like(HI); ZMAX = np.full_like(HI, -1e4)
    for ia, t in enumerate(ages):
        d = _cache.load(t)
        nlat, nlon = (int(v) for v in d['shape'])
        # cell-centre coordinates of the land-only cells from their flat (row-major) grid index
        idx = d['idx'].astype(np.int64); i = idx // nlon; j = idx % nlon
        la = d['lat'].astype(float)[i]; lo = d['lon'].astype(float)[j]
        # 1-degree bin of each cell: row 0 starts at 90 S, column 0 at 180 W
        I = np.clip(np.floor(la + 90.0).astype(int), 0, 179); J = np.floor((lo + 180.0) % 360.0).astype(int) % 360
        keep = lo < 180.0                                   # the dateline column is repeated in the 0.1-degree grid
        # erosion rate (mm/yr): NeuroLEM net erosion on the land of the adjusted surface, deposition set to zero
        E = np.where(d['mz'], np.maximum(-d['ero_z'].astype(float), 0.0), 0.0) * SCALE
        A = d['area'].astype(float) / 1e6                   # cell area, m2 -> km2
        land = d['mz'] & keep; hi = land & (E > 0) & (d['z_z'] > ZMIN)
        # unbuffered accumulation, so that several cells falling in one bin all count
        np.add.at(LAND[ia], (I[land], J[land]), A[land]); np.add.at(HI[ia], (I[hi], J[hi]), A[hi])
        np.maximum.at(ZMAX[ia], (I[land], J[land]), d['z_z'][land].astype(np.float32))
        print(t, 'Ma', f'{HI[ia].sum() / 1e6:.2f} M km2 above {ZMIN:.0f} m', flush=True)
    # written beside the target and renamed into place, so an interrupted run leaves the previous scan intact
    tmp = SCAN.with_name(SCAN.stem + '.partial.npz')
    np.savez_compressed(tmp, ages=np.array(ages), hi_km2=HI, land_km2=LAND, zmax=ZMAX)
    os.replace(tmp, SCAN); print('wrote', SCAN)


def _arc(la1, lo1, la2, lo2):
    """Great-circle distance in degrees of arc between points given in degrees (spherical law of cosines)."""
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))


def locate():
    """Age of the belt and reconstructed centre of the window of every event. Writes CATALOGUE.

    Each present-day reference point is assigned to the PALEOMAP plate polygon that holds it today and rotated with
    that plate to every age of its literature range present in the scan; the age with the largest area of eroding
    land above ZMIN within RADIUS of the reconstructed point is the age of the belt."""
    import pygplates
    S = np.load(SCAN); ages = S['ages']; HI = S['hi_km2']
    rot = pygplates.RotationModel(str(PLATE_DIR / 'Scotese_Wright_PlateModel_2023.rot'))
    part = pygplates.PlatePartitioner(pygplates.FeatureCollection(str(PLATE_DIR / 'Scotese_Wright_PlatePolygons_2023.gpml')), rot, reconstruction_time=0.0)
    LO, LA = np.meshgrid(np.arange(360) - 179.5, np.arange(180) - 89.5)     # centres of the 1-degree bins of the scan
    rows = []
    for key, label, la, lo, a_old, a_young in EVENTS:
        pp = part.partition_point(pygplates.PointOnSphere(la, lo))           # present-day plate of the reference point
        pid = pp.get_feature().get_reconstruction_plate_id() if pp else 0
        cand = []
        for t in ages[(ages <= a_old) & (ages >= a_young)]:
            # rotation of the plate from the present day (from_time 0) to age t
            p = (rot.get_rotation(float(t), int(pid)) * pygplates.PointOnSphere(la, lo)).to_lat_lon()
            # eroding land above ZMIN (km2 -> million km2) in the bins whose centres lie within RADIUS of the point
            h = float(HI[int(np.where(ages == t)[0][0])][_arc(LA, LO, p[0], p[1]) <= RADIUS].sum() / 1e6)
            cand.append((h, int(t), p[0], p[1]))
        h, t, plat, plon = max(cand)          # the age with the most such land (ties go to the older age)
        if h < MIN_AREA:
            use, note = 0, f'less than {MIN_AREA} M km2 of eroding land above {ZMIN:.0f} m at every age of its range'
        elif t >= int(ages.max()):
            # orogen_tracks.py compares the belt age with an earlier age, which the oldest reconstruction lacks
            use, note = 0, 'highest at the oldest reconstruction: no earlier surface to compare with'
        else:
            use, note = 1, ''
        rows.append(dict(key=key, label=label, ref_lat=la, ref_lon=lo, plate=pid, age_old=a_old, age_young=a_young, age=t,
                         lat=round(plat, 2), lon=round(plon, 2), radius=RADIUS, area_hi_Mkm2=round(h, 3), used=use, note=note))
        print(f'{key:15s} {t:3d} Ma ({plat:6.1f}, {plon:7.1f}) {h:5.2f} M km2  {"used" if use else "not used: " + note}', flush=True)
    for key in PAPER:            # marker rows: orogen_tracks.py takes the window and age of these belts from its BELTS
        rows.append(dict(key=key, label='', ref_lat='', ref_lon='', plate='', age_old='', age_young='', age='', lat='', lon='', radius='',
                         area_hi_Mkm2='', used=1, note='window and age of Fig. 4 (orogen_tracks.BELTS)'))
    with open(CATALOGUE, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote', CATALOGUE)


if __name__ == '__main__':
    {'scan': scan, 'locate': locate}[sys.argv[1]]()

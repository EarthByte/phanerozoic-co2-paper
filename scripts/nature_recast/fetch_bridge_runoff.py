#!/usr/bin/env python3
"""Precipitation and runoff of the HadCM3L simulations that supplied the climate forcing.

Reads the annual climatology of each of the 109 simulations (Valdes et al., 2021, suite
scotese_spinupa, the smoothed-CO2 sequence; one 1.3 MB netCDF per age on the BRIDGE server),
keeps precipitation and surface + subsurface runoff on the model grid and writes ONE table:

    data/nature_recast/hadcm3l_runoff_scotese_spinupa.npz
        ages (109), lat (73), lon (96), precip (109,73,96), runoff (109,73,96; NaN over ocean),
        puma_id (109); precipitation and runoff in mm/day as served

The table ships with the repository, so this script only needs to run to rebuild it.
Each netCDF is downloaded to a temporary file that is removed after reading;
use --nc-dir to read climatologies that are already on disk (<age>Ma_<puma_id>.nc).
"""
import argparse, csv, subprocess, tempfile
from pathlib import Path
import numpy as np, netCDF4 as nc
import _config as cfg

URL = 'https://www.paleo.bristol.ac.uk/ummodel/data/{pid}/climate/{pid}a.pdclann.nc'


def read(path):
    n = nc.Dataset(path)
    P = np.squeeze(n['precip_mm_srf'][:]).filled(np.nan).astype(np.float32)
    R = np.ma.filled(np.squeeze(n['srfRunoff_mm_srf'][:]) + np.squeeze(n['subsrfRunoff_mm_srf'][:]), np.nan).astype(np.float32)
    la = np.asarray(n['latitude'][:], np.float32); lo = np.asarray(n['longitude'][:], np.float32)
    n.close()
    return P, R, la, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='scotese_spinupa')
    ap.add_argument('--nc-dir', default=None)
    a = ap.parse_args()
    rows = sorted((r for r in csv.DictReader(open(cfg.EXPERIMENTS)) if r['suite'] == a.suite), key=lambda r: int(r['plate_age']))
    ages, pids, PP, RR = [], [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            age = int(r['plate_age']); pid = r['puma_id']
            if a.nc_dir:
                f = Path(a.nc_dir) / f'{age:03d}Ma_{pid}.nc'
            else:
                f = Path(tmp) / 'climatology.nc'
                subprocess.run(['curl', '-sS', '-f', '-m', '180', '--retry', '3', '-o', str(f), URL.format(pid=pid)], check=True)
            P, R, la, lo = read(f)
            ages.append(age); pids.append(pid); PP.append(P); RR.append(R)
            print(age, pid, f'{np.nansum(R) / np.nansum(np.where(np.isfinite(R), P, np.nan)):.3f}', flush=True)
    out = cfg.RUNOFF if a.suite == 'scotese_spinupa' else cfg.RUNOFF.with_name(f'hadcm3l_runoff_{a.suite}.npz')
    np.savez_compressed(out, ages=np.array(ages), lat=la, lon=lo, precip=np.stack(PP), runoff=np.stack(RR), puma_id=np.array(pids))
    print('wrote', out)


if __name__ == '__main__':
    main()

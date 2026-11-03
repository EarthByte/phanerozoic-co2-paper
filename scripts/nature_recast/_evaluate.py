"""Shared evaluation of the weathering quantities for the uncertainty experiments.

Depends on numpy only (the climate experiment must run where scipy, pandas and netCDF4 are absent).
The calibration amplitudes and the E_opt(runoff) table are read from the main run, never refitted,
so every experiment differs from the main model in exactly the input it perturbs.

  load_calibration(prefix)        amplitudes, organic amplitudes, run settings, qgrid, E_opt(q)
  Evaluator(...)(erodep, m, mref) metrics on own land and, suffix _common, on reference land
  ratio_at(age, lat, lon)         HadCM3L runoff ratio at points, from the filled model-grid table
  append_rows(path, header, rows, key)   ONE csv per experiment, one atomic write per block
"""
import ast, csv, io, os, re
from pathlib import Path
import numpy as np
import _config as cfg
import weathering_core as wc
import run_weathering as rw

KEEP = ['land_Mkm2', 'eroding_Mkm2', 'E_mean_eroding_mm', 'q_mean_mm', 'D_Gt',
        'F_sil', 'F_sulf', 'F_net', 'F_ocp', 'F_ocb', 'S_sil', 'S_sulf', 'S_net',
        'S_net_oc_rel_sil', 'S_total_rel_sil', 'Gamma_sil', 'WI', 'd7Li_riv',
        'beyond_frac_eroding', 'beyond_oc_frac_eroding', 'silflux_share_beyond', 'Eopt_mean_mm']
KEEP_COMMON = ['D_Gt', 'F_sil', 'F_net', 'S_sil', 'S_net', 'beyond_frac_eroding']
METRIC_COLUMNS = (KEEP + [f'{k}_common' for k in KEEP_COMMON] + [f'S_sil_{r}' for r in rw.SETS]
                  + [f'F_sil_{r}' for r in rw.SETS])
RATIO_FILLED = cfg.OUT / 'hadcm3l_runoff_ratio_filled.npz'


def load_calibration(main_prefix):
    lines = Path(f'{main_prefix}_amplitudes.txt').read_text().strip().splitlines()
    amps, amp_oc = ast.literal_eval(lines[0]), ast.literal_eval(lines[1])
    settings = dict(re.findall(r"(\w+)=(\{[^}]*\}|\S+)", lines[2]))     # values may be dict reprs with spaces (driver={...})
    # the experiments evaluate every surface with the weathering form of the main model; the forms the main run can take
    # beyond the published one (sulfide exponent, reference-runoff scaling, precipitation-driven sets, log-stretch) are not carried by
    # the shared evaluator yet, so a main run in one of those forms must not be silently paired with it
    # the sulfide runoff exponent is carried: every surface is evaluated with the exponent of the paired main run (the live main run uses
    # the pure West form; the preserved run of the direct goSPL surfaces, data/nature_recast/gospl_surfaces/, the published -0.75)
    wc.BETA['sulfide'] = float(settings.get('beta_sulf', '-0.75'))
    if (settings.get('qref_scale', 'none') != 'none' or "'precip'" in settings.get('driver', '')
            or float(settings.get('stretch', 1)) != 1.0):
        raise SystemExit(f'{main_prefix}: the main run uses a weathering form the experiments do not carry yet '
                         f'(beta_sulf={settings.get("beta_sulf")}, qref_scale={settings.get("qref_scale")}, driver={settings.get("driver")}, stretch={settings.get("stretch", 1)}); '
                         'generalise _evaluate.Evaluator and apply_ensemble_sensitivity.Evaluator before running them')
    with np.load(f'{main_prefix}_hist.npz') as h:
        qgrid, eopt_q = h['qgrid'].astype(float), h['eopt_q'].astype(float)
    if settings['qsrc'] != 'bridge':
        raise SystemExit('the main run must use the HadCM3L runoff ratio (--qsrc bridge)')
    return amps, amp_oc, settings, qgrid, eopt_q


class Evaluator:
    """Weathering of one age on a fixed set of cells (1-D arrays; any grid or mesh)."""

    def __init__(self, cells, A, cal, settings):
        self.cells, self.A = cells, np.asarray(A, float)[cells]
        self.amps, self.amp_oc, self.qgrid, self.eopt_q = cal
        self.mode, self.oc_form, self.scale = settings['mode'], settings['oc_form'], float(settings['scale'])
        self.fitsets = {r: wc.load_fits(rw.PARAMS, [r]) for r in rw.SETS}

    def __call__(self, erodep, mask, ref_mask, q, T):
        """erodep (mm/yr, negative = erosion), masks, runoff (mm/yr) and temperature (K) on the full arrays."""
        c = self.cells
        m = np.asarray(mask)[c]; mref = np.asarray(ref_mask)[c]
        q = np.asarray(q, float)[c]; T = np.asarray(T, float)[c]
        E = np.where(m | mref, np.maximum(-np.asarray(erodep)[c].astype(float), 0.0), 0.0) * self.scale
        terms = {r: rw.region_terms(E, q, T, self.fitsets[r], self.mode) for r in rw.SETS}
        cal = {r: {k: self.amps[r]['sil' if 'sil' in k else 'sulf'] * terms[r][k] for k in ('sil', 'dsil', 'sulf', 'dsulf')}
               for r in rw.SETS}
        ens = {k: sum(cal[r][k] for r in rw.SETS) / len(rw.SETS) for k in ('sil', 'dsil', 'sulf', 'dsulf')}
        g = sum(terms[r]['g'] * cal[r]['sil'] for r in rw.SETS) / np.maximum(sum(cal[r]['sil'] for r in rw.SETS), 1e-300)
        Jp, dJp = wc.oc_petro(E, self.oc_form); Jb, dJb = wc.oc_bio_burial(E)
        ocp, docp = self.amp_oc['ocp'] * Jp / 12.011, self.amp_oc['ocp'] * dJp / 12.011
        ocb, docb = self.amp_oc['ocb'] * Jb / 12.011, self.amp_oc['ocb'] * dJb / 12.011
        args = (ens['sil'], ens['dsil'], ens['sulf'], ens['dsulf'], g, ocp, docp, ocb, docb, self.eopt_q, self.qgrid)
        own = rw.summarise(E, q, m, self.A, *args)
        com = rw.summarise(E, q, mref, self.A, *args)
        out = {k: own[k] for k in KEEP}
        out.update({f'{k}_common': com[k] for k in KEEP_COMMON})
        Am = self.A[m]
        for r in rw.SETS:
            Fs = float(np.sum(cal[r]['sil'][m] * Am)); dFs = float(np.sum(cal[r]['dsil'][m] * Am))
            out[f'S_sil_{r}'] = dFs / Fs; out[f'F_sil_{r}'] = Fs * wc.GTC
        return out


_RATIO = None


def ratio_at(age, lat, lon):
    """Runoff ratio at points (degrees), bilinear from the nearest-land-filled model grid.
    Identical to the field build_cache.py puts on the 0.1-degree grid."""
    global _RATIO
    if _RATIO is None:
        if not RATIO_FILLED.exists():
            raise SystemExit(f'{RATIO_FILLED} missing: run fill_runoff_ratio.py')
        with np.load(RATIO_FILLED) as z:
            _RATIO = {k: z[k] for k in z.files}
    k = int(np.flatnonzero(_RATIO['ages'] == age)[0])
    f = _RATIO['ratio'][k].astype(float); la = _RATIO['lat'].astype(float); lo = _RATIO['lon'].astype(float)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    y = np.clip((lat - la[0]) / (la[1] - la[0]), 0, la.size - 1)
    x = np.clip((lon - lo[0]) / (lo[1] - lo[0]), 0, lo.size - 1)
    i = np.minimum(y.astype(int), la.size - 2); j = np.minimum(x.astype(int), lo.size - 2)
    ty = y - i; tx = x - j
    out = (f[i, j] * (1 - ty) * (1 - tx) + f[i + 1, j] * ty * (1 - tx) + f[i, j + 1] * (1 - ty) * tx + f[i + 1, j + 1] * ty * tx)
    return np.clip(out, 0.0, 1.0)


def ratio_grid(age, lat, lon):
    """The same on a regular grid given by its axes."""
    out = np.empty((lat.size, lon.size), np.float32)
    for a in range(0, lat.size, 200):
        LA, LO = np.meshgrid(lat[a:a + 200], lon, indexing='ij')
        out[a:a + 200] = ratio_at(age, LA, LO)
    return out


def blocks_done(path, key, n_rows):
    """Values of column `key` that hold exactly n_rows rows."""
    if not Path(path).exists():
        return set()
    count = {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            count[r[key]] = count.get(r[key], 0) + 1
    return {int(float(k)) for k, n in count.items() if n == n_rows}


def append_rows(path, header, rows, key):
    """Append one block (all rows share rows[0][key]). If the file already holds rows of that block, it is
    rebuilt beside the original without them and swapped in atomically; completed blocks are never at risk."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=header, lineterminator='\n')
    block = str(rows[0][key])
    if not path.exists():
        w.writeheader(); w.writerows(rows)
        tmp = path.with_name(f'.{path.name}.tmp'); tmp.write_text(buf.getvalue()); tmp.replace(path)
        return
    w.writerows(rows)
    with open(path, newline='') as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != list(header):
            raise SystemExit(f'{path} has different columns; move it aside before writing a new experiment')
        kept = [r for r in rd if str(int(float(r[key]))) != str(int(float(block)))]
        n_old = rd.line_num - 1
    if len(kept) != n_old:
        keep_buf = io.StringIO(); kw = csv.DictWriter(keep_buf, fieldnames=header, lineterminator='\n')
        kw.writeheader(); kw.writerows(kept)
        tmp = path.with_name(f'.{path.name}.tmp'); tmp.write_text(keep_buf.getvalue() + buf.getvalue()); tmp.replace(path)
    else:
        with open(path, 'a') as fh:
            fh.write(buf.getvalue()); fh.flush(); os.fsync(fh.fileno())

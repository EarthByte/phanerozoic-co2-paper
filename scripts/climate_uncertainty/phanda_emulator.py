#!/usr/bin/env python3
"""
phanda_emulator.py -- per-cell pattern-scaling emulator of HadCM3L palaeoclimate,
fitted on the PhanDA model-prior ensemble (Judd et al. 2024), that gives the
temperature and precipitation forcing of the erosion and weathering calculation at
any prescribed pCO2.

ROLE IN THE PAPER
-----------------
The precipitation and surface-air temperature that force NeuroLEM and the
Bufe et al. (2024) weathering calculation come from the HadCM3L simulations of
Valdes et al. (2021), run along one smoothed proxy CO2 history.  This module
supplies the climate of the same palaeogeography at other CO2 levels.  Its fitted
coefficients (``phanda_sensitivity.npz``) are read by:

  * the land-surface reactivity calculation (Methods, "Land-surface reactivity and
    the weathering feedback"; ``nature_recast/run_reactivity.py``): the silicate
    sink at 280, 560 and 1,120 ppm and the feedback strength n = d ln F / d ln C
    (Extended Data Fig. 9);
  * the test of the feedback at a larger climate sensitivity (Supplementary
    Information; ``nature_recast/aess_feedback_test.py``), which rescales the
    emulator's area-weighted warming per doubling (``ecs_area_weighted``,
    3.1-4.6 K across the 109 ages);
  * the climate-state uncertainty ensemble (Methods, "Uncertainty and
    sensitivity", item 6; Extended Data Fig. 6; ``nature_recast/climate_sensitivity.py``
    and ``mc_driver.py``): CO2 levels across the proxy envelope written by
    ``build_pco2_envelope.py``, combined with coherent draws of the climate-model
    spread (the sigma terms below).

WHY AN EMULATOR
---------------
The Valdes et al. (2021) Phanerozoic archive that supplies the forcing contains two
CO2 sequences (``scotese_spinupa``, smoothed CO2; ``scotese_02``, Foster et al.
2017 CO2) and no perturbed-parameter ensemble, so it has no low/high member.  The
PhanDA model priors (Judd et al. 2024; Zenodo 8237751) are an ensemble over the same
109 Scotese time slices: 8 suites x 109 ages x 5 sub-means = 4360 HadCM3L
simulation averages spanning 1x, 2x and 4x pre-industrial (PI) CO2, two proxy-CO2
curves, a GMST-matched curve and three model versions.  The forcing itself is the
``scotese_spinupa`` suite of this ensemble (at 240 Ma the packed temperature grid
matches it to an RMSE of 0.011 degC, against 0.49 degC for the next-closest suite).

The ensemble is enough to fit, per grid cell and per age, the local response of
temperature and precipitation to CO2:

    T(x)      = a_T(x) + lambda_T(x) * log2(CO2 / CO2_ref)          [K per doubling]
    ln P'(x)  = a_P(x) + kappa_P(x)  * log2(CO2 / CO2_ref)          [ln units per doubling]

with P' = P + P0, P in mm/yr and P0 = 10 mm/yr (keeps hyper-arid cells finite in
log space), and CO2_ref the CO2 of the reference suite at that age.  Applying a
pCO2 value or envelope through these two coefficients gives per-cell, per-age
perturbations of the reference forcing without further GCM runs.

UNCERTAINTY DECOMPOSITION
-------------------------
Four terms, kept separate so they can be reported separately:

  sigma_int     internal variability -- spread across the 5 x 20-yr sub-means (M1..M5)
                of a single simulation, pooled over suites.  The only term that needs
                no assumptions.
  sigma_fit     regression residual  -- how well a single log2(CO2) slope describes
                the suite response at that cell (large where the response is
                non-linear, e.g. sea-ice margins, monsoon edges).
  sigma_struct  structural/model     -- after normalising the suites to a common
                CO2 with the fitted slopes, the spread *between model versions*
                (Original / Update A / Update B).  This is the closest thing to a
                model-error term the archive supports.
  CO2 term      lambda * log2(CO2_hi / CO2_ref) -- the forcing uncertainty proper,
                driven by the pCO2 record supplied.

The CO2 term is the deterministic part of the envelope; the three sigmas are added
in quadrature and applied as +/- z * sigma_total (z = 1.645 for a 5-95% band).

PRECIPITATION, NOT RUNOFF
-------------------------
The ``rain`` channel in the erodep NPZs is PRECIPITATION -- not P - E, and not model
runoff.  Against the BRIDGE annual climatologies of the same simulation (texya at
0 Ma, texYw at 240 Ma), over land, for a channel averaging 0.646 / 0.552 m/yr:

    precip_mm_srf              0.649 / 0.532 m/yr   r = 0.997 / 0.982   slope 1.01 / 0.97
    P - E_land                 0.103 / 0.007        r = 0.52  / 0.40    slope 0.68 / 0.64
    srfRunoff + subsrfRunoff   0.205 / 0.194        r = 0.73  / 0.85    slope 1.65 / 1.56

PhanDA's ``pr`` is therefore the SAME variable as the forcing channel.  The
perturbation is emitted as a multiplicative factor on that channel:

    R_hi / R_ref = ( P'_hi / P'_ref ) ** epsilon

with ``--runoff-elasticity`` epsilon.  Because the two variables are identical,
epsilon is exactly 1 and is not a free assumption; the option exists only as a
sensitivity knob for an amplified hydrological response.  Throughout this module
"runoff" and "R" name this precipitation channel; the runoff ratio that the
weathering calculation applies downstream to turn precipitation into runoff is not
changed by the emulator.

ASSUMPTIONS
-----------
1. Log-linear in CO2.  Held per cell and per age, so the nonlinearity of global
   climate sensitivity through the Phanerozoic (Farnsworth et al. 2019) is carried
   by the age dependence of lambda, not assumed away.  ``sigma_fit`` and ``r2_T`` /
   ``r2_P`` report where the log-linear form is a poor description.
2. Relative perturbation.  The emulator is applied as a perturbation to the
   reference forcing (an additive dT and a multiplicative precipitation factor), so
   any constant offset between the PhanDA fields and the packed reference grids
   cancels.
3. Same palaeogeographic frame.  Both PhanDA and the Valdes et al. (2021) forcing
   use the Scotese (PALEOMAP) reconstructions, so the coarse-grid lambda fields and
   the 0.1 deg basegrid are in the same palaeo-coordinates at each age.  This does
   not hold for a reconstruction in another plate frame (e.g. Merdith et al. 2021).
4. lambda is interpolated from 3.75 x 2.5 deg to 0.1 deg.  The perturbation is
   long-wavelength by construction; it adds no orographic detail.

SUBCOMMANDS, INPUTS AND OUTPUTS
-------------------------------
``fetch``     download the 16 ``<suite>_{tas,pr}.zip`` archives (16 of the 32 in
              Zenodo record 8237751, ~6.5 GB) plus ``ExperimentInfo.xlsx``, with
              checksum verification.
``index``     list the members present per suite and variable and check that the
              plate ages of ``phanda_experiments.csv`` are 0..540 Ma in 5 Myr steps.
``fit``       read the archive (local zips, extracted directories, or streamed from
              Zenodo with ``--phanda-dir remote``) and write ``phanda_sensitivity.npz``
              on the native HadCM3L 96 x 73 grid, arrays (n_age, 73, 96), lat
              ascending, lon -180..176.25:
                lambda_T [K per doubling], kappa_P [ln(P') per doubling],
                intercept_T [K], intercept_lnP [ln(mm/yr)],
                sigma_{int,fit,struct}_T [K], sigma_{int,fit,struct}_lnP [ln units],
                r2_T, r2_P [-], T_ref_model [K], lnP_ref_model [ln(mm/yr)];
              per-age scalars co2_ref [ppm], ecs_area_weighted [K per doubling],
              dP_global_pct_per_doubling [%], n_struct_groups, n_fit_suites,
              co2_span_doublings; plus age [Ma], lat, lon [deg] and a JSON ``meta``
              record.  Per-age checkpoints let an interrupted fit resume.
``envelope``  per age, low / central / high temperature perturbation dT [K, to add
              to the reference temperature] and lnR [ln of the factor that
              multiplies the reference precipitation], int16-packed in
              ``climate_envelope_<age>Ma.npz`` (unpack x = x_q * x_scale +
              x_offset), on the native grid (``--emit coarse``) or on the 0.1 deg
              basegrid 1801 x 3601 (``--emit full``), plus ``envelope_summary.csv``.
              The pCO2 bounds come from ``--envelope csv`` (age,co2_lo,co2_hi from
              ``build_pco2_envelope.py``), ``factor`` (CO2_ref / f .. CO2_ref * f;
              the default) or ``records``.
``ensemble``  ``--n`` Monte-Carlo members per age on the native grid
              (``climate_ensemble_<age>Ma.npz``: dT and runoff_scale, (n, 73, 96),
              and the pCO2 draw of each member), for propagating the forcing
              uncertainty end-to-end through the surrogate.
``apply``     add the ``envelope`` perturbations to the reference forcing on the
              basegrid -- temperature [degC] from ``valdes_temperature/<age>Ma.npz``
              (written by ``compress_valdes_temp.py``) and precipitation [m/yr]
              from the ``rain`` channel of ``surrogate_erodep/erodep_<age>Ma.npz``
              -- and write packed low/high fields to ``forcing_<age>Ma.npz`` plus
              land-mean values to ``forcing_summary.csv``.
``selftest``  synthetic round-trip: recovers known lambda / kappa / sigmas and
              checks the regridding, the longitude georeferencing and the packing.

Requires numpy; ``fit`` also needs netCDF4, and streaming uses ``requests`` when
available.  ``phanda_experiments.csv`` (the archive's ExperimentInfo.xlsx as CSV,
872 rows) must sit beside this script.

USAGE
  python phanda_emulator.py selftest
  python phanda_emulator.py fetch     --dest ~/data/phanda [--dry-run]
  python phanda_emulator.py index     --phanda-dir ~/data/phanda
  python phanda_emulator.py fit       --phanda-dir ~/data/phanda --out phanda_sensitivity.npz
  python phanda_emulator.py fit       --phanda-dir remote --out phanda_sensitivity.npz \
                                      [--time-budget 3600]     # stream, resumable
  python phanda_emulator.py envelope  --sensitivity phanda_sensitivity.npz \
                                      --envelope csv --envelope-csv pco2_envelope.csv \
                                      --out-dir climate_envelope [--sigma-terms ""]
  python phanda_emulator.py apply     --envelope-dir climate_envelope \
                                      --temp-dir valdes_temperature \
                                      --erodep-dir surrogate_erodep --out-dir forcing_bands
  python phanda_emulator.py ensemble  --sensitivity phanda_sensitivity.npz \
                                      --envelope csv --envelope-csv pco2_envelope.csv \
                                      --n 200 --seed 0 --out-dir climate_ensemble

``--sigma-terms ""`` restricts the band to the CO2 term alone.  Note that
``--envelope`` defaults to ``factor``: pass ``--envelope csv`` for
``--envelope-csv`` to take effect.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Archive description (fixed by the Zenodo record; see module docstring)
# --------------------------------------------------------------------------------------

ZENODO_RECORD = "8237751"
ZENODO_FILES = "https://zenodo.org/records/8237751/files/{name}?download=1"

#: suite -> HadCM3L model version.  The five "Original" suites span 1x/2x/4x PI CO2
#: plus two proxy-CO2 curves and are what the slope is fitted on; the Update A /
#: Update B suites enter only the structural (between-version) term.  Comments give
#: the CO2 forcing of each suite.
MODEL_VERSION = {
    "scotese_solara":  "Original",   # 1 x PI
    "scotese_2co2a":   "Original",   # 2 x PI
    "scotese_4co2a":   "Original",   # 4 x PI
    "scotese_spinupa": "Original",   # smoothed Foster et al. (2017)  <- the Valdes et al. (2021) forcing
    "scotese_02":      "Original",   # Foster et al. (2017)
    "scotese_06":      "UpdateA",    # Foster et al. (2017)
    "scotese_07":      "UpdateB",    # Foster et al. (2017) + Rae et al. (2021)
    "scotese_08":      "UpdateB",    # tuned to match Scotese et al. (2021) GMST
}
SUITES = tuple(MODEL_VERSION)
FIT_SUITES_DEFAULT = tuple(s for s, v in MODEL_VERSION.items() if v == "Original")

#: the suite that produced the temperature and precipitation fields that force NeuroLEM
#: and the weathering calculation; its CO2 at each age is CO2_ref
REFERENCE_SUITE = "scotese_spinupa"

SUBMEANS = (1, 2, 3, 4, 5)          # M1..M5, the 5 x 20-yr sub-averages
VARS = ("tas", "pr")                # 2 m air temperature (K), precipitation (mm/yr)

#: uncertainty and goodness-of-fit fields stored as float16 in the output -- see cmd_fit
HALF_PRECISION_FIELDS = frozenset((
    "sigma_int_T", "sigma_fit_T", "sigma_struct_T",
    "sigma_int_lnP", "sigma_fit_lnP", "sigma_struct_lnP", "r2_T", "r2_P"))

NLON, NLAT = 96, 73                 # HadCM3L: 3.75 deg x 2.5 deg (73 latitudes include both poles)
PRECIP_FLOOR_MM_YR = 10.0           # P0 in ln(P + P0), mm/yr; small against any non-desert precipitation

#: member filename <suite>_<exp_num>_<stage age>Ma_<CO2 ppm>pCO2_M<sub-mean>_<var>.nc,
#: e.g. scotese_06_109_0Ma_276pCO2_M1_tas.nc
MEMBER_RE = re.compile(
    r"^(?P<suite>.+?)_(?P<num>\d{3})_(?P<age>\d+)Ma_(?P<co2>[\d.]+)pCO2_M(?P<mean>\d)_(?P<var>\w+)\.nc$"
)

def _is_real_member(path: str, suite: str, var: str) -> bool:
    """True only for a genuine data member of ``<suite>_<var>.zip``.

    The zips were built on macOS and carry ``__MACOSX/._<name>`` resource-fork
    twins.  Those match the member pattern too, and because they sort after the
    real entries they would silently SHADOW them in the index, and the fit would
    then fail on a 4-byte AppleDouble payload.  Require a real basename and an exact suite
    match, which also rejects any stray file from another suite.
    """
    if "__MACOSX" in path:
        return False
    base = path.rsplit("/", 1)[-1]
    if base.startswith("."):
        return False
    m = MEMBER_RE.match(base)
    return bool(m) and m.group("var") == var and m.group("suite") == suite


HERE = Path(__file__).resolve().parent
EXPERIMENTS_CSV = HERE / "phanda_experiments.csv"


# --------------------------------------------------------------------------------------
# Experiment table
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Experiment:
    """One PhanDA simulation: a row of ``phanda_experiments.csv``."""
    exp_num: int          # 1..109, 1 = oldest
    exp_age: int          # stage age of the simulation (Ma), e.g. 541
    plate_age: int        # palaeogeography age (Ma), 0..540 step 5 -- the ages of the basegrid fields
    suite: str            # e.g. "scotese_spinupa"; see MODEL_VERSION
    puma_id: str          # HadCM3L experiment identifier
    co2_ppm: float        # prescribed atmospheric CO2 of the simulation (ppm)


def load_experiments(path: Path = EXPERIMENTS_CSV) -> list[Experiment]:
    """Read ``phanda_experiments.csv``, the archive's ``ExperimentInfo.xlsx`` exported as CSV.

    Columns: exp_num, exp_age, plate_age, suite, puma_id, co2_ppm (872 rows,
    8 suites x 109 ages).
    """
    if not path.exists():
        raise SystemExit(f"missing experiment table: {path}")
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(Experiment(int(r["exp_num"]), int(r["exp_age"]), int(r["plate_age"]),
                                  r["suite"], r["puma_id"], float(r["co2_ppm"])))
    return out


def experiments_by_age(exps: list[Experiment]) -> dict[int, dict[str, Experiment]]:
    """Group experiments as {plate_age: {suite: Experiment}}."""
    d: dict[int, dict[str, Experiment]] = {}
    for e in exps:
        d.setdefault(e.plate_age, {})[e.suite] = e
    return d


# --------------------------------------------------------------------------------------
# Archive access -- reads straight out of the Zenodo zips, no extraction needed
# --------------------------------------------------------------------------------------

class PhanDAArchive:
    """Local copy of the PhanDA archive.

    Resolves (suite, var) to ``<root>/<suite>_<var>.zip`` or, failing that, to a
    directory holding the extracted ``<suite>_*_<var>.nc`` files, and reads members
    by (exp_num, sub-mean).
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise SystemExit(f"PhanDA directory not found: {self.root}")
        self._zips: dict[tuple[str, str], zipfile.ZipFile] = {}
        self._index: dict[tuple[str, str], dict[tuple[int, int], str]] = {}
        self._dirs: dict[tuple[str, str], Path] = {}

    # -- location -----------------------------------------------------------------
    def _locate(self, suite: str, var: str):
        """Find the zip or directory for (suite, var) and index its members by
        (exp_num, sub-mean); cached after the first call."""
        key = (suite, var)
        if key in self._index:
            return
        zp = self.root / f"{suite}_{var}.zip"
        if zp.exists():
            zf = zipfile.ZipFile(zp)
            names = zf.namelist()
            self._zips[key] = zf
        else:
            # extracted layout: try <root>/<suite>_<var>/, <root>/<suite>/, then <root>
            cand = [self.root / f"{suite}_{var}", self.root / suite, self.root]
            d = next((c for c in cand if c.is_dir() and any(c.glob(f"{suite}_*_{var}.nc"))), None)
            if d is None:
                raise SystemExit(
                    f"cannot find {suite}/{var}: expected {zp.name} or a directory of "
                    f"{suite}_*_{var}.nc under {self.root}"
                )
            self._dirs[key] = d
            names = [p.name for p in d.glob(f"{suite}_*_{var}.nc")]
        idx: dict[tuple[int, int], str] = {}
        for n in names:
            if not _is_real_member(n, suite, var):
                continue
            m = MEMBER_RE.match(Path(n).name)
            idx[(int(m.group("num")), int(m.group("mean")))] = n
        if not idx:
            raise SystemExit(f"no parseable members for {suite}/{var} in {self.root}")
        self._index[key] = idx

    def available(self) -> dict[tuple[str, str], int]:
        """Number of members found per (suite, var); absent pairs are omitted."""
        got = {}
        for s in SUITES:
            for v in VARS:
                try:
                    self._locate(s, v)
                    got[(s, v)] = len(self._index[(s, v)])
                except SystemExit:
                    pass
        return got

    # -- reading ------------------------------------------------------------------
    def read_field(self, suite: str, var: str, exp_num: int, sub_mean: int) -> np.ndarray:
        """Annual-mean field of one member as (73, 96) float32, latitude ascending,
        longitude -180..176.25 (see ``_read_nc_variable``).

        ``var`` is "tas" (K) or "pr" (mm/yr); ``sub_mean`` is 1..5 (M1..M5).
        HadCM3L runs a 360-day calendar with 12 equal 30-day months, so the plain mean
        over the monthly climatology is the exact annual mean.
        """
        self._locate(suite, var)
        key = (suite, var)
        name = self._index[key].get((exp_num, sub_mean))
        if name is None:
            raise KeyError(f"{suite}/{var}: no member exp_num={exp_num} M{sub_mean}")
        if key in self._zips:
            buf = self._zips[key].read(name)
            arr = _read_nc_variable(buf, var, source=name)
        else:
            arr = _read_nc_variable(self._dirs[key] / name, var, source=name)
        return arr


class RemotePhanDAArchive:
    """Read members straight out of the Zenodo zips over HTTP, without downloading them.

    Same reading interface as ``PhanDAArchive`` (``available``, ``read_field``), plus
    ``prefetch`` and ``release`` for the decoded-field cache.

    A zip's central directory lives in its last few hundred kB, so one ranged GET per
    (suite, variable) yields the offset and compressed size of all 545 members; each
    member is then a second ranged GET plus a raw-deflate inflate.  Fitting all 109
    ages touches 8720 members totalling ~8 GB of transfer, but peak disk use is zero
    -- nothing is written.  Use this when you want the emulator without keeping the
    6.5 GB archive, or on a machine where the download is the awkward part.

    ``prefetch`` pulls one age's members concurrently, which is what makes the full
    fit practical: the per-member latency dominates, not the bandwidth.
    """

    TAIL = 400_000                    # bytes fetched from the end of each zip for its central directory

    def __init__(self, record: str = ZENODO_RECORD, workers: int = 6, retries: int = 5):
        self.record = record
        self.workers = workers
        self.retries = retries
        self._index: dict[tuple[str, str], dict[tuple[int, int], tuple]] = {}
        self._cache: dict[tuple, np.ndarray] = {}

    # -- HTTP ---------------------------------------------------------------------
    #
    # A pooled session with keep-alive is essential here: one TLS handshake per
    # request across 8720 requests gets throttled and starts timing out
    # ("_ssl.c:999: The handshake operation timed out").  Connections are reused
    # and every request is retried with backoff.
    def _session(self):
        s = getattr(self, "_sess", None)
        if s is not None:
            return s
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
        except ImportError:
            self._sess = False
            return False
        s = requests.Session()
        ad = HTTPAdapter(pool_connections=self.workers * 2, pool_maxsize=self.workers * 2,
                         max_retries=Retry(total=self.retries, backoff_factor=1.0,
                                           status_forcelist=[429, 500, 502, 503, 504],
                                           allowed_methods=["GET", "HEAD"]))
        s.mount("https://", ad)
        s.mount("http://", ad)
        self._sess = s
        return s

    def _get(self, url: str, start: int | None = None, end: int | None = None) -> bytes:
        """GET ``url`` (bytes start..end inclusive when given), retrying with a growing pause.

        Uses the pooled ``requests`` session when available, otherwise urllib.
        """
        import time
        headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
        sess = self._session()
        last = None
        for k in range(self.retries):
            try:
                if sess:
                    r = sess.get(url, headers=headers, timeout=(30, 180))
                    r.raise_for_status()
                    return r.content
                import urllib.request
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers=headers), timeout=180) as r:
                    return r.read()
            except Exception as exc:                      # noqa: BLE001 -- retry anything
                last = exc
                time.sleep(2.0 * (k + 1))
        raise SystemExit(f"remote read failed after {self.retries} attempts: {url}\n  {last}")

    def _size(self, url: str) -> int:
        """Content-Length of ``url`` from a HEAD request, with the same retries as ``_get``."""
        import time
        sess = self._session()
        last = None
        for k in range(self.retries):
            try:
                if sess:
                    r = sess.head(url, timeout=(30, 90), allow_redirects=True)
                    r.raise_for_status()
                    return int(r.headers["Content-Length"])
                import urllib.request
                with urllib.request.urlopen(
                        urllib.request.Request(url, method="HEAD"), timeout=90) as r:
                    return int(r.headers["Content-Length"])
            except Exception as exc:                      # noqa: BLE001
                last = exc
                time.sleep(2.0 * (k + 1))
        raise SystemExit(f"remote HEAD failed: {url}\n  {last}")

    # -- zip central directory ------------------------------------------------------
    def _locate(self, suite: str, var: str):
        key = (suite, var)
        if key in self._index:
            return
        import struct
        url = ZENODO_FILES.format(name=f"{suite}_{var}.zip")
        size = self._size(url)
        tail = self._get(url, max(size - self.TAIL, 0), size - 1)
        # Scan the tail for central-directory file headers (signature PK\1\2).  Fixed
        # little-endian offsets within each header: +10 compression method, +20
        # compressed size, +28 file-name length, +42 offset of the local file header,
        # +46 file name.  32-bit fields only (no Zip64), which suffices for these zips.
        idx, p = {}, 0
        while True:
            p = tail.find(b"PK\x01\x02", p)
            if p < 0:
                break
            nlen = struct.unpack("<H", tail[p + 28:p + 30])[0]
            name = tail[p + 46:p + 46 + nlen].decode()
            off = struct.unpack("<I", tail[p + 42:p + 46])[0]
            csize = struct.unpack("<I", tail[p + 20:p + 24])[0]
            method = struct.unpack("<H", tail[p + 10:p + 12])[0]
            if _is_real_member(name, suite, var):
                m = MEMBER_RE.match(name.rsplit("/", 1)[-1])
                idx[(int(m.group("num")), int(m.group("mean")))] = (name, off, csize, method, url)
            p += 4
        if not idx:
            raise SystemExit(f"remote: no parseable members in {suite}_{var}.zip "
                             f"(central directory may exceed the {self.TAIL} byte tail)")
        self._index[key] = idx

    def available(self):
        got = {}
        for s in SUITES:
            for v in VARS:
                self._locate(s, v)
                got[(s, v)] = len(self._index[(s, v)])
        return got

    # -- member reads ---------------------------------------------------------------
    #: classic netCDF-3 and HDF5 (netCDF-4) file signatures
    _MAGIC = (b"CDF\x01", b"CDF\x02", b"\x89HDF")

    def _member_bytes(self, suite, var, exp_num, sub_mean) -> tuple[str, bytes]:
        """Download and inflate one member; returns (member name, netCDF bytes).

        Pure I/O + zlib, safe to run in a thread.  The request covers the 30-byte
        local file header, its name and extra fields (assumed to fit in 256 bytes)
        and the compressed data.

        The payload is validated before it is returned.  A server that ignores the
        Range header answers 200 with the WHOLE zip, and a truncated 206 yields a
        short read; either way the bytes inflate to garbage and netCDF then fails
        with an unhelpful "Unknown file format" much later.  Both are caught here
        and retried.
        """
        import struct
        import time
        import zlib
        rec = self._index[(suite, var)].get((exp_num, sub_mean))
        if rec is None:
            raise KeyError(f"{suite}/{var}: no member exp_num={exp_num} M{sub_mean}")
        name, off, csize, method, url = rec
        last = ""
        for attempt in range(self.retries):
            blob = self._get(url, off, off + csize + 256)
            if len(blob) < csize + 30:
                last = f"short read: {len(blob)} bytes for a {csize}-byte member"
            elif blob[:4] not in (b"PK\x03\x04",):
                last = f"not a local file header: {blob[:4]!r} (Range header ignored?)"
            else:
                nlen = struct.unpack("<H", blob[26:28])[0]
                elen = struct.unpack("<H", blob[28:30])[0]
                raw = blob[30 + nlen + elen:30 + nlen + elen + csize]
                try:
                    # method 8 = deflate; wbits -15 = raw deflate stream without a zlib header
                    data = zlib.decompress(raw, -15) if method == 8 else raw
                except zlib.error as exc:
                    last = f"inflate failed: {exc}"
                else:
                    if data[:4] in self._MAGIC or data[:3] == b"\x89HD":
                        return name, data
                    last = f"inflated payload is not netCDF: {data[:4]!r}"
            time.sleep(1.5 * (attempt + 1))
        raise SystemExit(f"remote member {name} failed validation after {self.retries} "
                         f"attempts ({last})")

    def prefetch(self, jobs):
        """Download one age's members concurrently, then decode them SERIALLY.

        ``jobs`` is a list of (suite, var, exp_num, sub_mean) tuples; decoded fields
        are cached for ``read_field`` until ``release``.

        netCDF4/HDF5 is not thread-safe: decoding in the worker threads corrupts the
        heap (``malloc(): unaligned tcache chunk detected``).  Only the network and
        zlib work is parallel, which is where the time goes anyway.
        """
        from concurrent.futures import ThreadPoolExecutor
        todo = [j for j in jobs if j not in self._cache]
        if not todo:
            return
        for s, v in {(j[0], j[1]) for j in todo}:      # serialise the directory reads
            self._locate(s, v)
        with ThreadPoolExecutor(self.workers) as ex:
            blobs = list(ex.map(lambda j: self._member_bytes(*j), todo))
        for j, (name, data) in zip(todo, blobs):       # decode in the main thread
            self._cache[j] = _read_nc_variable(data, j[1], source=name)

    def read_field(self, suite: str, var: str, exp_num: int, sub_mean: int) -> np.ndarray:
        """Annual-mean field (73, 96), from the cache or fetched on demand."""
        j = (suite, var, exp_num, sub_mean)
        if j not in self._cache:
            self._locate(suite, var)
            name, data = self._member_bytes(*j)
            self._cache[j] = _read_nc_variable(data, var, source=name)
        return self._cache[j]

    def release(self):
        """Drop the decoded fields of the current age to bound memory."""
        self._cache.clear()


def _read_nc_variable(src, var: str, source: str = "") -> np.ndarray:
    """Return the annual mean of *var* as (nlat, nlon), lat ascending, lon in [-180, 180).

    Handles both the short names (``tas`` (12,1,96,73) in K, ``pr`` (12,1,96,73) in
    mm/yr, dimensions (t, level, lon, lat)) and the Unified-Model long names
    (``temp_mm_1_5m`` (12,1,73,96) in K, ``precip_mm_srf`` (12,1,73,96) in kg/m2/s).

    LONGITUDE CONVENTION -- the one thing in this archive that will silently
    rotate your results by half the planet.  Each file carries TWO longitude
    axes: the Unified-Model ``longitude`` (0 .. 356.25 degE, describing
    ``temp_mm_1_5m`` / ``precip_mm_srf``) and the PhanDA-derived ``lon``
    (-180 .. 176.25, describing ``tas`` / ``pr``).  They differ by a 180 degree
    roll.  Reading ``tas`` while assuming the ``longitude`` axis puts every
    continent on the wrong side of the globe -- and it will not be obvious,
    because zonal means are unaffected.  This function always uses the axis that
    belongs to the array it read, normalises it to [-180, 180) and orders the data
    onto the canonical HadCM3L axis -180..176.25 (``hadcm3l_axes``).

    ``src`` is a path or the bytes of a netCDF file; ``source`` names it in error
    messages.  Returns a (73, 96) float32 array.
    """
    try:
        from netCDF4 import Dataset
    except ImportError:  # pragma: no cover
        raise SystemExit("netCDF4 is required to read the PhanDA archive: pip install netCDF4")

    if isinstance(src, (bytes, bytearray)):
        ds = Dataset("inmemory.nc", mode="r", memory=bytes(src))
    else:
        ds = Dataset(str(src), mode="r")
    try:
        long_name = {"tas": "temp_mm_1_5m", "pr": "precip_mm_srf"}[var]
        if var in ds.variables:                      # (t, level, lon, lat)
            a = np.asarray(ds.variables[var][:], dtype=np.float64)
            a = np.nanmean(a, axis=0)[0]             # (lon, lat)
            a = a.T                                  # (lat, lon)
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        elif long_name in ds.variables:              # (t, level, lat, lon)
            v = ds.variables[long_name]
            a = np.asarray(v[:], dtype=np.float64)
            a = np.nanmean(a, axis=0)[0]             # (lat, lon)
            if var == "pr":
                a = a * 31_536_000.0                 # kg m-2 s-1 -> mm/yr (1 kg m-2 = 1 mm; 365-day year)
            lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        else:
            raise SystemExit(f"{source}: neither '{var}' nor '{long_name}' present")
        if a.shape != (NLAT, NLON):
            raise SystemExit(f"{source}: unexpected shape {a.shape}, expected {(NLAT, NLON)}")
        if lat[0] > lat[-1]:                         # archive stores lat 90 -> -90; flip to ascending
            a, lat = a[::-1], lat[::-1]
        # normalise longitude to an ascending axis in [-180, 180) and roll the data
        # with it -- this is where the two archive conventions are reconciled
        lon = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon)
        if not np.array_equal(order, np.arange(lon.size)):
            a, lon = a[:, order], lon[order]
        # final guard: the result must sit on the canonical HadCM3L axis that every
        # other function in this module assumes
        if not np.allclose(lon, hadcm3l_axes()[1], atol=1e-6):
            raise SystemExit(f"{source}: unexpected longitude axis {lon[:3]} ... {lon[-1]}")
        return np.ascontiguousarray(a, dtype=np.float32)
    finally:
        ds.close()


def hadcm3l_axes() -> tuple[np.ndarray, np.ndarray]:
    """(lat ascending -90..90, lon ascending -180..176.25) for the 73 x 96 HadCM3L grid.

    Longitude follows the ``lon`` axis that the archive attaches to ``tas``/``pr``,
    not the Unified-Model ``longitude`` axis (0..356.25) attached to the raw
    ``temp_mm_1_5m``/``precip_mm_srf`` fields; the two differ by a 180 degree roll.
    """
    lat = np.linspace(-90.0, 90.0, NLAT)
    lon = -180.0 + np.arange(NLON) * (360.0 / NLON)
    return lat.astype(np.float64), lon.astype(np.float64)


# --------------------------------------------------------------------------------------
# Weighted least squares on the member axis
# --------------------------------------------------------------------------------------

def wls_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Per-cell weighted least squares of y on x, vectorised over the grid.

    x : (M,)            predictor, identical at every cell (log2 CO2 ratio)
    y : (M, ny, nx)     response (T in K, or ln P')
    w : (M, ny, nx)     weights, or None for unweighted

    Returns slope, intercept, sigma_fit, r2 -- each (ny, nx) float32.  Cells where
    the normal equations are degenerate get NaN slope and intercept.
    ``sigma_fit`` is the unweighted residual standard deviation with ddof = 2, in
    the units of y, so for temperature it reads directly as "K at this cell not
    explained by a single log2 slope".  ``r2`` is the unweighted coefficient of
    determination.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    M = x.size
    if M < 3:
        raise ValueError("need at least 3 members to fit a slope and report a residual")
    xb = x[:, None, None]
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)

    # closed-form weighted normal equations, one 2 x 2 system per cell
    Sw = w.sum(0)
    Swx = (w * xb).sum(0)
    Swy = (w * y).sum(0)
    Swxx = (w * xb * xb).sum(0)
    Swxy = (w * xb * y).sum(0)
    den = Sw * Swxx - Swx * Swx
    bad = ~np.isfinite(den) | (np.abs(den) < 1e-12)
    den = np.where(bad, 1.0, den)

    slope = (Sw * Swxy - Swx * Swy) / den
    intercept = (Swy - slope * Swx) / np.where(Sw == 0, 1.0, Sw)
    slope = np.where(bad, np.nan, slope)
    intercept = np.where(bad, np.nan, intercept)

    resid = y - (intercept[None] + slope[None] * xb)
    sse = np.nansum(resid ** 2, axis=0)
    sigma_fit = np.sqrt(sse / max(M - 2, 1))
    sst = np.nansum((y - np.nanmean(y, axis=0)[None]) ** 2, axis=0)
    r2 = np.where(sst > 0, 1.0 - sse / np.where(sst == 0, 1.0, sst), np.nan)
    return (slope.astype(np.float32), intercept.astype(np.float32),
            sigma_fit.astype(np.float32), r2.astype(np.float32))


# --------------------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------------------

def fit_age(arch: PhanDAArchive, byage: dict[str, Experiment], *,
            fit_suites: tuple[str, ...], reference_suite: str,
            precip_floor: float, weighting: str,
            struct_tol: float = 0.5) -> dict[str, np.ndarray]:
    """Fit lambda_T and kappa_P at one plate age and estimate the three sigma terms.

    arch            PhanDAArchive or RemotePhanDAArchive
    byage           {suite: Experiment} for this plate age (all suites present)
    fit_suites      suites the slopes are regressed on (the five Original suites)
    reference_suite suite whose CO2 is CO2_ref (x = 0 of the regression)
    precip_floor    P0 in ln(P + P0), mm/yr
    weighting       "internal" weights each suite by 1 / (internal variance) per cell;
                    "none" is unweighted
    struct_tol      doublings of CO2 beyond the fitted range within which a suite is
                    still normalised for the structural term

    Returns a dict of (73, 96) float32 fields -- lambda_T [K/doubling], kappa_P
    [ln units/doubling], intercepts, the three sigmas for T [K] and ln P', r2_T,
    r2_P, and the reference suite's own mean fields T_ref_model [K] and
    lnP_ref_model -- plus scalar diagnostics (co2_ref [ppm], area-weighted ECS
    [K/doubling], global precipitation change [% per doubling], group and suite
    counts, CO2 span [doublings]) and co2_fit, the CO2 of each fit suite [ppm].
    """
    # ---- load every suite: per-suite mean over M1..M5 and the internal spread ----
    mean_T, mean_lnP, sd_T, sd_lnP, co2 = {}, {}, {}, {}, {}
    for suite, exp in byage.items():
        t_stack, p_stack = [], []
        for m in SUBMEANS:
            t_stack.append(arch.read_field(suite, "tas", exp.exp_num, m))
            p_stack.append(arch.read_field(suite, "pr", exp.exp_num, m))
        T = np.stack(t_stack).astype(np.float64)                       # (5, nlat, nlon), K
        P = np.stack(p_stack).astype(np.float64)                       # (5, nlat, nlon), mm/yr
        lnP = np.log(np.maximum(P, 0.0) + precip_floor)             # ln(P + P0); negatives clipped
        mean_T[suite] = T.mean(0)                                      # 100-yr mean of the 5 sub-means
        mean_lnP[suite] = lnP.mean(0)
        sd_T[suite] = T.std(0, ddof=1)                                 # spread of one 20-yr sub-mean
        sd_lnP[suite] = lnP.std(0, ddof=1)
        co2[suite] = exp.co2_ppm

    co2_ref = co2[reference_suite]
    used = [s for s in fit_suites if s in mean_T]
    if len(used) < 3:
        raise SystemExit(f"only {len(used)} fit suites present; need >= 3")
    x = np.array([np.log2(co2[s] / co2_ref) for s in used])           # CO2 in doublings from CO2_ref
    if x.max() - x.min() < 0.5:
        raise SystemExit(f"CO2 spread across fit suites is only {x.max()-x.min():.2f} "
                         "doublings -- refusing to fit a slope")

    yT = np.stack([mean_T[s] for s in used])
    yP = np.stack([mean_lnP[s] for s in used])
    if weighting == "internal":
        # weight each suite by the precision of its 5 x 20-yr mean; only relative
        # weights matter, so the 1/5 factor of the variance of the mean is omitted.
        # The floors stop a cell with near-zero spread from taking all the weight.
        wT = 1.0 / np.maximum(np.stack([sd_T[s] for s in used]) ** 2, 1e-6)
        wP = 1.0 / np.maximum(np.stack([sd_lnP[s] for s in used]) ** 2, 1e-8)
    else:
        wT = wP = None

    lam_T, a_T, sfit_T, r2_T = wls_slope(x, yT, wT)
    kap_P, a_P, sfit_P, r2_P = wls_slope(x, yP, wP)

    # ---- internal variability: RMS over all suites present of the spread of a
    #      single 20-yr sub-mean ------------------------------------------------------
    sig_int_T = np.sqrt(np.mean(np.stack([sd_T[s] ** 2 for s in mean_T]), axis=0)).astype(np.float32)
    sig_int_P = np.sqrt(np.mean(np.stack([sd_lnP[s] ** 2 for s in mean_lnP]), axis=0)).astype(np.float32)

    # ---- structural: normalise the suites to CO2_ref with the fitted slopes, then
    #      take the spread between model-version group means.  Suites whose CO2 sits
    #      outside the fitted range (plus a small tolerance) are excluded: normalising
    #      them would extrapolate the slope and inflate the term with fit error rather
    #      than model error.  scotese_08 (GMST-tuned) is the usual casualty. ---------
    lo_x, hi_x = x.min() - struct_tol, x.max() + struct_tol
    in_range = [s for s in mean_T if lo_x <= np.log2(co2[s] / co2_ref) <= hi_x]

    def _version_groups(mean_map, slope):
        """Spread (ddof = 1) between model-version group means after normalising each
        in-range suite to CO2_ref; returns (sigma field, number of groups), or zeros
        when fewer than two versions are in range."""
        groups: dict[str, list[np.ndarray]] = {}
        for s in in_range:
            adj = mean_map[s] - slope * np.log2(co2[s] / co2_ref)
            groups.setdefault(MODEL_VERSION[s], []).append(adj)
        if len(groups) < 2:
            return np.zeros_like(slope), 0
        gm = np.stack([np.mean(np.stack(v), axis=0) for v in groups.values()])
        return gm.std(0, ddof=1), gm.shape[0]

    sig_str_T, n_groups = _version_groups(mean_T, lam_T.astype(np.float64))
    sig_str_P, _ = _version_groups(mean_lnP, kap_P.astype(np.float64))
    sig_str_T = sig_str_T.astype(np.float32)
    sig_str_P = sig_str_P.astype(np.float32)

    # ---- area-weighted global diagnostics, over all cells (land and ocean),
    #      weighted by cos(latitude) -------------------------------------------------
    #   ecs:    global-mean warming per CO2 doubling [K]
    #   dP_pct: change in global-mean precipitation per doubling [%], from the fitted
    #           ln(P + P0) at CO2_ref (a_P) and at twice CO2_ref (a_P + kappa_P)
    lat_ax, _ = hadcm3l_axes()
    aw = np.cos(np.deg2rad(lat_ax))[:, None] * np.ones((1, NLON))
    ecs = float(np.nansum(lam_T * aw) / np.nansum(aw))
    Pref_g = float(np.nansum((np.exp(a_P) - precip_floor) * aw) / np.nansum(aw))
    Pdbl_g = float(np.nansum((np.exp(a_P + kap_P) - precip_floor) * aw) / np.nansum(aw))
    dP_pct = 100.0 * (Pdbl_g / Pref_g - 1.0) if Pref_g > 0 else np.nan

    return dict(
        lambda_T=lam_T, kappa_P=kap_P,
        intercept_T=a_T, intercept_lnP=a_P,
        sigma_fit_T=sfit_T, sigma_fit_lnP=sfit_P,
        sigma_int_T=sig_int_T, sigma_int_lnP=sig_int_P,
        sigma_struct_T=sig_str_T, sigma_struct_lnP=sig_str_P,
        r2_T=r2_T, r2_P=r2_P,
        co2_ref=np.float32(co2_ref),
        ecs_area_weighted=np.float32(ecs),
        dP_global_pct_per_doubling=np.float32(dP_pct),
        n_struct_groups=np.int32(n_groups),
        n_fit_suites=np.int32(len(used)),
        co2_span_doublings=np.float32(x.max() - x.min()),
        co2_fit=np.array([co2[s] for s in used], dtype=np.float32),
        T_ref_model=mean_T[reference_suite].astype(np.float32),
        lnP_ref_model=mean_lnP[reference_suite].astype(np.float32),
    )


def cmd_fit(a):
    """``fit`` subcommand: fit every requested age and write the sensitivity NPZ.

    Each age is fitted by ``fit_age``, checkpointed to ``<out>.ckpt/age_<age>Ma.npz``
    and finally stacked along a leading age axis into ``--out``.
    """
    exps = load_experiments()
    byage = experiments_by_age(exps)
    remote = str(a.phanda_dir).lower() in ("remote", "zenodo")
    arch = RemotePhanDAArchive(workers=a.workers) if remote else PhanDAArchive(a.phanda_dir)
    if remote:
        print(f"[fit] streaming from Zenodo {ZENODO_RECORD} over HTTP range requests "
              f"({a.workers} workers, nothing written to disk)")
    ages = sorted(byage) if not a.ages else [int(v) for v in a.ages]
    fit_suites = tuple(a.fit_suites.split(",")) if a.fit_suites else FIT_SUITES_DEFAULT

    # Per-age checkpoints.  A 109-age remote fit is a ~40 minute network-bound job,
    # and an interruption (a killed process, a dropped connection, Ctrl-C) would
    # otherwise lose everything.  Each age is written as soon as it is fitted and
    # reloaded on the next run, so a restart costs at most one age.  --time-budget
    # makes the job drivable in chunks: run it repeatedly until it reports all ages
    # present.
    ckpt = Path(a.checkpoint_dir) if a.checkpoint_dir else Path(str(a.out) + ".ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)

    def _ckpt_path(age):
        return ckpt / f"age_{int(age)}Ma.npz"

    def _load_ckpt(age):
        """Checkpoint of ``age`` as a dict, or None if absent or unreadable."""
        p_ = _ckpt_path(age)
        if not p_.exists():
            return None
        try:
            with np.load(p_, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            return None

    # keys: output fields, taken from the first result; co2_fit is excluded because
    # its length (number of fit suites) need not be the same at every age
    keys = None
    stacks: dict[str, list[np.ndarray]] = {}
    done = []
    t0 = time.monotonic()
    n_new = n_cached = 0
    for age in ages:
        cached = _load_ckpt(age)
        if cached is not None:
            res, n_cached = cached, n_cached + 1
            if keys is None:
                keys = [k for k in res if k != "co2_fit"]
                stacks = {k: [] for k in keys}
            for k in keys:
                stacks[k].append(res[k])
            done.append(age)
            continue
        if a.time_budget and time.monotonic() - t0 > a.time_budget:
            print(f"  [{age:>4} Ma] time budget reached -- stopping; re-run to continue",
                  flush=True)
            break
        try:
            if remote:
                arch.prefetch([(s_, v_, e_.exp_num, m_) for s_, e_ in byage[age].items()
                               for v_ in VARS for m_ in SUBMEANS])
            res = fit_age(arch, byage[age], fit_suites=fit_suites,
                          reference_suite=a.reference_suite,
                          precip_floor=a.precip_floor, weighting=a.weighting,
                          struct_tol=a.struct_tol)
        except (KeyError, SystemExit) as exc:
            # a missing member or a failed check skips this age; the run continues
            print(f"  [{age:>4} Ma] SKIPPED: {exc}", file=sys.stderr, flush=True)
            continue
        tmp = _ckpt_path(age).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **{k: v for k, v in res.items() if k != "co2_fit"})
        tmp.replace(_ckpt_path(age))          # atomic: never leave a half-written ckpt
        n_new += 1
        if keys is None:
            keys = [k for k in res if k != "co2_fit"]
            stacks = {k: [] for k in keys}
        for k in keys:
            stacks[k].append(res[k])
        done.append(age)
        print(f"  [{age:>4} Ma] ECS {float(res['ecs_area_weighted']):.2f} K/doubling, "
              f"dP {float(res['dP_global_pct_per_doubling']):+.1f} %/doubling, "
              f"median R2_T {np.nanmedian(res['r2_T']):.3f}, "
              f"CO2_ref {float(res['co2_ref']):.0f} ppm, "
              f"{int(res['n_struct_groups'])} version groups in range", flush=True)
        if remote:
            arch.release()

    missing = sorted(set(ages) - set(done))
    print(f"\n{len(done)}/{len(ages)} ages available ({n_new} fitted now, {n_cached} from "
          f"checkpoints in {ckpt})")
    if missing:
        print(f"  still missing {len(missing)}: {missing[:12]}"
              f"{' ...' if len(missing) > 12 else ''}\n  re-run the same command to continue.")
    if not done:
        raise SystemExit("nothing fitted")
    lat, lon = hadcm3l_axes()
    # Stack the 2-D fields to (n_age, 73, 96) and the scalars to (n_age,).
    # The sigma and R^2 fields are uncertainty estimates, not quantities anything is
    # divided by: float16 carries ~3 significant digits, far more than they mean, and
    # halves their storage.  lambda/kappa/intercepts stay float32 -- they multiply
    # log2 ratios of up to ~3, where the extra precision does matter.
    out = {k: np.stack(v).astype(np.float16 if k in HALF_PRECISION_FIELDS else np.float32)
           for k, v in stacks.items() if np.ndim(v[0]) == 2}
    for k, v in stacks.items():
        if np.ndim(v[0]) == 0:
            out[k] = np.array([float(x) for x in v], dtype=np.float32)
    out["age"] = np.array(done, dtype=np.int32)
    out["lat"] = lat.astype(np.float32)
    out["lon"] = lon.astype(np.float32)
    out["meta"] = np.bytes_(json.dumps(dict(
        source=f"PhanDA HadCM3L model priors, Zenodo {ZENODO_RECORD} (Judd et al. 2024)",
        fit_suites=list(fit_suites), reference_suite=a.reference_suite,
        precip_floor_mm_yr=a.precip_floor, weighting=a.weighting,
        model_version=MODEL_VERSION, n_ages=len(done),
    )))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}  ({Path(a.out).stat().st_size/1e6:.1f} MB, {len(done)} ages)")


# --------------------------------------------------------------------------------------
# Regridding: HadCM3L 73 x 96 (lat x lon) -> 0.1 deg basegrid 1801 x 3601
# --------------------------------------------------------------------------------------

def regrid_bilinear(field: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
                    dst_lat: np.ndarray, dst_lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation, periodic in longitude, clamped at the poles.

    field : (nlat, nlon) with src_lat ascending and src_lon ascending, spanning any
    360-degree window (the archive uses -180..176.25; 0..356.25 also works).
    dst_lat, dst_lon : 1-D target axes in degrees (any longitude convention).
    Returns (dst_lat.size, dst_lon.size) float32.  Interpolation is in degree
    space, which is adequate for the smooth, long-wavelength fields it is used on.
    """
    f = np.asarray(field, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    # -- wrap longitude so the last -> first seam interpolates correctly
    lon_ext = np.concatenate(([src_lon[-1] - 360.0], src_lon, [src_lon[0] + 360.0]))
    f_ext = np.concatenate((f[:, -1:], f, f[:, :1]), axis=1)

    # fold the target longitudes into [src_lon[0], src_lon[0] + 360) -- works for a
    # 0-based or a -180-based source axis alike
    lon0 = src_lon[0]
    dlon = lon0 + np.mod(np.asarray(dst_lon, dtype=np.float64) - lon0, 360.0)
    ix = np.clip(np.searchsorted(lon_ext, dlon) - 1, 0, lon_ext.size - 2)
    tx = (dlon - lon_ext[ix]) / (lon_ext[ix + 1] - lon_ext[ix])

    # latitude is not periodic: targets beyond the source range take the edge row
    dlat = np.clip(np.asarray(dst_lat, dtype=np.float64), src_lat[0], src_lat[-1])
    iy = np.clip(np.searchsorted(src_lat, dlat) - 1, 0, src_lat.size - 2)
    ty = (dlat - src_lat[iy]) / (src_lat[iy + 1] - src_lat[iy])

    # the four corners of each target cell's enclosing source cell; f_ext carries
    # one wrapped column at each end, so ix indexes it directly
    f00 = f_ext[np.ix_(iy, ix)]
    f01 = f_ext[np.ix_(iy, ix + 1)]
    f10 = f_ext[np.ix_(iy + 1, ix)]
    f11 = f_ext[np.ix_(iy + 1, ix + 1)]
    TX = tx[None, :]
    TY = ty[:, None]
    out = (f00 * (1 - TX) * (1 - TY) + f01 * TX * (1 - TY)
           + f10 * (1 - TX) * TY + f11 * TX * TY)
    return out.astype(np.float32)


def basegrid_axes() -> tuple[np.ndarray, np.ndarray]:
    """The 0.1 deg basegrid of the NeuroLEM / goSPL NPZs (Salles et al. 2022 fields).

    Returns (lat, lon) float32: lat -90..90 (1801 points), lon -180..180 (3601
    points; the dateline column appears at both ends).
    """
    return (np.linspace(-90.0, 90.0, 1801).astype(np.float32),
            np.linspace(-180.0, 180.0, 3601).astype(np.float32))


def _clip_scale(rscale: np.ndarray, clip: tuple[float, float] | None):
    """Bound the precipitation scaling factor to ``clip`` = (lo, hi); None = no bound.

    Hyper-arid cells have tiny P, so ln(P + P0) has a large model-version spread and
    exp() of it produces meaningless factors (x100 on a 10 mm/yr desert).  Those cells
    carry almost no weathering flux, but an unbounded factor can still blow up a
    high-erosion, low-runoff cell.  Clipping is a guard, not a correction: report the
    clipped fraction alongside the result.
    """
    if clip is None:
        return rscale
    lo, hi = float(min(clip)), float(max(clip))
    return np.clip(rscale, lo, hi)


# --------------------------------------------------------------------------------------
# The emulator
# --------------------------------------------------------------------------------------

class ClimateEmulator:
    """Per-cell log2(CO2) sensitivities, with helpers to build envelopes and draw members.

    Wraps the contents of ``phanda_sensitivity.npz`` (see ``cmd_fit``): fields are
    indexed by plate age, on the HadCM3L grid (73, 96).  ``perturbation`` and ``draw``
    return a temperature perturbation dT [K], to be added to the reference
    temperature, and a precipitation scaling factor [-], to multiply the reference
    precipitation, on the HadCM3L grid or the 0.1 deg basegrid.
    """

    def __init__(self, d: dict):
        self.d = d
        self.age = np.asarray(d["age"])
        self.lat = np.asarray(d["lat"], dtype=np.float64)
        self.lon = np.asarray(d["lon"], dtype=np.float64)
        self.meta = json.loads(bytes(d["meta"]).decode()) if "meta" in d else {}
        self.precip_floor = float(self.meta.get("precip_floor_mm_yr", PRECIP_FLOOR_MM_YR))

    @classmethod
    def load(cls, path) -> "ClimateEmulator":
        """Load a ``phanda_sensitivity.npz`` written by ``fit``."""
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def _i(self, age: int) -> int:
        """Index of ``age`` (Ma) along the fitted age axis; KeyError if not fitted."""
        hit = np.nonzero(self.age == int(age))[0]
        if hit.size == 0:
            raise KeyError(f"age {age} Ma not in the fitted set")
        return int(hit[0])

    def co2_ref(self, age: int) -> float:
        """CO2 of the reference simulation at ``age`` (ppm), the point where dT = 0."""
        return float(self.d["co2_ref"][self._i(age)])

    SIGMA_TERMS = ("int", "fit", "struct")

    def sigma_total(self, age: int, terms: tuple[str, ...] = SIGMA_TERMS
                    ) -> tuple[np.ndarray, np.ndarray]:
        """(sigma_T [K], sigma_lnP [ln units]) -- selected terms added in quadrature.

        ``terms=()`` isolates the pure CO2 contribution, which is usually the number a
        reader wants first; the structural term is large at high latitudes where model
        versions disagree about sea ice, and should be quoted separately rather than
        buried in a single band.
        """
        i = self._i(age)
        sT = np.zeros_like(self.d["sigma_int_T"][i], dtype=np.float64)
        sP = np.zeros_like(self.d["sigma_int_lnP"][i], dtype=np.float64)
        for t in terms:
            if t not in self.SIGMA_TERMS:
                raise SystemExit(f"unknown sigma term {t!r}; choose from {self.SIGMA_TERMS}")
            sT = sT + self.d[f"sigma_{t}_T"][i].astype(np.float64) ** 2
            sP = sP + self.d[f"sigma_{t}_lnP"][i].astype(np.float64) ** 2
        return np.sqrt(sT), np.sqrt(sP)

    def perturbation(self, age: int, co2_ppm: float, *, z: float = 0.0,
                     runoff_elasticity: float = 1.0, coarse: bool = False,
                     terms: tuple[str, ...] = SIGMA_TERMS,
                     runoff_clip: tuple[float, float] | None = None):
        """Perturbation of the reference forcing at ``age`` for a given pCO2.

        Returns (dT [K], runoff_scale [-]): dT adds to the reference temperature and
        runoff_scale multiplies the reference precipitation.  ``co2_ppm`` is the
        target pCO2.  ``z`` shifts by z * sigma_total: use z = +/-1.645 for a 5-95%
        band, 0 for the pure CO2 term.  ``terms`` selects the sigma terms,
        ``runoff_elasticity`` is epsilon and ``runoff_clip`` bounds the factor.
        ``coarse`` keeps the result on the 73 x 96 HadCM3L grid instead of the
        1801 x 3601 basegrid.
        """
        i = self._i(age)
        r = np.log2(float(co2_ppm) / self.co2_ref(age))                 # doublings from CO2_ref
        sT, sP = self.sigma_total(age, terms)
        dT = self.d["lambda_T"][i] * r + z * sT
        dlnP = self.d["kappa_P"][i] * r + z * sP                         # change in ln(P + P0)
        # (P' / P'_ref) ** epsilon; clipped on the coarse grid before regridding
        rscale = _clip_scale(np.exp(runoff_elasticity * dlnP), runoff_clip)
        if coarse:
            return dT.astype(np.float32), rscale.astype(np.float32)
        blat, blon = basegrid_axes()
        return (regrid_bilinear(dT, self.lat, self.lon, blat, blon),
                regrid_bilinear(rscale, self.lat, self.lon, blat, blon))

    def draw(self, age: int, co2_ppm: float, rng: np.random.Generator, *,
             runoff_elasticity: float = 1.0, coarse: bool = False,
             correlated: bool = True, terms: tuple[str, ...] = SIGMA_TERMS,
             runoff_clip: tuple[float, float] | None = None):
        """One Monte-Carlo member: CO2 term plus a noise realisation of sigma_total.

        ``correlated=True`` draws a single spatially-uniform standard normal per member
        (the conservative choice: it treats the sigma field as a coherent bias, so it
        does not average away under global integration).  ``correlated=False`` draws
        white noise per cell, which will very nearly cancel in a global sum.  The
        temperature and precipitation draws are independent of each other.  Returns
        (dT [K], runoff_scale [-]) as in ``perturbation``.
        """
        i = self._i(age)
        r = np.log2(float(co2_ppm) / self.co2_ref(age))
        sT, sP = self.sigma_total(age, terms)
        if correlated:
            eT = float(rng.standard_normal())
            eP = float(rng.standard_normal())
        else:
            eT = rng.standard_normal(sT.shape)
            eP = rng.standard_normal(sP.shape)
        dT = self.d["lambda_T"][i] * r + eT * sT
        dlnP = self.d["kappa_P"][i] * r + eP * sP
        rscale = _clip_scale(np.exp(runoff_elasticity * dlnP), runoff_clip)
        if coarse:
            return dT.astype(np.float32), rscale.astype(np.float32)
        blat, blon = basegrid_axes()
        return (regrid_bilinear(dT, self.lat, self.lon, blat, blon),
                regrid_bilinear(rscale, self.lat, self.lon, blat, blon))


# --------------------------------------------------------------------------------------
# pCO2 envelopes
# --------------------------------------------------------------------------------------

def load_envelope(kind: str, ages: np.ndarray, *, csv_path: str | None = None,
                  factor: float = 1.5, data_dir: Path | None = None,
                  age_tol: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (co2_lo, co2_hi) in ppm at each of ``ages`` (Ma), as two (n_age,) arrays.

    kind = 'csv'     read age,co2_lo,co2_hi from --envelope-csv and interpolate linearly
                     in age (the file written by ``build_pco2_envelope.py``)
    kind = 'factor'  multiplicative bracket CO2_ref / f .. CO2_ref * f -- a crude but
                     explicit stand-in.  CO2_ref is not known here, so (-f, +f) is
                     returned as a sentinel and ``_resolve_co2`` builds the bracket.
    kind = 'records' bracket the Foster et al. (2017) and Judd et al. (2024) curves
                     (``foster.npy`` / ``judd.npy``, columns age [Ma], CO2 [ppm]) at
                     each age; data-based, but a between-record spread, not a
                     posterior uncertainty.  Ages that no record covers return NaN.
    """
    ages = np.asarray(ages, dtype=float)
    if kind == "csv":
        if not csv_path:
            raise SystemExit("--envelope csv requires --envelope-csv")
        tab = np.genfromtxt(csv_path, delimiter=",", names=True)
        a = np.asarray(tab["age"], dtype=float)
        lo = np.interp(ages, a, np.asarray(tab["co2_lo"], dtype=float))
        hi = np.interp(ages, a, np.asarray(tab["co2_hi"], dtype=float))
        return lo, hi
    if kind == "factor":
        return np.full(ages.shape, -factor), np.full(ages.shape, factor)  # sentinel, see caller
    if kind == "records":
        dd = Path(data_dir) if data_dir else HERE
        cur = []
        for name in ("foster.npy", "judd.npy"):
            p = dd / name
            if p.exists():
                arr = np.asarray(np.load(p), dtype=float)
                arr = arr[np.argsort(arr[:, 0])]
                v = np.interp(ages, arr[:, 0], arr[:, 1])
                # only snap to a record's endpoint within age_tol; beyond that the
                # record simply does not cover the age and must not be extrapolated
                out_of_range = (ages < arr[0, 0] - age_tol) | (ages > arr[-1, 0] + age_tol)
                v[out_of_range] = np.nan
                cur.append(v)
        if not cur:
            raise SystemExit(f"no foster.npy / judd.npy under {dd}; pass --envelope-data-dir")
        stack = np.vstack(cur)
        allnan = np.all(~np.isfinite(stack), axis=0)
        lo = np.full(ages.shape, np.nan)
        hi = np.full(ages.shape, np.nan)
        ok = ~allnan
        lo[ok] = np.nanmin(stack[:, ok], axis=0)
        hi[ok] = np.nanmax(stack[:, ok], axis=0)
        return lo, hi
    raise SystemExit(f"unknown envelope kind: {kind}")


# --------------------------------------------------------------------------------------
# envelope / ensemble commands
# --------------------------------------------------------------------------------------

def _pack_int16(a: np.ndarray, step: float):
    """Quantise ``a`` to int16 as (a - offset) / step, offset = nanmin(a).

    Returns (q int16, step float32, offset float32); unpack with q * step + offset.
    If the range does not fit in 32760 steps the step is widened to fit.  NaNs are
    stored as the minimum.
    """
    off = float(np.nanmin(a))
    q = np.round((np.nan_to_num(a, nan=off) - off) / step)
    if q.max() >= 32760:
        step = float((np.nanmax(a) - off) / 32760.0)
        q = np.round((np.nan_to_num(a, nan=off) - off) / step)
    return q.astype(np.int16), np.float32(step), np.float32(off)


def _resolve_co2(a, emu, ages, lo, hi, k, age):
    """(CO2_ref, co2_lo, co2_hi) in ppm for the k-th age of the envelope.

    Builds the ``factor`` bracket about CO2_ref, orders lo/hi, and returns NaN bounds
    where the envelope has no value at this age.
    """
    ref = emu.co2_ref(int(age))
    if a.envelope == "factor":
        return ref, ref / a.envelope_factor, ref * a.envelope_factor
    c_lo, c_hi = float(lo[k]), float(hi[k])
    if not np.isfinite(c_lo) or not np.isfinite(c_hi):
        return ref, np.nan, np.nan
    return ref, min(c_lo, c_hi), max(c_lo, c_hi)


def cmd_envelope(a):
    """``envelope`` subcommand: low / central / high perturbation fields per age.

    Low = CO2 at the lower envelope bound and z = -sigma_multiplier; high = CO2 at the
    upper bound and z = +sigma_multiplier; central = CO2_ref with z = 0, which is zero
    perturbation by construction.
    """
    emu = ClimateEmulator.load(a.sensitivity)
    ages = emu.age if not a.ages else np.array([int(v) for v in a.ages])
    lo, hi = load_envelope(a.envelope, ages, csv_path=a.envelope_csv,
                           factor=a.envelope_factor, data_dir=a.envelope_data_dir)
    terms = tuple(t for t in a.sigma_terms.split(",") if t) if a.sigma_terms else ()
    clip = (a.runoff_scale_clip[0], a.runoff_scale_clip[1])
    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    blat, blon = basegrid_axes()
    lat, lon = (emu.lat, emu.lon) if a.emit == "coarse" else (blat, blon)
    summary = []
    for k, age in enumerate(ages):
        ref, c_lo, c_hi = _resolve_co2(a, emu, ages, lo, hi, k, age)
        if not np.isfinite(c_lo):
            print(f"  [{age:>4} Ma] no pCO2 envelope value at this age, skipped", file=sys.stderr)
            continue
        kw = dict(runoff_elasticity=a.runoff_elasticity, coarse=(a.emit == "coarse"),
                  terms=terms, runoff_clip=clip)
        dT_lo, rs_lo = emu.perturbation(int(age), c_lo, z=-a.sigma_multiplier, **kw)
        dT_hi, rs_hi = emu.perturbation(int(age), c_hi, z=+a.sigma_multiplier, **kw)
        dT_c, rs_c = emu.perturbation(int(age), ref, z=0.0, **kw)
        # fraction of grid cells (unweighted, land and ocean) at which either band
        # edge reached the clip bound
        clipped = float(np.mean((rs_lo <= clip[0] + 1e-9) | (rs_hi >= clip[1] - 1e-9)))

        # int16 packing: 0.01 K steps for dT, 1e-4 steps for ln(runoff scale)
        pack = {}
        for nm, arr, step in (("dT_lo", dT_lo, 0.01), ("dT_central", dT_c, 0.01),
                              ("dT_hi", dT_hi, 0.01),
                              ("lnR_lo", np.log(rs_lo), 1e-4),
                              ("lnR_central", np.log(rs_c), 1e-4),
                              ("lnR_hi", np.log(rs_hi), 1e-4)):
            q, sc, off = _pack_int16(np.asarray(arr, dtype=np.float64), step)
            pack[f"{nm}_q"], pack[f"{nm}_scale"], pack[f"{nm}_offset"] = q, sc, off
        np.savez_compressed(
            outd / f"climate_envelope_{int(age)}Ma.npz", **pack,
            co2_ref=np.float32(ref), co2_lo=np.float32(c_lo), co2_hi=np.float32(c_hi),
            sigma_multiplier=np.float32(a.sigma_multiplier),
            sigma_terms=np.bytes_(",".join(terms)),
            runoff_elasticity=np.float32(a.runoff_elasticity),
            runoff_scale_clip=np.array(clip, dtype=np.float32),
            clipped_fraction=np.float32(clipped),
            age=np.int32(age), lat=lat.astype(np.float32), lon=lon.astype(np.float32),
            grid=np.bytes_(a.emit),
            readme=np.bytes_(
                "dT = temperature perturbation (K) to ADD to the HadCM3 reference field; "
                "runoff_scale = exp(lnR) is the factor to MULTIPLY the reference runoff by. "
                "Unpack: x = x_q.astype('f4') * x_scale + x_offset."),
        )
        summary.append((int(age), ref, c_lo, c_hi,
                        float(np.nanmean(dT_lo)), float(np.nanmean(dT_hi)),
                        float(np.nanmedian(rs_lo)), float(np.nanmedian(rs_hi)), clipped))
        print(f"  [{age:>4} Ma] CO2 {c_lo:7.0f} / {ref:7.0f} / {c_hi:7.0f} ppm  ->  "
              f"dT {np.nanmean(dT_lo):+6.2f} .. {np.nanmean(dT_hi):+6.2f} K   "
              f"runoff x {np.nanmedian(rs_lo):.3f} .. {np.nanmedian(rs_hi):.3f}"
              + (f"   [{100*clipped:.1f}% of cells clipped]" if clipped > 0.001 else ""))

    with open(outd / "envelope_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age_Ma", "co2_ref_ppm", "co2_lo_ppm", "co2_hi_ppm",
                    "mean_dT_lo_K", "mean_dT_hi_K",
                    "median_runoff_scale_lo", "median_runoff_scale_hi", "clipped_fraction"])
        w.writerows(summary)
    print(f"\nwrote {len(summary)} ages to {outd}  (grid: {a.emit}, "
          f"sigma terms: {','.join(terms) or 'none -- CO2 term only'})")


def cmd_ensemble(a):
    """``ensemble`` subcommand: ``--n`` Monte-Carlo members per age on the native grid.

    Each member draws pCO2 log-uniformly within the envelope and a noise realisation
    of sigma_total (``ClimateEmulator.draw``).  Writes climate_ensemble_<age>Ma.npz
    with dT [K] and runoff_scale [-], each (n, 73, 96), and co2_ppm (n,).
    """
    emu = ClimateEmulator.load(a.sensitivity)
    ages = emu.age if not a.ages else np.array([int(v) for v in a.ages])
    lo, hi = load_envelope(a.envelope, ages, csv_path=a.envelope_csv,
                           factor=a.envelope_factor, data_dir=a.envelope_data_dir)
    rng = np.random.default_rng(a.seed)
    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    terms = tuple(t for t in a.sigma_terms.split(",") if t) if a.sigma_terms else ()
    clip = (a.runoff_scale_clip[0], a.runoff_scale_clip[1])
    for k, age in enumerate(ages):
        ref, c_lo, c_hi = _resolve_co2(a, emu, ages, lo, hi, k, age)
        if not np.isfinite(c_lo):
            print(f"  [{age:>4} Ma] no pCO2 envelope value at this age, skipped", file=sys.stderr)
            continue
        # log-uniform CO2 draws across the envelope: no distributional claim beyond
        # "somewhere in the band", and uniform in log CO2, the variable in which the
        # emulator's response is linear
        u = rng.uniform(np.log(c_lo), np.log(c_hi), size=a.n)
        co2 = np.exp(u)
        dTs, rss = [], []
        for c in co2:
            dT, rs = emu.draw(int(age), float(c), rng, runoff_elasticity=a.runoff_elasticity,
                              coarse=True, correlated=not a.white_noise,
                              terms=terms, runoff_clip=clip)
            dTs.append(dT)
            rss.append(rs)
        np.savez_compressed(
            outd / f"climate_ensemble_{int(age)}Ma.npz",
            dT=np.stack(dTs).astype(np.float32),
            runoff_scale=np.stack(rss).astype(np.float32),
            co2_ppm=co2.astype(np.float32), co2_ref=np.float32(ref),
            age=np.int32(age), lat=emu.lat.astype(np.float32), lon=emu.lon.astype(np.float32),
            seed=np.int64(a.seed), grid=np.bytes_("coarse"),
        )
        print(f"  [{age:>4} Ma] {a.n} members, CO2 {c_lo:.0f}-{c_hi:.0f} ppm, "
              f"dT spread {np.std([d.mean() for d in dTs]):.2f} K")
    print(f"\nwrote ensembles to {outd}  (coarse grid; expand with "
          f"phanda_emulator.regrid_bilinear or ClimateEmulator.draw(coarse=False))")


# --------------------------------------------------------------------------------------
# apply -- turn the perturbations into ready-to-run forcing fields
# --------------------------------------------------------------------------------------

def _unpack(z, name):
    """Unpack the int16 field ``name`` of an envelope NPZ: q * scale + offset."""
    return z[f"{name}_q"].astype(np.float32) * z[f"{name}_scale"] + z[f"{name}_offset"]


def load_reference_temperature(temp_dir: Path, age: int) -> np.ndarray:
    """Reference surface-air temperature (degC) on the basegrid, (1801, 3601).

    Reads ``<temp_dir>/<age>Ma.npz`` as written by ``compress_valdes_temp.py``.
    """
    p = Path(temp_dir) / f"{int(age)}Ma.npz"
    with np.load(p) as z:
        return z["temp_q"].astype(np.float32) * z["scale"] + z["offset"]


def load_reference_runoff(erodep_dir: Path, age: int) -> np.ndarray:
    """Reference precipitation (the 'rain' channel, m/yr) on the basegrid, (1801, 3601).

    Reads ``<erodep_dir>/erodep_<age>Ma.npz``.  The channel is precipitation, not
    runoff (see the module docstring); "runoff" in the name follows the naming of
    the scaling factor used throughout this module.
    """
    p = Path(erodep_dir) / f"erodep_{int(age)}Ma.npz"
    with np.load(p) as z:
        return np.asarray(z["rain"], dtype=np.float32)


def cmd_apply(a):
    """``apply`` subcommand: add the ``envelope`` perturbations to the reference forcing.

    For every climate_envelope_<age>Ma.npz in ``--envelope-dir``, regrids coarse
    fields to the basegrid, forms T = T_ref + dT [degC] and R = R_ref * exp(lnR)
    [m/yr] for the low and high edges, and writes them int16-packed (temperature in
    0.01 degC steps, ln R in 1e-4 steps) to forcing_<age>Ma.npz.  Land means
    (cells with R_ref > 0) go to forcing_summary.csv.
    """
    envd = Path(a.envelope_dir)
    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    blat, blon = basegrid_axes()
    files = sorted(envd.glob("climate_envelope_*Ma.npz"),
                   key=lambda p: int(re.search(r"_(\d+)Ma", p.name).group(1)))
    if a.ages:
        keep = {int(v) for v in a.ages}
        files = [f for f in files if int(re.search(r"_(\d+)Ma", f.name).group(1)) in keep]
    rows = []
    for f in files:
        age = int(re.search(r"_(\d+)Ma", f.name).group(1))
        with np.load(f) as z:
            coarse = bytes(z["grid"]).decode() == "coarse"
            lat, lon = np.asarray(z["lat"], float), np.asarray(z["lon"], float)
            fields = {}
            for nm in ("dT_lo", "dT_central", "dT_hi", "lnR_lo", "lnR_central", "lnR_hi"):
                v = _unpack(z, nm)
                fields[nm] = regrid_bilinear(v, lat, lon, blat, blon) if coarse else v
        T_ref = load_reference_temperature(a.temp_dir, age)
        R_ref = load_reference_runoff(a.erodep_dir, age)
        # 'central' is by construction identical to the reference fields, so only the
        # two band edges are written; use the existing valdes_temperature / erodep
        # files for the central member.
        out = {}
        for tag in ("lo", "hi"):
            T = T_ref + fields[f"dT_{tag}"]
            R = R_ref * np.exp(fields[f"lnR_{tag}"])
            q, sc, off = _pack_int16(T.astype(np.float64), 0.01)
            out[f"temp_{tag}_q"], out[f"temp_{tag}_scale"], out[f"temp_{tag}_offset"] = q, sc, off
            # pack ln R; the 1e-6 m/yr floor keeps zero-precipitation cells finite
            q, sc, off = _pack_int16(np.log(np.maximum(R, 1e-6)).astype(np.float64), 1e-4)
            out[f"lnrain_{tag}_q"], out[f"lnrain_{tag}_scale"], out[f"lnrain_{tag}_offset"] = q, sc, off
        np.savez_compressed(
            outd / f"forcing_{age}Ma.npz", **out,
            age=np.int32(age), lat=blat, lon=blon,
            units=np.bytes_("temp_<tag> = degC via q*scale+offset; "
                            "rain = exp(lnrain_<tag> via q*scale+offset), m/yr; "
                            "the central member is the unmodified reference forcing"),
            provenance=np.bytes_(f"HadCM3 reference (Valdes et al. 2021) + PhanDA "
                                 f"pattern-scaling perturbation from {f.name}"),
        )
        land = R_ref > 0                       # cells with reference precipitation, used as the land mask
        rows.append((age, float(T_ref[land].mean()),
                     float((T_ref + fields["dT_lo"])[land].mean()),
                     float((T_ref + fields["dT_hi"])[land].mean()),
                     float(R_ref[land].mean()),
                     float((R_ref * np.exp(fields["lnR_lo"]))[land].mean()),
                     float((R_ref * np.exp(fields["lnR_hi"]))[land].mean())))
        print(f"  [{age:>4} Ma] land-mean T {rows[-1][1]:6.2f} degC "
              f"({rows[-1][2]:+6.2f} .. {rows[-1][3]:+6.2f} band), "
              f"runoff {rows[-1][4]:.3f} m/yr ({rows[-1][5]:.3f} .. {rows[-1][6]:.3f})")
    with open(outd / "forcing_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["age_Ma", "T_land_ref_C", "T_land_lo_C", "T_land_hi_C",
                    "runoff_land_ref_m_yr", "runoff_land_lo_m_yr", "runoff_land_hi_m_yr"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} ages to {outd}")


# --------------------------------------------------------------------------------------
# index / fetch
# --------------------------------------------------------------------------------------

def cmd_index(a):
    """``index`` subcommand: report members per suite/variable and check the age mapping."""
    arch = (RemotePhanDAArchive() if str(a.phanda_dir).lower() in ("remote", "zenodo")
            else PhanDAArchive(a.phanda_dir))
    got = arch.available()
    exps = load_experiments()
    byage = experiments_by_age(exps)
    print(f"PhanDA archive at {arch.root}")
    missing = []
    for s in SUITES:
        row = []
        for v in VARS:
            n = got.get((s, v))
            row.append(f"{v}:{'-' if n is None else n}")
            if n is None:
                missing.append(f"{s}_{v}.zip")
            elif n != 545:                    # 109 ages x 5 sub-means
                print(f"  WARNING {s}/{v}: {n} members, expected 545", file=sys.stderr)
        print(f"  {s:<18} {MODEL_VERSION[s]:<9} {'  '.join(row)}")
    ages = sorted(byage)
    print(f"\n{len(ages)} plate ages {ages[0]}..{ages[-1]} Ma "
          f"(step {ages[1]-ages[0]}); matches the 0.1 deg basegrid age set: "
          f"{ages == list(range(0, 545, 5))}")
    for age in (0, 240, 540):
        if age in byage:
            c = {s: e.co2_ppm for s, e in byage[age].items()}
            span = np.log2(max(c.values()) / min(c.values()))
            print(f"  {age:>4} Ma  CO2 {min(c.values()):.0f}-{max(c.values()):.0f} ppm "
                  f"({span:.1f} doublings across suites)")
    if missing:
        print("\nMISSING (fetch these):", " ".join(sorted(set(missing))), file=sys.stderr)


def cmd_fetch(a):
    """``fetch`` subcommand: download the tas/pr zips and ExperimentInfo.xlsx from Zenodo.

    Files already present at the expected size are skipped; each download is checked
    against the checksum in the Zenodo record.
    """
    import urllib.request
    dest = Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORD}") as r:
        rec = json.load(r)
    want = {f"{s}_{v}.zip" for s in SUITES for v in VARS} | {"ExperimentInfo.xlsx"}
    files = [f for f in rec.get("files", []) if f["key"] in want]
    total = sum(f["size"] for f in files)
    print(f"{len(files)} files, {total/1e9:.1f} GB -> {dest}")
    if a.dry_run:
        for f in sorted(files, key=lambda f: f["key"]):
            print(f"  {f['key']:<28} {f['size']/1e6:8.0f} MB")
        return
    for f in sorted(files, key=lambda f: f["key"]):
        out = dest / f["key"]
        if out.exists() and out.stat().st_size == f["size"]:
            print(f"  have {f['key']}")
            continue
        print(f"  get  {f['key']} ({f['size']/1e6:.0f} MB) ...", flush=True)
        urllib.request.urlretrieve(ZENODO_FILES.format(name=f["key"]), out)
        algo, _, want_sum = f["checksum"].partition(":")
        h = hashlib.new(algo)
        with open(out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != want_sum:
            raise SystemExit(f"checksum mismatch for {f['key']}")
    print("done")


# --------------------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------------------

def _synthetic_truth(nlat=NLAT, nlon=NLON, seed=0):
    """Known lambda_T [K/doubling], kappa_P, base temperature T0 [K] and base
    precipitation P0 [mm/yr, not the floor of the same name] on the HadCM3L grid,
    plus a seeded RNG, for the selftest."""
    rng = np.random.default_rng(seed)
    lat, lon = hadcm3l_axes()
    LA = lat[:, None] * np.ones((1, nlon))
    # polar-amplified warming, drying in the subtropics and wetting at the equator/poles
    lam_T = 2.0 + 3.0 * (np.abs(LA) / 90.0) ** 2
    kap_P = 0.12 * np.cos(np.deg2rad(2.2 * LA)) - 0.02
    T0 = 300.0 - 45.0 * (np.abs(LA) / 90.0) ** 1.6
    P0 = 1800.0 * np.exp(-((LA / 22.0) ** 2)) + 250.0
    return lat, lon, lam_T, kap_P, T0, P0, rng


def cmd_selftest(a):
    """``selftest`` subcommand: synthetic checks of every step; returns 0 if all pass.

    Builds eight synthetic suites at known CO2 with imposed slopes, internal noise
    and inter-version biases, and checks that the fit recovers them; then checks
    regridding (accuracy, seam continuity, longitude georeferencing), envelope
    ordering and packing, the spread of Monte-Carlo draws, and the experiment table.
    """
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    print("1. WLS slope recovery on synthetic suites")
    lat, lon, lam_T, kap_P, T0, P0, rng = _synthetic_truth()
    co2 = {"scotese_solara": 276.01, "scotese_2co2a": 560.0, "scotese_4co2a": 1120.0,
           "scotese_spinupa": 800.0, "scotese_02": 1500.0,
           "scotese_06": 1500.0, "scotese_07": 1500.0, "scotese_08": 1800.0}
    ref = co2["scotese_spinupa"]
    sig_int_T, sig_int_lnP = 0.25, 0.03
    version_bias = {"Original": 0.0, "UpdateA": 0.6, "UpdateB": -0.4}   # K, structural
    mean_T, mean_lnP, sd_T, sd_lnP = {}, {}, {}, {}
    for s, c in co2.items():
        r = np.log2(c / ref)
        trueT = T0 + lam_T * r + version_bias[MODEL_VERSION[s]]
        truelnP = np.log(P0 + PRECIP_FLOOR_MM_YR) + kap_P * r
        Ts = trueT[None] + rng.normal(0, sig_int_T, (5,) + trueT.shape)
        Ps = truelnP[None] + rng.normal(0, sig_int_lnP, (5,) + truelnP.shape)
        mean_T[s], sd_T[s] = Ts.mean(0), Ts.std(0, ddof=1)
        mean_lnP[s], sd_lnP[s] = Ps.mean(0), Ps.std(0, ddof=1)

    used = FIT_SUITES_DEFAULT
    x = np.array([np.log2(co2[s] / ref) for s in used])
    lamhat, _, sfitT, r2T = wls_slope(x, np.stack([mean_T[s] for s in used]),
                                      1.0 / np.stack([sd_T[s] for s in used]) ** 2)
    kaphat, _, _, _ = wls_slope(x, np.stack([mean_lnP[s] for s in used]),
                                1.0 / np.stack([sd_lnP[s] for s in used]) ** 2)
    eT = float(np.abs(lamhat - lam_T).mean())
    eP = float(np.abs(kaphat - kap_P).mean())
    check("lambda_T recovered", eT < 0.06, f"mean |err| {eT:.4f} K/doubling")
    check("kappa_P recovered", eP < 0.01, f"mean |err| {eP:.5f} /doubling")
    check("R2 near 1 where linear", np.nanmedian(r2T) > 0.99, f"median R2 {np.nanmedian(r2T):.4f}")
    check("sigma_fit ~ sigma_int/sqrt(5)", abs(float(np.mean(sfitT)) - sig_int_T / np.sqrt(5)) < 0.05,
          f"{float(np.mean(sfitT)):.3f} vs {sig_int_T/np.sqrt(5):.3f} K")

    print("2. structural term recovers the imposed inter-version bias")
    groups = {}
    for s in co2:
        adj = mean_T[s] - lamhat * np.log2(co2[s] / ref)
        groups.setdefault(MODEL_VERSION[s], []).append(adj)
    gm = np.stack([np.mean(np.stack(v), 0) for v in groups.values()])
    sig_str = float(gm.std(0, ddof=1).mean())
    expect = float(np.std(list(version_bias.values()), ddof=1))
    check("sigma_struct", abs(sig_str - expect) < 0.1, f"{sig_str:.3f} vs imposed {expect:.3f} K")

    print("3. regridding")
    blat, blon = basegrid_axes()
    f = np.cos(np.deg2rad(lat))[:, None] * np.cos(np.deg2rad(lon))[None, :]
    g = regrid_bilinear(f, lat, lon, blat, blon)
    truth = np.cos(np.deg2rad(blat))[:, None] * np.cos(np.deg2rad(blon))[None, :]
    check("shape", g.shape == (1801, 3601), str(g.shape))
    check("bilinear accuracy on a smooth field", np.abs(g - truth).max() < 0.02,
          f"max |err| {np.abs(g-truth).max():.4f}")
    seam = np.abs(g[:, 0] - g[:, -1]).max()
    check("longitude seam is continuous", seam < 1e-5, f"max |f(-180)-f(180)| {seam:.2e}")
    # asymmetric in longitude: catches a half-globe roll, which cos(lon) alone would not
    fa = np.ones((NLAT, 1)) * np.sin(np.deg2rad(lon - 37.0))[None, :]
    ga = regrid_bilinear(fa, lat, lon, blat, blon)
    ta = np.ones((1801, 1)) * np.sin(np.deg2rad(blon - 37.0))[None, :]
    check("longitude georeferencing (no 180 deg roll)", np.abs(ga - ta).max() < 0.01,
          f"max |err| {np.abs(ga-ta).max():.4f}")
    check("hadcm3l lon axis is the archive's tas/pr axis", abs(lon[0] + 180.0) < 1e-9,
          f"lon[0] = {lon[0]}")
    const = regrid_bilinear(np.full((NLAT, NLON), 3.0), lat, lon, blat, blon)
    check("constants preserved", np.allclose(const, 3.0), "")

    print("4. envelope monotonicity and packing")
    d = dict(age=np.array([100]), lat=lat.astype(np.float32), lon=lon.astype(np.float32),
             co2_ref=np.array([ref], dtype=np.float32),
             lambda_T=lam_T[None].astype(np.float32), kappa_P=kap_P[None].astype(np.float32),
             sigma_int_T=np.full((1,) + lam_T.shape, sig_int_T, np.float32),
             sigma_fit_T=np.zeros((1,) + lam_T.shape, np.float32),
             sigma_struct_T=np.full((1,) + lam_T.shape, expect, np.float32),
             sigma_int_lnP=np.full((1,) + lam_T.shape, sig_int_lnP, np.float32),
             sigma_fit_lnP=np.zeros((1,) + lam_T.shape, np.float32),
             sigma_struct_lnP=np.full((1,) + lam_T.shape, 0.01, np.float32),
             meta=np.bytes_(json.dumps({"precip_floor_mm_yr": PRECIP_FLOOR_MM_YR})))
    emu = ClimateEmulator(d)
    dT_lo, rs_lo = emu.perturbation(100, ref / 2, z=-1.645, coarse=True)
    dT_c, rs_c = emu.perturbation(100, ref, z=0.0, coarse=True)
    dT_hi, rs_hi = emu.perturbation(100, ref * 2, z=+1.645, coarse=True)
    check("dT ordered lo < central < hi", (dT_lo < dT_c).all() and (dT_c < dT_hi).all(), "")
    check("central CO2 with z=0 gives zero dT", np.allclose(dT_c, 0.0, atol=1e-5), "")
    check("runoff scale is 1 at the reference", np.allclose(rs_c, 1.0, atol=1e-4), "")
    check("doubling CO2 gives lambda_T", np.allclose(emu.perturbation(100, ref*2, coarse=True)[0],
                                                     lam_T, atol=1e-3), "")
    e2 = emu.perturbation(100, ref * 2, runoff_elasticity=2.0, coarse=True)[1]
    e1 = emu.perturbation(100, ref * 2, runoff_elasticity=1.0, coarse=True)[1]
    check("runoff elasticity squares the ratio", np.allclose(e2, e1 ** 2, rtol=1e-5), "")
    q, sc, off = _pack_int16(dT_hi, 0.01)
    err = float(np.abs(q.astype(np.float32) * sc + off - dT_hi).max())
    check("int16 packing round-trip <= 0.01 K", err <= 0.01 + 1e-6, f"max err {err:.5f} K")

    print("5. Monte-Carlo draws")
    rng2 = np.random.default_rng(0)
    ms = [emu.draw(100, ref, rng2, coarse=True)[0].mean() for _ in range(400)]
    sT, _ = emu.sigma_total(100)
    check("draw spread ~ sigma_total", abs(np.std(ms) - float(sT.mean())) < 0.15,
          f"{np.std(ms):.3f} vs {float(sT.mean()):.3f} K")
    check("draws are unbiased", abs(np.mean(ms)) < 0.1, f"mean {np.mean(ms):+.3f} K")

    print("6. experiment table")
    exps = load_experiments()
    byage = experiments_by_age(exps)
    check("872 experiments", len(exps) == 872, str(len(exps)))
    check("109 plate ages 0..540 step 5", sorted(byage) == list(range(0, 545, 5)), "")
    span = [np.log2(max(e.co2_ppm for e in v.values()) / min(e.co2_ppm for e in v.values()))
            for v in byage.values()]
    check("every age spans >= 1.5 CO2 doublings", min(span) >= 1.5, f"min {min(span):.2f}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------

def main(argv=None):
    """Command-line entry point; see the module docstring for the subcommands."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("fetch", help="download the tas + pr archives from Zenodo (~6.5 GB)")
    q.add_argument("--dest", required=True)
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_fetch)

    q = sub.add_parser("index", help="report what is present and check the age mapping")
    q.add_argument("--phanda-dir", required=True,
                   help="directory of the archive, or 'remote' to stream from Zenodo")
    q.set_defaults(func=cmd_index)

    q = sub.add_parser("fit", help="fit per-cell lambda_T / kappa_P and the sigma terms")
    q.add_argument("--phanda-dir", required=True,
                   help="directory of the archive, or 'remote' to stream the members "
                        "straight out of the Zenodo zips over HTTP (no download, no disk)")
    q.add_argument("--checkpoint-dir",
                   help="per-age checkpoints (default: <out>.ckpt). Completed ages are "
                        "reloaded instead of refetched, so an interrupted run resumes.")
    q.add_argument("--time-budget", type=float,
                   help="stop after this many seconds and write what is done; re-run to "
                        "continue. Use to drive a long remote fit in chunks.")
    q.add_argument("--workers", type=int, default=6,
                   help="concurrent range requests when streaming (default 12)")
    q.add_argument("--out", default="phanda_sensitivity.npz")
    q.add_argument("--ages", nargs="*", help="plate ages to fit (default: all 109)")
    q.add_argument("--fit-suites", help=f"comma list (default: {','.join(FIT_SUITES_DEFAULT)})")
    q.add_argument("--reference-suite", default=REFERENCE_SUITE,
                   help="suite whose CO2 defines the reference point (default: the "
                        "Valdes 2021 smoothed-CO2 run that supplies our forcing)")
    q.add_argument("--precip-floor", type=float, default=PRECIP_FLOOR_MM_YR,
                   help="P0 in ln(P + P0), mm/yr")
    q.add_argument("--weighting", choices=["none", "internal"], default="internal")
    q.add_argument("--struct-tol", type=float, default=0.5,
                   help="doublings of CO2 extrapolation allowed when normalising a suite "
                        "for the structural term (default 0.5; excludes scotese_08 at ages "
                        "where its GMST-tuned CO2 is far from the fitted range)")
    q.set_defaults(func=cmd_fit)

    for name, fn, extra in (("envelope", cmd_envelope, True), ("ensemble", cmd_ensemble, True)):
        q = sub.add_parser(name, help=f"{name} products on the basegrid")
        q.add_argument("--sensitivity", default="phanda_sensitivity.npz")
        q.add_argument("--out-dir", required=True)
        q.add_argument("--ages", nargs="*")
        q.add_argument("--envelope", choices=["csv", "factor", "records"], default="factor")
        q.add_argument("--envelope-csv", help="age,co2_lo,co2_hi")
        q.add_argument("--envelope-factor", type=float, default=1.5,
                       help="multiplicative pCO2 bracket when --envelope factor")
        q.add_argument("--envelope-data-dir", help="dir holding foster.npy / judd.npy")
        q.add_argument("--runoff-elasticity", type=float, default=1.0,
                       help="epsilon in R/R_ref = (P/P_ref)**epsilon; 1 = pass-through "
                            "(default), ~2 = observed runoff elasticity")
        q.add_argument("--sigma-terms", default="int,fit,struct",
                       help="which sigma terms enter the band, comma separated from "
                            "int,fit,struct; empty string = CO2 term only")
        q.add_argument("--runoff-scale-clip", type=float, nargs=2, default=[0.2, 5.0],
                       metavar=("LO", "HI"),
                       help="bound the runoff scaling factor; the clipped fraction is "
                            "recorded in the output and printed")
        if name == "envelope":
            q.add_argument("--sigma-multiplier", type=float, default=1.645,
                           help="z in +/- z*sigma_total (1.645 = 5-95%%; 0 = CO2 term only)")
            q.add_argument("--emit", choices=["coarse", "full"], default="coarse",
                           help="'coarse' = native 96x73 (default, ~30 kB/age; expand with "
                                "regrid_bilinear), 'full' = 0.1 deg basegrid (~20 MB/age)")
        else:
            q.add_argument("--n", type=int, default=200)
            q.add_argument("--seed", type=int, default=0)
            q.add_argument("--white-noise", action="store_true",
                           help="per-cell noise instead of a spatially coherent draw "
                                "(will cancel under global integration)")
        q.set_defaults(func=fn)

    q = sub.add_parser("apply", help="add the perturbations to the real HadCM3 forcing")
    q.add_argument("--envelope-dir", required=True, help="output of the 'envelope' command")
    q.add_argument("--temp-dir", required=True, help="valdes_temperature/ (<age>Ma.npz)")
    q.add_argument("--erodep-dir", required=True, help="surrogate_erodep/ (erodep_<age>Ma.npz)")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--ages", nargs="*")
    q.set_defaults(func=cmd_apply)

    q = sub.add_parser("selftest", help="synthetic round-trip of every step")
    q.set_defaults(func=cmd_selftest)

    a = p.parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())

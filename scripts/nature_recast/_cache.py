"""One-file cache of the per-age land-only fields (written by build_cache.py).

The cache is a single uncompressed zip archive holding one .npz member per age, so that
building it writes one file and reading an age costs one sequential read. Its location is
`_config.CACHE` (environment variable NATURE_CACHE).

Each member, `cache_<age>Ma.npz` with the age zero-padded to three digits, holds the cells that
are land on either palaeoelevation surface (see build_cache.py for the full list of arrays):
`idx` (flat indices of those cells in the full 0.1-degree grid), `shape` (1801, 3601), and per cell
the erosion-deposition rates (mm/yr, negative = erosion), emergent masks, precipitation (m/yr),
temperature (deg C), runoff ratio, palaeoelevations (m) and cell area (m2).
"""
import io, zipfile
import numpy as np
import _config as cfg

# the archive is opened once per process and kept open for later reads
_zf = None


def member(age):
    """Name of the archive member of one age, e.g. cache_005Ma.npz."""
    return f'cache_{int(age):03d}Ma.npz'


def ages_present():
    """Set of ages (Ma) already in the archive; empty if the archive does not exist."""
    if not cfg.CACHE.exists():
        return set()
    with zipfile.ZipFile(cfg.CACHE) as z:
        # characters 6-8 of 'cache_<age>Ma.npz' hold the zero-padded age
        return {int(n[6:9]) for n in z.namelist()}


def load(age):
    """Fields of one age as a dict of arrays."""
    global _zf
    if _zf is None:
        if not cfg.CACHE.exists():
            raise FileNotFoundError(f'{cfg.CACHE} not found: run build_cache.py first')
        _zf = zipfile.ZipFile(cfg.CACHE)
    with np.load(io.BytesIO(_zf.read(member(age)))) as d:
        return {k: d[k] for k in d.files}


def append(_age, **arrays):
    """Add the arrays of one age to the archive as one stored (uncompressed) .npz member."""
    buf = io.BytesIO(); np.savez(buf, **arrays)
    with zipfile.ZipFile(cfg.CACHE, 'a', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr(member(_age), buf.getvalue())

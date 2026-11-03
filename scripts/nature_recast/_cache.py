"""One-file cache of the per-age land-only fields (written by build_cache.py).

The cache is a single uncompressed zip archive holding one .npz member per age, so that
building it writes one file and reading an age costs one sequential read."""
import io, zipfile
import numpy as np
import _config as cfg

_zf = None


def member(age):
    return f'cache_{int(age):03d}Ma.npz'


def ages_present():
    if not cfg.CACHE.exists():
        return set()
    with zipfile.ZipFile(cfg.CACHE) as z:
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
    buf = io.BytesIO(); np.savez(buf, **arrays)
    with zipfile.ZipFile(cfg.CACHE, 'a', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr(member(_age), buf.getvalue())

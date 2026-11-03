"""Path resolution for the neurolem_CO2 paper scripts.

Two data tiers:

- **Small derived series** (a few kB) are vendored into the repo under
  ``neurolem_CO2/data/`` and are git-tracked, so the light-weight analyses
  (pCO2 bootstrap, Fig S2) reproduce from a fresh clone with no download.
- **Large grids and the surrogate weights** live in the companion Zenodo data
  archive, ``neurolem_CO2_zenodo_data``. They are intentionally excluded from
  this Git repository. Download and extract the archive inside the NeuroLEM
  working folder as a sibling of this repo. Override that default location with
  the ``NEUROLEM_CO2_DATA`` environment variable.

Layout expected inside the Zenodo archive::

    neurolem_CO2_zenodo_data/
        surrogate_erodep/   erodep_<age>Ma.npz  -- multilayer per age:
                              ero_scotese, ero_zhou (NeuroLEM surrogate),
                              zpaleo_scotese, zpaleo_zhou, delta (paleo-elevation),
                              rain (runoff forcing), mask_scotese, mask_zhou, lat, lon, age
        gospl_salles/       salles_<age>Ma.npz  -- goSPL reference (target) + inputs
        valdes_temperature/ <age>Ma.npz         -- HadCM3 temperature (int16 + scale/offset)
        consolidated/       mapdata_pygmt.npz, mapdata_cal.npz, scfsf_maps.npz
        dem_scotese_adjusted/ rock-chemistry-adjusted 1-degree palaeo-DEMs (craton-excluded correction, 400 km support) used by apply_dem.py
        surrogate_model/    best.pt             -- NeuroLEM surrogate weights
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[1]              # neurolem_CO2/
DATA = REPO / "data"                 # vendored small series (git-tracked)


def zenodo_root() -> Path:
    """Root of the extracted Zenodo grid archive."""
    env = os.environ.get("NEUROLEM_CO2_DATA")
    if env:
        return Path(env).expanduser()
    return REPO.parent / "neurolem_CO2_zenodo_data"


FIGURES = REPO / "figures"                        # canonical paper figures (output)
PARAMS = DATA / "bufe_2024_params.json"           # Bufe (2024) regional fits (vendored)

ZENODO = zenodo_root()
ERODEP_DIR = ZENODO / "surrogate_erodep"          # multilayer erosion/elevation/forcing
GOSPL_DIR = ZENODO / "gospl_salles"               # goSPL reference + inputs
TEMP_DIR = ZENODO / "valdes_temperature"          # HadCM3 temperature
CONSOLIDATED = ZENODO / "consolidated"            # precomputed map/analysis grids
MODEL = ZENODO / "surrogate_model" / "best.pt"    # surrogate weights

# Calibration constants (see bufe_co2_from_dem.py / Methods)
ERATE_SCALE = 2.0     # surrogate erate -> denudation (mm/yr), sediment-flux calibrated
F_SIL = 2.4           # silicate amplitude factor (-> ~0.10 GtC/yr modern silicate sink)
F_SULF = 0.2          # sulfide amplitude factor (-> ~0.02 GtC/yr modern pyrite source)
BUFE_OPTIMUM = (0.04, 0.11)   # mm/yr, Bufe CO2-drawdown optimum band


def require(path: Path) -> Path:
    """Return *path* if it exists, else raise a helpful error pointing at Zenodo."""
    if not Path(path).exists():
        raise SystemExit(
            f"Missing data: {path}\n"
            "Large grids are not in the git repo. Download the companion Zenodo\n"
            "archive, extract it as 'neurolem_CO2_zenodo_data/' next to this repo,\n"
            "or set NEUROLEM_CO2_DATA to its location. See the repo README."
        )
    return Path(path)

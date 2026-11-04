"""Path resolution and shared calibration constants for the scripts in this folder.

Imported by the scripts in this folder (for example make_main_map_panels.py and
analyse_gadi_validation.py) so that every script finds the repository data and
the companion Zenodo archive in the same place; ``nature_recast/_config.py``
resolves the same layout for the analysis in ``nature_recast/``.  No computation
happens here.

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
        surrogate_erodep/   erodep_<age>Ma.npz  -- multilayer per age, written by apply_dem.py:
                              ero_scotese, ero_zhou (NeuroLEM erosion-deposition rate,
                              mm/yr, negative = erosion; reference and rock-chemistry-
                              adjusted surface), zpaleo_scotese, zpaleo_zhou (elevation, m),
                              delta (palaeoelevation increment, m), rain (precipitation
                              forcing, m/yr), mask_scotese, mask_zhou (emergent land),
                              lat, lon, age
        gospl_salles/       salles_<age>Ma.npz  -- goSPL reference (target) + inputs
                              (elevation, precipitation) on the 0.1-degree grid
        valdes_temperature/ <age>Ma.npz         -- HadCM3L temperature (Valdes et al., 2021),
                              stored as int16 with scale/offset
        consolidated/       mapdata_pygmt.npz, mapdata_cal.npz, scfsf_maps.npz
        dem_scotese_adjusted/ rock-chemistry-adjusted 1-degree palaeo-DEMs
                              (<age>Ma_corrected_SW.nc: craton samples excluded, 400 km
                              support, residual reach 250-500 km, samples on PALEOMAP sea
                              dropped) used by apply_dem.py
        surrogate_model/    best.pt             -- NeuroLEM surrogate weights

The names ``scotese`` and ``zhou`` denote the reference PALEOMAP surface and the
surface adjusted with chemical mohometry (Zhou et al., 2025); ``salles`` denotes
the published goSPL simulations (Salles et al., 2023).
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
TEMP_DIR = ZENODO / "valdes_temperature"          # HadCM3L temperature
CONSOLIDATED = ZENODO / "consolidated"            # precomputed map/analysis grids
MODEL = ZENODO / "surrogate_model" / "best.pt"    # surrogate weights

# Calibration constants of the single-fit weathering calculation in bufe_co2_from_dem.py
# (median regional fit of Bufe et al. (2024), one amplitude per phase), used by the
# scripts in this folder.  The analysis in nature_recast/ calibrates its own amplitudes
# for each of the three regional parameter sets (Methods, "Parameter sets").
ERATE_SCALE = 2.0     # surrogate erate -> denudation (mm/yr), sediment-flux calibrated
F_SIL = 2.4           # silicate amplitude factor (-> ~0.10 GtC/yr modern silicate sink)
F_SULF = 0.1          # sulfide amplitude factor (-> ~0.02 GtC/yr modern pyrite source at 1 mol C per mol sulfate)
BUFE_OPTIMUM = (0.04, 0.11)   # mm/yr, erosion-rate band of maximum net CO2 drawdown (Bufe et al., 2024)


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

"""
ce2coast_config.py
==================
Shared path configuration for CE2COAST ML pipeline.
Import this at the top of every script.

Required environment variables (set in ~/.bashrc or .env):
    CE2COAST_BASE    — root data directory
    CE2COAST_COLLOC  — ML collocation outputs (default: BASE/ML_colloc)
    CE2COAST_ROMS    — ROMS AVG files directory
    CE2COAST_VALID   — validation / satellite obs directory
    CE2COAST_INSITU  — in-situ BGC profile directory

Example ~/.bashrc additions:
    export CE2COAST_BASE=/scratch/ulg/mast/eivanov
    export CE2COAST_COLLOC=$CE2COAST_BASE/ML_colloc
    export CE2COAST_ROMS=$CE2COAST_BASE/Output/CE2COAST_2006
    export CE2COAST_VALID=$CE2COAST_BASE/Data_for_validation
    export CE2COAST_INSITU=$CE2COAST_VALID/insitu_bgc/profiles
"""

import os
import sys
from pathlib import Path


def _require(varname: str, default_relative: str = None) -> Path:
    """
    Get an environment variable as a Path.
    If not set and no default, print a helpful message and exit.
    """
    val = os.environ.get(varname)
    if val:
        return Path(val)
    if default_relative and 'CE2COAST_BASE' in os.environ:
        return Path(os.environ['CE2COAST_BASE']) / default_relative
    print(f'\n[ce2coast_config] ERROR: ${varname} is not set.')
    print(f'Add to ~/.bashrc:')
    print(f'  export CE2COAST_BASE=/your/data/root')
    print(f'  export CE2COAST_COLLOC=$CE2COAST_BASE/ML_colloc')
    print(f'  export CE2COAST_ROMS=$CE2COAST_BASE/Output/CE2COAST_2006')
    print(f'  export CE2COAST_VALID=$CE2COAST_BASE/Data_for_validation')
    print(f'  export CE2COAST_INSITU=$CE2COAST_VALID/insitu_bgc/profiles')
    sys.exit(1)


# ── Resolved paths ────────────────────────────────────────────
BASE_DIR    = _require('CE2COAST_BASE')
COLLOC_DIR  = _require('CE2COAST_COLLOC',  'ML_colloc')
ROMS_DIR    = _require('CE2COAST_ROMS',    'Output/CE2COAST_2006')
VALID_DIR   = _require('CE2COAST_VALID',   'Data_for_validation')
INSITU_DIR  = _require('CE2COAST_INSITU',  'Data_for_validation/insitu_bgc/profiles')

# ── Derived paths (no env var needed) ────────────────────────
FIG_DIR     = COLLOC_DIR  / 'figures'
KALMAN_DIR  = COLLOC_DIR  / 'kalman'

# ── File patterns ─────────────────────────────────────────────
ROMS_AVG_PATTERN = str(ROMS_DIR / 'Hindcast_CE2COAST_AVG_{year}_2c_atm3.nc')
ROMS_BIO_PATTERN = str(ROMS_DIR / 'Hindcast_CE2COAST_AVG_{year}_2c_atm3.nc')
ROMS_GRID_FILE   = str(ROMS_DIR / 'Hindcast_CE2COAST_AVG_2010_2c_atm3.nc')

SST_OBS_FILE = str(VALID_DIR / 'Validation' /
                   'Satellite_validation_temp_{year}.nc')
CHL_OBS_FILE = str(VALID_DIR /
                   'cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_'
                   'P1M_1698156332793.nc')


if __name__ == '__main__':
    print('CE2COAST path configuration:')
    print(f'  BASE_DIR   : {BASE_DIR}')
    print(f'  COLLOC_DIR : {COLLOC_DIR}')
    print(f'  ROMS_DIR   : {ROMS_DIR}')
    print(f'  VALID_DIR  : {VALID_DIR}')
    print(f'  INSITU_DIR : {INSITU_DIR}')
    print(f'  FIG_DIR    : {FIG_DIR}')
    print(f'  KALMAN_DIR : {KALMAN_DIR}')

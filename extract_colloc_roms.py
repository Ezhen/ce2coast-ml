"""
extract_colloc_roms.py
======================
Collocate CE2COAST ROMS monthly AVG outputs (SST, Chl) against
CMEMS satellite observations at ROMS ocean grid points.

Strategy: interpolate obs (denser grid) onto ROMS rho-points.
One record per ROMS ocean point per month.

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from ce2coast_config import (
    COLLOC_DIR, ROMS_DIR, VALID_DIR, INSITU_DIR,
    FIG_DIR, KALMAN_DIR, ROMS_AVG_PATTERN, ROMS_BIO_PATTERN,
    ROMS_GRID_FILE, SST_OBS_FILE, CHL_OBS_FILE,
)
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

ROMS_AVG_PATTERN = (
    str(ROMS_DIR) + '/'
    'Hindcast_CE2COAST_AVG_{year}_2c_atm3.nc'
)
ROMS_BIO_PATTERN = (
    str(ROMS_DIR) + '/'
    'Hindcast_CE2COAST_AVG_{year}_2c_atm3.nc'
)
SST_OBS_FILE = (
    str(VALID_DIR / 'Validation') + '/'
    'Satellite_validation_temp_{year}.nc'
)
CHL_OBS_FILE = (
    str(VALID_DIR) + '/'
    'cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M_'
    '1698156332793.nc'
)

OUT_DIR = COLLOC_DIR

# TEST: 2010 only — change to list(range(2010, 2021)) for full run
YEARS = list(np.arange(2010,2021))

# ─────────────────────────────────────────────────────────────
# ROMS TIME DECODER
# Pure Python — bypasses pandas/numpy nanosecond architecture
# ─────────────────────────────────────────────────────────────

ROMS_EPOCH_DT = datetime(1858, 11, 17)


def roms_time_to_datetime(ds):
    """
    Let xarray decode ocean_time via CF convention, then convert
    to plain Python datetimes via pandas. Bypasses timedelta overflow.
    """
    raw_vals   = ds['ocean_time'].values.flatten()
    timestamps = pd.to_datetime(raw_vals).to_pydatetime().tolist()
    return timestamps


# ─────────────────────────────────────────────────────────────
# ROMS GRID UTILITIES
# ─────────────────────────────────────────────────────────────

def build_roms_grid(ds_roms):
    """Extract ROMS rho-point coordinates and ocean point indices."""
    lat       = ds_roms['lat_rho'].values
    lon       = ds_roms['lon_rho'].values
    mask      = ds_roms['mask_rho'].values
    lat_flat  = lat.ravel()
    lon_flat  = lon.ravel()
    ocean_idx = np.where(mask.ravel() == 1)[0]
    print(f'  ROMS grid: {lat.shape} | ocean points: {len(ocean_idx):,}')
    print(f'  ROMS lat: {lat_flat[ocean_idx].min():.2f} -> {lat_flat[ocean_idx].max():.2f}')
    print(f'  ROMS lon: {lon_flat[ocean_idx].min():.2f} -> {lon_flat[ocean_idx].max():.2f}')
    return lat_flat, lon_flat, ocean_idx


def roms_field_at_ocean(ds, varname, t_idx, ocean_idx,
                        surface=True, regrid_to_rho=False):
    """Extract a ROMS field at ocean grid points only."""
    if regrid_to_rho:
        raw       = ds[varname].isel(ocean_time=t_idx).values
        rho_shape = ds['lat_rho'].shape
        out       = np.zeros(rho_shape)
        if raw.shape[1] == rho_shape[1] - 1:       # xi-staggered (sustr)
            out[:, 1:-1] = 0.5 * (raw[:, :-1] + raw[:, 1:])
            out[:, 0]    = out[:, 1]
            out[:, -1]   = out[:, -2]
        elif raw.shape[0] == rho_shape[0] - 1:     # eta-staggered (svstr)
            out[1:-1, :] = 0.5 * (raw[:-1, :] + raw[1:, :])
            out[0, :]    = out[1, :]
            out[-1, :]   = out[-2, :]
        return out.ravel()[ocean_idx]
    elif surface:
        return ds[varname].isel(ocean_time=t_idx, s_rho=-1).values.ravel()[ocean_idx]
    else:
        return ds[varname].isel(ocean_time=t_idx).values.ravel()[ocean_idx]


# ─────────────────────────────────────────────────────────────
# LOAD SST OBSERVATIONS
# ─────────────────────────────────────────────────────────────

def load_sst_obs(year):
    """
    Load CMEMS satellite SST for one year.
    Confirmed: analysed_sst, units=kelvin, no scale factor,
               lat 48-59.6N, lon -5.8-10.8E, 366 daily timesteps.
    Returns xarray DataArray of 12 monthly means in degC.
    """
    fpath = SST_OBS_FILE.format(year=year)
    if not Path(fpath).exists():
        print(f'  [WARN] SST file not found: {fpath}')
        return None
    ds          = xr.open_dataset(fpath)
    sst         = ds['analysed_sst'].astype(np.float32) - 273.15
    sst_monthly = sst.resample(time='1MS').mean()
    print(f'  SST obs {year}: {len(sst_monthly.time)} monthly snapshots')
    return sst_monthly


# ─────────────────────────────────────────────────────────────
# LOAD CHL OBSERVATIONS
# ─────────────────────────────────────────────────────────────

def load_chl_obs(year):
    """
    Load CMEMS ocean colour Chl for one year from multi-year file.
    Already monthly (P1M). Returns DataArray (n_months, lat, lon).
    """
    if not Path(CHL_OBS_FILE).exists():
        print(f'  [WARN] Chl file not found: {CHL_OBS_FILE}')
        return None
    ds = xr.open_dataset(CHL_OBS_FILE)
    try:
        ds_year = ds.sel(time=str(year))
    except KeyError:
        print(f'  [WARN] No Chl obs for {year}')
        return None
    if ds_year.dims.get('time', 0) == 0:
        print(f'  [WARN] Empty Chl obs for {year}')
        return None
    chl_var = None
    for candidate in ['CHL', 'chl', 'chlorophyll', 'Chl', 'chla']:
        if candidate in ds_year.data_vars:
            chl_var = candidate
            break
    if chl_var is None:
        print(f'  [WARN] Chl var not found. Available: {list(ds_year.data_vars)}')
        return None
    chl = ds_year[chl_var].astype(np.float32)
    chl = chl.where(chl > 0)
    print(f'  Chl obs {year}: {len(chl.time)} monthly snapshots')
    return chl


# ─────────────────────────────────────────────────────────────
# INTERPOLATE OBS GRID ONTO ROMS OCEAN POINTS
# ─────────────────────────────────────────────────────────────

def interp_obs_to_roms(obs_2d, obs_lat_1d, obs_lon_1d,
                       roms_lat_ocean, roms_lon_ocean):
    """Bilinear interpolation of a 2D obs field onto ROMS ocean points."""
    interp = RegularGridInterpolator(
        (obs_lat_1d, obs_lon_1d),
        obs_2d,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    return interp(np.column_stack([roms_lat_ocean, roms_lon_ocean]))


# ─────────────────────────────────────────────────────────────
# COLLOCATION — ONE YEAR
# ─────────────────────────────────────────────────────────────

def colloc_year(year):
    print(f'\n{"="*55}')
    print(f'[{year}] Starting collocation')
    print(f'{"="*55}')

    roms_avg_path = ROMS_AVG_PATTERN.format(year=year)
    roms_bio_path = ROMS_BIO_PATTERN.format(year=year)

    if not Path(roms_avg_path).exists():
        print(f'  [WARN] ROMS AVG not found: {roms_avg_path}')
        return None, None

    ds_avg = xr.open_dataset(roms_avg_path)
    ds_bio = xr.open_dataset(roms_bio_path) if Path(roms_bio_path).exists() else None
    if ds_bio is None:
        print('  [WARN] ROMS BIO not found — Chl skipped')

    lat_flat, lon_flat, ocean_idx = build_roms_grid(ds_avg)
    roms_lat_ocean = lat_flat[ocean_idx]
    roms_lon_ocean = lon_flat[ocean_idx]

    roms_times = roms_time_to_datetime(ds_avg)
    print(f'  ROMS: {roms_times[0].strftime("%Y-%m")} -> '
          f'{roms_times[-1].strftime("%Y-%m")} ({len(roms_times)} snapshots)')

    sst_monthly = load_sst_obs(year)
    chl_monthly = load_chl_obs(year)

    sst_lat = sst_monthly['lat'].values if sst_monthly is not None else None
    sst_lon = sst_monthly['lon'].values if sst_monthly is not None else None
    chl_lat = chl_monthly['lat'].values if chl_monthly is not None else None
    chl_lon = chl_monthly['lon'].values if chl_monthly is not None else None

    records_sst = []
    records_chl = []

    for t_idx, roms_time in enumerate(roms_times):
        mstr = roms_time.strftime('%Y-%m')
        print(f'\n  [{mstr}]', end=' ')

        # ── SST ─────────────────────────────────────────────
        if sst_monthly is not None:
            try:
                sst_snap = sst_monthly.sel(
                    time=mstr, method='nearest'
                ).values.astype(np.float32)
            except Exception as e:
                print(f'SST sel failed ({e})', end=' ')
                sst_snap = None

            if sst_snap is not None:
                sst_obs_at_roms = interp_obs_to_roms(
                    sst_snap, sst_lat, sst_lon,
                    roms_lat_ocean, roms_lon_ocean
                )
                sst_roms   = roms_field_at_ocean(ds_avg, 'temp',  t_idx, ocean_idx, surface=True)
                salt_roms  = roms_field_at_ocean(ds_avg, 'salt',  t_idx, ocean_idx, surface=True)
                zeta_roms  = roms_field_at_ocean(ds_avg, 'zeta',  t_idx, ocean_idx, surface=False)
                sustr_roms = roms_field_at_ocean(ds_avg, 'sustr', t_idx, ocean_idx, surface=False, regrid_to_rho=True)
                svstr_roms = roms_field_at_ocean(ds_avg, 'svstr', t_idx, ocean_idx, surface=False, regrid_to_rho=True)

                valid   = ~np.isnan(sst_obs_at_roms) & ~np.isnan(sst_roms)
                n_valid = int(valid.sum())
                print(f'SST={n_valid:,}', end=' ')

                for i in np.where(valid)[0]:
                    records_sst.append({
                        'time':         roms_time,
                        'lat':          float(roms_lat_ocean[i]),
                        'lon':          float(roms_lon_ocean[i]),
                        'sst_obs':      float(sst_obs_at_roms[i]),
                        'sst_roms':     float(sst_roms[i]),
                        'residual_sst': float(sst_obs_at_roms[i] - sst_roms[i]),
                        'salt_roms':    float(salt_roms[i]),
                        'zeta_roms':    float(zeta_roms[i]),
                        'sustr_roms':   float(sustr_roms[i]),
                        'svstr_roms':   float(svstr_roms[i]),
                        'month':        roms_time.month,
                        'year':         roms_time.year,
                    })

        # ── Chl ─────────────────────────────────────────────
        if chl_monthly is not None and ds_bio is not None:
            try:
                chl_snap = chl_monthly.sel(
                    time=mstr, method='nearest'
                ).values.astype(np.float32)
            except Exception as e:
                print(f'Chl sel failed ({e})', end=' ')
                chl_snap = None

            if chl_snap is not None:
                chl_obs_at_roms = interp_obs_to_roms(
                    chl_snap, chl_lat, chl_lon,
                    roms_lat_ocean, roms_lon_ocean
                )
                chl_roms = roms_field_at_ocean(
                    ds_bio, 'chlorophyll', t_idx, ocean_idx, surface=True)

                valid = (
                    ~np.isnan(chl_obs_at_roms) &
                    ~np.isnan(chl_roms) &
                    (chl_obs_at_roms > 0) &
                    (chl_roms > 0)
                )
                n_valid = int(valid.sum())
                print(f'Chl={n_valid:,}', end=' ')

                for i in np.where(valid)[0]:
                    co = float(chl_obs_at_roms[i])
                    cr = float(chl_roms[i])
                    records_chl.append({
                        'time':             roms_time,
                        'lat':              float(roms_lat_ocean[i]),
                        'lon':              float(roms_lon_ocean[i]),
                        'chl_obs':          co,
                        'chl_roms':         cr,
                        'residual_chl':     co - cr,
                        'log_residual_chl': np.log(co) - np.log(cr),
                        'month':            roms_time.month,
                        'year':             roms_time.year,
                    })

    ds_avg.close()
    if ds_bio is not None:
        ds_bio.close()

    df_sst = pd.DataFrame(records_sst) if records_sst else None
    df_chl = pd.DataFrame(records_chl) if records_chl else None
    return df_sst, df_chl


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sst, all_chl = [], []

    for year in YEARS:
        df_sst, df_chl = colloc_year(year)

        if df_sst is not None and len(df_sst) > 0:
            df_sst['time'] = pd.to_datetime(df_sst['time'])
            out = OUT_DIR / f'colloc_sst_{year}.parquet'
            df_sst.to_parquet(out, index=False)
            print(f'\n  SST {year}: {len(df_sst):,} rows -> {out.name}')
            print(f'    residual mean: {df_sst["residual_sst"].mean():.3f} degC')
            print(f'    residual std:  {df_sst["residual_sst"].std():.3f} degC')
            all_sst.append(df_sst)
        else:
            print(f'\n  [WARN] No SST records for {year}')

        if df_chl is not None and len(df_chl) > 0:
            df_chl['time'] = pd.to_datetime(df_chl['time'])
            out = OUT_DIR / f'colloc_chl_{year}.parquet'
            df_chl.to_parquet(out, index=False)
            print(f'  Chl {year}: {len(df_chl):,} rows -> {out.name}')
            print(f'    log-residual mean: {df_chl["log_residual_chl"].mean():.3f}')
            print(f'    log-residual std:  {df_chl["log_residual_chl"].std():.3f}')
            all_chl.append(df_chl)
        else:
            print(f'  [WARN] No Chl records for {year}')

    if len(all_sst) > 1:
        df_full = pd.concat(all_sst, ignore_index=True)
        out = OUT_DIR / f'colloc_sst_{YEARS[0]}_{YEARS[-1]}.parquet'
        df_full.to_parquet(out, index=False)
        print(f'\nSST full: {len(df_full):,} rows -> {out.name}')

    if len(all_chl) > 1:
        df_full = pd.concat(all_chl, ignore_index=True)
        out = OUT_DIR / f'colloc_chl_{YEARS[0]}_{YEARS[-1]}.parquet'
        df_full.to_parquet(out, index=False)
        print(f'Chl full: {len(df_full):,} rows -> {out.name}')

    print('\nDone.')


if __name__ == '__main__':
    main()

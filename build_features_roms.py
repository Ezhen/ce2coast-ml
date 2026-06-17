"""
build_features_roms.py
======================
Feature engineering on top of the collocation parquets.
Adds spatial, temporal, and lagged features before XGBoost training.

Inputs:
    colloc_sst_2010_2020.parquet
    colloc_chl_2010_2020.parquet

Outputs:
    features_sst_2010_2020.parquet
    features_chl_2010_2020.parquet

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from ce2coast_config import (
    COLLOC_DIR, ROMS_DIR, VALID_DIR, INSITU_DIR,
    FIG_DIR, KALMAN_DIR, ROMS_AVG_PATTERN, ROMS_BIO_PATTERN,
    ROMS_GRID_FILE, SST_OBS_FILE, CHL_OBS_FILE,
)
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
OUT_DIR    = COLLOC_DIR

# ROMS grid file — any AVG file works, grid is static
ROMS_GRID_FILE = (
    str(ROMS_DIR) + '/'
    'Hindcast_CE2COAST_AVG_2010_2c_atm3.nc'
)

# Chl outlier clip — linear space (mg/m3)
# Points beyond these are bloom extremes or artefacts
CHL_RESIDUAL_CLIP = 20.0   # |residual_chl| > 20 flagged

# SST outlier clip (degC)
SST_RESIDUAL_CLIP = 6.0    # |residual_sst| > 6 flagged

# ─────────────────────────────────────────────────────────────
# LOAD ROMS BATHYMETRY AND COAST DISTANCE
# ─────────────────────────────────────────────────────────────

def build_spatial_lookup(grid_file):
    """
    Extract from ROMS grid:
      - h       : bottom depth (m) at rho points
      - coast_dist : distance to nearest land point (degrees, approx)

    Returns a DataFrame indexed by (lat, lon) rounded to 4dp.
    """
    print('Building spatial lookup from ROMS grid...')
    ds   = xr.open_dataset(grid_file)

    lat  = ds['lat_rho'].values.ravel()
    lon  = ds['lon_rho'].values.ravel()
    h    = ds['h'].values.ravel()          # bottom depth (m)
    mask = ds['mask_rho'].values.ravel()   # 1=ocean, 0=land

    ocean_idx = np.where(mask == 1)[0]
    land_idx  = np.where(mask == 0)[0]

    lat_ocean = lat[ocean_idx]
    lon_ocean = lon[ocean_idx]
    h_ocean   = h[ocean_idx]

    # Distance to nearest land point (degrees)
    if len(land_idx) > 0:
        land_tree   = cKDTree(np.column_stack([lat[land_idx], lon[land_idx]]))
        coast_dist, _ = land_tree.query(
            np.column_stack([lat_ocean, lon_ocean]),
            workers=-1
        )
    else:
        coast_dist = np.full(len(ocean_idx), np.nan)

    spatial = pd.DataFrame({
        'lat':        np.round(lat_ocean, 4),
        'lon':        np.round(lon_ocean, 4),
        'depth_m':    h_ocean,
        'coast_dist': coast_dist,
    })

    ds.close()
    print(f'  Spatial lookup: {len(spatial):,} ocean points')
    print(f'  Depth range: {h_ocean.min():.1f} -> {h_ocean.max():.1f} m')
    print(f'  Coast dist range: {coast_dist.min():.3f} -> {coast_dist.max():.3f} deg')
    return spatial


# ─────────────────────────────────────────────────────────────
# TEMPORAL FEATURES
# ─────────────────────────────────────────────────────────────

def add_temporal_features(df):
    """
    Add cyclical encoding of month and day-of-year.
    Cyclical encoding preserves continuity (Dec->Jan).
    """
    df = df.copy()

    # Day of year from time column
    df['doy'] = pd.to_datetime(df['time']).dt.dayofyear

    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Cyclical doy encoding
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

    return df


# ─────────────────────────────────────────────────────────────
# LAGGED RESIDUAL FEATURE
# ─────────────────────────────────────────────────────────────

def add_lagged_residual(df, residual_col, lag_months=1):
    """
    Add lagged residual — residual at same location previous month.
    Captures persistence of systematic ROMS bias.

    Groups by (lat_round, lon_round), sorts by time, shifts by lag.
    NaN for first month of each location's record.
    """
    print(f'  Adding lag-{lag_months} residual...')
    df = df.copy()

    # Round coords for grouping key
    df['lat_r'] = np.round(df['lat'], 3)
    df['lon_r'] = np.round(df['lon'], 3)

    df = df.sort_values(['lat_r', 'lon_r', 'time']).reset_index(drop=True)

    lag_col = f'{residual_col}_lag{lag_months}'
    df[lag_col] = (
        df.groupby(['lat_r', 'lon_r'])[residual_col]
        .shift(lag_months)
    )

    # Drop helper columns
    df = df.drop(columns=['lat_r', 'lon_r'])

    n_valid = df[lag_col].notna().sum()
    print(f'  Lag residual valid: {n_valid:,} / {len(df):,} rows')
    return df


# ─────────────────────────────────────────────────────────────
# OUTLIER FLAGGING
# ─────────────────────────────────────────────────────────────

def flag_outliers(df, residual_col, clip_val):
    """
    Add boolean flag for extreme residuals.
    Does NOT remove rows — lets XGBoost decide, but flags them.
    """
    flag_col = f'{residual_col}_extreme'
    df[flag_col] = (df[residual_col].abs() > clip_val).astype(int)
    n_extreme = df[flag_col].sum()
    pct = 100 * n_extreme / len(df)
    print(f'  Extreme {residual_col}: {n_extreme:,} ({pct:.2f}%) flagged '
          f'(|residual| > {clip_val})')
    return df


# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT — TEMPORAL
# ─────────────────────────────────────────────────────────────

def add_split_column(df, test_years=(2019, 2020)):
    """
    Temporal train/test split.
    Test = last 2 years. Train = everything before.
    Never random split on time series.
    """
    df = df.copy()
    df['split'] = np.where(df['year'].isin(test_years), 'test', 'train')
    train_n = (df['split'] == 'train').sum()
    test_n  = (df['split'] == 'test').sum()
    print(f'  Split: train={train_n:,} ({100*train_n/len(df):.0f}%)'
          f' | test={test_n:,} ({100*test_n/len(df):.0f}%)'
          f' | test years={test_years}')
    return df


# ─────────────────────────────────────────────────────────────
# MAIN FEATURE BUILD
# ─────────────────────────────────────────────────────────────

def build_features(df, spatial, residual_col, clip_val, label):
    """Full feature engineering pipeline for one variable."""
    print(f'\n--- Building features: {label} ---')
    print(f'  Input rows: {len(df):,}')

    # ── Merge spatial features ──────────────────────────────
    print('  Merging spatial features...')
    df['lat_r'] = np.round(df['lat'], 4)
    df['lon_r'] = np.round(df['lon'], 4)
    spatial_r   = spatial.rename(columns={'lat': 'lat_r', 'lon': 'lon_r'})
    df = df.merge(spatial_r, on=['lat_r', 'lon_r'], how='left')
    df = df.drop(columns=['lat_r', 'lon_r'])

    n_missing_depth = df['depth_m'].isna().sum()
    if n_missing_depth > 0:
        print(f'  [WARN] {n_missing_depth:,} rows missing depth '
              f'(grid rounding tolerance) — filling with median')
        df['depth_m']    = df['depth_m'].fillna(df['depth_m'].median())
        df['coast_dist'] = df['coast_dist'].fillna(df['coast_dist'].median())

    # ── Temporal features ───────────────────────────────────
    print('  Adding temporal features...')
    df = add_temporal_features(df)

    # ── Lagged residual ─────────────────────────────────────
    df = add_lagged_residual(df, residual_col, lag_months=1)

    # ── Outlier flag ────────────────────────────────────────
    df = flag_outliers(df, residual_col, clip_val)

    # ── Train/test split ────────────────────────────────────
    df = add_split_column(df, test_years=(2019, 2020))

    print(f'  Output columns: {list(df.columns)}')
    print(f'  Output rows: {len(df):,}')
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build spatial lookup once ───────────────────────────
    spatial = build_spatial_lookup(ROMS_GRID_FILE)

    # ── SST ─────────────────────────────────────────────────
    print('\n' + '='*55)
    print('SST FEATURE BUILD')
    print('='*55)
    sst = pd.read_parquet(
        COLLOC_DIR / 'colloc_sst_2010_2020.parquet'
    )
    sst = build_features(
        sst, spatial,
        residual_col='residual_sst',
        clip_val=SST_RESIDUAL_CLIP,
        label='SST'
    )
    out_sst = OUT_DIR / 'features_sst_2010_2020.parquet'
    sst.to_parquet(out_sst, index=False)
    print(f'\n  Saved: {out_sst.name} ({out_sst.stat().st_size/1e6:.1f} MB)')

    # Summary statistics
    print(f'\n  SST residual by month:')
    print(sst.groupby('month')['residual_sst'].agg(['mean','std']).round(3))

    # ── Chl ─────────────────────────────────────────────────
    print('\n' + '='*55)
    print('CHL FEATURE BUILD')
    print('='*55)
    chl = pd.read_parquet(
        COLLOC_DIR / 'colloc_chl_2010_2020.parquet'
    )
    chl = build_features(
        chl, spatial,
        residual_col='log_residual_chl',
        clip_val=3.0,   # log-space: |log(obs/roms)| > 3 → ratio > 20x
        label='Chl'
    )
    out_chl = OUT_DIR / 'features_chl_2010_2020.parquet'
    chl.to_parquet(out_chl, index=False)
    print(f'\n  Saved: {out_chl.name} ({out_chl.stat().st_size/1e6:.1f} MB)')

    print(f'\n  Chl log-residual by month:')
    print(chl.groupby('month')['log_residual_chl'].agg(['mean','std']).round(3))

    print('\nDone. Ready for XGBoost training.')


if __name__ == '__main__':
    main()

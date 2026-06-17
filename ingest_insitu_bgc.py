"""
ingest_insitu_bgc.py
====================
Ingest CMEMS in-situ BGC profiles (INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030).
Extracts TEMP and CPHL at near-surface (depth < 10m), QC flag = 1.
Resamples to monthly means per platform.
Collocates against ROMS monthly grid via KDTree.

Handles large files (>20MB) via xarray chunking to avoid OOM.

Outputs:
    insitu_temp_northsea.parquet
    insitu_chl_northsea.parquet
    insitu_colloc_temp.parquet
    insitu_colloc_chl.parquet

Author : Eugène Ivanov / Twin
Project: CE2COAST in-situ sensor fusion — DEME portfolio
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from pathlib import Path
from ce2coast_config import (
    COLLOC_DIR, ROMS_DIR, VALID_DIR, INSITU_DIR,
    FIG_DIR, KALMAN_DIR, ROMS_AVG_PATTERN, ROMS_BIO_PATTERN,
    ROMS_GRID_FILE, SST_OBS_FILE, CHL_OBS_FILE,
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROFILE_DIR = INSITU_DIR
PROFILE_DIR = INSITU_DIR
PROFILE_DIR = INSITU_DIR
ROMS_GRID_FILE = (
    str(ROMS_DIR) + '/'
    'Hindcast_CE2COAST_AVG_2010_2c_atm3.nc'
)
COLLOC_DIR = COLLOC_DIR
OUT_DIR    = COLLOC_DIR

MAX_DEPTH     = 10.0   # surface layer (m)
GOOD_QC       = 1      # accepted QC flag
MAX_DIST_DEG  = 0.2    # KDTree match threshold (~20 km)
YEAR_START    = 2010
YEAR_END      = 2020
BATCH_SIZE    = 50     # files per RAM flush
LARGE_FILE_MB = 20     # threshold for chunked loading


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def find_var(ds, candidates):
    """Return first matching variable or coordinate name."""
    for c in candidates:
        if c in ds.data_vars or c in ds.coords:
            return c
    return None


def get_values(da):
    """Extract numpy array from DataArray, computing dask if needed."""
    if hasattr(da, 'compute'):
        return da.compute().values
    return da.values


def decode_time(ds, time_var):
    """
    Decode CMEMS time to pandas DatetimeIndex.
    Handles 'days since' and 'seconds since' reference epochs.
    Returns None on failure.
    """
    try:
        raw   = get_values(ds[time_var]).ravel().astype(np.float64)
        units = ds[time_var].attrs.get('units', 'days since 1950-01-01')
        if 'days since' in units:
            ref = pd.Timestamp(units.replace('days since', '').strip()[:10])
            return pd.to_datetime(
                [ref + pd.Timedelta(days=float(t)) for t in raw]
            )
        elif 'seconds since' in units:
            ref = pd.Timestamp(units.replace('seconds since', '').strip()[:10])
            return pd.to_datetime(
                [ref + pd.Timedelta(seconds=float(t)) for t in raw]
            )
    except Exception:
        pass
    return None


def build_roms_tree(grid_file):
    """Build KDTree from ROMS ocean rho-points."""
    ds        = xr.open_dataset(grid_file)
    lat       = ds['lat_rho'].values.ravel()
    lon       = ds['lon_rho'].values.ravel()
    mask      = ds['mask_rho'].values.ravel()
    h         = ds['h'].values.ravel()
    ocean_idx = np.where(mask == 1)[0]
    tree      = cKDTree(np.column_stack([lat[ocean_idx], lon[ocean_idx]]))
    ds.close()
    print(f'ROMS grid: ocean points = {len(ocean_idx):,}')
    return tree, lat, lon, h, ocean_idx


# ─────────────────────────────────────────────────────────────
# INGEST ONE FILE
# ─────────────────────────────────────────────────────────────

def ingest_file(fpath):
    """
    Extract monthly-mean surface TEMP and CPHL from one profile file.
    Large files loaded with dask chunks to avoid OOM.
    Returns (records_temp, records_chl) as lists of dicts.
    """
    fsize_mb = fpath.stat().st_size / 1e6

    # Open — chunked for large files to avoid loading into RAM
    try:
        if fsize_mb > LARGE_FILE_MB:
            ds = xr.open_dataset(fpath, decode_times=False,
                                 chunks={'TIME': 10000})
        else:
            ds = xr.open_dataset(fpath, decode_times=False)
    except Exception:
        return [], []

    # ── Detect variable names ────────────────────────────────
    lat_var   = find_var(ds, ['LATITUDE',  'latitude',  'lat'])
    lon_var   = find_var(ds, ['LONGITUDE', 'longitude', 'lon'])
    time_var  = find_var(ds, ['TIME', 'time', 'JULD'])
    depth_var = find_var(ds, ['DEPH', 'DEPTH', 'depth', 'PRES'])
    temp_var  = find_var(ds, ['TEMP', 'THETA', 'PTMP'])
    temp_qc   = find_var(ds, ['TEMP_QC', 'THETA_QC'])
    chl_var   = find_var(ds, ['CPHL', 'CHLA', 'CHL'])
    chl_qc    = find_var(ds, ['CPHL_QC', 'CHLA_QC', 'CHL_QC'])

    if not all([lat_var, lon_var, time_var, depth_var]):
        ds.close()
        return [], []

    # ── Decode time ──────────────────────────────────────────
    times = decode_time(ds, time_var)
    if times is None:
        ds.close()
        return [], []

    n = len(times)

    # ── Load coordinates ─────────────────────────────────────
    try:
        lats   = get_values(ds[lat_var]).ravel().astype(np.float64)
        lons   = get_values(ds[lon_var]).ravel().astype(np.float64)
        depths = get_values(ds[depth_var]).ravel().astype(np.float64)
    except Exception:
        ds.close()
        return [], []

    # Broadcast scalar coords to time length
    if lats.size   == 1: lats   = np.repeat(lats[0],   n)
    if lons.size   == 1: lons   = np.repeat(lons[0],   n)
    if depths.size == 1: depths = np.repeat(depths[0], n)
    if depths.size != n: depths = np.repeat(
        depths[0] if depths.size > 0 else 0.0, n
    )

    # ── Base DataFrame ───────────────────────────────────────
    df = pd.DataFrame({
        'time':  times,
        'lat':   lats[:n],
        'lon':   lons[:n],
        'depth': depths[:n],
    })

    # Early filter — surface + period (reduces RAM before variable load)
    df = df[
        (df['depth'] <= MAX_DEPTH) &
        (df['time'].dt.year >= YEAR_START) &
        (df['time'].dt.year <= YEAR_END)
    ].copy()

    if len(df) == 0:
        ds.close()
        return [], []

    idx = df.index

    # ── Helper: extract variable → monthly means ─────────────
    def extract_monthly(data_var, qc_var, vmin, vmax, col_name):
        if data_var is None:
            return []
        try:
            vals = get_values(ds[data_var])
            if vals.ndim == 2:
                vals = vals[:, 0]    # shallowest depth level
            vals = vals.ravel().astype(np.float64)

            qc = (get_values(ds[qc_var]).ravel().astype(int)
                  if qc_var else np.ones(len(vals), dtype=int))

            # Align to surface-filtered index
            v_filt = np.where(
                (qc[idx] == GOOD_QC) &
                (vals[idx] > vmin) &
                (vals[idx] < vmax),
                vals[idx], np.nan
            )
            df[col_name] = v_filt

            sub = df.dropna(subset=[col_name]).copy()
            if len(sub) == 0:
                return []

            # Monthly mean — collapses high-frequency data efficiently
            sub['ym'] = sub['time'].dt.to_period('M')
            monthly = (sub.groupby('ym')
                       .agg(time=(col_name,   'first'),
                            lat=('lat',       'mean'),
                            lon=('lon',       'mean'),
                            **{col_name: (col_name, 'mean')})
                       .reset_index(drop=True))
            # fix time column — use first timestamp not the value
            monthly['time'] = (sub.groupby('ym')['time']
                               .first().values)
            monthly['platform'] = fpath.stem
            return monthly.to_dict('records')
        except Exception:
            return []

    records_temp = extract_monthly(temp_var, temp_qc, -5,  40,  'temp_insitu')
    records_chl  = extract_monthly(chl_var,  chl_qc,   0, 100,  'chl_insitu')

    ds.close()
    return records_temp, records_chl


# ─────────────────────────────────────────────────────────────
# COLLOC IN-SITU AGAINST ROMS MONTHLY GRID
# ─────────────────────────────────────────────────────────────

def colloc_insitu_to_roms(df, tree, lat_flat, lon_flat, h_flat,
                          ocean_idx, roms_parquet,
                          roms_obs_col, roms_model_col, label):
    """
    Match each in-situ obs to nearest ROMS ocean point.
    Join against existing ROMS collocation parquet on
    (lat_r, lon_r, month, year).
    """
    print(f'\n  Collocating {label} against ROMS...')

    if len(df) == 0:
        print(f'  [WARN] No {label} records')
        return None

    dist, tidx = tree.query(
        np.column_stack([df['lat'].values, df['lon'].values]),
        workers=-1
    )
    full_idx       = ocean_idx[tidx]
    df             = df.copy()
    df['roms_lat'] = lat_flat[full_idx]
    df['roms_lon'] = lon_flat[full_idx]
    df['depth_m']  = h_flat[full_idx]
    df['dist_deg'] = dist
    df             = df[df['dist_deg'] < MAX_DIST_DEG].copy()

    print(f'  Within {MAX_DIST_DEG}°: {len(df):,} obs')
    if len(df) == 0:
        return None

    df['month'] = pd.to_datetime(df['time']).dt.month
    df['year']  = pd.to_datetime(df['time']).dt.year
    df['lat_r'] = np.round(df['roms_lat'], 3)
    df['lon_r'] = np.round(df['roms_lon'], 3)

    if not roms_parquet.exists():
        print(f'  [WARN] ROMS parquet not found: {roms_parquet.name}')
        return df

    roms = pd.read_parquet(roms_parquet)
    roms['lat_r'] = np.round(roms['lat'], 3)
    roms['lon_r'] = np.round(roms['lon'], 3)

    keep = ['lat_r', 'lon_r', 'month', 'year', roms_obs_col, roms_model_col]
    keep = [c for c in keep if c in roms.columns]

    merged = df.merge(roms[keep],
                      on=['lat_r', 'lon_r', 'month', 'year'],
                      how='left')
    merged = merged.drop(columns=['lat_r', 'lon_r'])

    n_matched = (merged[roms_obs_col].notna().sum()
                 if roms_obs_col in merged.columns else 0)
    print(f'  Matched to ROMS colloc: {n_matched:,} / {len(merged):,}')
    return merged


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── ROMS KDTree ──────────────────────────────────────────
    tree, lat_flat, lon_flat, h_flat, ocean_idx = build_roms_tree(
        ROMS_GRID_FILE
    )

    # ── Find profile files ───────────────────────────────────
    nc_files = sorted(PROFILE_DIR.glob('*.nc'))
    print(f'Found {len(nc_files):,} profile files\n')

    # ── Ingest loop — flush to parquet every BATCH_SIZE files ──
    batch_temp, batch_chl = [], []
    n_temp_total = 0
    n_chl_total  = 0
    part_idx     = 0

    for i, fpath in enumerate(nc_files):
        rt, rc = ingest_file(fpath)
        batch_temp.extend(rt)
        batch_chl.extend(rc)

        flush = ((i + 1) % BATCH_SIZE == 0) or ((i + 1) == len(nc_files))

        if flush:
            if batch_temp:
                p = OUT_DIR / f'_part_temp_{part_idx:04d}.parquet'
                pd.DataFrame(batch_temp).to_parquet(p, index=False)
                n_temp_total += len(batch_temp)
                batch_temp    = []
            if batch_chl:
                p = OUT_DIR / f'_part_chl_{part_idx:04d}.parquet'
                pd.DataFrame(batch_chl).to_parquet(p, index=False)
                n_chl_total += len(batch_chl)
                batch_chl    = []
            part_idx += 1
            print(f'  [{i+1:4d}/{len(nc_files)}]  '
                  f'temp={n_temp_total:,}  chl={n_chl_total:,}',
                  flush=True)

    print(f'\nIngestion complete:')
    print(f'  TEMP monthly records: {n_temp_total:,}')
    print(f'  CHL  monthly records: {n_chl_total:,}')

    # ── Concatenate parts ────────────────────────────────────
    def concat_parts(pattern, out_path, time_col='time'):
        parts = sorted(OUT_DIR.glob(pattern))
        if not parts:
            print(f'  No parts for {pattern}')
            return pd.DataFrame()
        print(f'  Concatenating {len(parts)} parts -> {out_path.name}')
        df = pd.concat([pd.read_parquet(p) for p in parts],
                       ignore_index=True)
        df[time_col] = pd.to_datetime(df[time_col])
        df.to_parquet(out_path, index=False)
        for p in parts:
            p.unlink()
        print(f'  Saved: {out_path.name}  '
              f'({out_path.stat().st_size/1e6:.1f} MB)  '
              f'{len(df):,} rows')
        return df

    df_temp = concat_parts('_part_temp_*.parquet',
                           OUT_DIR / 'insitu_temp_northsea.parquet')
    df_chl  = concat_parts('_part_chl_*.parquet',
                           OUT_DIR / 'insitu_chl_northsea.parquet')

    # ── Print summaries ──────────────────────────────────────
    if not df_temp.empty:
        print(f'\n  TEMP platforms : {df_temp["platform"].nunique()}')
        print(f'  TEMP time range: {df_temp["time"].min().date()} -> '
              f'{df_temp["time"].max().date()}')
        print(f'  TEMP value range: {df_temp["temp_insitu"].min():.2f} -> '
              f'{df_temp["temp_insitu"].max():.2f} degC')

    if not df_chl.empty:
        print(f'\n  CHL  platforms : {df_chl["platform"].nunique()}')
        print(f'  CHL  time range: {df_chl["time"].min().date()} -> '
              f'{df_chl["time"].max().date()}')
        print(f'  CHL  range: {df_chl["chl_insitu"].min():.4f} -> '
              f'{df_chl["chl_insitu"].max():.4f} mg/m3')

    # ── Colloc against ROMS ──────────────────────────────────
    if not df_temp.empty:
        merged = colloc_insitu_to_roms(
            df_temp, tree, lat_flat, lon_flat, h_flat, ocean_idx,
            roms_parquet   = COLLOC_DIR / 'colloc_sst_2010_2020.parquet',
            roms_obs_col   = 'sst_obs',
            roms_model_col = 'sst_roms',
            label          = 'TEMP'
        )
        if merged is not None:
            out = OUT_DIR / 'insitu_colloc_temp.parquet'
            merged.to_parquet(out, index=False)
            print(f'  Saved: {out.name}  '
                  f'({out.stat().st_size/1e6:.1f} MB)')

    if not df_chl.empty:
        merged = colloc_insitu_to_roms(
            df_chl, tree, lat_flat, lon_flat, h_flat, ocean_idx,
            roms_parquet   = COLLOC_DIR / 'colloc_chl_2010_2020.parquet',
            roms_obs_col   = 'chl_obs',
            roms_model_col = 'chl_roms',
            label          = 'CHL'
        )
        if merged is not None:
            out = OUT_DIR / 'insitu_colloc_chl.parquet'
            merged.to_parquet(out, index=False)
            print(f'  Saved: {out.name}  '
                  f'({out.stat().st_size/1e6:.1f} MB)')

    # ── Coverage by year ─────────────────────────────────────
    print('\n' + '='*50)
    print('COVERAGE BY YEAR')
    print('='*50)
    if not df_temp.empty:
        print('\n  TEMP:')
        print(df_temp.groupby(df_temp['time'].dt.year)['temp_insitu']
              .count().rename('n_obs').to_string())
    if not df_chl.empty:
        print('\n  CHL:')
        print(df_chl.groupby(df_chl['time'].dt.year)['chl_insitu']
              .count().rename('n_obs').to_string())

    print('\nDone.')


if __name__ == '__main__':
    main()

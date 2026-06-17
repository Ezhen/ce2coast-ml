"""
kalman_update_roms.py
=====================
Optimal Interpolation (Kalman-style) update of XGBoost-corrected
ROMS fields using in-situ observations.

Physics:
    x_a = x_b + K (y - H x_b)
    K   = B Hᵀ (H B Hᵀ + R)⁻¹

where:
    x_b  = XGBoost-corrected ROMS field (background state)
    y    = in-situ observation
    H    = observation operator (nearest-neighbour interpolation)
    B    = background error covariance (Gaussian, length scale L_b)
    R    = observation error variance (diagonal)
    x_a  = analysis (corrected) field

Applied independently per month on the ROMS ocean grid.

Outputs:
    analysis_sst_YYYY_MM.parquet   — monthly SST analysis fields
    analysis_chl_YYYY_MM.parquet   — monthly Chl analysis fields
    figures/kalman_sst_YYYY_MM.png — comparison maps
    figures/kalman_chl_YYYY_MM.png

Author : Eugène Ivanov / Twin
Project: CE2COAST in-situ sensor fusion — DEME portfolio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
OUT_DIR    = KALMAN_DIR
FIG_DIR    = FIG_DIR

# ── Covariance parameters ────────────────────────────────────
# Background error length scale (degrees, ~50 km for SST)
###############  L_B_SST  = 0.5
L_B_CHL  = 0.5

# Background error variance (sigma_b²)
# Set to observed residual variance from XGBoost output
SIGMA_B_SST = 0.73 ** 2   # from XGB RMSE on test set
SIGMA_B_CHL = 0.91 ** 2   # from XGB RMSE on test set

# Observation error variance (sigma_o²)
# In-situ measurement uncertainty
############ SIGMA_O_SST = 0.3 ** 2    # ~0.3 degC instrument error
SIGMA_O_CHL = 0.5 ** 2    # ~0.5 mg/m3 fluorometer uncertainty

# Max radius for obs influence (degrees)
# Points beyond this contribute negligible covariance

############## OI_RADIUS = 2.0


L_B_SST     = 0.3     # tighter — 30km instead of 50km
SIGMA_O_SST = 0.5**2  # trust obs less
OI_RADIUS   = 1.0     # smaller influence zone



# ── Aesthetics ───────────────────────────────────────────────
DARK_BG  = '#0d1117'
SURFACE  = '#161b22'
BORDER   = '#30363d'
TEXT     = '#e6edf3'
MUTED    = '#7d8590'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT,
    'axes.titlecolor':   TEXT,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        TEXT,
    'grid.color':        BORDER,
    'font.family':       'monospace',
    'font.size':         9,
    'savefig.facecolor': DARK_BG,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})


# ─────────────────────────────────────────────────────────────
# OPTIMAL INTERPOLATION — CORE
# ─────────────────────────────────────────────────────────────

def gaussian_cov(dist, length_scale, sigma2):
    """Gaussian covariance: C(d) = sigma² * exp(-d²/2L²)"""
    return sigma2 * np.exp(-(dist ** 2) / (2 * length_scale ** 2))


def oi_update(grid_lat, grid_lon, x_b,
              obs_lat, obs_lon, obs_val,
              length_scale, sigma_b2, sigma_o2,
              radius=OI_RADIUS):
    """
    Optimal Interpolation update on a 1D grid.

    Parameters
    ----------
    grid_lat, grid_lon : 1D arrays of background grid point coords
    x_b                : 1D background field at grid points
    obs_lat, obs_lon   : 1D observation locations
    obs_val            : 1D observation values
    length_scale       : Gaussian covariance length scale (degrees)
    sigma_b2           : background error variance
    sigma_o2           : observation error variance (scalar, diagonal R)
    radius             : max influence radius (degrees)

    Returns
    -------
    x_a    : 1D analysis field at grid points
    spread : 1D analysis error spread (sqrt of diagonal of Pa)
    """
    n_grid = len(grid_lat)
    n_obs  = len(obs_lat)

    x_a    = x_b.copy()
    spread = np.full(n_grid, np.sqrt(sigma_b2))

    if n_obs == 0:
        return x_a, spread

    # Build KDTree on obs locations
    obs_tree = cKDTree(np.column_stack([obs_lat, obs_lon]))

    # For each grid point, find nearby obs within radius
    for i in range(n_grid):
        nearby = obs_tree.query_ball_point(
            [grid_lat[i], grid_lon[i]], r=radius
        )
        if len(nearby) == 0:
            continue

        # Extract nearby obs
        j       = np.array(nearby)
        y_j     = obs_val[j]
        lat_j   = obs_lat[j]
        lon_j   = obs_lon[j]

        # H x_b at obs locations — nearest grid point
        # (simplified: background already interpolated to obs via colloc)
        # Innovation: y - H x_b
        # We need x_b at obs locations — use mean of nearby background
        # For simplicity use the obs-collocated background value
        # passed through obs_val already as (y - H x_b)
        # so obs_val here IS the innovation vector

        # B Hᵀ — covariance between grid point i and each obs j
        d_grid_obs = np.sqrt(
            (grid_lat[i] - lat_j) ** 2 +
            (grid_lon[i] - lon_j) ** 2
        )
        b_hT = gaussian_cov(d_grid_obs, length_scale, sigma_b2)

        # H B Hᵀ + R — obs-obs covariance matrix
        d_obs_obs = np.sqrt(
            (lat_j[:, None] - lat_j[None, :]) ** 2 +
            (lon_j[:, None] - lon_j[None, :]) ** 2
        )
        HBHt_R = (gaussian_cov(d_obs_obs, length_scale, sigma_b2)
                  + np.eye(len(j)) * sigma_o2)

        # Kalman gain K = B Hᵀ (H B Hᵀ + R)⁻¹
        try:
            K = b_hT @ np.linalg.solve(HBHt_R, np.eye(len(j)))
        except np.linalg.LinAlgError:
            continue

        # Analysis increment
        x_a[i]    = x_b[i] + K @ y_j
        # Analysis spread (diagonal of Pa = B - K H B)
        spread[i] = np.sqrt(max(0, sigma_b2 - K @ b_hT))

    return x_a, spread


# ─────────────────────────────────────────────────────────────
# PROCESS ONE MONTH
# ─────────────────────────────────────────────────────────────

def process_month(bg_month, insitu_month,
                  bg_col, obs_col, roms_col,
                  length_scale, sigma_b2, sigma_o2,
                  label):
    """
    Run OI update for one month.

    bg_month     : DataFrame of background field (XGB-corrected ROMS grid)
    insitu_month : DataFrame of in-situ obs for this month
    bg_col       : column name of XGB-corrected background in bg_month
    obs_col      : column name of in-situ observation
    roms_col     : column name of raw ROMS value
    Returns enriched bg_month with 'analysis' and 'spread' columns.
    """
    bg = bg_month.copy()

    if len(insitu_month) == 0:
        bg['analysis'] = bg[bg_col]
        bg['spread']   = np.sqrt(sigma_b2)
        bg['n_obs']    = 0
        return bg

    # Innovation: y - H x_b
    # H x_b at obs location = background interpolated to obs point
    # We use the XGB-corrected value already stored in insitu_colloc
    # as the background at obs location
    if bg_col in insitu_month.columns:
        innov = insitu_month[obs_col].values - insitu_month[bg_col].values
    else:
        # Fallback: use raw ROMS as background at obs
        innov = insitu_month[obs_col].values - insitu_month[roms_col].values

    # Remove NaN innovations
    valid = np.isfinite(innov)
    if valid.sum() == 0:
        bg['analysis'] = bg[bg_col]
        bg['spread']   = np.sqrt(sigma_b2)
        bg['n_obs']    = 0
        return bg

    obs_lat  = insitu_month['lat'].values[valid]
    obs_lon  = insitu_month['lon'].values[valid]
    innov_v  = innov[valid]

    # Run OI
    x_a, spread = oi_update(
        bg['lat'].values, bg['lon'].values,
        bg[bg_col].values,
        obs_lat, obs_lon, innov_v,
        length_scale, sigma_b2, sigma_o2
    )

    bg['analysis'] = x_a
    bg['spread']   = spread
    bg['n_obs']    = valid.sum()

    return bg


# ─────────────────────────────────────────────────────────────
# FIGURE — THREE-COLUMN COMPARISON MAP
# ─────────────────────────────────────────────────────────────

def make_comparison_figure(bg_month, insitu_month,
                           roms_col, bg_col,
                           obs_col, year, month,
                           label, units, cmap='RdYlBu_r'):
    """
    Three-panel map: ROMS raw | XGB corrected | OI analysis
    with in-situ obs overlaid as scatter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'CE2COAST {label} — Three-layer correction  |  '
        f'{year}-{month:02d}',
        fontsize=12, fontweight='bold', color=TEXT
    )

    cols   = [roms_col, bg_col, 'analysis']
    titles = ['ROMS raw', 'XGB corrected', 'OI analysis']

    # Common colour scale
    all_vals = pd.concat([bg_month[c] for c in cols
                          if c in bg_month.columns]).dropna()
    vmin = np.percentile(all_vals, 2)
    vmax = np.percentile(all_vals, 98)

    for ax, col, title in zip(axes, cols, titles):
        if col not in bg_month.columns:
            ax.set_visible(False)
            continue

        sc = ax.scatter(
            bg_month['lon'], bg_month['lat'],
            c=bg_month[col], cmap=cmap,
            vmin=vmin, vmax=vmax,
            s=4, rasterized=True
        )
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                     label=units)

        # Overlay in-situ obs
        if len(insitu_month) > 0 and obs_col in insitu_month.columns:
            valid = insitu_month[obs_col].notna()
            ax.scatter(
                insitu_month.loc[valid, 'lon'],
                insitu_month.loc[valid, 'lat'],
                c=insitu_month.loc[valid, obs_col],
                cmap=cmap, vmin=vmin, vmax=vmax,
                s=60, edgecolors='white', linewidths=0.8,
                zorder=5
            )

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    n_obs = len(insitu_month[insitu_month[obs_col].notna()]) \
        if len(insitu_month) > 0 and obs_col in insitu_month.columns else 0
    fig.text(0.5, 0.01,
             f'In-situ obs: {n_obs} | '
             f'White-outlined circles = in-situ stations',
             ha='center', fontsize=8, color=MUTED)

    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    print('Loading data...')

    # XGBoost test predictions (background state)
    pred_sst = pd.read_parquet(
        COLLOC_DIR / 'predictions_sst_test.parquet'
    )
    pred_chl = pd.read_parquet(
        COLLOC_DIR / 'predictions_chl_test.parquet'
    )

    # In-situ collocated observations
    insitu_t = pd.read_parquet(
        COLLOC_DIR / 'insitu_colloc_temp.parquet'
    )
    insitu_c = pd.read_parquet(
        COLLOC_DIR / 'insitu_colloc_chl.parquet'
    )

    # Ensure datetime
    for df in [pred_sst, pred_chl, insitu_t, insitu_c]:
        df['time'] = pd.to_datetime(df['time'])

    print(f'  Background SST: {len(pred_sst):,} rows')
    print(f'  Background Chl: {len(pred_chl):,} rows')
    print(f'  In-situ temp:   {len(insitu_t):,} rows')
    print(f'  In-situ chl:    {len(insitu_c):,} rows')

    # ── Add month/year ───────────────────────────────────────
    for df in [pred_sst, pred_chl, insitu_t, insitu_c]:
        df['month'] = df['time'].dt.month
        df['year']  = df['time'].dt.year

    # ── Get unique months in test period ─────────────────────
    months_sst = (pred_sst[['year','month']]
                  .drop_duplicates()
                  .sort_values(['year','month']))

    # ── Process SST ──────────────────────────────────────────
    print('\nProcessing SST OI updates...')
    results_sst = []

    for _, row in months_sst.iterrows():
        yr, mo = int(row['year']), int(row['month'])

        bg_m = pred_sst[
            (pred_sst['year'] == yr) & (pred_sst['month'] == mo)
        ].copy()

        ins_m = insitu_t[
            (insitu_t['year'] == yr) & (insitu_t['month'] == mo)
        ].copy()

        bg_m = process_month(
            bg_m, ins_m,
            bg_col    = 'sst_corrected',
            obs_col   = 'temp_insitu',
            roms_col  = 'sst_roms',
            length_scale = L_B_SST,
            sigma_b2     = SIGMA_B_SST,
            sigma_o2     = SIGMA_O_SST,
            label        = 'SST'
        )
        results_sst.append(bg_m)

        n = int(bg_m['n_obs'].iloc[0]) if 'n_obs' in bg_m.columns else 0
        print(f'  {yr}-{mo:02d}  grid={len(bg_m):,}  in-situ={n}',
              flush=True)

        # Figure for months with in-situ data
        if n > 0:
            fig = make_comparison_figure(
                bg_m, ins_m,
                roms_col='sst_roms',
                bg_col='sst_corrected',
                obs_col='temp_insitu',
                year=yr, month=mo,
                label='SST', units='°C',
                cmap='RdYlBu_r'
            )
            out = FIG_DIR / f'kalman_sst_{yr}_{mo:02d}.png'
            fig.savefig(out)
            plt.close(fig)

    df_sst_analysis = pd.concat(results_sst, ignore_index=True)
    out = OUT_DIR / 'analysis_sst.parquet'
    df_sst_analysis.to_parquet(out, index=False)
    print(f'\n  SST analysis saved: {out.name}  '
          f'({out.stat().st_size/1e6:.1f} MB)')

    # ── Process Chl ──────────────────────────────────────────
    print('\nProcessing Chl OI updates...')
    results_chl = []

    months_chl = (pred_chl[['year','month']]
                  .drop_duplicates()
                  .sort_values(['year','month']))

    for _, row in months_chl.iterrows():
        yr, mo = int(row['year']), int(row['month'])

        bg_m = pred_chl[
            (pred_chl['year'] == yr) & (pred_chl['month'] == mo)
        ].copy()

        ins_m = insitu_c[
            (insitu_c['year'] == yr) & (insitu_c['month'] == mo)
        ].copy()

        # Work in log space for Chl
        if 'chl_corrected' in bg_m.columns:
            bg_m['log_chl_corrected'] = np.log(
                np.clip(bg_m['chl_corrected'], 0.001, None)
            )
        if len(ins_m) > 0 and 'chl_insitu' in ins_m.columns:
            ins_m = ins_m.copy()
            ins_m['log_chl_insitu'] = np.log(
                np.clip(ins_m['chl_insitu'], 0.001, None)
            )

        bg_m = process_month(
            bg_m, ins_m,
            bg_col    = 'log_chl_corrected',
            obs_col   = 'log_chl_insitu',
            roms_col  = 'chl_roms',
            length_scale = L_B_CHL,
            sigma_b2     = SIGMA_B_CHL,
            sigma_o2     = SIGMA_O_CHL,
            label        = 'Chl'
        )

        # Back-transform analysis to linear space
        if 'analysis' in bg_m.columns:
            bg_m['chl_analysis'] = np.exp(bg_m['analysis'])

        results_chl.append(bg_m)

        n = int(bg_m['n_obs'].iloc[0]) if 'n_obs' in bg_m.columns else 0
        print(f'  {yr}-{mo:02d}  grid={len(bg_m):,}  in-situ={n}',
              flush=True)

        if n > 0 and 'chl_analysis' in bg_m.columns:
            fig = make_comparison_figure(
                bg_m, ins_m,
                roms_col='chl_roms',
                bg_col='chl_corrected',
                obs_col='chl_insitu',
                year=yr, month=mo,
                label='Chl', units='mg/m³',
                cmap='YlGn'
            )
            out = FIG_DIR / f'kalman_chl_{yr}_{mo:02d}.png'
            fig.savefig(out)
            plt.close(fig)

    df_chl_analysis = pd.concat(results_chl, ignore_index=True)
    out = OUT_DIR / 'analysis_chl.parquet'
    df_chl_analysis.to_parquet(out, index=False)
    print(f'\n  Chl analysis saved: {out.name}  '
          f'({out.stat().st_size/1e6:.1f} MB)')

    # ── Summary metrics ──────────────────────────────────────
    print('\n' + '='*55)
    print('OI ANALYSIS SUMMARY')
    print('='*55)

    for label, df, obs_col, roms_col, bg_col, ana_col in [
        ('SST', df_sst_analysis,
         'sst_obs', 'sst_roms', 'sst_corrected', 'analysis'),
        ('Chl', df_chl_analysis,
         'chl_obs', 'chl_roms', 'chl_corrected', 'chl_analysis'),
    ]:
        cols_present = [c for c in [obs_col, roms_col, bg_col, ana_col]
                        if c in df.columns]
        sub = df[cols_present].dropna()
        if len(sub) == 0:
            continue

        from sklearn.metrics import r2_score, mean_squared_error
        print(f'\n  {label} (n={len(sub):,}):')
        for col, name in [(roms_col, 'ROMS raw    '),
                          (bg_col,   'XGB corrected'),
                          (ana_col,  'OI analysis ')]:
            if col not in sub.columns:
                continue
            r2   = r2_score(sub[obs_col], sub[col])
            rmse = np.sqrt(mean_squared_error(sub[obs_col], sub[col]))
            bias = float(np.mean(sub[col] - sub[obs_col]))
            print(f'    {name}  R²={r2:.3f}  RMSE={rmse:.3f}  '
                  f'bias={bias:+.3f}')

    print('\nFigures saved to:', FIG_DIR)
    print('Done.')


if __name__ == '__main__':
    main()

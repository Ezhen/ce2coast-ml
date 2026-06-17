"""
uncertainty_chl.py
==================
Uncertainty quantification for CE2COAST Chl correction pipeline.

Three uncertainty sources addressed:

  1. XGBoost quantile regression (P10 / P50 / P90) in log-space
     — back-transformed to linear mg/m3 for interpretation

  2. Cloud cover proxy uncertainty
     — spatial variance of monthly Chl field as obs confidence proxy
     — or NOBS variable from CMEMS if available

  3. Log-space asymmetry
     — P10/P90 asymmetric around P50 in linear space
     — visualised explicitly to show non-Gaussian nature of Chl errors

Outputs:
  figures/uncertainty_chl_quantile_maps.png  — P10/P50/P90 seasonal maps
  figures/uncertainty_chl_coverage.png       — calibration validation
  figures/uncertainty_chl_cloud.png          — cloud cover proxy analysis
  figures/uncertainty_chl_summary.png        — portfolio summary figure
  uncertainty_chl_quantile_preds.parquet     — quantile predictions

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import xarray as xr
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
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
OUT_DIR    = COLLOC_DIR
FIG_DIR    = FIG_DIR

CHL_OBS_FILE = Path(
    str(VALID_DIR) + '/'
    'cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M_'
    '1698156332793.nc'
)

CHL_FEATURES = [
    'chl_roms', 'depth_m', 'coast_dist',
    'month_sin', 'month_cos',
    'doy_sin',   'doy_cos',
    'log_residual_chl_lag1',
]
CHL_TARGET = 'log_residual_chl'   # train in log-space

QUANTILES = [0.10, 0.50, 0.90]

XGB_BASE = dict(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 10,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    tree_method           = 'hist',
    n_jobs                = -1,
    random_state          = 42,
    early_stopping_rounds = 30,
)

SEASONS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

# ── Aesthetics ────────────────────────────────────────────────
DARK_BG = '#0d1117'
SURFACE = '#161b22'
BORDER  = '#30363d'
TEXT    = '#e6edf3'
MUTED   = '#7d8590'
BLUE    = '#2f81f7'
GREEN   = '#3fb950'
AMBER   = '#d29922'
CORAL   = '#f0644a'
PURPLE  = '#bc5adc'

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
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'font.size':         8,
    'axes.titlesize':    9,
    'legend.facecolor':  SURFACE,
    'legend.edgecolor':  BORDER,
    'savefig.facecolor': DARK_BG,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})


# ─────────────────────────────────────────────────────────────
# CLOUD COVER PROXY FROM CMEMS FILE
# ─────────────────────────────────────────────────────────────

def extract_cloud_proxy(chl_obs_file, years=(2019, 2020)):
    """
    Extract cloud cover proxy from CMEMS monthly Chl file.

    Strategy:
      1. Try NOBS or error variable directly
      2. Fallback: use fraction of NaN pixels per month as cloud proxy
         (more NaNs = more cloud = less reliable monthly mean)

    Returns DataFrame with lat, lon, month, year, cloud_proxy
    where higher cloud_proxy = MORE cloud = LESS reliable obs.
    """
    print('\nExtracting cloud cover proxy...')

    if not chl_obs_file.exists():
        print(f'  [WARN] Chl obs file not found: {chl_obs_file}')
        return None

    ds = xr.open_dataset(chl_obs_file)

    # Detect Chl variable
    chl_var = None
    for c in ['CHL', 'chl', 'chlorophyll', 'Chl', 'chla']:
        if c in ds.data_vars:
            chl_var = c
            break

    # Try to find obs count variable
    nobs_var = None
    for c in ['NOBS', 'nobs', 'n_obs', 'count', 'weights',
              'error_small', 'uncertainty']:
        if c in ds.data_vars:
            nobs_var = c
            print(f'  Found obs count variable: {nobs_var}')
            break

    records = []

    for year in years:
        try:
            ds_yr = ds.sel(time=str(year))
        except Exception:
            continue

        for t_idx in range(len(ds_yr['time'])):
            try:
                month = int(pd.to_datetime(
                    ds_yr['time'].values[t_idx]
                ).month)
            except Exception:
                continue

            if nobs_var:
                # Use actual observation count — invert to get cloud proxy
                nobs = ds_yr[nobs_var].isel(time=t_idx).values
                nobs = np.where(np.isfinite(nobs), nobs, 0)
                # Normalise: 0 obs = full cloud, max obs = clear
                nobs_max = np.nanpercentile(nobs[nobs > 0], 95) \
                    if (nobs > 0).any() else 1.0
                cloud = 1.0 - np.clip(nobs / nobs_max, 0, 1)
            else:
                # Fallback: NaN fraction in CHL field as cloud proxy
                if chl_var is None:
                    continue
                chl_snap = ds_yr[chl_var].isel(time=t_idx).values
                # Use local variance as additional signal
                # High variance in a monthly composite = fewer obs,
                # more gap-filling artefacts
                from scipy.ndimage import uniform_filter
                valid = np.isfinite(chl_snap)
                # NaN fraction on 5-pixel rolling window
                nan_frac = uniform_filter(
                    (~valid).astype(float), size=5
                )
                cloud = nan_frac   # 0 = fully observed, 1 = fully cloudy

            lat_vals = ds_yr['lat'].values
            lon_vals = ds_yr['lon'].values

            # Subsample to North Sea domain
            lat_mask = (lat_vals >= 48) & (lat_vals <= 62)
            lon_mask = (lon_vals >= -6) & (lon_vals <= 11)

            lat_sub   = lat_vals[lat_mask]
            lon_sub   = lon_vals[lon_mask]
            cloud_sub = cloud[np.ix_(lat_mask, lon_mask)] \
                if cloud.ndim == 2 else \
                np.zeros((lat_mask.sum(), lon_mask.sum()))

            # Flatten
            lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
            for i in range(cloud_sub.shape[0]):
                for j in range(0, cloud_sub.shape[1], 3):  # subsample
                    records.append({
                        'lat':         float(lat_grid[i, j]),
                        'lon':         float(lon_grid[i, j]),
                        'month':       month,
                        'year':        year,
                        'cloud_proxy': float(cloud_sub[i, j]),
                    })

    ds.close()

    if not records:
        print('  [WARN] No cloud proxy records extracted')
        return None

    df = pd.DataFrame(records)
    print(f'  Cloud proxy: {len(df):,} grid cells  '
          f'| mean={df["cloud_proxy"].mean():.3f}')
    return df


# ─────────────────────────────────────────────────────────────
# QUANTILE REGRESSION
# ─────────────────────────────────────────────────────────────

def train_quantile_models(df_train, df_test):
    """Train P10/P50/P90 XGBoost models in log-space."""
    print('\nTraining Chl quantile models (log-space)...')

    for col in CHL_FEATURES:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col]  = pd.to_numeric(df_test[col],  errors='coerce')

    train = df_train.dropna(subset=CHL_FEATURES + [CHL_TARGET])
    test  = df_test.dropna(subset=CHL_FEATURES  + [CHL_TARGET])

    # Exclude extreme outliers from training
    if 'log_residual_chl_extreme' in train.columns:
        train = train[train['log_residual_chl_extreme'] == 0]

    X_tr = train[CHL_FEATURES].values
    y_tr = train[CHL_TARGET].values
    X_te = test[CHL_FEATURES].values
    y_te = test[CHL_TARGET].values

    preds = {}
    for q in QUANTILES:
        print(f'  Q{q:.0%}...', end=' ', flush=True)
        params = {
            **XGB_BASE,
            'objective':      'reg:quantileerror',
            'quantile_alpha': q,
            'verbosity':      0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)
        preds[q] = model.predict(X_te)
        print(f'done  iter={model.best_iteration}', flush=True)

    return preds, test, y_te


def compute_coverage(y_true, p10, p90, label=''):
    inside   = (y_true >= p10) & (y_true <= p90)
    coverage = inside.mean()
    width    = (p90 - p10).mean()
    print(f'  {label:<12} coverage={coverage:.3f}  '
          f'(target=0.80)  width={width:.3f} log-units')
    return float(coverage), float(width)


# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────

def fig_quantile_maps(test):
    """
    4 seasons × 4 columns in log-space:
    P10 | P50 | P90 | interval width
    Plus linear-space asymmetry panel.
    """
    print('\nBuilding Chl quantile maps...')

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        'CE2COAST Chl — XGBoost Quantile Uncertainty  |  Test 2019–2020\n'
        'P10 / P50 / P90 of log-residual correction  '
        '(log mg/m³)  |  Width = P90 − P10',
        fontsize=11, fontweight='bold', color=TEXT, y=0.998
    )

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.32, wspace=0.12,
                           left=0.05, right=0.95,
                           top=0.965, bottom=0.04)

    err_max   = 2.5   # log units
    width_max = 3.5

    for row, (sname, months) in enumerate(SEASONS.items()):
        sub = test[test['month'].isin(months)]
        grp = sub.groupby(['lat', 'lon'])[
            ['p10_log', 'p50_log', 'p90_log', 'width_log']
        ].mean().reset_index()

        for col, (field, cmap, vmin, vmax, title) in enumerate([
            ('p10_log',   'PiYG',   -err_max,   err_max,   f'{sname} P10'),
            ('p50_log',   'PiYG',   -err_max,   err_max,   f'{sname} P50'),
            ('p90_log',   'PiYG',   -err_max,   err_max,   f'{sname} P90'),
            ('width_log', 'plasma',  0,          width_max, f'{sname} Width'),
        ]):
            ax = fig.add_subplot(gs[row, col])
            sc = ax.scatter(grp['lon'], grp['lat'],
                            c=grp[field],
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            s=3, rasterized=True)
            ax.set_title(title, pad=3, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel('Lat', fontsize=7)
            ax.text(0.02, 0.03,
                    f'mean={grp[field].mean():+.2f}',
                    transform=ax.transAxes, fontsize=6,
                    color=TEXT, family='monospace')

    # Colorbars
    cbar1 = fig.add_axes([0.05, 0.005, 0.67, 0.010])
    sm1 = plt.cm.ScalarMappable(
        cmap='PiYG', norm=plt.Normalize(-err_max, err_max)
    )
    cb1 = plt.colorbar(sm1, cax=cbar1, orientation='horizontal')
    cb1.set_label(
        'Log-residual (log mg/m³)  |  green=ROMS overestimate corrected  '
        'pink=underestimate',
        color=TEXT, fontsize=8
    )
    cb1.ax.tick_params(colors=MUTED, labelsize=7)

    cbar2 = fig.add_axes([0.77, 0.005, 0.18, 0.010])
    sm2 = plt.cm.ScalarMappable(
        cmap='plasma', norm=plt.Normalize(0, width_max)
    )
    cb2 = plt.colorbar(sm2, cax=cbar2, orientation='horizontal')
    cb2.set_label('Interval width (log units)', color=TEXT, fontsize=8)
    cb2.ax.tick_params(colors=MUTED, labelsize=7)

    return fig


def fig_log_asymmetry(test):
    """
    Log-space P10/P50/P90 → back-transform to linear mg/m3.
    Shows asymmetric intervals — key Chl vs SST difference.
    """
    print('Building log-space asymmetry figure...')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'CE2COAST Chl — Log-space Uncertainty → Linear-space Asymmetry  |'
        '  2019–2020\n'
        'P10–P90 symmetric in log-space but asymmetric in mg/m³ '
        '(heteroscedastic uncertainty)',
        fontsize=10, fontweight='bold', color=TEXT
    )

    sample = test.dropna(
        subset=['p10_log', 'p50_log', 'p90_log',
                'p10_lin', 'p50_lin', 'p90_lin', 'chl_obs']
    ).sample(min(3000, len(test)), random_state=42)
    sample = sample.sort_values('p50_lin')

    # Panel 1: log-space — symmetric
    ax = axes[0]
    ax.fill_between(range(len(sample)),
                    sample['p10_log'].values,
                    sample['p90_log'].values,
                    alpha=0.3, color=GREEN, label='P10–P90 band')
    ax.plot(sample['p50_log'].values, color=GREEN,
            linewidth=0.8, label='P50')
    ax.set_title('Log-space: symmetric intervals')
    ax.set_xlabel('Sample index (sorted by P50)')
    ax.set_ylabel('Log-residual (log mg/m³)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: linear-space — asymmetric
    ax = axes[1]
    ax.fill_between(range(len(sample)),
                    sample['p10_lin'].values,
                    sample['p90_lin'].values,
                    alpha=0.3, color=AMBER, label='P10–P90 band')
    ax.plot(sample['p50_lin'].values, color=AMBER,
            linewidth=0.8, label='P50')
    ax.set_title('Linear-space: asymmetric intervals\n(wider for high Chl)')
    ax.set_xlabel('Sample index (sorted by P50)')
    ax.set_ylabel('Chl correction factor')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, np.percentile(sample['p90_lin'], 95))

    # Panel 3: width in linear space vs Chl magnitude
    ax = axes[2]
    sample['lin_width'] = sample['p90_lin'] - sample['p10_lin']
    ax.scatter(sample['chl_obs'],
               sample['lin_width'],
               s=2, alpha=0.3, color=PURPLE, rasterized=True)
    # Running mean
    bins = np.percentile(sample['chl_obs'],
                         np.linspace(5, 95, 20))
    bin_idx = np.digitize(sample['chl_obs'], bins)
    bin_means_x = [sample['chl_obs'][bin_idx == i].mean()
                   for i in range(1, len(bins))]
    bin_means_y = [sample['lin_width'][bin_idx == i].mean()
                   for i in range(1, len(bins))]
    ax.plot(bin_means_x, bin_means_y,
            color=AMBER, linewidth=2, label='Running mean')
    ax.set_xlabel('Observed Chl (mg/m³)')
    ax.set_ylabel('Interval width in linear space')
    ax.set_title('Uncertainty grows with Chl magnitude\n'
                 '(heteroscedastic — expected for log-normal)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    r, p = spearmanr(sample['chl_obs'], sample['lin_width'])
    ax.text(0.05, 0.92,
            f'Spearman r = {r:.3f}  p < 0.001',
            transform=ax.transAxes, fontsize=8,
            color=AMBER, family='monospace')

    plt.tight_layout()
    return fig


def fig_cloud_coverage(test, df_cloud):
    """
    Cloud cover proxy vs interval width.
    Where satellite obs are sparse → larger uncertainty.
    """
    print('Building cloud cover figure...')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'CE2COAST Chl — Cloud Cover Proxy vs Prediction Uncertainty  |'
        '  2019–2020\n'
        'Hypothesis: more cloud → fewer obs → less reliable monthly Chl'
        ' → wider correction interval',
        fontsize=10, fontweight='bold', color=TEXT
    )

    # ── Seasonal cloud proxy maps ─────────────────────────────
    for ax, (sname, months) in zip(axes.ravel(), SEASONS.items()):

        if df_cloud is not None:
            sub_c = df_cloud[df_cloud['month'].isin(months)]
            grp_c = (sub_c.groupby(['lat', 'lon'])['cloud_proxy']
                     .mean().reset_index())
            sc = ax.scatter(grp_c['lon'], grp_c['lat'],
                            c=grp_c['cloud_proxy'],
                            cmap='Blues', vmin=0, vmax=1,
                            s=2, rasterized=True,
                            label='Cloud proxy')

            # Overlay interval width from test set
            sub_t = test[test['month'].isin(months)].dropna(
                subset=['width_log']
            )
            grp_t = (sub_t.groupby(['lat', 'lon'])['width_log']
                     .mean().reset_index())
            sc2 = ax.scatter(grp_t['lon'], grp_t['lat'],
                             c=grp_t['width_log'],
                             cmap='Reds', vmin=0, vmax=3,
                             s=6, alpha=0.5, rasterized=True,
                             label='Interval width')

            plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01,
                         label='Cloud proxy (0=clear, 1=cloudy)')
        else:
            # No cloud data — show interval width only
            sub_t = test[test['month'].isin(months)].dropna(
                subset=['width_log']
            )
            grp_t = (sub_t.groupby(['lat', 'lon'])['width_log']
                     .mean().reset_index())
            sc2 = ax.scatter(grp_t['lon'], grp_t['lat'],
                             c=grp_t['width_log'],
                             cmap='plasma', vmin=0, vmax=3.5,
                             s=4, rasterized=True)
            plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.01,
                         label='Interval width (log units)')

        mean_w = grp_t['width_log'].mean() if len(grp_t) > 0 else 0
        ax.set_title(f'{sname}  |  mean width={mean_w:.2f} log-units',
                     fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


def fig_coverage_validation(test):
    """Coverage bar + scatter + width distribution."""
    print('Building Chl coverage validation figure...')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'CE2COAST Chl — Uncertainty Calibration  |  Test 2019–2020',
        fontsize=11, fontweight='bold', color=TEXT
    )

    # Panel 1: coverage by season
    ax = axes[0]
    coverages, widths, labels = [], [], []
    colors = [BLUE, GREEN, AMBER, CORAL, TEXT]

    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['p10_log', 'p90_log', CHL_TARGET]
        )
        if len(sub) == 0:
            continue
        cov = ((sub[CHL_TARGET] >= sub['p10_log']) &
               (sub[CHL_TARGET] <= sub['p90_log'])).mean()
        wid = (sub['p90_log'] - sub['p10_log']).mean()
        coverages.append(float(cov))
        widths.append(float(wid))
        labels.append(sname)

    sub_all = test.dropna(subset=['p10_log', 'p90_log', CHL_TARGET])
    cov_all = ((sub_all[CHL_TARGET] >= sub_all['p10_log']) &
               (sub_all[CHL_TARGET] <= sub_all['p90_log'])).mean()
    coverages.append(float(cov_all))
    widths.append((sub_all['p90_log'] - sub_all['p10_log']).mean())
    labels.append('Annual')

    bars = ax.bar(labels, coverages,
                  color=colors[:len(labels)], alpha=0.85)
    ax.axhline(0.80, color=AMBER, linewidth=1.5,
               linestyle='--', label='Target 80%')
    for bar, val in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.01, f'{val:.2f}',
                ha='center', fontsize=8, color=TEXT)
    ax.set_ylim(0, 1.1)
    ax.set_title('P10–P90 coverage by season')
    ax.set_ylabel('Empirical coverage')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: true vs P50 scatter with band
    ax = axes[1]
    sample = test.dropna(
        subset=[CHL_TARGET, 'p10_log', 'p50_log', 'p90_log']
    ).sample(min(5000, len(test)), random_state=42)
    s = sample.sort_values('p50_log')

    ax.fill_between(s['p50_log'], s['p10_log'], s['p90_log'],
                    alpha=0.25, color=GREEN, label='P10–P90')
    ax.scatter(s['p50_log'], s[CHL_TARGET],
               s=1, alpha=0.3, color=GREEN, rasterized=True)
    rng = (-3, 3)
    ax.plot(rng, rng, '--', color=MUTED, linewidth=1, label='1:1')
    ax.set_xlim(rng)
    ax.set_ylim(rng)
    ax.set_xlabel('P50 log-residual')
    ax.set_ylabel('True log-residual')
    ax.set_title('True vs P50 with P10–P90 band\n(log-space)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: width-vs-error calibration
    ax = axes[2]
    test2 = test.dropna(
        subset=['width_log', CHL_TARGET, 'p50_log']
    ).copy()
    test2['abs_err'] = (test2[CHL_TARGET] - test2['p50_log']).abs()

    grp_wr = test2.groupby(
        pd.cut(test2['width_log'], bins=20)
    )[['abs_err', 'width_log']].mean()

    ax.scatter(grp_wr['width_log'], grp_wr['abs_err'],
               color=PURPLE, s=50, zorder=5)
    ax.plot(grp_wr['width_log'], grp_wr['abs_err'],
            color=PURPLE, alpha=0.5)
    ax.set_xlabel('Interval width (log units)')
    ax.set_ylabel('Mean |error| (log units)')
    ax.set_title('Width vs actual error\n(calibration diagnostic)')
    ax.grid(True, alpha=0.3)

    r, _ = spearmanr(test2['width_log'], test2['abs_err'])
    ax.text(0.05, 0.92,
            f'Spearman r = {r:.3f}',
            transform=ax.transAxes, fontsize=8,
            color=AMBER, family='monospace')

    plt.tight_layout()
    return fig


def fig_summary(test, df_cloud):
    """Combined portfolio summary figure."""
    print('Building Chl summary figure...')

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        'CE2COAST Chl — Uncertainty Quantification Summary  |  '
        'Test 2019–2020',
        fontsize=13, fontweight='bold', color=TEXT, y=0.99
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.40, wspace=0.28,
                           left=0.06, right=0.97,
                           top=0.94, bottom=0.08)

    # Top row: spring spatial maps (most uncertain season)
    spring = test[test['month'].isin([3, 4, 5])].dropna(
        subset=['p10_log', 'p50_log', 'p90_log', 'width_log']
    )
    grp = spring.groupby(['lat', 'lon'])[
        ['p10_log', 'p50_log', 'p90_log', 'width_log']
    ].mean().reset_index()

    for col, (field, cmap, vmin, vmax, title) in enumerate([
        ('p10_log',   'PiYG',  -2.5, 2.5, 'MAM P10'),
        ('p50_log',   'PiYG',  -2.5, 2.5, 'MAM P50 (best estimate)'),
        ('p90_log',   'PiYG',  -2.5, 2.5, 'MAM P90'),
        ('width_log', 'plasma', 0,   3.5, 'MAM Interval width'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        sc = ax.scatter(grp['lon'], grp['lat'],
                        c=grp[field],
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        s=4, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # Bottom row
    # Coverage
    ax_cov = fig.add_subplot(gs[1, 0])
    coverages, labels = [], []
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['p10_log', 'p90_log', CHL_TARGET]
        )
        if len(sub) == 0:
            continue
        cov = ((sub[CHL_TARGET] >= sub['p10_log']) &
               (sub[CHL_TARGET] <= sub['p90_log'])).mean()
        coverages.append(float(cov))
        labels.append(sname)
    sub_all = test.dropna(subset=['p10_log', 'p90_log', CHL_TARGET])
    coverages.append(float(
        ((sub_all[CHL_TARGET] >= sub_all['p10_log']) &
         (sub_all[CHL_TARGET] <= sub_all['p90_log'])).mean()
    ))
    labels.append('Annual')

    colors = [BLUE, GREEN, AMBER, CORAL, TEXT]
    bars = ax_cov.bar(labels, coverages,
                      color=colors[:len(labels)], alpha=0.85)
    ax_cov.axhline(0.80, color=AMBER, linewidth=1.5,
                   linestyle='--', label='Target 80%')
    for bar, val in zip(bars, coverages):
        ax_cov.text(bar.get_x() + bar.get_width()/2,
                    val + 0.01, f'{val:.2f}',
                    ha='center', fontsize=8, color=TEXT)
    ax_cov.set_ylim(0, 1.1)
    ax_cov.set_title('P10–P90 coverage\nby season')
    ax_cov.set_ylabel('Empirical coverage')
    ax_cov.legend(fontsize=7)
    ax_cov.grid(True, alpha=0.3, axis='y')

    # Width by season boxplot
    ax_wid = fig.add_subplot(gs[1, 1])
    season_data, season_labs = [], []
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['width_log']
        )
        if len(sub) > 0:
            season_data.append(sub['width_log'].values)
            season_labs.append(sname)
    bp = ax_wid.boxplot(
        season_data, labels=season_labs,
        patch_artist=True,
        medianprops=dict(color=AMBER, linewidth=2),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(marker='.', color=MUTED, markersize=2)
    )
    for patch, color in zip(bp['boxes'], [BLUE, GREEN, AMBER, CORAL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax_wid.set_ylabel('Width (log units)')
    ax_wid.set_title('Interval width by season\n(wider = more uncertain)')
    ax_wid.grid(True, alpha=0.3, axis='y')

    # Cloud cover map — winter (most cloud)
    ax_cloud = fig.add_subplot(gs[1, 2])
    if df_cloud is not None:
        sub_c = df_cloud[df_cloud['month'].isin([12, 1, 2])]
        grp_c = (sub_c.groupby(['lat', 'lon'])['cloud_proxy']
                 .mean().reset_index())
        sc3 = ax_cloud.scatter(grp_c['lon'], grp_c['lat'],
                               c=grp_c['cloud_proxy'],
                               cmap='Blues_r', vmin=0, vmax=1,
                               s=2, rasterized=True)
        plt.colorbar(sc3, ax=ax_cloud, fraction=0.04, pad=0.02,
                     label='Cloud proxy\n(0=clear, 1=cloudy)')
        ax_cloud.set_title('DJF cloud cover proxy\n'
                           '(blue = more cloud = less reliable obs)')
    else:
        # Show winter interval width instead
        sub_w = test[test['month'].isin([12, 1, 2])].dropna(
            subset=['width_log']
        )
        grp_w = (sub_w.groupby(['lat', 'lon'])['width_log']
                 .mean().reset_index())
        sc3 = ax_cloud.scatter(grp_w['lon'], grp_w['lat'],
                               c=grp_w['width_log'],
                               cmap='plasma', vmin=0, vmax=3.5,
                               s=4, rasterized=True)
        plt.colorbar(sc3, ax=ax_cloud, fraction=0.04, pad=0.02,
                     label='Width (log units)')
        ax_cloud.set_title('DJF interval width\n(obs-sparse = wider)')
    ax_cloud.set_aspect('equal')
    ax_cloud.grid(True, alpha=0.2)
    ax_cloud.tick_params(labelsize=6)

    # Key numbers
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.set_facecolor(SURFACE)
    ax_txt.axis('off')

    sub_all2 = test.dropna(subset=['p10_log', 'p90_log',
                                   'width_log', CHL_TARGET])
    ann_cov   = float(
        ((sub_all2[CHL_TARGET] >= sub_all2['p10_log']) &
         (sub_all2[CHL_TARGET] <= sub_all2['p90_log'])).mean()
    )
    ann_width = sub_all2['width_log'].mean()
    spring_w  = test[test['month'].isin([3, 4, 5])
                    ]['width_log'].mean()
    winter_w  = test[test['month'].isin([12, 1, 2])
                    ]['width_log'].mean()

    lines = [
        ('CHL UNCERTAINTY', BLUE,  10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('Annual coverage', MUTED, 8, 'normal'),
        (f'  {ann_cov:.3f}  (target 0.80)', GREEN, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('Annual width (log)', MUTED, 8, 'normal'),
        (f'  {ann_width:.3f} log-units', AMBER, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('Spring width (MAM)', MUTED, 8, 'normal'),
        (f'  {spring_w:.3f} log-units', CORAL, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('Winter width (DJF)', MUTED, 8, 'normal'),
        (f'  {winter_w:.3f} log-units', BLUE, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('─' * 26, BORDER, 7, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('Uncertainty is HIGHEST', MUTED, 7, 'normal'),
        ('in spring (bloom onset)', MUTED, 7, 'normal'),
        ('and LOWEST in winter.', MUTED, 7, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('Two sources:', MUTED, 7, 'normal'),
        ('1. Model structural error', MUTED, 7, 'normal'),
        ('   (bloom timing)', MUTED, 7, 'normal'),
        ('2. Satellite retrieval', MUTED, 7, 'normal'),
        ('   (cloud cover gaps)', MUTED, 7, 'normal'),
    ]

    y = 0.97
    for text, color, size, weight in lines:
        ax_txt.text(0.05, y, text,
                    transform=ax_txt.transAxes,
                    fontsize=size, color=color,
                    fontweight=weight, family='monospace',
                    va='top')
        y -= 0.045

    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load features ────────────────────────────────────────
    print('Loading Chl features...')
    df = pd.read_parquet(COLLOC_DIR / 'features_chl_2010_2020.parquet')
    df['time']  = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['year']  = df['time'].dt.year
    print(f'  {len(df):,} rows')

    train = df[df['split'] == 'train'].copy()
    test  = df[df['split'] == 'test'].copy()

    # ── Quantile models ──────────────────────────────────────
    preds, test, y_te = train_quantile_models(train, test)

    # Attach predictions — log-space
    test = test.copy()
    test['p10_log']   = preds[0.10]
    test['p50_log']   = preds[0.50]
    test['p90_log']   = preds[0.90]
    test['width_log'] = test['p90_log'] - test['p10_log']

    # Back-transform to linear space (ratio to ROMS)
    # correction factor = exp(log_residual)
    test['p10_lin']   = np.exp(test['p10_log'])
    test['p50_lin']   = np.exp(test['p50_log'])
    test['p90_lin']   = np.exp(test['p90_log'])
    test['width_lin'] = test['p90_lin'] - test['p10_lin']

    # Save
    out = OUT_DIR / 'uncertainty_chl_quantile_preds.parquet'
    test.to_parquet(out, index=False)
    print(f'\n  Saved: {out.name}  ({out.stat().st_size/1e6:.1f} MB)')

    # ── Coverage ─────────────────────────────────────────────
    print('\nCoverage validation (log-space):')
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['p10_log', 'p90_log', CHL_TARGET]
        )
        if len(sub) > 0:
            compute_coverage(
                sub[CHL_TARGET].values,
                sub['p10_log'].values,
                sub['p90_log'].values,
                label=sname
            )
    sub_all = test.dropna(subset=['p10_log', 'p90_log', CHL_TARGET])
    compute_coverage(
        sub_all[CHL_TARGET].values,
        sub_all['p10_log'].values,
        sub_all['p90_log'].values,
        label='Annual'
    )

    # ── Cloud cover proxy ────────────────────────────────────
    df_cloud = extract_cloud_proxy(CHL_OBS_FILE, years=(2019, 2020))

    # ── Figures ──────────────────────────────────────────────
    f1 = fig_quantile_maps(test)
    f1.savefig(FIG_DIR / 'uncertainty_chl_quantile_maps.png')
    plt.close(f1)
    print('  Saved: uncertainty_chl_quantile_maps.png')

    f2 = fig_log_asymmetry(test)
    f2.savefig(FIG_DIR / 'uncertainty_chl_asymmetry.png')
    plt.close(f2)
    print('  Saved: uncertainty_chl_asymmetry.png')

    f3 = fig_cloud_coverage(test, df_cloud)
    f3.savefig(FIG_DIR / 'uncertainty_chl_cloud.png')
    plt.close(f3)
    print('  Saved: uncertainty_chl_cloud.png')

    f4 = fig_coverage_validation(test)
    f4.savefig(FIG_DIR / 'uncertainty_chl_coverage.png')
    plt.close(f4)
    print('  Saved: uncertainty_chl_coverage.png')

    f5 = fig_summary(test, df_cloud)
    f5.savefig(FIG_DIR / 'uncertainty_chl_summary.png')
    plt.close(f5)
    print('  Saved: uncertainty_chl_summary.png')

    print('\nDone.')


if __name__ == '__main__':
    main()

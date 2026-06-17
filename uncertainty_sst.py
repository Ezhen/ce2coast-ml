"""
uncertainty_sst.py
==================
Uncertainty quantification for CE2COAST SST correction pipeline.

Three complementary uncertainty layers:

  1. XGBoost quantile regression (P10 / P50 / P90)
     — prediction interval on the residual correction itself

  2. OI analysis spread
     — spatially-varying posterior uncertainty from Kalman update
     — already computed by kalman_update_roms.py

  3. Coverage validation
     — empirical check: do 80% of true residuals fall inside P10-P90?
     — seasonal breakdown

Outputs:
  figures/uncertainty_quantile_maps.png   — spatial P10/P50/P90 maps
  figures/uncertainty_oi_spread.png       — OI spread seasonal maps
  figures/uncertainty_coverage.png        — coverage validation figure
  figures/uncertainty_summary.png         — combined portfolio figure
  uncertainty_quantile_preds.parquet      — quantile predictions

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import xgboost as xgb
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
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
KALMAN_DIR = KALMAN_DIR
OUT_DIR    = COLLOC_DIR
FIG_DIR    = FIG_DIR

SST_FEATURES = [
    'sst_roms', 'salt_roms', 'zeta_roms',
    'sustr_roms', 'svstr_roms',
    'depth_m', 'coast_dist',
    'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'residual_sst_lag1',
]
SST_TARGET = 'residual_sst'

# Quantiles to predict
QUANTILES = [0.10, 0.50, 0.90]

XGB_BASE = dict(
    n_estimators        = 500,
    max_depth           = 6,
    learning_rate       = 0.05,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    min_child_weight    = 10,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    tree_method         = 'hist',
    n_jobs              = -1,
    random_state        = 42,
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
# STEP 1 — QUANTILE REGRESSION
# ─────────────────────────────────────────────────────────────

def train_quantile_models(df_train, df_test):
    """
    Train one XGBoost model per quantile (P10, P50, P90).
    Returns dict of {quantile: predictions_on_test}.
    """
    print('\nTraining quantile models...')

    # Force numeric
    for col in SST_FEATURES:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col]  = pd.to_numeric(df_test[col],  errors='coerce')

    train = df_train.dropna(subset=SST_FEATURES + [SST_TARGET])
    test  = df_test.dropna(subset=SST_FEATURES + [SST_TARGET])

    X_tr = train[SST_FEATURES].values
    y_tr = train[SST_TARGET].values
    X_te = test[SST_FEATURES].values
    y_te = test[SST_TARGET].values

    preds = {}
    for q in QUANTILES:
        print(f'  Quantile {q:.0%}...', end=' ', flush=True)
        params = {**XGB_BASE,
                  'objective':  'reg:quantileerror',
                  'quantile_alpha': q,
                  'verbosity':  0}
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)
        preds[q] = model.predict(X_te)
        print(f'done  (best iter={model.best_iteration})', flush=True)

    return preds, test, y_te


def compute_coverage(y_true, p10, p90, label=''):
    """Empirical coverage: fraction of true values inside [P10, P90]."""
    inside    = (y_true >= p10) & (y_true <= p90)
    coverage  = inside.mean()
    mean_width = (p90 - p10).mean()
    print(f'  {label} coverage={coverage:.3f}  '
          f'(target=0.80)  mean width={mean_width:.3f}°C')
    return float(coverage), float(mean_width)


# ─────────────────────────────────────────────────────────────
# STEP 2 — FIGURES
# ─────────────────────────────────────────────────────────────

def fig_quantile_maps(test, preds):
    """
    Figure 1: Spatial maps of P10, P50, P90 and interval width.
    One row per season, 4 columns: P10 | P50 | P90 | width.
    """
    print('\nBuilding quantile maps figure...')

    test = test.copy()
    test['p10']   = preds[0.10]
    test['p50']   = preds[0.50]
    test['p90']   = preds[0.90]
    test['width'] = test['p90'] - test['p10']

    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(
        'CE2COAST SST — XGBoost Quantile Uncertainty  |  Test 2019–2020\n'
        'P10 / P50 / P90 of residual correction  |  '
        'Width = P90 − P10 (prediction interval)',
        fontsize=11, fontweight='bold', color=TEXT, y=0.998
    )

    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.32, wspace=0.12,
                           left=0.05, right=0.95,
                           top=0.965, bottom=0.04)

    err_max   = 3.0
    width_max = 4.0

    for row, (sname, months) in enumerate(SEASONS.items()):
        sub = test[test['month'].isin(months)]
        grp = sub.groupby(['lat', 'lon'])[
            ['p10', 'p50', 'p90', 'width']
        ].mean().reset_index()

        sc_ref = None
        for col, (col_name, cmap, vmin, vmax, title) in enumerate([
            ('p10',   'RdBu_r', -err_max,   err_max,   f'{sname} P10'),
            ('p50',   'RdBu_r', -err_max,   err_max,   f'{sname} P50 (median)'),
            ('p90',   'RdBu_r', -err_max,   err_max,   f'{sname} P90'),
            ('width', 'plasma',  0,          width_max, f'{sname} Interval width'),
        ]):
            ax = fig.add_subplot(gs[row, col])
            sc = ax.scatter(grp['lon'], grp['lat'],
                            c=grp[col_name],
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            s=3, rasterized=True)
            ax.set_title(title, pad=3, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)
            if col < 3:
                sc_ref = sc
            if col == 0:
                ax.set_ylabel('Lat', fontsize=7)
            # Annotate mean
            ax.text(0.02, 0.03,
                    f'mean={grp[col_name].mean():+.2f}°C',
                    transform=ax.transAxes, fontsize=6,
                    color=TEXT, family='monospace')

    # Colorbars
    cbar1 = fig.add_axes([0.05, 0.005, 0.67, 0.010])
    sm1 = plt.cm.ScalarMappable(
        cmap='RdBu_r',
        norm=plt.Normalize(-err_max, err_max)
    )
    cb1 = plt.colorbar(sm1, cax=cbar1, orientation='horizontal')
    cb1.set_label('Residual correction (°C)  |  P10 / P50 / P90',
                  color=TEXT, fontsize=8)
    cb1.ax.tick_params(colors=MUTED, labelsize=7)

    cbar2 = fig.add_axes([0.77, 0.005, 0.18, 0.010])
    sm2 = plt.cm.ScalarMappable(
        cmap='plasma',
        norm=plt.Normalize(0, width_max)
    )
    cb2 = plt.colorbar(sm2, cax=cbar2, orientation='horizontal')
    cb2.set_label('Interval width (°C)', color=TEXT, fontsize=8)
    cb2.ax.tick_params(colors=MUTED, labelsize=7)

    return fig, test


def fig_coverage_validation(test):
    """
    Figure 2: Coverage validation.
    Row 1: reliability diagram (observed coverage vs nominal)
    Row 2: interval width distribution by season
    Row 3: scatter of true residual vs P50, shaded P10-P90
    """
    print('Building coverage validation figure...')

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'CE2COAST SST — Uncertainty Calibration  |  Test 2019–2020',
        fontsize=12, fontweight='bold', color=TEXT
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.38, wspace=0.30,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.08)

    ax_rel  = fig.add_subplot(gs[0, 0])
    ax_wid  = fig.add_subplot(gs[0, 1])
    ax_scat = fig.add_subplot(gs[0, 2])
    ax_map  = fig.add_subplot(gs[1, :])

    # ── Panel 1: Coverage by season ──────────────────────────
    coverages = []
    widths    = []
    labels    = []
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['p10', 'p90', SST_TARGET]
        )
        if len(sub) == 0:
            continue
        cov  = ((sub[SST_TARGET] >= sub['p10']) &
                (sub[SST_TARGET] <= sub['p90'])).mean()
        wid  = (sub['p90'] - sub['p10']).mean()
        coverages.append(float(cov))
        widths.append(float(wid))
        labels.append(sname)

    # Annual
    sub_all = test.dropna(subset=['p10', 'p90', SST_TARGET])
    cov_all = ((sub_all[SST_TARGET] >= sub_all['p10']) &
               (sub_all[SST_TARGET] <= sub_all['p90'])).mean()
    coverages.append(float(cov_all))
    widths.append((sub_all['p90'] - sub_all['p10']).mean())
    labels.append('Annual')

    colors = [BLUE, GREEN, AMBER, CORAL, TEXT]
    x      = np.arange(len(labels))
    bars   = ax_rel.bar(x, coverages, color=colors[:len(labels)],
                        alpha=0.85, width=0.55)
    ax_rel.axhline(0.80, color=AMBER, linewidth=1.5,
                   linestyle='--', label='Target 80%')
    ax_rel.axhline(1.00, color=MUTED, linewidth=0.8,
                   linestyle=':', alpha=0.5)
    for bar, val in zip(bars, coverages):
        ax_rel.text(bar.get_x() + bar.get_width()/2,
                    val + 0.01, f'{val:.2f}',
                    ha='center', fontsize=8, color=TEXT)
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(labels)
    ax_rel.set_ylim(0, 1.1)
    ax_rel.set_ylabel('Empirical coverage')
    ax_rel.set_title('P10–P90 coverage by season\n(target = 0.80)')
    ax_rel.legend(fontsize=7)
    ax_rel.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Width distribution by season ────────────────
    season_data = []
    season_labs = []
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(subset=['width'])
        if len(sub) > 0:
            season_data.append(sub['width'].values)
            season_labs.append(sname)

    bp = ax_wid.boxplot(
        season_data,
        labels=season_labs,
        patch_artist=True,
        medianprops=dict(color=AMBER, linewidth=2),
        whiskerprops=dict(color=MUTED),
        capprops=dict(color=MUTED),
        flierprops=dict(marker='.', color=MUTED, markersize=2)
    )
    for patch, color in zip(bp['boxes'],
                             [BLUE, GREEN, AMBER, CORAL]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax_wid.set_ylabel('Prediction interval width (°C)')
    ax_wid.set_title('P10–P90 width distribution\nby season')
    ax_wid.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Scatter true vs P50 with P10-P90 band ───────
    sample = test.dropna(
        subset=[SST_TARGET, 'p10', 'p50', 'p90']
    ).sample(min(5000, len(test)), random_state=42)

    # Sort by P50 for clean band
    s = sample.sort_values('p50')

    ax_scat.fill_between(
        s['p50'], s['p10'], s['p90'],
        alpha=0.25, color=BLUE, label='P10–P90 band'
    )
    ax_scat.scatter(
        s['p50'], s[SST_TARGET],
        s=1, alpha=0.3, color=BLUE, rasterized=True
    )
    rng = (-4, 4)
    ax_scat.plot(rng, rng, '--', color=MUTED,
                 linewidth=1, label='1:1')
    ax_scat.set_xlim(rng)
    ax_scat.set_ylim(rng)
    ax_scat.set_xlabel('P50 predicted residual (°C)')
    ax_scat.set_ylabel('True residual (°C)')
    ax_scat.set_title('True residual vs P50\nwith P10–P90 band')
    ax_scat.legend(fontsize=7)
    ax_scat.grid(True, alpha=0.3)

    # ── Panel 4: Spatial map of interval width ────────────────
    grp = test.dropna(subset=['width']).groupby(
        ['lat', 'lon']
    )['width'].mean().reset_index()

    sc = ax_map.scatter(
        grp['lon'], grp['lat'],
        c=grp['width'],
        cmap='plasma', vmin=0, vmax=4,
        s=4, rasterized=True
    )
    plt.colorbar(sc, ax=ax_map, fraction=0.015, pad=0.01,
                 label='Mean interval width (°C)')
    ax_map.set_title(
        'Spatial distribution of prediction uncertainty\n'
        'Wide intervals = model uncertain here  |  '
        'Narrow = confident correction'
    )
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)

    return fig


def fig_oi_spread(df_analysis):
    """
    Figure 3: OI analysis spread — posterior uncertainty after
    Kalman update. Four seasonal panels.
    """
    print('Building OI spread figure...')

    if 'spread' not in df_analysis.columns:
        print('  [WARN] No spread column in analysis — skipping')
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'CE2COAST SST — OI Analysis Spread (posterior uncertainty)  |'
        '  2019–2020\n'
        'σ_a = √(B − KHB)  |  '
        'Narrow = well-constrained by obs  |  '
        'Wide = obs-sparse region',
        fontsize=10, fontweight='bold', color=TEXT
    )

    axes_flat = axes.ravel()
    vmax      = 1.0   # °C — spread should be < background σ_b

    for ax, (sname, months) in zip(axes_flat, SEASONS.items()):
        sub = df_analysis[df_analysis['month'].isin(months)]
        grp = sub.groupby(['lat', 'lon'])['spread'].mean().reset_index()

        sc = ax.scatter(
            grp['lon'], grp['lat'],
            c=grp['spread'],
            cmap='YlOrRd_r', vmin=0, vmax=vmax,
            s=4, rasterized=True
        )
        mean_sp = grp['spread'].mean()
        min_sp  = grp['spread'].min()

        ax.set_title(f'{sname}  |  mean σ_a = {mean_sp:.3f}°C',
                     fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.03,
                f'min σ_a = {min_sp:.3f}°C  (near stations)',
                transform=ax.transAxes, fontsize=7,
                color=TEXT, family='monospace')

    plt.colorbar(sc, ax=axes_flat, fraction=0.02, pad=0.02,
                 label='Analysis spread σ_a (°C)  |  '
                       'yellow = low uncertainty, red = high')
    plt.tight_layout()
    return fig


def fig_summary_uncertainty(test, df_analysis):
    """
    Figure 4: Combined portfolio figure.
    Two-row: top = uncertainty landscape, bottom = calibration summary.
    """
    print('Building summary figure...')

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        'CE2COAST SST — Uncertainty Quantification Summary  |  '
        'Test 2019–2020',
        fontsize=13, fontweight='bold', color=TEXT, y=0.99
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.40, wspace=0.28,
                           left=0.06, right=0.97,
                           top=0.94, bottom=0.08)

    # ── Top row: JJA spatial uncertainty ─────────────────────
    summer = test[test['month'].isin([6, 7, 8])].dropna(
        subset=['p10', 'p50', 'p90', 'width', SST_TARGET]
    )
    grp = summer.groupby(['lat', 'lon'])[
        ['p10', 'p50', 'p90', 'width']
    ].mean().reset_index()

    for col, (field, cmap, vmin, vmax, title) in enumerate([
        ('p10',   'RdBu_r', -3,  3,  'P10  (lower bound)'),
        ('p50',   'RdBu_r', -3,  3,  'P50  (best estimate)'),
        ('p90',   'RdBu_r', -3,  3,  'P90  (upper bound)'),
        ('width', 'plasma',  0,  4,  'Interval width'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        sc = ax.scatter(grp['lon'], grp['lat'],
                        c=grp[field],
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        s=4, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                     label='°C')
        ax.set_title(f'JJA  |  {title}', fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # ── Bottom row: calibration metrics ──────────────────────

    # Coverage bar
    ax_cov = fig.add_subplot(gs[1, 0])
    coverages, labels = [], []
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['p10', 'p90', SST_TARGET]
        )
        if len(sub) == 0:
            continue
        cov = ((sub[SST_TARGET] >= sub['p10']) &
               (sub[SST_TARGET] <= sub['p90'])).mean()
        coverages.append(float(cov))
        labels.append(sname)
    annual = test.dropna(subset=['p10', 'p90', SST_TARGET])
    cov_ann = ((annual[SST_TARGET] >= annual['p10']) &
               (annual[SST_TARGET] <= annual['p90'])).mean()
    coverages.append(float(cov_ann))
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
    ax_cov.set_title('P10–P90 coverage\n(empirical vs 80% target)')
    ax_cov.set_ylabel('Coverage')
    ax_cov.legend(fontsize=7)
    ax_cov.grid(True, alpha=0.3, axis='y')

    # Width vs RMSE scatter — uncertainty width vs actual error
    ax_wr = fig.add_subplot(gs[1, 1])
    sub   = test.dropna(subset=['width', SST_TARGET, 'p50'])
    sub   = sub.copy()
    sub['abs_err'] = (sub[SST_TARGET] - sub['p50']).abs()
    grp_wr = sub.groupby(
        pd.cut(sub['width'], bins=20)
    )[['abs_err', 'width']].mean()

    ax_wr.scatter(grp_wr['width'], grp_wr['abs_err'],
                  color=BLUE, s=40, zorder=5)
    ax_wr.plot(grp_wr['width'], grp_wr['abs_err'],
               color=BLUE, alpha=0.5, linewidth=1)
    ax_wr.set_xlabel('Prediction interval width (°C)')
    ax_wr.set_ylabel('Mean |error| (°C)')
    ax_wr.set_title('Width vs actual error\n(wider = less confident = larger error?)')
    ax_wr.grid(True, alpha=0.3)

    corr = sub[['width', 'abs_err']].corr().iloc[0, 1]
    ax_wr.text(0.05, 0.92,
               f'Pearson r = {corr:.3f}',
               transform=ax_wr.transAxes,
               fontsize=8, color=AMBER, family='monospace')

    # OI spread map — summer
    if df_analysis is not None and 'spread' in df_analysis.columns:
        ax_sp = fig.add_subplot(gs[1, 2])
        sub_s = df_analysis[df_analysis['month'].isin([6, 7, 8])]
        grp_s = sub_s.groupby(['lat', 'lon'])['spread'].mean().reset_index()
        sc2   = ax_sp.scatter(grp_s['lon'], grp_s['lat'],
                              c=grp_s['spread'],
                              cmap='YlOrRd_r', vmin=0, vmax=1.0,
                              s=4, rasterized=True)
        plt.colorbar(sc2, ax=ax_sp, fraction=0.04, pad=0.02,
                     label='σ_a (°C)')
        ax_sp.set_title('JJA OI analysis spread σ_a\n'
                        '(posterior uncertainty after in-situ update)')
        ax_sp.set_aspect('equal')
        ax_sp.grid(True, alpha=0.2)
        ax_sp.tick_params(labelsize=6)

    # Key numbers panel
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.set_facecolor(SURFACE)
    ax_txt.axis('off')

    annual_cov   = cov_ann
    annual_width = test.dropna(subset=['width'])['width'].mean()
    sigma_b      = np.sqrt(0.73**2)

    lines = [
        ('UNCERTAINTY METRICS', '#2f81f7', 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        (f'Annual P10–P90 coverage', MUTED, 8, 'normal'),
        (f'  {annual_cov:.3f}  (target 0.80)', GREEN, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        (f'Mean interval width', MUTED, 8, 'normal'),
        (f'  {annual_width:.3f} °C', AMBER, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        (f'Background σ_b (XGB RMSE)', MUTED, 8, 'normal'),
        (f'  {sigma_b:.3f} °C', BLUE, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        (f'Width / σ_b ratio', MUTED, 8, 'normal'),
        (f'  {annual_width/sigma_b:.2f}×', AMBER, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('─' * 28, BORDER, 7, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('Wider interval = model in', MUTED, 7, 'normal'),
        ('unfamiliar regime.', MUTED, 7, 'normal'),
        ('Calibration check: observed', MUTED, 7, 'normal'),
        ('coverage should equal nominal', MUTED, 7, 'normal'),
        ('quantile level (80%).', MUTED, 7, 'normal'),
    ]

    y = 0.97
    for text, color, size, weight in lines:
        ax_txt.text(0.05, y, text,
                    transform=ax_txt.transAxes,
                    fontsize=size, color=color,
                    fontweight=weight, family='monospace',
                    va='top')
        y -= 0.05

    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load features ────────────────────────────────────────
    print('Loading SST features...')
    df = pd.read_parquet(COLLOC_DIR / 'features_sst_2010_2020.parquet')
    df['time']  = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['year']  = df['time'].dt.year
    print(f'  {len(df):,} rows')

    train = df[df['split'] == 'train'].copy()
    test  = df[df['split'] == 'test'].copy()

    # Remove extreme outliers from training
    train = train[train['residual_sst_extreme'] == 0]

    # ── Quantile regression ──────────────────────────────────
    preds, test, y_te = train_quantile_models(train, test)

    # Save predictions
    test = test.copy()
    test['p10']   = preds[0.10]
    test['p50']   = preds[0.50]
    test['p90']   = preds[0.90]
    test['width'] = test['p90'] - test['p10']

    out = OUT_DIR / 'uncertainty_quantile_preds.parquet'
    test.to_parquet(out, index=False)
    print(f'\n  Saved: {out.name}  ({out.stat().st_size/1e6:.1f} MB)')

    # ── Coverage ─────────────────────────────────────────────
    print('\nCoverage validation:')
    for sname, months in SEASONS.items():
        sub = test[test['month'].isin(months)].dropna(
            subset=['p10', 'p90', SST_TARGET]
        )
        if len(sub) > 0:
            compute_coverage(
                sub[SST_TARGET].values,
                sub['p10'].values,
                sub['p90'].values,
                label=sname
            )
    compute_coverage(
        test.dropna(subset=['p10', 'p90', SST_TARGET])[SST_TARGET].values,
        test.dropna(subset=['p10', 'p90', SST_TARGET])['p10'].values,
        test.dropna(subset=['p10', 'p90', SST_TARGET])['p90'].values,
        label='Annual'
    )

    # ── Load OI analysis ─────────────────────────────────────
    df_analysis = None
    ap = KALMAN_DIR / 'analysis_sst.parquet'
    if ap.exists():
        df_analysis = pd.read_parquet(ap)
        df_analysis['time']  = pd.to_datetime(df_analysis['time'])
        df_analysis['month'] = df_analysis['time'].dt.month
        df_analysis['year']  = df_analysis['time'].dt.year

    # ── Figures ──────────────────────────────────────────────
    f1, test = fig_quantile_maps(test, preds)
    f1.savefig(FIG_DIR / 'uncertainty_quantile_maps.png')
    plt.close(f1)
    print(f'  Saved: uncertainty_quantile_maps.png')

    f2 = fig_coverage_validation(test)
    f2.savefig(FIG_DIR / 'uncertainty_coverage.png')
    plt.close(f2)
    print(f'  Saved: uncertainty_coverage.png')

    if df_analysis is not None:
        f3 = fig_oi_spread(df_analysis)
        if f3:
            f3.savefig(FIG_DIR / 'uncertainty_oi_spread.png')
            plt.close(f3)
            print(f'  Saved: uncertainty_oi_spread.png')

    f4 = fig_summary_uncertainty(test, df_analysis)
    f4.savefig(FIG_DIR / 'uncertainty_summary.png')
    plt.close(f4)
    print(f'  Saved: uncertainty_summary.png')

    print('\nDone.')


if __name__ == '__main__':
    main()

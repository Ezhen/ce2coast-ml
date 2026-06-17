"""
visualise_results_roms.py
=========================
Diagnostic visualisation of XGBoost ROMS bias correction.

Produces two figure sets:
  figures/sst_diagnostics.png  — 6-panel SST results
  figures/chl_diagnostics.png  — 6-panel Chl results

Panels per figure:
  1. Scatter: obs vs ROMS raw
  2. Scatter: obs vs XGB corrected
  3. Monthly bias before/after correction
  4. Residual map (spatial bias, annual mean)
  5. Residual histogram before/after
  6. Feature importance bar chart

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from ce2coast_config import (
    COLLOC_DIR, ROMS_DIR, VALID_DIR, INSITU_DIR,
    FIG_DIR, KALMAN_DIR, ROMS_AVG_PATTERN, ROMS_BIO_PATTERN,
    ROMS_GRID_FILE, SST_OBS_FILE, CHL_OBS_FILE,
)
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
FIG_DIR    = FIG_DIR

# ── Aesthetics ───────────────────────────────────────────────
DARK_BG   = '#0d1117'
SURFACE   = '#161b22'
BORDER    = '#30363d'
TEXT      = '#e6edf3'
MUTED     = '#7d8590'
ACCENT    = '#2f81f7'    # blue  — XGB corrected
ACCENT2   = '#f0644a'    # red   — ROMS raw
ACCENT3   = '#3fb950'    # green — observations
WARN      = '#d29922'    # amber — feature importance

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
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'legend.fontsize':   8,
    'legend.facecolor':  SURFACE,
    'legend.edgecolor':  BORDER,
    'savefig.facecolor': DARK_BG,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})

MONTH_LABELS = ['J','F','M','A','M','J','J','A','S','O','N','D']

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def metrics_str(obs, pred):
    from sklearn.metrics import r2_score, mean_squared_error
    r2   = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    bias = np.mean(pred - obs)
    return f'R²={r2:.3f}  RMSE={rmse:.3f}  bias={bias:+.3f}'


def density_scatter(ax, x, y, color, label, n_sample=30_000, alpha=0.6):
    """Scatter with kernel density colouring — readable at 2M points."""
    if len(x) > n_sample:
        idx = np.random.choice(len(x), n_sample, replace=False)
        x, y = x[idx], y[idx]
    try:
        xy  = np.vstack([x, y])
        z   = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(x[idx], y[idx], c=z[idx], s=2, alpha=alpha,
                   cmap='plasma', rasterized=True, label=label)
    except Exception:
        ax.scatter(x, y, s=2, alpha=0.3, color=color,
                   rasterized=True, label=label)


def add_diagonal(ax, vmin, vmax):
    ax.plot([vmin, vmax], [vmin, vmax], '--', color=MUTED,
            linewidth=1, zorder=5, label='1:1')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)


def style_ax(ax, title='', xlabel='', ylabel='', grid=True):
    ax.set_title(title, pad=6, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=8)


# ─────────────────────────────────────────────────────────────
# PANEL BUILDERS
# ─────────────────────────────────────────────────────────────

def panel_scatter_raw(ax, obs, roms, vmin, vmax, units):
    density_scatter(ax, obs, roms, ACCENT2, 'ROMS raw')
    add_diagonal(ax, vmin, vmax)
    style_ax(ax, title='ROMS raw vs obs',
             xlabel=f'Observed ({units})', ylabel=f'ROMS raw ({units})')
    ax.text(0.03, 0.97, metrics_str(obs, roms),
            transform=ax.transAxes, fontsize=7,
            va='top', color=ACCENT2, family='monospace')


def panel_scatter_corrected(ax, obs, corrected, vmin, vmax, units):
    density_scatter(ax, obs, corrected, ACCENT, 'XGB corrected')
    add_diagonal(ax, vmin, vmax)
    style_ax(ax, title='XGB corrected vs obs',
             xlabel=f'Observed ({units})', ylabel=f'XGB corrected ({units})')
    ax.text(0.03, 0.97, metrics_str(obs, corrected),
            transform=ax.transAxes, fontsize=7,
            va='top', color=ACCENT, family='monospace')


def panel_monthly_bias(ax, df, obs_col, roms_col, corrected_col, units):
    grp_raw  = df.groupby('month')[roms_col].mean() - df.groupby('month')[obs_col].mean()
    grp_corr = df.groupby('month')[corrected_col].mean() - df.groupby('month')[obs_col].mean()

    months = np.arange(1, 13)
    ax.bar(months - 0.2, grp_raw.values,  width=0.35,
           color=ACCENT2, alpha=0.8, label='ROMS raw')
    ax.bar(months + 0.2, grp_corr.values, width=0.35,
           color=ACCENT,  alpha=0.8, label='XGB corrected')
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle='--')
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)
    style_ax(ax, title='Monthly bias (model − obs)',
             xlabel='Month', ylabel=f'Bias ({units})')
    ax.legend(loc='upper right')


def panel_spatial_bias(ax, df, residual_col, title, cmap='RdBu_r',
                       vmax=None, log_space=False):
    """Mean annual residual on ROMS grid — scatter plot coloured by bias."""
    annual = df.groupby(['lat', 'lon'])[residual_col].mean().reset_index()

    if vmax is None:
        vmax = np.percentile(np.abs(annual[residual_col]), 95)
    vmin = -vmax

    sc = ax.scatter(
        annual['lon'], annual['lat'],
        c=annual[residual_col],
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=3, rasterized=True
    )
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02,
                 label='mean residual')
    style_ax(ax, title=title, xlabel='Longitude', ylabel='Latitude')
    ax.set_aspect('equal')


def panel_histogram(ax, df, residual_col, corrected_residual, units, log_space=False):
    """Residual distributions before and after correction."""
    raw  = df[residual_col].dropna().values
    corr = corrected_residual.dropna().values

    clip = np.percentile(np.abs(raw), 99)
    bins = np.linspace(-clip, clip, 80)

    ax.hist(raw,  bins=bins, color=ACCENT2, alpha=0.6,
            density=True, label='ROMS raw residual')
    ax.hist(corr, bins=bins, color=ACCENT,  alpha=0.6,
            density=True, label='XGB residual')
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle='--')

    ax.set_xlabel(f'Residual ({units})')
    ax.set_ylabel('Density')
    style_ax(ax, title='Residual distribution')
    ax.legend()

    # Stats annotation
    ax.text(0.97, 0.97,
            f'raw  σ={np.std(raw):.3f}\ncorr σ={np.std(corr):.3f}',
            transform=ax.transAxes, fontsize=7,
            va='top', ha='right', color=TEXT, family='monospace')


def panel_feature_importance(ax, df_imp):
    """Horizontal bar chart of feature importance."""
    df_imp = df_imp.sort_values('gain_pct')
    colors = [WARN if p > df_imp['gain_pct'].median() else MUTED
              for p in df_imp['gain_pct']]

    bars = ax.barh(df_imp['feature'], df_imp['gain_pct'],
                   color=colors, alpha=0.85, height=0.6)

    # Value labels
    for bar, val in zip(bars, df_imp['gain_pct']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=7, color=TEXT)

    style_ax(ax, title='Feature importance (XGBoost gain)',
             xlabel='Gain %', grid=False)
    ax.set_xlim(0, df_imp['gain_pct'].max() * 1.2)
    ax.tick_params(axis='y', labelsize=7)


# ─────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────

def build_sst_figure(df, df_imp):
    print('  Building SST figure...')

    obs       = df['sst_obs'].values
    roms      = df['sst_roms'].values
    corrected = df['sst_corrected'].values

    vmin = np.percentile(obs, 1)
    vmax = np.percentile(obs, 99)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        'CE2COAST ROMS SST — XGBoost Bias Correction  |  Test period 2019–2020',
        fontsize=13, fontweight='bold', color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    panel_scatter_raw(ax1, obs, roms, vmin, vmax, '°C')
    panel_scatter_corrected(ax2, obs, corrected, vmin, vmax, '°C')
    panel_monthly_bias(ax3, df, 'sst_obs', 'sst_roms', 'sst_corrected', '°C')

    # Spatial bias map
    panel_spatial_bias(ax4, df,
                       residual_col='residual_sst',
                       title='Annual mean SST bias (ROMS − obs)',
                       vmax=3.0)

    # Histogram — corrected residual = obs - corrected
    df['corrected_residual_sst'] = df['sst_obs'] - df['sst_corrected']
    panel_histogram(ax5, df,
                    residual_col='residual_sst',
                    corrected_residual=df['corrected_residual_sst'],
                    units='°C')

    panel_feature_importance(ax6, df_imp)

    return fig


def build_chl_figure(df, df_imp):
    print('  Building Chl figure...')

    obs       = df['chl_obs'].values
    roms      = df['chl_roms'].values
    corrected = df['chl_corrected'].values

    # Log-space for scatter plots
    log_obs  = np.log10(np.clip(obs,  0.01, None))
    log_roms = np.log10(np.clip(roms, 0.01, None))
    log_corr = np.log10(np.clip(corrected, 0.01, None))

    vmin = np.percentile(log_obs, 1)
    vmax = np.percentile(log_obs, 99)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        'CE2COAST ROMS Chl — XGBoost Bias Correction  |  Test period 2019–2020',
        fontsize=13, fontweight='bold', color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    panel_scatter_raw(ax1, log_obs, log_roms, vmin, vmax, 'log₁₀ mg/m³')
    panel_scatter_corrected(ax2, log_obs, log_corr, vmin, vmax, 'log₁₀ mg/m³')
    panel_monthly_bias(ax3, df, 'chl_obs', 'chl_roms', 'chl_corrected', 'mg/m³')

    panel_spatial_bias(ax4, df,
                       residual_col='log_residual_chl',
                       title='Annual mean Chl log-bias (log obs/roms)',
                       cmap='PiYG', vmax=1.5)

    df['corrected_log_residual_chl'] = (
        np.log(np.clip(df['chl_obs'],       0.001, None)) -
        np.log(np.clip(df['chl_corrected'], 0.001, None))
    )
    panel_histogram(ax5, df,
                    residual_col='log_residual_chl',
                    corrected_residual=df['corrected_log_residual_chl'],
                    units='log mg/m³')

    panel_feature_importance(ax6, df_imp)

    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── SST ─────────────────────────────────────────────────
    print('Loading SST predictions...')
    sst     = pd.read_parquet(COLLOC_DIR / 'predictions_sst_test.parquet')
    imp_sst = pd.read_parquet(COLLOC_DIR / 'importance_sst.parquet')
    print(f'  {len(sst):,} test rows')

    fig_sst = build_sst_figure(sst, imp_sst)
    out_sst = FIG_DIR / 'sst_diagnostics.png'
    fig_sst.savefig(out_sst)
    plt.close(fig_sst)
    print(f'  Saved: {out_sst}')

    # ── Chl ─────────────────────────────────────────────────
    print('\nLoading Chl predictions...')
    chl     = pd.read_parquet(COLLOC_DIR / 'predictions_chl_test.parquet')
    imp_chl = pd.read_parquet(COLLOC_DIR / 'importance_chl.parquet')
    print(f'  {len(chl):,} test rows')

    fig_chl = build_chl_figure(chl, imp_chl)
    out_chl = FIG_DIR / 'chl_diagnostics.png'
    fig_chl.savefig(out_chl)
    plt.close(fig_chl)
    print(f'  Saved: {out_chl}')

    print('\nDone. Figures in:')
    print(f'  {FIG_DIR}')


if __name__ == '__main__':
    main()

"""
seasonal_error_maps.py
======================
Seasonal error maps for CE2COAST SST bias correction.

Four-season × three-layer grid:
  Rows: DJF (winter), MAM (spring), JJA (summer), SON (autumn)
  Cols: ROMS error | XGB error | XGB - ROMS error reduction

Error defined as model - obs (positive = model too warm).

Also produces:
  - XGB corrected - OI analysis difference map (where OI matters)

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
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

COLLOC_DIR = COLLOC_DIR
KALMAN_DIR = KALMAN_DIR
FIG_DIR    = FIG_DIR

SEASONS = {
    'DJF (Winter)': [12, 1, 2],
    'MAM (Spring)': [3, 4, 5],
    'JJA (Summer)': [6, 7, 8],
    'SON (Autumn)': [9, 10, 11],
}

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
    'grid.linewidth':    0.5,
    'font.family':       'monospace',
    'font.size':         8,
    'axes.titlesize':    9,
    'savefig.facecolor': DARK_BG,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})


# ─────────────────────────────────────────────────────────────
# SPATIAL MEAN PER GRID POINT
# ─────────────────────────────────────────────────────────────

def seasonal_mean_error(df, months, error_col):
    """
    Mean error per ROMS grid point for a given season.
    Returns DataFrame with lat, lon, mean_error.
    """
    sub = df[df['month'].isin(months)].copy()
    out = (sub.groupby(['lat', 'lon'])[error_col]
             .mean()
             .reset_index()
             .rename(columns={error_col: 'error'}))
    return out


def scatter_map(ax, df_map, vmin, vmax, cmap, title, s=3):
    """Scatter plot of error field on lat/lon grid."""
    sc = ax.scatter(
        df_map['lon'], df_map['lat'],
        c=df_map['error'],
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=s, rasterized=True
    )
    ax.set_title(title, pad=4, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    return sc


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — SEASONAL ERROR MAPS (4 × 3)
# ─────────────────────────────────────────────────────────────

def build_seasonal_figure(df):
    """
    4-row × 3-col figure:
      Rows: DJF / MAM / JJA / SON
      Cols: ROMS error | XGB error | error reduction (ROMS - XGB)
    """
    print('Building seasonal error figure...')

    # Compute errors (model - obs, positive = too warm)
    df = df.copy()
    df['roms_error'] = df['sst_roms']      - df['sst_obs']
    df['xgb_error']  = df['sst_corrected'] - df['sst_obs']
    df['reduction']  = df['roms_error']    - df['xgb_error']  # positive = improvement

    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(
        'CE2COAST SST — Seasonal Error Maps  |  Test period 2019–2020\n'
        'Error = model − obs  (positive = model too warm)',
        fontsize=12, fontweight='bold', color=TEXT, y=0.995
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.35, wspace=0.15,
        left=0.06, right=0.94,
        top=0.965, bottom=0.04
    )

    # Colour scale — symmetric around zero for errors
    err_max  = 3.0   # °C
    red_max  = 2.5   # error reduction max

    cmap_err = 'RdBu_r'    # red = too warm, blue = too cold
    cmap_red = 'PiYG'      # green = improvement, pink = degradation

    col_titles = ['ROMS error (°C)', 'XGB error (°C)',
                  'Error reduction (°C)\nROMS − XGB']

    # Column header row (text only)
    for col, title in enumerate(col_titles):
        ax_dummy = fig.add_subplot(gs[0, col])
        ax_dummy.set_visible(False)

    sc_err = None
    sc_red = None

    for row, (season_name, months) in enumerate(SEASONS.items()):

        roms_map = seasonal_mean_error(df, months, 'roms_error')
        xgb_map  = seasonal_mean_error(df, months, 'xgb_error')
        red_map  = seasonal_mean_error(df, months, 'reduction')

        # Mean statistics for annotation
        roms_bias = df[df['month'].isin(months)]['roms_error'].mean()
        xgb_bias  = df[df['month'].isin(months)]['xgb_error'].mean()
        roms_rmse = np.sqrt((df[df['month'].isin(months)]['roms_error']**2).mean())
        xgb_rmse  = np.sqrt((df[df['month'].isin(months)]['xgb_error']**2).mean())

        # Col 0 — ROMS error
        ax0 = fig.add_subplot(gs[row, 0])
        sc  = scatter_map(ax0, roms_map, -err_max, err_max, cmap_err,
                          f'{season_name}  |  ROMS error')
        ax0.set_ylabel('Latitude')
        ax0.text(0.02, 0.03,
                 f'bias={roms_bias:+.2f}°C  RMSE={roms_rmse:.2f}°C',
                 transform=ax0.transAxes, fontsize=7,
                 color='#f0644a', family='monospace')
        sc_err = sc

        # Col 1 — XGB error
        ax1 = fig.add_subplot(gs[row, 1])
        scatter_map(ax1, xgb_map, -err_max, err_max, cmap_err,
                    f'{season_name}  |  XGB error')
        ax1.text(0.02, 0.03,
                 f'bias={xgb_bias:+.2f}°C  RMSE={xgb_rmse:.2f}°C',
                 transform=ax1.transAxes, fontsize=7,
                 color='#2f81f7', family='monospace')

        # Col 2 — error reduction
        ax2 = fig.add_subplot(gs[row, 2])
        sc2 = scatter_map(ax2, red_map, -red_max, red_max, cmap_red,
                          f'{season_name}  |  Error reduction')
        ax2.text(0.02, 0.03,
                 f'mean reduction={red_map["error"].mean():+.2f}°C',
                 transform=ax2.transAxes, fontsize=7,
                 color='#3fb950', family='monospace')
        sc_red = sc2

    # Shared colorbars
    cbar_ax1 = fig.add_axes([0.07, 0.005, 0.55, 0.012])
    cb1 = fig.colorbar(sc_err, cax=cbar_ax1, orientation='horizontal')
    cb1.set_label('Model error (°C)  |  red = too warm, blue = too cold',
                  color=TEXT, fontsize=8)
    cb1.ax.tick_params(colors=MUTED, labelsize=7)

    cbar_ax2 = fig.add_axes([0.67, 0.005, 0.27, 0.012])
    cb2 = fig.colorbar(sc_red, cax=cbar_ax2, orientation='horizontal')
    cb2.set_label('Error reduction (°C)  |  green = improvement',
                  color=TEXT, fontsize=8)
    cb2.ax.tick_params(colors=MUTED, labelsize=7)

    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — XGB vs OI DIFFERENCE MAP
# ─────────────────────────────────────────────────────────────

def build_oi_difference_figure(df_pred, df_analysis):
    """
    Where does the OI update change the XGB field?
    Shows XGB corrected - OI analysis per season.
    Green = OI warmer than XGB, Pink = OI cooler.
    """
    print('Building OI difference figure...')

    # Merge on lat, lon, month, year
    df_pred     = df_pred.copy()
    df_analysis = df_analysis.copy()

    df_pred['lat_r'] = np.round(df_pred['lat'], 3)
    df_pred['lon_r'] = np.round(df_pred['lon'], 3)
    df_analysis['lat_r'] = np.round(df_analysis['lat'], 3)
    df_analysis['lon_r'] = np.round(df_analysis['lon'], 3)

    merged = df_pred.merge(
        df_analysis[['lat_r', 'lon_r', 'month', 'year', 'analysis']],
        on=['lat_r', 'lon_r', 'month', 'year'],
        how='inner'
    )
    merged['oi_increment'] = merged['analysis'] - merged['sst_corrected']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'CE2COAST SST — OI Increment (analysis − XGB corrected)  |  2019–2020\n'
        'Green = OI warmer than XGB  |  Pink = OI cooler than XGB',
        fontsize=11, fontweight='bold', color=TEXT
    )

    axes_flat = axes.ravel()
    sc_ref    = None
    vmax      = 0.5   # °C — OI increments should be small

    for ax, (season_name, months) in zip(axes_flat, SEASONS.items()):
        sub = merged[merged['month'].isin(months)]
        grp = (sub.groupby(['lat', 'lon'])['oi_increment']
               .mean().reset_index())

        sc = ax.scatter(
            grp['lon'], grp['lat'],
            c=grp['oi_increment'],
            cmap='PiYG', vmin=-vmax, vmax=vmax,
            s=4, rasterized=True
        )
        ax.set_title(season_name, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

        mean_inc = grp['oi_increment'].mean()
        n_sig = (grp['oi_increment'].abs() > 0.05).sum()
        ax.text(0.02, 0.03,
                f'mean increment={mean_inc:+.3f}°C  '
                f'|  significant cells={n_sig:,}',
                transform=ax.transAxes, fontsize=7,
                color=TEXT, family='monospace')
        sc_ref = sc

    plt.colorbar(sc_ref, ax=axes_flat, fraction=0.02, pad=0.02,
                 label='OI increment (°C)')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — SUMMARY STAT TABLE AS FIGURE
# ─────────────────────────────────────────────────────────────

def build_stats_table(df):
    """
    Clean table figure: seasonal RMSE and bias for all three layers.
    """
    print('Building stats table...')

    df = df.copy()
    df['roms_error'] = df['sst_roms']      - df['sst_obs']
    df['xgb_error']  = df['sst_corrected'] - df['sst_obs']

    rows = []
    for season_name, months in SEASONS.items():
        sub = df[df['month'].isin(months)]
        rows.append({
            'Season':        season_name,
            'ROMS RMSE':     f'{np.sqrt((sub["roms_error"]**2).mean()):.3f}',
            'XGB RMSE':      f'{np.sqrt((sub["xgb_error"]**2).mean()):.3f}',
            'ROMS bias':     f'{sub["roms_error"].mean():+.3f}',
            'XGB bias':      f'{sub["xgb_error"].mean():+.3f}',
            'RMSE reduc.':   f'{np.sqrt((sub["roms_error"]**2).mean()) - np.sqrt((sub["xgb_error"]**2).mean()):.3f}',
        })

    # Annual
    rows.append({
        'Season':      'Annual',
        'ROMS RMSE':   f'{np.sqrt((df["roms_error"]**2).mean()):.3f}',
        'XGB RMSE':    f'{np.sqrt((df["xgb_error"]**2).mean()):.3f}',
        'ROMS bias':   f'{df["roms_error"].mean():+.3f}',
        'XGB bias':    f'{df["xgb_error"].mean():+.3f}',
        'RMSE reduc.': f'{np.sqrt((df["roms_error"]**2).mean()) - np.sqrt((df["xgb_error"]**2).mean()):.3f}',
    })

    df_table = pd.DataFrame(rows)
    cols = df_table.columns.tolist()

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_facecolor(SURFACE)
    ax.axis('off')

    tbl = ax.table(
        cellText=df_table.values,
        colLabels=cols,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)

    # Style cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(SURFACE if row > 0 else '#1f2937')
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=TEXT, family='monospace')
        if row == 0:
            cell.set_text_props(fontweight='bold', color='#2f81f7')
        # Highlight annual row
        if row == len(rows):
            cell.set_facecolor('#1a2332')
            cell.set_text_props(fontweight='bold', color=TEXT)
        # Colour RMSE reduction green
        if col == 5 and row > 0:
            try:
                val = float(df_table.values[row-1][col])
                cell.set_text_props(
                    color='#3fb950' if val > 0 else '#f0644a'
                )
            except Exception:
                pass

    ax.set_title(
        'CE2COAST SST — Seasonal Skill Summary  |  Test 2019–2020  '
        '(RMSE in °C, bias in °C)',
        color=TEXT, fontsize=10, fontweight='bold', pad=12
    )
    fig.tight_layout()
    return fig, df_table


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    print('Loading SST predictions...')
    pred = pd.read_parquet(COLLOC_DIR / 'predictions_sst_test.parquet')
    pred['time']  = pd.to_datetime(pred['time'])
    pred['month'] = pred['time'].dt.month
    pred['year']  = pred['time'].dt.year
    print(f'  {len(pred):,} rows')

    # Load OI analysis if available
    analysis_path = KALMAN_DIR / 'analysis_sst.parquet'
    df_analysis   = None
    if analysis_path.exists():
        df_analysis = pd.read_parquet(analysis_path)
        df_analysis['time']  = pd.to_datetime(df_analysis['time'])
        df_analysis['month'] = df_analysis['time'].dt.month
        df_analysis['year']  = df_analysis['time'].dt.year
        print(f'  OI analysis: {len(df_analysis):,} rows')
    else:
        print('  [INFO] No OI analysis found — skipping OI difference figure')

    # ── Figure 1 — Seasonal error maps ───────────────────────
    fig1 = build_seasonal_figure(pred)
    out1 = FIG_DIR / 'seasonal_error_maps.png'
    fig1.savefig(out1)
    plt.close(fig1)
    print(f'\nSaved: {out1}')

    # ── Figure 2 — OI increment ──────────────────────────────
    if df_analysis is not None:
        fig2 = build_oi_difference_figure(pred, df_analysis)
        out2 = FIG_DIR / 'oi_increment_maps.png'
        fig2.savefig(out2)
        plt.close(fig2)
        print(f'Saved: {out2}')

    # ── Figure 3 — Stats table ───────────────────────────────
    fig3, df_table = build_stats_table(pred)
    out3 = FIG_DIR / 'seasonal_stats_table.png'
    fig3.savefig(out3)
    plt.close(fig3)
    print(f'Saved: {out3}')

    print('\nSeasonal statistics:')
    print(df_table.to_string(index=False))
    print('\nDone.')


if __name__ == '__main__':
    main()

"""
anomaly_detection.py
====================
Two-layer anomaly detection for CE2COAST SST and Chl correction pipeline.

Layer 1 — Residual anomaly (climatological z-score)
    For each grid point × month, compute 2010-2018 climatological mean
    and std of the residual. Flag 2019-2020 where |z| > 2.
    Answers: "Was ROMS unusually wrong here?"

Layer 2 — Isolation Forest (feature-space anomaly)
    Train on 2010-2018 physical feature matrix. Score 2019-2020.
    Answers: "Was the physical state outside the training distribution?"

Outputs:
  figures/anomaly_residual_sst.png      — z-score maps by season
  figures/anomaly_isoforest_sst.png     — isolation forest score maps
  figures/anomaly_combined_sst.png      — combined two-layer dashboard
  figures/anomaly_residual_chl.png      — same for Chl
  figures/anomaly_combined_chl.png      — Chl combined dashboard
  anomaly_sst_results.parquet           — full anomaly scores
  anomaly_chl_results.parquet

Author : Eugène Ivanov / Twin
Project: CE2COAST anomaly detection — DEME portfolio
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
OUT_DIR    = COLLOC_DIR
FIG_DIR    = FIG_DIR

SST_FEATURES = [
    'sst_roms', 'salt_roms', 'zeta_roms',
    'sustr_roms', 'svstr_roms',
    'depth_m', 'coast_dist',
    'month_sin', 'month_cos',
    'doy_sin',   'doy_cos',
    'residual_sst_lag1',
]
SST_RESIDUAL = 'residual_sst'

CHL_FEATURES = [
    'chl_roms', 'depth_m', 'coast_dist',
    'month_sin', 'month_cos',
    'doy_sin',   'doy_cos',
    'log_residual_chl_lag1',
]
CHL_RESIDUAL = 'log_residual_chl'

# Anomaly thresholds
ZSCORE_THRESHOLD   = 2.0    # |z| > 2 = residual anomaly
ISO_CONTAMINATION  = 0.05   # 5% of training data treated as anomalous

SEASONS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

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
# LAYER 1 — RESIDUAL ANOMALY (Z-SCORE)
# ─────────────────────────────────────────────────────────────

def compute_residual_anomaly(train, test, residual_col):
    """
    Compute climatological mean + std of residual per
    (lat_r, lon_r, month) from training set.
    Apply z-score to test set. Flag |z| > threshold.
    """
    print(f'  Computing residual climatology ({residual_col})...')

    train = train.copy()
    test  = test.copy()

    # Round coordinates for grouping
    for df in [train, test]:
        df['lat_r'] = np.round(df['lat'], 3)
        df['lon_r'] = np.round(df['lon'], 3)

    # Climatology from training years
    clim = (train.groupby(['lat_r', 'lon_r', 'month'])[residual_col]
            .agg(['mean', 'std'])
            .reset_index()
            .rename(columns={'mean': 'clim_mean', 'std': 'clim_std'}))

    # Fill zero std with small value to avoid division by zero
    clim['clim_std'] = clim['clim_std'].replace(0, np.nan).fillna(0.01)

    # Merge into test
    test = test.merge(clim, on=['lat_r', 'lon_r', 'month'], how='left')

    # Z-score
    test['residual_zscore'] = (
        (test[residual_col] - test['clim_mean']) / test['clim_std']
    )
    test['anomaly_residual'] = (
        test['residual_zscore'].abs() > ZSCORE_THRESHOLD
    ).astype(int)

    n_anom = test['anomaly_residual'].sum()
    pct    = 100 * n_anom / len(test)
    print(f'  Residual anomalies: {n_anom:,} ({pct:.1f}%)'
          f'  (threshold |z|>{ZSCORE_THRESHOLD})')

    return test


# ─────────────────────────────────────────────────────────────
# LAYER 2 — ISOLATION FOREST
# ─────────────────────────────────────────────────────────────

def compute_isolation_forest(train, test, features):
    """
    Train Isolation Forest on training feature matrix.
    Score test set. Return anomaly score and binary flag.
    Higher score = more normal. Lower score = more anomalous.
    """
    print(f'  Training Isolation Forest '
          f'(contamination={ISO_CONTAMINATION})...')

    # Force numeric
    for col in features:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col]  = pd.to_numeric(test[col],  errors='coerce')

    tr = train.dropna(subset=features).copy()
    te = test.dropna(subset=features).copy()

    X_tr = tr[features].values
    X_te = te[features].values

    # Standardise features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    iso = IsolationForest(
        contamination = ISO_CONTAMINATION,
        n_estimators  = 200,
        random_state  = 42,
        n_jobs        = -1
    )
    iso.fit(X_tr_s)

    # decision_function: negative = anomalous, positive = normal
    test = test.copy()
    scores = np.full(len(test), np.nan)
    flags  = np.zeros(len(test), dtype=int)

    te_idx = te.index
    scores_raw  = iso.decision_function(X_te_s)
    flags_raw   = (iso.predict(X_te_s) == -1).astype(int)

    test.loc[te_idx, 'iso_score'] = scores_raw
    test.loc[te_idx, 'anomaly_iso'] = flags_raw

    n_anom = flags_raw.sum()
    pct    = 100 * n_anom / len(flags_raw)
    print(f'  Isolation Forest anomalies: {n_anom:,} ({pct:.1f}%)')

    # Combined flag: anomalous in EITHER layer
    if 'anomaly_residual' in test.columns:
        test['anomaly_combined'] = (
            (test['anomaly_residual'] == 1) |
            (test['anomaly_iso'] == 1)
        ).astype(int)
        n_comb = test['anomaly_combined'].sum()
        print(f'  Combined anomalies: {n_comb:,} '
              f'({100*n_comb/len(test):.1f}%)')

    return test


# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────

def fig_residual_anomaly(test, residual_col, label, units):
    """
    4-panel seasonal z-score maps.
    Background = z-score magnitude. Overlay = anomaly flag.
    """
    print(f'  Building residual anomaly figure ({label})...')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'CE2COAST {label} — Residual Anomaly (Z-score)  |  '
        f'Test 2019–2020\n'
        f'|z| = |residual − climatology| / σ_climatology  |  '
        f'Red circles = anomaly flag (|z|>{ZSCORE_THRESHOLD})',
        fontsize=10, fontweight='bold', color=TEXT
    )

    axes_flat = axes.ravel()
    vmax = 4.0

    for ax, (sname, months) in zip(axes_flat, SEASONS.items()):
        sub = test[test['month'].isin(months)].dropna(
            subset=['residual_zscore']
        )
        grp = (sub.groupby(['lat', 'lon'])
               .agg(zscore_mean=('residual_zscore', 'mean'),
                    anom_frac=('anomaly_residual', 'mean'))
               .reset_index())

        # Background: mean z-score magnitude
        sc = ax.scatter(
            grp['lon'], grp['lat'],
            c=grp['zscore_mean'].abs(),
            cmap='YlOrRd', vmin=0, vmax=vmax,
            s=4, rasterized=True, alpha=0.8
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01,
                     label='Mean |z-score|')

        # Overlay: anomaly hotspots
        hot = grp[grp['anom_frac'] > 0.3]
        if len(hot) > 0:
            ax.scatter(hot['lon'], hot['lat'],
                       c=CORAL, s=8, alpha=0.6,
                       rasterized=True, label=f'anom>30%')

        # Annotate most anomalous month
        sub_months = sub.groupby('month')['anomaly_residual'].mean()
        most_anom  = sub_months.idxmax() if len(sub_months) > 0 else None
        frac_anom  = sub['anomaly_residual'].mean()

        ax.set_title(
            f'{sname}  |  anomaly fraction={frac_anom:.2f}',
            fontweight='bold'
        )
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Lon', fontsize=7)
        ax.set_ylabel('Lat', fontsize=7)

    plt.tight_layout()
    return fig


def fig_isolation_forest(test, label):
    """
    4 panels: seasonal anomaly score maps.
    """
    print(f'  Building Isolation Forest figure ({label})...')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'CE2COAST {label} — Isolation Forest Anomaly  |  '
        f'Test 2019–2020\n'
        f'Score: negative = anomalous physical state  |  '
        f'Red = outside training distribution',
        fontsize=10, fontweight='bold', color=TEXT
    )

    axes_flat = axes.ravel()

    for ax, (sname, months) in zip(axes_flat, SEASONS.items()):
        sub = test[test['month'].isin(months)].dropna(
            subset=['iso_score']
        )
        grp = (sub.groupby(['lat', 'lon'])
               .agg(score_mean=('iso_score',   'mean'),
                    anom_frac =('anomaly_iso',  'mean'))
               .reset_index())

        vmin = np.percentile(grp['score_mean'], 2)
        vmax = np.percentile(grp['score_mean'], 98)

        sc = ax.scatter(
            grp['lon'], grp['lat'],
            c=grp['score_mean'],
            cmap='RdYlGn', vmin=vmin, vmax=vmax,
            s=4, rasterized=True
        )
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01,
                     label='Anomaly score\n(red=anomalous)')

        # Flag hotspots
        hot = grp[grp['anom_frac'] > 0.4]
        if len(hot) > 0:
            ax.scatter(hot['lon'], hot['lat'],
                       c=CORAL, s=10, alpha=0.7,
                       rasterized=True)

        frac = sub['anomaly_iso'].mean()
        ax.set_title(f'{sname}  |  anomaly fraction={frac:.2f}',
                     fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


def fig_combined_dashboard(test, residual_col, features, label, units):
    """
    Combined two-layer anomaly dashboard.
    Row 1: monthly anomaly timeline
    Row 2: spatial combined anomaly map
    Row 3: anomaly score distributions + overlap Venn
    """
    print(f'  Building combined dashboard ({label})...')

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f'CE2COAST {label} — Two-Layer Anomaly Detection  |  '
        f'Test 2019–2020\n'
        f'Layer 1: residual z-score  |  '
        f'Layer 2: Isolation Forest  |  '
        f'Combined: either layer flags',
        fontsize=11, fontweight='bold', color=TEXT, y=0.99
    )

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.42, wspace=0.28,
                           left=0.06, right=0.97,
                           top=0.94, bottom=0.06)

    # ── Row 1: Monthly anomaly timeline ──────────────────────
    ax_time = fig.add_subplot(gs[0, :3])

    test2 = test.copy()
    test2['year_month'] = (test2['year'].astype(str) + '-' +
                           test2['month'].astype(str).str.zfill(2))

    monthly = test2.groupby(['year', 'month']).agg(
        anom_resid   = ('anomaly_residual', 'mean'),
        anom_iso     = ('anomaly_iso',      'mean'),
        anom_combined= ('anomaly_combined', 'mean'),
        zscore_mean  = ('residual_zscore',  lambda x:
                        x.abs().mean()),
    ).reset_index().sort_values(['year', 'month'])

    months_x = range(len(monthly))
    ax_time.bar(months_x,
                monthly['anom_combined'].values * 100,
                color=CORAL, alpha=0.5, label='Combined anomaly %')
    ax_time.plot(months_x,
                 monthly['anom_resid'].values * 100,
                 color=AMBER, linewidth=1.5,
                 marker='o', markersize=4, label='Residual anomaly %')
    ax_time.plot(months_x,
                 monthly['anom_iso'].values * 100,
                 color=BLUE, linewidth=1.5,
                 marker='s', markersize=4, label='Isolation Forest %')

    ax_time.set_xticks(list(months_x))
    ax_time.set_xticklabels(
        [f'{int(r.year)}-{MONTH_NAMES[int(r.month)]}'
         for _, r in monthly.iterrows()],
        rotation=45, ha='right', fontsize=7
    )
    ax_time.set_ylabel('Anomaly fraction (%)')
    ax_time.set_title(
        'Monthly anomaly fraction — both detection layers',
        fontweight='bold'
    )
    ax_time.legend(fontsize=7)
    ax_time.grid(True, alpha=0.3, axis='y')

    # Highlight extreme months
    threshold_pct = monthly['anom_combined'].quantile(0.75) * 100
    for i, row in enumerate(monthly.itertuples()):
        if row.anom_combined * 100 > threshold_pct:
            ax_time.axvspan(i - 0.4, i + 0.4,
                            alpha=0.15, color=CORAL)

    # ── Row 1 col 3: Layer agreement ─────────────────────────
    ax_venn = fig.add_subplot(gs[0, 3])
    ax_venn.set_facecolor(SURFACE)
    ax_venn.axis('off')

    both  = ((test['anomaly_residual'] == 1) &
             (test['anomaly_iso'] == 1)).sum()
    only_r= ((test['anomaly_residual'] == 1) &
              (test['anomaly_iso'] == 0)).sum()
    only_i= ((test['anomaly_residual'] == 0) &
              (test['anomaly_iso'] == 1)).sum()
    neither= ((test['anomaly_residual'] == 0) &
               (test['anomaly_iso'] == 0)).sum()
    total  = len(test)

    lines = [
        ('LAYER AGREEMENT', BLUE, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('Both layers:', MUTED, 8, 'normal'),
        (f'  {both:,}  ({100*both/total:.1f}%)', CORAL, 10, 'bold'),
        ('', TEXT, 8, 'normal'),
        ('Residual only:', MUTED, 8, 'normal'),
        (f'  {only_r:,}  ({100*only_r/total:.1f}%)', AMBER, 9, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('IsoForest only:', MUTED, 8, 'normal'),
        (f'  {only_i:,}  ({100*only_i/total:.1f}%)', BLUE, 9, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('Neither:', MUTED, 8, 'normal'),
        (f'  {neither:,}  ({100*neither/total:.1f}%)', GREEN, 9, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('─'*24, BORDER, 7, 'normal'),
        ('', TEXT, 8, 'normal'),
        ('Both layers agree =', MUTED, 7, 'normal'),
        ('highest confidence', MUTED, 7, 'normal'),
        ('anomaly flag.', MUTED, 7, 'normal'),
    ]

    y = 0.97
    for text, color, size, weight in lines:
        ax_venn.text(0.05, y, text,
                     transform=ax_venn.transAxes,
                     fontsize=size, color=color,
                     fontweight=weight, family='monospace',
                     va='top')
        y -= 0.055

    # ── Row 2: Spatial combined anomaly ──────────────────────
    for col, (sname, months) in enumerate(SEASONS.items()):
        ax = fig.add_subplot(gs[1, col])
        sub = test[test['month'].isin(months)].dropna(
            subset=['anomaly_combined', 'iso_score']
        )
        grp = (sub.groupby(['lat', 'lon'])
               .agg(anom_frac=('anomaly_combined', 'mean'),
                    iso_mean =('iso_score',         'mean'))
               .reset_index())

        sc = ax.scatter(
            grp['lon'], grp['lat'],
            c=grp['anom_frac'],
            cmap='hot_r', vmin=0, vmax=0.4,
            s=4, rasterized=True
        )
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                     label='Anomaly fraction')

        frac = sub['anomaly_combined'].mean()
        ax.set_title(f'{sname}  |  {frac:.2f} anomalous',
                     fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # ── Row 3: Score distributions ───────────────────────────
    ax_z = fig.add_subplot(gs[2, 0:2])

    # Z-score distribution: normal vs anomalous months
    # Split by top/bottom anomaly months
    monthly_anom = monthly.nlargest(4, 'anom_combined')
    monthly_norm = monthly.nsmallest(4, 'anom_combined')

    for _, row in monthly_anom.iterrows():
        sub = test[
            (test['year'] == int(row['year'])) &
            (test['month'] == int(row['month']))
        ]['residual_zscore'].dropna()
        if len(sub) > 0:
            ax_z.hist(sub.values, bins=50, alpha=0.3,
                      color=CORAL, density=True)

    for _, row in monthly_norm.iterrows():
        sub = test[
            (test['year'] == int(row['year'])) &
            (test['month'] == int(row['month']))
        ]['residual_zscore'].dropna()
        if len(sub) > 0:
            ax_z.hist(sub.values, bins=50, alpha=0.3,
                      color=BLUE, density=True)

    ax_z.axvline(-ZSCORE_THRESHOLD, color=AMBER,
                 linewidth=1.5, linestyle='--',
                 label=f'|z|={ZSCORE_THRESHOLD} threshold')
    ax_z.axvline(ZSCORE_THRESHOLD, color=AMBER,
                 linewidth=1.5, linestyle='--')
    ax_z.set_xlabel('Residual z-score')
    ax_z.set_ylabel('Density')
    ax_z.set_title(
        'Z-score distribution\nred=top anomaly months  '
        'blue=normal months'
    )
    ax_z.legend(fontsize=7)
    ax_z.grid(True, alpha=0.3)

    # Isolation Forest score distribution
    ax_iso = fig.add_subplot(gs[2, 2:4])
    sub_anom = test[test['anomaly_combined'] == 1][
        'iso_score'
    ].dropna()
    sub_norm = test[test['anomaly_combined'] == 0][
        'iso_score'
    ].dropna()

    if len(sub_norm) > 0:
        ax_iso.hist(sub_norm.values, bins=60, alpha=0.5,
                    color=BLUE, density=True, label='Normal')
    if len(sub_anom) > 0:
        ax_iso.hist(sub_anom.values, bins=60, alpha=0.5,
                    color=CORAL, density=True, label='Anomalous')

    ax_iso.set_xlabel('Isolation Forest score\n'
                      '(negative = anomalous)')
    ax_iso.set_ylabel('Density')
    ax_iso.set_title(
        'Isolation Forest score distribution\n'
        'blue=normal  red=flagged by either layer'
    )
    ax_iso.legend(fontsize=7)
    ax_iso.grid(True, alpha=0.3)

    return fig


def fig_extreme_months(test, residual_col, label, units, n_top=6):
    """
    Spatial maps of the N most anomalous months.
    Shows the actual residual field for those months.
    """
    print(f'  Building extreme months figure ({label})...')

    # Find most anomalous months by combined anomaly fraction
    monthly_rank = (test.groupby(['year', 'month'])
                    ['anomaly_combined'].mean()
                    .reset_index()
                    .sort_values('anomaly_combined', ascending=False)
                    .head(n_top))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f'CE2COAST {label} — Most Anomalous Months  |  Test 2019–2020\n'
        f'Ranked by combined anomaly fraction  |  '
        f'Color = residual ({units})',
        fontsize=11, fontweight='bold', color=TEXT
    )

    vmax = 3.0 if 'sst' in label.lower() else 2.0
    cmap = 'RdBu_r' if 'sst' in label.lower() else 'PiYG'

    for ax, (_, row) in zip(axes.ravel(),
                             monthly_rank.iterrows()):
        yr, mo = int(row['year']), int(row['month'])
        sub = test[
            (test['year'] == yr) & (test['month'] == mo)
        ].dropna(subset=[residual_col, 'anomaly_combined'])

        grp = (sub.groupby(['lat', 'lon'])
               .agg(resid=    (residual_col,     'mean'),
                    anom_frac=('anomaly_combined', 'mean'))
               .reset_index())

        sc = ax.scatter(
            grp['lon'], grp['lat'],
            c=grp['resid'],
            cmap=cmap, vmin=-vmax, vmax=vmax,
            s=4, rasterized=True
        )
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                     label=units)

        # Overlay combined anomaly hotspots
        hot = grp[grp['anom_frac'] > 0.5]
        if len(hot) > 0:
            ax.scatter(hot['lon'], hot['lat'],
                       c=CORAL, s=8, alpha=0.6,
                       rasterized=True)

        frac = sub['anomaly_combined'].mean()
        ax.set_title(
            f'{yr}-{MONTH_NAMES[mo]}  |  '
            f'anom={frac:.2f}',
            fontweight='bold'
        )
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # SST
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*55)
    print('SST ANOMALY DETECTION')
    print('='*55)

    df_sst = pd.read_parquet(
        COLLOC_DIR / 'features_sst_2010_2020.parquet'
    )
    df_sst['time']  = pd.to_datetime(df_sst['time'])
    df_sst['month'] = df_sst['time'].dt.month
    df_sst['year']  = df_sst['time'].dt.year

    train_sst = df_sst[df_sst['split'] == 'train'].copy()
    test_sst  = df_sst[df_sst['split'] == 'test'].copy()

    print(f'  Train: {len(train_sst):,}  Test: {len(test_sst):,}')

    # Layer 1
    test_sst = compute_residual_anomaly(
        train_sst, test_sst, SST_RESIDUAL
    )

    # Layer 2
    test_sst = compute_isolation_forest(
        train_sst, test_sst, SST_FEATURES
    )

    # Save
    out = OUT_DIR / 'anomaly_sst_results.parquet'
    test_sst.to_parquet(out, index=False)
    print(f'  Saved: {out.name}  ({out.stat().st_size/1e6:.1f} MB)')

    # Figures
    f1 = fig_residual_anomaly(
        test_sst, SST_RESIDUAL, 'SST', '°C'
    )
    f1.savefig(FIG_DIR / 'anomaly_residual_sst.png')
    plt.close(f1)
    print('  Saved: anomaly_residual_sst.png')

    f2 = fig_isolation_forest(test_sst, 'SST')
    f2.savefig(FIG_DIR / 'anomaly_isoforest_sst.png')
    plt.close(f2)
    print('  Saved: anomaly_isoforest_sst.png')

    f3 = fig_combined_dashboard(
        test_sst, SST_RESIDUAL, SST_FEATURES, 'SST', '°C'
    )
    f3.savefig(FIG_DIR / 'anomaly_combined_sst.png')
    plt.close(f3)
    print('  Saved: anomaly_combined_sst.png')

    f4 = fig_extreme_months(test_sst, SST_RESIDUAL, 'SST', '°C')
    f4.savefig(FIG_DIR / 'anomaly_extreme_sst.png')
    plt.close(f4)
    print('  Saved: anomaly_extreme_sst.png')

    # ══════════════════════════════════════════════════════════
    # CHL
    # ══════════════════════════════════════════════════════════
    print('\n' + '='*55)
    print('CHL ANOMALY DETECTION')
    print('='*55)

    df_chl = pd.read_parquet(
        COLLOC_DIR / 'features_chl_2010_2020.parquet'
    )
    df_chl['time']  = pd.to_datetime(df_chl['time'])
    df_chl['month'] = df_chl['time'].dt.month
    df_chl['year']  = df_chl['time'].dt.year

    train_chl = df_chl[df_chl['split'] == 'train'].copy()
    test_chl  = df_chl[df_chl['split'] == 'test'].copy()

    print(f'  Train: {len(train_chl):,}  Test: {len(test_chl):,}')

    test_chl = compute_residual_anomaly(
        train_chl, test_chl, CHL_RESIDUAL
    )
    test_chl = compute_isolation_forest(
        train_chl, test_chl, CHL_FEATURES
    )

    out = OUT_DIR / 'anomaly_chl_results.parquet'
    test_chl.to_parquet(out, index=False)
    print(f'  Saved: {out.name}  ({out.stat().st_size/1e6:.1f} MB)')

    f5 = fig_residual_anomaly(
        test_chl, CHL_RESIDUAL, 'Chl', 'log mg/m³'
    )
    f5.savefig(FIG_DIR / 'anomaly_residual_chl.png')
    plt.close(f5)
    print('  Saved: anomaly_residual_chl.png')

    f6 = fig_isolation_forest(test_chl, 'Chl')
    f6.savefig(FIG_DIR / 'anomaly_isoforest_chl.png')
    plt.close(f6)
    print('  Saved: anomaly_isoforest_chl.png')

    f7 = fig_combined_dashboard(
        test_chl, CHL_RESIDUAL, CHL_FEATURES, 'Chl', 'log mg/m³'
    )
    f7.savefig(FIG_DIR / 'anomaly_combined_chl.png')
    plt.close(f7)
    print('  Saved: anomaly_combined_chl.png')

    f8 = fig_extreme_months(
        test_chl, CHL_RESIDUAL, 'Chl', 'log mg/m³'
    )
    f8.savefig(FIG_DIR / 'anomaly_extreme_chl.png')
    plt.close(f8)
    print('  Saved: anomaly_extreme_chl.png')

    # ── Summary ──────────────────────────────────────────────
    print('\n' + '='*55)
    print('ANOMALY SUMMARY')
    print('='*55)

    for label, test_df in [('SST', test_sst), ('Chl', test_chl)]:
        print(f'\n  {label}:')
        for col, name in [
            ('anomaly_residual', 'Residual z-score'),
            ('anomaly_iso',      'Isolation Forest'),
            ('anomaly_combined', 'Combined'),
        ]:
            if col in test_df.columns:
                n   = test_df[col].sum()
                pct = 100 * n / len(test_df)
                print(f'    {name:<20} {n:>8,} ({pct:.1f}%)')

        if 'anomaly_combined' in test_df.columns:
            print(f'\n  {label} most anomalous months:')
            top = (test_df.groupby(['year','month'])
                   ['anomaly_combined'].mean()
                   .sort_values(ascending=False).head(5))
            for (yr, mo), frac in top.items():
                print(f'    {yr}-{MONTH_NAMES[int(mo)]}'
                      f'  {frac:.3f}')

    print('\nDone.')


if __name__ == '__main__':
    main()

"""
ablation_baseline.py
====================
Ablation study for CE2COAST XGBoost residual correction.

Compares three feature sets to quantify how much of the correction
comes from seasonality vs physics-aware features:

  Model A — climatology only  (month/doy encoding)
  Model B — physics only      (ROMS state, no seasonality)
  Model C — full model        (all features, matches train_xgboost_roms.py)

Outputs:
  ablation_results.csv        — metrics table for all models × variables
  figures/ablation_sst.png    — SST comparison figure
  figures/ablation_chl.png    — Chl comparison figure

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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
OUT_DIR    = COLLOC_DIR
FIG_DIR    = FIG_DIR

# ── Feature sets ─────────────────────────────────────────────

SST_FEATURES = {
    'climatology': [
        'month_sin', 'month_cos',
        'doy_sin',   'doy_cos',
    ],
    'physics': [
        'sst_roms', 'salt_roms', 'zeta_roms',
        'sustr_roms', 'svstr_roms',
        'depth_m', 'coast_dist',
    ],
    'full': [
        'sst_roms', 'salt_roms', 'zeta_roms',
        'sustr_roms', 'svstr_roms',
        'depth_m', 'coast_dist',
        'month_sin', 'month_cos',
        'doy_sin',   'doy_cos',
        'residual_sst_lag1',
    ],
}
SST_TARGET    = 'residual_sst'
SST_OBS       = 'sst_obs'
SST_ROMS      = 'sst_roms'

CHL_FEATURES = {
    'climatology': [
        'month_sin', 'month_cos',
        'doy_sin',   'doy_cos',
    ],
    'physics': [
        'chl_roms',
        'depth_m', 'coast_dist',
    ],
    'full': [
        'chl_roms',
        'depth_m', 'coast_dist',
        'month_sin', 'month_cos',
        'doy_sin',   'doy_cos',
        'log_residual_chl_lag1',
    ],
}
CHL_TARGET    = 'log_residual_chl'
CHL_OBS       = 'chl_obs'
CHL_ROMS      = 'chl_roms'

# ── XGBoost params (same as main training) ───────────────────
XGB_PARAMS = {
    'n_estimators':       500,
    'max_depth':          6,
    'learning_rate':      0.05,
    'subsample':          0.8,
    'colsample_bytree':   0.8,
    'min_child_weight':   10,
    'reg_alpha':          0.1,
    'reg_lambda':         1.0,
    'objective':          'reg:squarederror',
    'eval_metric':        'rmse',
    'tree_method':        'hist',
    'n_jobs':             -1,
    'random_state':       42,
    'early_stopping_rounds': 30,
}

# ── Aesthetics ───────────────────────────────────────────────
DARK_BG  = '#0d1117'
SURFACE  = '#161b22'
BORDER   = '#30363d'
TEXT     = '#e6edf3'
MUTED    = '#7d8590'

COLORS = {
    'roms_raw':     '#f0644a',   # red
    'climatology':  '#d29922',   # amber
    'physics':      '#3fb950',   # green
    'full':         '#2f81f7',   # blue
}

MONTH_LABELS = ['J','F','M','A','M','J','J','A','S','O','N','D']

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
    'legend.fontsize':   8,
    'legend.facecolor':  SURFACE,
    'legend.edgecolor':  BORDER,
    'savefig.facecolor': DARK_BG,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
})


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(obs, pred, label=''):
    rmse = np.sqrt(mean_squared_error(obs, pred))
    mae  = mean_absolute_error(obs, pred)
    r2   = r2_score(obs, pred)
    bias = float(np.mean(pred - obs))
    return {'label': label, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'bias': bias}


# ─────────────────────────────────────────────────────────────
# TRAIN ONE ABLATION MODEL
# ─────────────────────────────────────────────────────────────

def train_ablation(df_train, df_test, features, target, model_label):
    """Train one ablation model, return predictions on test set."""
    print(f'    [{model_label}] features={len(features)}', end=' ')

    # Force numeric
    for col in features:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_test[col]  = pd.to_numeric(df_test[col],  errors='coerce')

    tr = df_train.dropna(subset=features + [target])
    te = df_test.dropna(subset=features + [target])

    X_tr = tr[features].values
    y_tr = tr[target].values
    X_te = te[features].values
    y_te = te[target].values

    model = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    r2_tr = r2_score(y_tr, pred_tr)
    r2_te = r2_score(y_te, pred_te)
    print(f'| train R²={r2_tr:.3f}  test R²={r2_te:.3f}')

    # Return test predictions aligned to te index
    result = te[['lat', 'lon', 'month', 'year']].copy()
    result['residual_pred'] = pred_te
    result['residual_true'] = y_te
    return model, result


# ─────────────────────────────────────────────────────────────
# RUN ABLATION FOR ONE VARIABLE
# ─────────────────────────────────────────────────────────────

def run_ablation(df, feature_sets, target, obs_col, roms_col,
                 extreme_flag, log_space, label):
    """
    Train all three ablation models and collect metrics.
    Returns metrics DataFrame and per-model test prediction DataFrames.
    """
    print(f'\n{"="*55}')
    print(f'{label} ABLATION')
    print(f'{"="*55}')

    train = df[df['split'] == 'train'].copy()
    test  = df[df['split'] == 'test'].copy()

    # Remove extreme outliers from training
    if extreme_flag in train.columns:
        n_before = len(train)
        train = train[train[extreme_flag] == 0]
        print(f'  Excluded {n_before - len(train):,} extremes from train')

    print(f'  Train: {len(train):,} | Test: {len(test):,}')

    all_metrics  = []
    all_preds    = {}

    # ── ROMS raw baseline ───────────────────────────────────
    te_clean = test.dropna(subset=[obs_col, roms_col])
    obs_vals  = te_clean[obs_col].values
    roms_vals = te_clean[roms_col].values

    m = compute_metrics(obs_vals, roms_vals, 'ROMS raw')
    m['r2_residual'] = np.nan
    all_metrics.append(m)
    print(f'  ROMS raw       | R²={m["r2"]:.3f}  RMSE={m["rmse"]:.3f}')

    # ── Three ablation models ───────────────────────────────
    for model_name, features in feature_sets.items():
        print(f'  Training {model_name}...')
        _, result = train_ablation(train, test, features, target, model_name)

        # Corrected variable
        te_sub = test.loc[result.index].copy()
        if log_space:
            corrected = te_sub[roms_col] * np.exp(result['residual_pred'].values)
        else:
            corrected = te_sub[roms_col] + result['residual_pred'].values

        obs_sub  = te_sub[obs_col].values
        roms_sub = te_sub[roms_col].values

        # Metrics on corrected variable vs obs
        valid = (
            np.isfinite(obs_sub) &
            np.isfinite(corrected.values) &
            (corrected.values > 0 if log_space else True)
        )
        m_corr = compute_metrics(
            obs_sub[valid], corrected.values[valid], model_name
        )
        # Also residual R²
        m_corr['r2_residual'] = r2_score(
            result['residual_true'].values,
            result['residual_pred'].values
        )
        all_metrics.append(m_corr)

        # Monthly bias
        result['corrected'] = corrected.values
        result['obs']       = obs_sub
        result['roms']      = roms_sub
        all_preds[model_name] = result

    metrics_df = pd.DataFrame(all_metrics).set_index('label')
    print(f'\n  --- Summary ({label}) ---')
    print(metrics_df[['r2', 'rmse', 'bias', 'r2_residual']].round(4))

    return metrics_df, all_preds


# ─────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────

def build_ablation_figure(metrics_df, all_preds, label, units,
                          obs_col, roms_col, log_space):
    """
    3-panel ablation figure:
      1. R² comparison bar chart
      2. RMSE comparison bar chart
      3. Monthly bias — all models overlaid
    """
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f'CE2COAST {label} — Ablation Study  |  '
        f'Test period 2019–2020',
        fontsize=13, fontweight='bold', color=TEXT, y=1.02
    )

    gs = gridspec.GridSpec(1, 3, figure=fig,
                           hspace=0.3, wspace=0.32,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.12)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    model_order  = ['ROMS raw', 'climatology', 'physics', 'full']
    model_labels = ['ROMS raw', 'Clim. only', 'Physics only', 'Full model']
    bar_colors   = [
        COLORS['roms_raw'],
        COLORS['climatology'],
        COLORS['physics'],
        COLORS['full'],
    ]

    # Filter to models that exist in metrics
    valid_order  = [m for m in model_order if m in metrics_df.index]
    valid_labels = [model_labels[model_order.index(m)] for m in valid_order]
    valid_colors = [bar_colors[model_order.index(m)] for m in valid_order]

    r2_vals   = [metrics_df.loc[m, 'r2']   for m in valid_order]
    rmse_vals = [metrics_df.loc[m, 'rmse'] for m in valid_order]

    x = np.arange(len(valid_order))

    # ── Panel 1: R² ─────────────────────────────────────────
    bars = ax1.bar(x, r2_vals, color=valid_colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, r2_vals):
        va = 'bottom' if val >= 0 else 'top'
        offset = 0.01 if val >= 0 else -0.01
        ax1.text(bar.get_x() + bar.get_width()/2,
                 val + offset,
                 f'{val:.3f}',
                 ha='center', fontsize=8, color=TEXT)
    ax1.axhline(0, color=MUTED, linewidth=0.8, linestyle='--')
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_labels, rotation=15, ha='right')
    ax1.set_ylabel('R²')
    ax1.set_title(f'{label} R² (corrected vs obs)')
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: RMSE ────────────────────────────────────────
    bars = ax2.bar(x, rmse_vals, color=valid_colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + max(rmse_vals) * 0.01,
                 f'{val:.3f}',
                 ha='center', fontsize=8, color=TEXT)
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_labels, rotation=15, ha='right')
    ax2.set_ylabel(f'RMSE ({units})')
    ax2.set_title(f'{label} RMSE (corrected vs obs)')
    ax2.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Monthly bias ────────────────────────────────
    months = np.arange(1, 13)

    # ROMS raw monthly bias
    roms_data = all_preds.get('full')
    if roms_data is not None:
        grp_roms = (roms_data.groupby('month')['roms'].mean()
                    - roms_data.groupby('month')['obs'].mean())
        ax3.plot(months, grp_roms.values,
                 color=COLORS['roms_raw'], linewidth=2,
                 marker='o', markersize=4, label='ROMS raw',
                 zorder=5)

    for model_name, col, zord in [
        ('climatology', COLORS['climatology'], 4),
        ('physics',     COLORS['physics'],     3),
        ('full',        COLORS['full'],         6),
    ]:
        if model_name not in all_preds:
            continue
        pred_df = all_preds[model_name]
        grp = (pred_df.groupby('month')['corrected'].mean()
               - pred_df.groupby('month')['obs'].mean())
        lbl = {'climatology': 'Clim. only',
               'physics':     'Physics only',
               'full':        'Full model'}[model_name]
        ax3.plot(months, grp.values,
                 color=col, linewidth=2,
                 marker='o', markersize=4,
                 label=lbl, zorder=zord)

    ax3.axhline(0, color=MUTED, linewidth=0.8, linestyle='--')
    ax3.set_xticks(months)
    ax3.set_xticklabels(MONTH_LABELS)
    ax3.set_xlabel('Month')
    ax3.set_ylabel(f'Bias ({units})')
    ax3.set_title(f'{label} monthly bias — model comparison')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    return fig


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── SST ─────────────────────────────────────────────────
    print('Loading SST features...')
    sst = pd.read_parquet(COLLOC_DIR / 'features_sst_2010_2020.parquet')
    print(f'  {len(sst):,} rows')

    metrics_sst, preds_sst = run_ablation(
        sst,
        feature_sets   = SST_FEATURES,
        target         = SST_TARGET,
        obs_col        = SST_OBS,
        roms_col       = SST_ROMS,
        extreme_flag   = 'residual_sst_extreme',
        log_space      = False,
        label          = 'SST'
    )
    all_results['SST'] = metrics_sst

    fig_sst = build_ablation_figure(
        metrics_sst, preds_sst,
        label='SST', units='°C',
        obs_col=SST_OBS, roms_col=SST_ROMS,
        log_space=False
    )
    out_sst = FIG_DIR / 'ablation_sst.png'
    fig_sst.savefig(out_sst)
    plt.close(fig_sst)
    print(f'\n  Figure saved: {out_sst}')

    # ── Chl ─────────────────────────────────────────────────
    print('\nLoading Chl features...')
    chl = pd.read_parquet(COLLOC_DIR / 'features_chl_2010_2020.parquet')
    print(f'  {len(chl):,} rows')

    metrics_chl, preds_chl = run_ablation(
        chl,
        feature_sets   = CHL_FEATURES,
        target         = CHL_TARGET,
        obs_col        = CHL_OBS,
        roms_col       = CHL_ROMS,
        extreme_flag   = 'log_residual_chl_extreme',
        log_space      = True,
        label          = 'Chl'
    )
    all_results['Chl'] = metrics_chl

    fig_chl = build_ablation_figure(
        metrics_chl, preds_chl,
        label='Chl', units='mg/m³',
        obs_col=CHL_OBS, roms_col=CHL_ROMS,
        log_space=True
    )
    out_chl = FIG_DIR / 'ablation_chl.png'
    fig_chl.savefig(out_chl)
    plt.close(fig_chl)
    print(f'  Figure saved: {out_chl}')

    # ── Summary table ────────────────────────────────────────
    print('\n' + '='*55)
    print('ABLATION SUMMARY TABLE')
    print('='*55)

    rows = []
    for var, metrics_df in all_results.items():
        for model_name in metrics_df.index:
            row = {'variable': var, 'model': model_name}
            row.update(metrics_df.loc[model_name].to_dict())
            rows.append(row)

    df_summary = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'ablation_results.csv'
    df_summary.to_csv(out_csv, index=False, float_format='%.4f')
    print(df_summary[['variable','model','r2','rmse','bias']].to_string(index=False))
    print(f'\n  CSV saved: {out_csv}')

    # ── The Vergote answer ────────────────────────────────────
    print('\n' + '='*55)
    print('KEY FINDING FOR INTERVIEW')
    print('='*55)
    for var in ['SST', 'Chl']:
        m = all_results[var]
        if all(k in m.index for k in ['climatology', 'physics', 'full']):
            r2_clim = m.loc['climatology', 'r2']
            r2_phys = m.loc['physics',     'r2']
            r2_full = m.loc['full',         'r2']
            r2_raw  = m.loc['ROMS raw',     'r2']
            print(f'\n  {var}:')
            print(f'    ROMS raw       R² = {r2_raw:.3f}')
            print(f'    Clim. only     R² = {r2_clim:.3f}  '
                  f'(seasonal signal)')
            print(f'    Physics only   R² = {r2_phys:.3f}  '
                  f'(state-dependent signal)')
            print(f'    Full model     R² = {r2_full:.3f}  '
                  f'(combined)')
            phys_add = r2_full - r2_clim
            print(f'    Physics adds   ΔR² = {phys_add:+.3f} '
                  f'beyond climatology')

    print('\nDone.')


if __name__ == '__main__':
    main()

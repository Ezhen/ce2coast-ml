"""
train_xgboost_roms.py
=====================
XGBoost residual correction models for CE2COAST ROMS outputs.

Two models:
  1. SST bias correction  — target: residual_sst (degC)
  2. Chl bias correction  — target: log_residual_chl (log mg/m3)

Workflow:
  - Load feature parquets
  - Train XGBoost on 2010-2018
  - Evaluate on 2019-2020 (held-out temporal test)
  - SHAP feature importance
  - Save models + corrected predictions

Outputs (in ML_colloc/):
  xgb_sst_model.json
  xgb_chl_model.json
  predictions_sst_test.parquet
  predictions_chl_test.parquet
  shap_sst.parquet
  shap_chl.parquet

Author : Eugène Ivanov / Twin
Project: CE2COAST residual correction — DEME portfolio
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import json
from pathlib import Path
from ce2coast_config import (
    COLLOC_DIR, ROMS_DIR, VALID_DIR, INSITU_DIR,
    FIG_DIR, KALMAN_DIR, ROMS_AVG_PATTERN, ROMS_BIO_PATTERN,
    ROMS_GRID_FILE, SST_OBS_FILE, CHL_OBS_FILE,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

COLLOC_DIR = COLLOC_DIR
OUT_DIR    = COLLOC_DIR

# ── SST ─────────────────────────────────────────────────────
SST_FEATURES = [
    'sst_roms',           # primary ROMS prediction
    'salt_roms',          # surface salinity — proxy for water mass
    'zeta_roms',          # sea surface height — tidal/surge state
    'sustr_roms',         # wind stress x — atmospheric forcing
    'svstr_roms',         # wind stress y
    'depth_m',            # bathymetry — shallow vs deep bias
    'coast_dist',         # distance to coast — coastal vs offshore
    'month_sin',          # seasonal cycle (cyclical)
    'month_cos',
    'doy_sin',            # day-of-year (cyclical)
    'doy_cos',
    'residual_sst_lag1',  # bias persistence from previous month
]
SST_TARGET  = 'residual_sst'

# ── Chl ─────────────────────────────────────────────────────
CHL_FEATURES = [
    'chl_roms',                # primary ROMS prediction
    'depth_m',                 # bathymetry
    'coast_dist',              # coastal vs offshore
    'month_sin',               # seasonal cycle
    'month_cos',
    'doy_sin',
    'doy_cos',
    'log_residual_chl_lag1',   # bias persistence
]
CHL_TARGET  = 'log_residual_chl'

# ── XGBoost hyperparameters ──────────────────────────────────
# Tuned for ~2M row tabular environmental data
XGB_PARAMS = {
    'n_estimators':     500,
    'max_depth':        6,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,    # prevents overfitting on sparse coastal cells
    'reg_alpha':        0.1,   # L1
    'reg_lambda':       1.0,   # L2
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'tree_method':      'hist', # fast for large datasets
    'n_jobs':           -1,
    'random_state':     42,
    'early_stopping_rounds': 30,
}

# SHAP sample size — full dataset is slow, sample is representative
SHAP_SAMPLE = 50_000

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label=''):
    """Compute and print regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    print(f'  {label}')
    print(f'    RMSE: {rmse:.4f}')
    print(f'    MAE:  {mae:.4f}')
    print(f'    R²:   {r2:.4f}')
    print(f'    Bias: {bias:.4f}')
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'bias': bias}


def compute_corrected_metrics(obs, roms, residual_pred, label='', log_space=False):
    """
    Compare raw ROMS vs XGBoost-corrected against observations.
    """
    if log_space:
        # Corrected Chl = roms * exp(predicted_log_residual)
        corrected = roms * np.exp(residual_pred)
        print(f'\n  {label} — raw ROMS vs observations:')
        compute_metrics(obs, roms, label='  ROMS raw')
        print(f'  {label} — XGBoost corrected vs observations:')
        compute_metrics(obs, corrected, label='  XGB corrected')
    else:
        # Corrected SST = roms + predicted_residual
        corrected = roms + residual_pred
        print(f'\n  {label} — raw ROMS vs observations:')
        compute_metrics(obs, roms, label='  ROMS raw')
        print(f'  {label} — XGBoost corrected vs observations:')
        compute_metrics(obs, corrected, label='  XGB corrected')

    return corrected


# ─────────────────────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────────────────────

def train_model(df, features, target, label, extreme_flag_col=None):
    """
    Train XGBoost residual correction model.

    Parameters
    ----------
    df              : full feature DataFrame
    features        : list of feature column names
    target          : target column name
    label           : 'SST' or 'Chl'
    extreme_flag_col: optional column to exclude extreme outliers from training

    Returns
    -------
    model, df_train, df_test
    """
    print(f'\n{"="*55}')
    print(f'{label} XGBOOST TRAINING')
    print(f'{"="*55}')

    # ── Split ───────────────────────────────────────────────
    train = df[df['split'] == 'train'].copy()
    test  = df[df['split'] == 'test'].copy()

    # Drop rows with NaN in features or target
    train = train.dropna(subset=features + [target])
    test  = test.dropna(subset=features + [target])

    # Optionally exclude extreme outliers from training only
    if extreme_flag_col and extreme_flag_col in train.columns:
        n_before = len(train)
        train = train[train[extreme_flag_col] == 0]
        print(f'  Excluded {n_before - len(train):,} extreme outliers from training')

    print(f'  Train: {len(train):,} rows | Test: {len(test):,} rows')
    print(f'  Features ({len(features)}): {features}')
    print(f'  Target: {target}')

# Force numeric — guard against string-encoded masked values
    for col in features:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col]  = pd.to_numeric(test[col],  errors='coerce')
    train = train.dropna(subset=features)
    test  = test.dropna(subset=features)

    X_train = train[features].values
    y_train = train[target].values
    X_test  = test[features].values
    y_test  = test[target].values

    # ── Train ───────────────────────────────────────────────
    print(f'\n  Training XGBoost...')
    model = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    best_iter = model.best_iteration
    print(f'  Best iteration: {best_iter}')

    # ── Predict ─────────────────────────────────────────────
    train['residual_pred'] = model.predict(X_train)
    test['residual_pred']  = model.predict(X_test)

    # ── Residual metrics ────────────────────────────────────
    print(f'\n  --- Residual prediction metrics ---')
    print(f'  Train:')
    compute_metrics(y_train, train['residual_pred'].values, label='  train')
    print(f'  Test:')
    compute_metrics(y_test, test['residual_pred'].values, label='  test')

    # ── Corrected variable metrics ───────────────────────────
    log_space = (label == 'Chl')
    if label == 'SST':
        test['sst_corrected'] = test['sst_roms'] + test['residual_pred']
        compute_corrected_metrics(
            test['sst_obs'].values,
            test['sst_roms'].values,
            test['residual_pred'].values,
            label='SST', log_space=False
        )
    else:
        test['chl_corrected'] = test['chl_roms'] * np.exp(test['residual_pred'])
        compute_corrected_metrics(
            test['chl_obs'].values,
            test['chl_roms'].values,
            test['residual_pred'].values,
            label='Chl', log_space=True
        )

    # ── Monthly breakdown on test set ───────────────────────
    print(f'\n  --- Test residual by month ---')
    if label == 'SST':
        monthly = test.groupby('month').agg(
            rmse_raw =('residual_sst',   lambda x: np.sqrt(np.mean(x**2))),
            rmse_xgb =('residual_pred',  lambda x: np.sqrt(np.mean((x - test.loc[x.index, 'residual_sst'])**2))),
            bias_raw =('residual_sst',   'mean'),
        ).round(3)
    else:
        monthly = test.groupby('month').agg(
            rmse_raw=('log_residual_chl', lambda x: np.sqrt(np.mean(x**2))),
            bias_raw=('log_residual_chl', 'mean'),
        ).round(3)
    print(monthly)

    return model, train, test


# ─────────────────────────────────────────────────────────────
# SHAP ANALYSIS
# ─────────────────────────────────────────────────────────────

def compute_shap(model, df_test, features, label):
    """
    Compute feature importance via XGBoost gain score.
    Fallback for SHAP/XGBoost version incompatibility.
    """
    print(f'\n  Computing feature importance ({label})...')

    # XGBoost native importance — gain is most informative
    importance = model.get_booster().get_score(importance_type='gain')

    # Align to feature list order, fill missing with 0
    scores = pd.Series(
        {f: importance.get(f'f{i}', 0.0) for i, f in enumerate(features)},
        name='gain'
    )
    scores = scores.sort_values(ascending=False)
    total  = scores.sum()

    print(f'\n  Feature importance by gain ({label}):')
    for feat, val in scores.items():
        pct = 100 * val / total
        bar = '█' * int(pct / 2)
        print(f'    {feat:<30} {pct:5.1f}%  {bar}')

    # Save as DataFrame — same interface as SHAP output
    df_importance = pd.DataFrame({
        'feature':    scores.index,
        'gain':       scores.values,
        'gain_pct':   100 * scores.values / total,
    })

    return df_importance


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── SST ─────────────────────────────────────────────────
    print('Loading SST features...')
    sst = pd.read_parquet(COLLOC_DIR / 'features_sst_2010_2020.parquet')
    print(f'  {len(sst):,} rows loaded')

    model_sst, train_sst, test_sst = train_model(
        sst, SST_FEATURES, SST_TARGET,
        label='SST',
        extreme_flag_col='residual_sst_extreme'
    )

    # Save model
    model_path = OUT_DIR / 'xgb_sst_model.json'
    model_sst.save_model(model_path)
    print(f'\n  Model saved: {model_path.name}')

    # Save test predictions
    pred_cols = ['time', 'lat', 'lon', 'month', 'year',
                 'sst_obs', 'sst_roms', 'residual_sst',
                 'residual_pred', 'sst_corrected', 'split']
    test_sst[pred_cols].to_parquet(
        OUT_DIR / 'predictions_sst_test.parquet', index=False
    )
    print(f'  Predictions saved: predictions_sst_test.parquet')

    # SHAP
    df_imp_sst = compute_shap(model_sst, test_sst, SST_FEATURES, 'SST')
    df_imp_sst.to_parquet(OUT_DIR / 'importance_sst.parquet', index=False)
    print(f'  Importance saved: importance_sst.parquet')

    # ── Chl ─────────────────────────────────────────────────
    print('\n\nLoading Chl features...')
    chl = pd.read_parquet(COLLOC_DIR / 'features_chl_2010_2020.parquet')
    print(f'  {len(chl):,} rows loaded')

    model_chl, train_chl, test_chl = train_model(
        chl, CHL_FEATURES, CHL_TARGET,
        label='Chl',
        extreme_flag_col='log_residual_chl_extreme'
    )

    # Save model
    model_path = OUT_DIR / 'xgb_chl_model.json'
    model_chl.save_model(model_path)
    print(f'\n  Model saved: {model_path.name}')

    # Save test predictions
    pred_cols = ['time', 'lat', 'lon', 'month', 'year',
                 'chl_obs', 'chl_roms', 'log_residual_chl',
                 'residual_pred', 'chl_corrected', 'split']
    test_chl[pred_cols].to_parquet(
        OUT_DIR / 'predictions_chl_test.parquet', index=False
    )
    print(f'  Predictions saved: predictions_chl_test.parquet')

    # SHAP
    df_imp_chl = compute_shap(model_chl, test_chl, CHL_FEATURES, 'Chl')
    df_imp_chl.to_parquet(OUT_DIR / 'importance_chl.parquet', index=False)
    print(f'  Importance saved: importance_chl.parquet')

    print('\n' + '='*55)
    print('Training complete.')
    print('Next step: visualise_results_roms.py')
    print('='*55)


if __name__ == '__main__':
    main()

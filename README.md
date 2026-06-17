# CE2COAST — ML Bias Correction Pipeline

**Physics-grounded machine learning correction of North Sea ROMS model outputs (SST and Chlorophyll-a), with optimal interpolation data assimilation and uncertainty quantification.**

*Eugène Ivanov — MAST, University of Liège*

---

## Overview

CE2COAST is a regional ocean model (ROMS) running North Sea simulations under the SSP3-7.0 climate scenario (MAR v3.14 / MPI-ESM forcing, 1980–2100). Despite capturing dominant physical dynamics, the model exhibits systematic seasonal biases in sea surface temperature and chlorophyll concentration.

This pipeline builds a three-layer correction system:

```
ROMS raw output
      ↓
XGBoost residual correction    (removes systematic seasonal + state-dependent bias)
      ↓
Optimal Interpolation update   (assimilates sparse in-situ observations)
      ↓
Analysis field with uncertainty bounds
```

---

## Key Results

| Variable | Metric | ROMS raw | XGB corrected | Improvement |
|---|---|---|---|---|
| SST | RMSE (°C) | 1.949 | 0.731 | −63% |
| SST | R² | 0.71 | 0.96 | +35% |
| SST | Bias (°C) | −0.47 | +0.02 | −96% |
| Chl | RMSE (mg/m³) | 2.952 | 0.913 | −69% |
| Chl | R² | −2.45 | 0.67 | +3.12 |

**Ablation study** shows spring SST bias reduction of 1.72°C — the dominant error mode is thermocline onset timing, not random noise.

**Uncertainty quantification**: P10–P90 prediction intervals with 63% empirical coverage (SST) and 77% (Chl, log-space). OI posterior spread shows local uncertainty reduction near in-situ stations.

**Anomaly detection**: Two-layer system (residual z-score + Isolation Forest) identifies 2019-Mar/Apr as the most anomalous months — consistent with the 2019 European spring heatwave. Both layers independently flag the Norwegian Trench as physically anomalous in 2019–2020.

---

## Pipeline

```
extract_colloc_roms.py    — ROMS × satellite collocation (KDTree / RegularGridInterpolator)
build_features_roms.py    — Feature engineering (depth, coast dist, doy cycle, lag residual)
train_xgboost_roms.py     — XGBoost quantile + point models, SHAP/gain importance
ablation_baseline.py      — Climatology-only vs physics-only vs full model
ingest_insitu_bgc.py      — CMEMS in-situ BGC profile ingestion (monthly resampling)
kalman_update_roms.py     — Optimal Interpolation: x_a = x_b + K(y − Hx_b)
visualise_results_roms.py — Scatter, monthly bias, spatial maps, feature importance
seasonal_error_maps.py    — Seasonal error decomposition + OI increment maps
uncertainty_sst.py        — Quantile regression P10/P50/P90 + OI spread (SST)
uncertainty_chl.py        — Quantile regression + cloud cover proxy (Chl)
anomaly_detection.py      — Residual z-score + Isolation Forest anomaly detection
ce2coast_config.py        — Shared path configuration (env-var based)
```

---

## Setup

### 1. Environment variables

```bash
cp .env.example .env
# Edit .env with your data paths
source .env
```

Required variables:

| Variable | Description |
|---|---|
| `CE2COAST_BASE` | Root data directory |
| `CE2COAST_COLLOC` | ML collocation outputs |
| `CE2COAST_ROMS` | ROMS AVG NetCDF files |
| `CE2COAST_VALID` | Satellite validation data |
| `CE2COAST_INSITU` | In-situ BGC profiles |

### 2. Dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, XGBoost 2.0, scikit-learn 1.3 on NIC5/CECI HPC (conda env Yoda).

### 3. Verify config

```bash
python ce2coast_config.py
```

### 4. Run full pipeline

```bash
bash run_pipeline.sh

# Or resume from a specific step
bash run_pipeline.sh --from 7
```

---

## Data

### ROMS outputs
CE2COAST monthly AVG files (`Hindcast_CE2COAST_AVG_{year}_2c_atm3.nc`), 2010–2020.
Grid: 240×180 rho-points, 30 vertical levels, DT=300s.
Forcing: MAR v3.14-ecRad / MPI-ESM SSP3-7.0.

### Satellite observations
- **SST**: CMEMS `OSTIA` L4 reprocessed daily (1/20° grid), resampled to monthly
- **Chl**: CMEMS `cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M`, monthly L4

### In-situ BGC
- CMEMS `INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030`
- 1182 platform files (Argo, CTD, Ferrybox)
- North Sea domain: 48–62°N, −6–11°E, 2010–2020
- Extracted: 2,867 monthly surface temperature obs, 223 monthly surface Chl obs

---

## Methods

### Collocation
ROMS (curvilinear grid, ~15km) and satellite (regular grid, 4km) are collocated via `scipy.spatial.cKDTree` nearest-neighbour (SST) and `scipy.interpolate.RegularGridInterpolator` bilinear (gridded satellite → ROMS rho-points).

### XGBoost residual correction
Target: `residual = obs − ROMS`. Features: ROMS physical state (SST/Chl, salinity, wind stress, SSH), bathymetry, coast distance, cyclical season encoding (sin/cos), lagged residual. Train: 2010–2018. Test: 2019–2020 (temporal holdout).

### Optimal Interpolation
Standard OI with Gaussian background error covariance:
`x_a = x_b + K(y − Hx_b)`, `K = BHᵀ(HBHᵀ + R)⁻¹`
Parameters: `L_b = 0.3°` (~30km), `σ_b² = 0.73²°C²` (from XGB RMSE), `σ_o² = 0.25°C²`.

### Uncertainty quantification
XGBoost quantile regression (`reg:quantileerror`) for P10/P50/P90 of residual correction. OI posterior spread `σ_a = √(B − KHB)`. Cloud cover proxy from NaN fraction in monthly satellite composites.

### Anomaly detection
- **Layer 1**: Residual z-score vs 2010–2018 climatology per grid point × month. Flag `|z| > 2`.
- **Layer 2**: Isolation Forest on 12-feature physical state matrix. Contamination = 5%.

---

## Scientific context

The dominant ROMS error modes identified:

| Season | ROMS error | Physical cause | XGB RMSE reduction |
|---|---|---|---|
| MAM | −2.05°C cold | Spring thermocline onset too late | 1.72°C |
| JJA | −1.63°C cold | Summer stratification underestimated | 1.33°C |
| SON | +1.25°C warm | Mixed layer deepening too slow | 0.95°C |
| DJF | +0.56°C warm | Winter convection overestimated | 0.71°C |

Chl: ROMS/FABM overestimates spring bloom by ~2.5× (log-residual mean −0.90). Physics features dominate correction (ΔR²=+0.57 beyond pure seasonality).

---

## Repository structure

```
.
├── ce2coast_config.py          # Path configuration (env-var based)
├── extract_colloc_roms.py      # Step 1: collocation
├── build_features_roms.py      # Step 2: feature engineering
├── train_xgboost_roms.py       # Step 3: XGBoost training
├── ablation_baseline.py        # Step 4: ablation study
├── ingest_insitu_bgc.py        # Step 5: in-situ ingestion
├── kalman_update_roms.py       # Step 6: OI analysis
├── visualise_results_roms.py   # Step 7: diagnostics
├── seasonal_error_maps.py      # Step 8: seasonal maps
├── uncertainty_sst.py          # Step 9: SST uncertainty
├── uncertainty_chl.py          # Step 10: Chl uncertainty
├── anomaly_detection.py        # Step 11: anomaly detection
├── run_pipeline.sh             # Full pipeline runner
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

MIT License. Data products derived from Copernicus Marine Service — see [marine.copernicus.eu](https://marine.copernicus.eu) for data attribution requirements.

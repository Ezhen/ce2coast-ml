#!/bin/bash
# run_pipeline.sh
# ===============
# Run the full CE2COAST ML bias correction pipeline in order.
# Source your environment first: source .env
#
# Usage:
#   bash run_pipeline.sh           # full pipeline
#   bash run_pipeline.sh --from 3  # resume from step 3

set -e

STEP_FROM=${2:-1}

run_step() {
    local n=$1
    local label=$2
    local script=$3
    if [ "$n" -ge "$STEP_FROM" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Step $n — $label"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python -u "$script"
    else
        echo "  [skip] Step $n — $label"
    fi
}

echo "CE2COAST ML Bias Correction Pipeline"
echo "====================================="
python ce2coast_config.py   # validate env vars first

run_step 1 "Collocation: ROMS vs satellite obs"     extract_colloc_roms.py
run_step 2 "Feature engineering"                    build_features_roms.py
run_step 3 "XGBoost training (SST + Chl)"           train_xgboost_roms.py
run_step 4 "Ablation study"                         ablation_baseline.py
run_step 5 "In-situ BGC ingestion"                  ingest_insitu_bgc.py
run_step 6 "Kalman / OI analysis"                   kalman_update_roms.py
run_step 7 "Diagnostic visualisation"               visualise_results_roms.py
run_step 8 "Seasonal error maps"                    seasonal_error_maps.py
run_step 9 "Uncertainty SST"                        uncertainty_sst.py
run_step 10 "Uncertainty Chl"                       uncertainty_chl.py
run_step 11 "Anomaly detection"                     anomaly_detection.py

echo ""
echo "Pipeline complete. Figures in: $CE2COAST_COLLOC/figures/"

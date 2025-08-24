#!/bin/bash

# Set environment variables
export PYTHONPATH=/app
export FLASK_APP=app.py
export FLASK_ENV=production
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export OMP_NUM_THREADS=1  # Limit OpenMP threads
export MKL_NUM_THREADS=1  # Limit MKL threads
export NUMEXPR_NUM_THREADS=1  # Limit NumExpr threads

# Handle PORT environment variable
if [ -z "$PORT" ]; then
    PORT=5000
fi

echo "Starting DisastroScope Backend with all advanced features..."
echo "Using port: $PORT"

# Initialize advanced libraries
echo "Initializing advanced libraries..."

# Test TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test XGBoost
python -c "import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')"

# Test LightGBM
python -c "import lightgbm as lgb; print(f'LightGBM version: {lgb.__version__}')"

# Test Optuna
python -c "import optuna; print(f'Optuna version: {optuna.__version__}')"

# Test SHAP
python -c "import shap; print(f'SHAP version: {shap.__version__}')"

# Test spaCy
python -c "import spacy; print(f'spaCy version: {spacy.__version__}')"

echo "All advanced libraries initialized successfully!"

# Check AI models
echo "Checking AI models..."

# Train advanced models first
echo "Training advanced AI models..."
python -c "
from ai_models import ai_prediction_service
try:
    results = ai_prediction_service.train_advanced_models(epochs=50)
    print('Advanced AI models trained successfully!')
    for hazard_type, result in results.items():
        if result['status'] == 'success':
            print(f'  ✓ {hazard_type}: {result.get(\"accuracy\", 0):.3f} accuracy')
        else:
            print(f'  ✗ {hazard_type}: {result.get(\"error\", \"Unknown error\")}')
except Exception as e:
    print(f'Advanced training failed: {e}')
    print('Falling back to simple models...')
    try:
        results = ai_prediction_service.train_simple_models()
        print('Simple AI models trained successfully as fallback!')
    except Exception as e2:
        print(f'Simple training also failed: {e2}')
        print('Using heuristic models only...')
"

# Start Gunicorn with optimized settings for heavy ML workloads
echo "Starting Gunicorn server..."
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --worker-class sync \
    --timeout 300 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app

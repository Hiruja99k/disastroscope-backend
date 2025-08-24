#!/bin/bash

# Startup script for Railway deployment
# This ensures we use sync workers and avoid eventlet dependency issues

echo "Starting DisastroScope Backend with sync workers..."

# Set environment variables
export PYTHONPATH=/app
export FLASK_APP=app.py
export FLASK_ENV=production

# Set AI model configuration
export AI_AUTO_TRAIN_ON_STARTUP=true
export AI_STARTUP_TRAIN_EPOCHS=50
export ENSEMBLE_ENABLED=true
export DEEP_LEARNING_ENABLED=true
export FEATURE_ENGINEERING_ENABLED=true

# Use default port 5000 if PORT is not set
PORT=${PORT:-5000}
echo "Starting server on port $PORT"

# Check if AI models need training
echo "Checking AI models..."
python -c "
import os
import sys
sys.path.insert(0, '/app')

try:
    from ai_models import ai_prediction_service
    print('AI prediction service loaded successfully')
    
    # Check if models exist
    if not hasattr(ai_prediction_service, 'models') or not ai_prediction_service.models:
        print('No AI models found. Training models...')
        ai_prediction_service.train_advanced_models(epochs=50)
        print('AI models trained successfully!')
    else:
        print('AI models already exist and are loaded')
        
except Exception as e:
    print(f'Error during AI model initialization: {e}')
    print('Continuing with heuristic predictions only')
"

# Start Gunicorn with explicit sync worker configuration
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --worker-class sync \
    --timeout 30 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app

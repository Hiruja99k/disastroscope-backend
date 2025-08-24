#!/bin/bash

# Startup script for Railway deployment
# This ensures we use sync workers and avoid eventlet dependency issues

echo "Starting DisastroScope Backend with sync workers..."

# Set environment variables
export PYTHONPATH=/app
export FLASK_APP=app.py
export FLASK_ENV=production

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

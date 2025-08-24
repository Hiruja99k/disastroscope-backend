#!/bin/bash

# Set environment variables
export PYTHONPATH=/app
export FLASK_APP=app.py
export FLASK_ENV=production

# Handle PORT environment variable
if [ -z "$PORT" ]; then
    PORT=5000
fi

echo "Starting DisastroScope Backend..."
echo "Using port: $PORT"

# Start Gunicorn with simple settings
echo "Starting Gunicorn server..."
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --worker-class eventlet \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app

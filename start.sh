#!/bin/bash

# DisastroScope Backend Startup Script
# This script ensures gunicorn runs with the correct configuration

echo "Starting DisastroScope Backend..."

# Set environment variables
export GUNICORN_CMD_ARGS="--worker-class sync --workers 1 --bind 0.0.0.0:$PORT --timeout 120 --max-requests 1000"

# Start the application with explicit sync workers
exec gunicorn --worker-class sync --workers 1 --bind 0.0.0.0:$PORT --timeout 120 app:app

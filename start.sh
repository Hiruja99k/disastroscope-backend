#!/bin/bash

# Ensure we're using sync workers, not eventlet
export GUNICORN_CMD_ARGS="--worker-class sync --workers 1 --bind 0.0.0.0:$PORT --timeout 120"

# Start the application
exec gunicorn app:app

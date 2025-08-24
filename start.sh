#!/bin/bash

# DisastroScope Backend Startup Script
# Production-ready startup with monitoring and error handling

set -e  # Exit on any error

echo "🚀 Starting DisastroScope Backend v2.0.0..."

# Environment validation
if [ -z "$PORT" ]; then
    echo "⚠️  PORT not set, using default 5000"
    export PORT=5000
fi

# Set production environment variables
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export MAX_REQUESTS_PER_MINUTE=${MAX_REQUESTS_PER_MINUTE:-100}

echo "📋 Environment Configuration:"
echo "   - Environment: $ENVIRONMENT"
echo "   - Port: $PORT"
echo "   - Log Level: $LOG_LEVEL"
echo "   - Rate Limit: $MAX_REQUESTS_PER_MINUTE req/min"

# Health check function
health_check() {
    echo "🏥 Performing health check..."
    sleep 5
    if curl -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ Health check passed"
        return 0
    else
        echo "❌ Health check failed"
        return 1
    fi
}

# Signal handler for graceful shutdown
cleanup() {
    echo "🛑 Received shutdown signal, cleaning up..."
    # Add any cleanup logic here
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start the application with explicit configuration
echo "🔧 Starting Gunicorn with production configuration..."

exec gunicorn \
    --config gunicorn.conf.py \
    --worker-class sync \
    --workers 1 \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app:app

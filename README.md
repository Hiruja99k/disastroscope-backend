# DisastroScope Enterprise Backend

A high-performance, enterprise-grade disaster prediction and monitoring system built with advanced AI models, real-time monitoring, and comprehensive observability.

## 🚀 Enterprise Features

### 🤖 Advanced AI Models
- **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Neural Networks
- **Auto-Training**: Automatic model retraining with new data
- **Model Versioning**: Track model performance and versions
- **Multi-Hazard Support**: Flood, Wildfire, Storm, Tornado, Landslide, Drought, Earthquake
- **Real-time Predictions**: Sub-second prediction times with performance monitoring

### 📊 Monitoring & Observability
- **Prometheus Metrics**: Comprehensive system and application metrics
- **OpenTelemetry Tracing**: Distributed tracing with Jaeger
- **Health Checks**: Automated health monitoring and alerting
- **Performance Monitoring**: Real-time performance tracking
- **Structured Logging**: JSON-structured logs with correlation IDs

### 🔒 Security & Performance
- **Rate Limiting**: Configurable API rate limiting
- **CORS Management**: Secure cross-origin resource sharing
- **API Key Authentication**: Optional API key-based authentication
- **Performance Tuning**: Optimized for high-throughput scenarios
- **Caching**: Redis-based caching for improved performance

### 🌐 Data Sources
- **Weather Data**: OpenWeatherMap API integration
- **Disaster Events**: NASA EONET, GDACS, FIRMS
- **Government Data**: OpenFEMA disaster declarations
- **Real-time Updates**: WebSocket-based real-time data streaming

## 🛠️ Technology Stack

- **Framework**: Flask 2.3.3 with SocketIO
- **AI/ML**: Scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **Monitoring**: Prometheus, OpenTelemetry, Jaeger
- **Caching**: Redis
- **Database**: PostgreSQL (with SQLite fallback)
- **Deployment**: Railway, Docker-ready
- **Logging**: Structlog with JSON formatting

## 📋 Prerequisites

- Python 3.9+
- Redis (optional, for caching)
- PostgreSQL (optional, SQLite for development)
- Railway account (for deployment)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Hiruja99k/disastroscope-backend.git
cd disastroscope-backend
```

### 2. Environment Configuration

Create a `.env` file with your configuration:

```env
# Core Configuration
ENVIRONMENT=production
SECRET_KEY=your-secure-secret-key-here
DEBUG=false

# API Keys
WEATHER_API_KEY=your-openweathermap-api-key
FIRMS_API_TOKEN=your-firms-api-token
GEMINI_API_KEY=your-gemini-api-key
OPENFEMA_API_KEY=your-openfema-api-key
EONET_API_KEY=your-eonet-api-key

# Database
DATABASE_URL=postgresql://user:password@localhost/disastroscope
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Monitoring
PROMETHEUS_ENABLED=true
OPENTELEMETRY_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=6831

# AI Configuration
ENSEMBLE_ENABLED=true
AI_AUTO_TRAIN_ON_STARTUP=true
MODEL_VERSION=2.0.0
AI_STARTUP_TRAIN_EPOCHS=100

# Security
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
API_KEY_REQUIRED=false
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Performance
WORKER_PROCESSES=4
WORKER_THREADS=2
MAX_CONNECTIONS=1000
CACHE_TTL=300
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize AI Models

```bash
python -c "from ai_models import ai_prediction_service; ai_prediction_service.train_advanced_models()"
```

### 5. Run the Application

#### Development
```bash
python app.py
```

#### Production with Gunicorn
```bash
gunicorn -w 4 -k eventlet -b 0.0.0.0:5000 app:app
```

## 📡 API Endpoints

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns system health status and metrics.

#### Metrics (Prometheus)
```http
GET /metrics
```
Returns Prometheus-formatted metrics.

#### AI Models Status
```http
GET /api/models
```
Returns comprehensive AI model status and performance metrics.

#### Disaster Predictions
```http
POST /api/ai/predict
Content-Type: application/json

{
  "lat": 37.7749,
  "lon": -122.4194,
  "location_name": "San Francisco, CA"
}
```

#### Train AI Models
```http
POST /api/ai/train
Content-Type: application/json

{
  "epochs": 100,
  "auto_train": true
}
```

#### Weather Data
```http
GET /api/weather
```
Returns weather data for monitored locations.

#### Current Weather
```http
GET /api/weather/current?lat=37.7749&lon=-122.4194&name=San Francisco
```

### Real-time WebSocket Events

Connect to `/socket.io` for real-time updates:

- `new_event`: New disaster event detected
- `new_prediction`: New AI prediction generated
- `weather_update`: Weather data updates
- `system_alert`: System alerts and notifications

## 🔧 Configuration

### Environment Variables

The system uses a comprehensive configuration system. See `config.py` for all available options.

### AI Model Configuration

```python
# Enable/disable ensemble learning
ENSEMBLE_ENABLED=true

# Auto-training configuration
AI_AUTO_TRAIN_ON_STARTUP=true
AI_STARTUP_TRAIN_EPOCHS=100

# Model versioning
MODEL_VERSION=2.0.0

# Prediction thresholds
PREDICTION_THRESHOLD=0.1
MAX_PREDICTION_TIME=5.0
```

### Monitoring Configuration

```python
# Prometheus metrics
PROMETHEUS_ENABLED=true

# OpenTelemetry tracing
OPENTELEMETRY_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=6831

# Metrics collection interval
METRICS_INTERVAL=30
HEALTH_CHECK_INTERVAL=60
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The application exposes comprehensive metrics at `/metrics`:

- **Request Metrics**: Total requests, duration, status codes
- **AI Metrics**: Prediction counts, durations, accuracy
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Weather requests, disaster events

### Health Checks

Access `/health` for system health status:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "checks": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.1,
      "status": "healthy"
    },
    "application": {
      "requests_per_second": 12.5,
      "error_rate": 0.02,
      "average_response_time": 0.8,
      "status": "healthy"
    }
  }
}
```

### Structured Logging

All logs are structured JSON with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "info",
  "logger": "app",
  "message": "Enterprise AI prediction completed",
  "location": "San Francisco, CA",
  "weather_fetch_duration_ms": 245.67,
  "prediction_duration_ms": 123.45,
  "total_duration_ms": 369.12,
  "predictions_count": 7
}
```

## 🚀 Deployment

### Railway Deployment

1. **Connect Repository**: Link your GitHub repository to Railway
2. **Environment Variables**: Set all required environment variables
3. **Deploy**: Railway will automatically deploy on push to main branch

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-k", "eventlet", "-b", "0.0.0.0:5000", "app:app"]
```

### Production Considerations

1. **Database**: Use PostgreSQL for production
2. **Caching**: Enable Redis for improved performance
3. **Monitoring**: Set up Prometheus and Jaeger
4. **SSL**: Configure HTTPS with proper certificates
5. **Rate Limiting**: Enable and configure rate limiting
6. **Backup**: Set up automated database backups

## 🔍 Troubleshooting

### Common Issues

1. **AI Models Not Loading**
   - Check model files exist in `models/` directory
   - Verify model training completed successfully
   - Check logs for model loading errors

2. **High Response Times**
   - Monitor system resources (CPU, memory, disk)
   - Check database connection pool settings
   - Verify Redis connectivity (if using caching)

3. **API Errors**
   - Check API key configuration
   - Verify rate limiting settings
   - Review CORS configuration

### Debug Mode

Enable debug mode for detailed error information:

```env
DEBUG=true
ENVIRONMENT=development
```

### Log Analysis

Use structured logging for better debugging:

```bash
# Filter logs by level
grep '"level":"error"' app.log

# Filter by specific operation
grep '"message":"AI prediction"' app.log

# Filter by duration
grep '"total_duration_ms":[0-9]{4,}' app.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

## 🔄 Changelog

### Version 2.0.0 (Enterprise)
- ✨ Advanced ensemble AI models
- 📊 Comprehensive monitoring and observability
- 🔒 Enhanced security features
- ⚡ Performance optimizations
- 🌐 Real-time WebSocket support
- 📈 Prometheus metrics integration
- 🔍 Distributed tracing with OpenTelemetry
- 🚀 Railway deployment ready

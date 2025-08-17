# DisastroScope Backend API

A Flask-based backend API for disaster prediction and monitoring using AI models and real-time weather data.

## Features

- **AI-Powered Disaster Prediction**: Neural network models for flood, wildfire, storm, earthquake, tornado, landslide, and drought prediction
- **Real-time Weather Integration**: OpenWeatherMap API with Open-Meteo fallback
- **Geocoding Services**: Worldwide location search and coordinate conversion
- **Real-time Updates**: WebSocket support for live data streaming
- **FEMA Integration**: Real-time disaster declarations from OpenFEMA
- **NASA EONET**: Natural event monitoring from NASA's Earth Observatory

## API Endpoints

### Health Check
- `GET /api/health` - Service health status

### Weather
- `GET /api/weather` - Weather for monitored locations
- `GET /api/weather/current` - Current weather by coordinates
- `GET /api/weather/by-city` - Weather by city name
- `GET /api/weather/forecast` - Weather forecast
- `GET /api/geocode` - Location geocoding

### AI Predictions
- `POST /api/ai/predict` - Generate disaster predictions for location
- `POST /api/ai/train` - Train AI models

### Events & Data
- `GET /api/events` - Disaster events
- `GET /api/predictions` - AI predictions
- `GET /api/sensors` - Sensor data
- `GET /api/stats` - System statistics
- `GET /api/disasters` - FEMA disaster declarations
- `GET /api/eonet` - NASA EONET events

## Environment Variables

```bash
OPENWEATHER_API_KEY=your_openweather_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Deployment

This backend is configured for deployment on Railway with:
- Gunicorn + Eventlet for WebSocket support
- Python 3.12 runtime
- Automatic dependency installation

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

The API will be available at `http://localhost:5000`

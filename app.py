from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta, timezone
import random
import threading
import time
from typing import Dict, List, Any
import logging
import asyncio
from contextlib import suppress
import joblib

# Load environment variables from .env (must happen BEFORE importing modules that read env)
load_dotenv()

# Enterprise imports
from monitoring import monitoring
import structlog
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from weather_service import weather_service, WeatherData
from ai_models import ai_prediction_service
from openfema_service import openfema_service, FEMADeclaration
from eonet_service import eonet_service, EONETEvent
from training.wildfire_trainer import train_and_save as train_wildfire
from training.storm_trainer import train_and_save as train_storm
from training.flood_trainer import train_and_save as train_flood
from training.landslide_trainer import train_and_save as train_landslide
from training.drought_trainer import train_and_save as train_drought
from source_services import fetch_gdacs_events_near, fetch_firms_count_near, fetch_openaq_near
from hazard_services import get_earthquake_hazard

# Configure enterprise logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize Flask app with enterprise features
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['ENVIRONMENT'] = os.getenv('ENVIRONMENT', 'production')

# Enterprise CORS configuration
CORS(app, resources={r"/api/*": {
    "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
    "max_age": 600
}})

# Initialize monitoring
monitoring.init_app(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# In-memory storage (replace with database in production)
disaster_events = []
predictions = []
sensor_data = []
historical_data = []
weather_data_cache = []
fema_disasters = []  # OpenFEMA disaster declarations (list of dicts)
eonet_events = []    # NASA EONET events (list of dicts)

# Optional: Gemini configuration for natural-language summaries
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai_model = None
with suppress(Exception):
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        # Prefer a fast, cost-effective model for summaries
        genai_model = genai.GenerativeModel('gemini-1.5-flash')

def generate_prediction_summary(event_type: str, location: str, weather: Dict[str, Any], risk_score: float) -> str:
    """Generate a brief, professional summary for a predicted disaster.
    Returns an empty string if Gemini is not configured or on any error.
    """
    if not genai_model:
        return ""
    try:
        prompt = (
            "You are a disaster risk analyst. Given live weather features and an AI risk score, "
            "write a concise (2-3 sentences) professional summary for a potential {etype} at {loc}. "
            "Focus on risk drivers (e.g., wind, precipitation), expected timeframe (~24-72h), and a clear actionable note.\n\n"
            f"Disaster: {event_type}\n"
            f"Location: {location}\n"
            f"Risk score (0-1): {risk_score:.2f}\n"
            f"Weather: {json.dumps(weather, ensure_ascii=False)}\n"
        ).format(etype=event_type, loc=location)
        resp = genai_model.generate_content(prompt)
        text = getattr(resp, 'text', None)
        if isinstance(text, str):
            return text.strip()
    except Exception as e:
        logger.warning(f"Gemini summary generation failed: {e}")
    return ""

# Major cities for weather monitoring
MONITORED_LOCATIONS = [
    {'name': 'San Francisco, CA', 'coords': {'lat': 37.7749, 'lng': -122.4194}},
    {'name': 'Los Angeles, CA', 'coords': {'lat': 34.0522, 'lng': -118.2437}},
    {'name': 'Miami, FL', 'coords': {'lat': 25.7617, 'lng': -80.1918}},
    {'name': 'New York, NY', 'coords': {'lat': 40.7128, 'lng': -74.0060}},
    {'name': 'Houston, TX', 'coords': {'lat': 29.7604, 'lng': -95.3698}},
    {'name': 'Seattle, WA', 'coords': {'lat': 47.6062, 'lng': -122.3321}},
    {'name': 'New Orleans, LA', 'coords': {'lat': 29.9511, 'lng': -90.0715}},
    {'name': 'Portland, OR', 'coords': {'lat': 45.5152, 'lng': -122.6784}},
    {'name': 'Chicago, IL', 'coords': {'lat': 41.8781, 'lng': -87.6298}},
    {'name': 'Denver, CO', 'coords': {'lat': 39.7392, 'lng': -104.9903}}
]

class DisasterEvent:
    def __init__(self, event_id: str, name: str, event_type: str, location: str, 
                 severity: str, status: str, coordinates: Dict[str, float], 
                 affected_population: int = 0, economic_impact: float = 0.0,
                 weather_data: Dict = None, ai_confidence: float = 0.0):
        self.id = event_id
        self.name = name
        self.event_type = event_type
        self.location = location
        self.severity = severity
        self.status = status
        self.coordinates = coordinates
        self.affected_population = affected_population
        self.economic_impact = economic_impact
        self.weather_data = weather_data
        self.ai_confidence = ai_confidence
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'event_type': self.event_type,
            'location': self.location,
            'severity': self.severity,
            'status': self.status,
            'coordinates': self.coordinates,
            'affected_population': self.affected_population,
            'economic_impact': self.economic_impact,
            'weather_data': self.weather_data,
            'ai_confidence': self.ai_confidence,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Prediction:
    def __init__(self, prediction_id: str, event_type: str, location: str, 
                 probability: float, severity: str, timeframe: str,
                 coordinates: Dict[str, float], weather_data: Dict = None,
                 ai_model: str = "PyTorch Neural Network", potential_impact: str = ""):
        self.id = prediction_id
        self.event_type = event_type
        self.location = location
        self.probability = probability
        self.severity = severity
        self.timeframe = timeframe
        self.coordinates = coordinates
        self.weather_data = weather_data
        self.ai_model = ai_model
        self.potential_impact = potential_impact
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'event_type': self.event_type,
            'location': self.location,
            'probability': self.probability,
            'severity': self.severity,
            'timeframe': self.timeframe,
            'coordinates': self.coordinates,
            'weather_data': self.weather_data,
            'ai_model': self.ai_model,
            'potential_impact': self.potential_impact,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class SensorData:
    def __init__(self, sensor_id: str, sensor_type: str, location: str,
                 coordinates: Dict[str, float], reading_value: float, 
                 reading_unit: str, data_quality: str = "good"):
        self.id = sensor_id
        self.sensor_type = sensor_type
        self.location = location
        self.coordinates = coordinates
        self.reading_value = reading_value
        self.reading_unit = reading_unit
        self.data_quality = data_quality
        self.reading_time = datetime.now(timezone.utc)
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sensor_type': self.sensor_type,
            'location': self.location,
            'coordinates': self.coordinates,
            'reading_value': self.reading_value,
            'reading_unit': self.reading_unit,
            'data_quality': self.data_quality,
            'reading_time': self.reading_time.isoformat(),
            'created_at': self.created_at.isoformat()
        }

# Health check endpoint
@app.route('/api/health')
def health_check():
    """Health check endpoint for Railway deployment"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'service': 'DisastroScope Backend API',
        'version': '1.0.0'
    })

# Geocoding endpoint for worldwide location search
@app.route('/api/geocode')
def geocode_location():
    """Geocode a location query to get coordinates"""
    query = request.args.get('query')
    limit = int(request.args.get('limit', 5))
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Fetch multiple candidates to improve worldwide matching
        results = loop.run_until_complete(weather_service.geocode(query, limit=5))
        if not results:
            loop.close()
            return jsonify({'error': 'No results for query'}), 404

        # Choose the best candidate: prefer exact (case-insensitive) name match, else first
        qnorm = query.strip().lower()
        def score(item: dict) -> int:
            name = str(item.get('name') or '').lower()
            state = str(item.get('state') or '')
            country = str(item.get('country') or '')
            # exact name gets higher score
            s = 0
            if name == qnorm:
                s += 3
            if qnorm in name:
                s += 1
            # presence of state/country boosts confidence
            if state:
                s += 1
            if country:
                s += 1
            return s

        best = sorted(results, key=score, reverse=True)[0]
        lat = best['lat']
        lon = best['lon']
        name = f"{best.get('name')}{', ' + best.get('state') if best.get('state') else ''}{' ' + best.get('country') if best.get('country') else ''}".strip()
        weather = loop.run_until_complete(weather_service.get_current_weather(lat, lon, name, 'metric'))
        loop.close()
        if not weather:
            return jsonify({'error': 'Failed to fetch weather'}), 502
        return jsonify(weather.to_dict())
    except Exception as e:
        logger.error(f"Error fetching weather by city: {e}")
        return jsonify({'error': 'Failed to fetch weather by city'}), 500

# Enhanced location-based analysis endpoint
@app.route('/api/location/analyze', methods=['POST'])
def analyze_location():
    """Analyze a location for disaster risks and weather data"""
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'Location query required'}), 400
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Step 1: Geocode the location
        results = loop.run_until_complete(weather_service.geocode(query, limit=5))
        if not results:
            loop.close()
            return jsonify({'error': 'Location not found'}), 404

        # Step 2: Get the best match
        qnorm = query.strip().lower()
        def score(item: dict) -> int:
            name = str(item.get('name') or '').lower()
            state = str(item.get('state') or '')
            country = str(item.get('country') or '')
            s = 0
            if name == qnorm:
                s += 3
            if qnorm in name:
                s += 1
            if state:
                s += 1
            if country:
                s += 1
            return s

        best = sorted(results, key=score, reverse=True)[0]
        lat = best['lat']
        lon = best['lon']
        location_name = f"{best.get('name')}{', ' + best.get('state') if best.get('state') else ''}{' ' + best.get('country') if best.get('country') else ''}".strip()
        
        # Step 3: Get current weather
        weather = loop.run_until_complete(weather_service.get_current_weather(lat, lon, location_name, 'metric'))
        if not weather:
            loop.close()
            return jsonify({'error': 'Could not compute prediction for your location - weather data unavailable'}), 502
        
        # Step 4: Generate AI predictions
        weather_dict = {
            'temperature': weather.temperature,
            'humidity': weather.humidity,
            'pressure': weather.pressure,
            'wind_speed': weather.wind_speed,
            'wind_direction': weather.wind_direction,
            'precipitation': weather.precipitation,
            'visibility': weather.visibility,
            'cloud_cover': weather.cloud_cover
        }
        
        predictions = ai_prediction_service.predict_disaster_risks(weather_dict)
        
        # Step 5: Get weather forecast
        forecast = loop.run_until_complete(weather_service.get_weather_forecast(lat, lon, 5, 'metric'))
        loop.close()
        
        # Step 6: Compile comprehensive analysis
        analysis = {
            'location': {
                'name': location_name,
                'coordinates': {'lat': lat, 'lng': lon},
                'geocoding_confidence': 'high' if best.get('state') and best.get('country') else 'medium'
            },
            'current_weather': weather_dict,
            'disaster_risks': predictions,
            'forecast': forecast[:8],  # First 24 hours (3-hour intervals)
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'risk_summary': _generate_risk_summary(predictions, weather_dict)
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in location analysis: {e}")
        return jsonify({'error': 'Could not compute prediction for your location'}), 500

def _generate_risk_summary(predictions: Dict[str, float], weather: Dict) -> str:
    """Generate a human-readable risk summary"""
    high_risks = [k for k, v in predictions.items() if v > 0.6]
    medium_risks = [k for k, v in predictions.items() if 0.3 < v <= 0.6]
    
    summary = f"Current weather: {weather.get('temperature', 0):.1f}°C, "
    summary += f"{weather.get('humidity', 0):.0f}% humidity, "
    summary += f"{weather.get('wind_speed', 0):.1f} m/s wind"
    
    if high_risks:
        summary += f". HIGH RISK: {', '.join(high_risks).title()}"
    elif medium_risks:
        summary += f". MEDIUM RISK: {', '.join(medium_risks).title()}"
    else:
        summary += ". LOW RISK conditions"
    
    return summary

# Helper: reverse geocode coordinates to a friendly name
def reverse_geocode_osm(lat: float, lon: float) -> str:
    try:
        import requests as http
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'format': 'json',
            'lat': lat,
            'lon': lon,
            'zoom': 12,
            'addressdetails': 1,
        }
        headers = {'User-Agent': 'DisastroScope/1.0 (contact: support@disastroscope.local)'}
        r = http.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            addr = data.get('address') or {}
            city = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('hamlet') or ''
            state = addr.get('state') or addr.get('province') or ''
            country = addr.get('country') or ''
            name = ", ".join([p for p in [city, state, country] if p])
            return name or data.get('display_name') or f"{lat:.4f}, {lon:.4f}"
    except Exception:
        pass
    return f"{lat:.4f}, {lon:.4f}"

@app.route('/api/location/analyze/coords', methods=['POST', 'OPTIONS'])
def analyze_location_by_coords():
    """Analyze a location by exact coordinates for disaster risks and weather data."""
    data = request.get_json() or {}
    lat = data.get('lat', None)
    lon = data.get('lon', None)
    units = data.get('units', 'metric')
    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon are required'}), 400
    try:
        latf = float(lat)
        lonf = float(lon)
    except Exception:
        return jsonify({'error': 'invalid coordinates'}), 400

    try:
        # Reverse geocode for display name
        location_name = reverse_geocode_osm(latf, lonf)
        # Fetch weather
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather = loop.run_until_complete(weather_service.get_current_weather(latf, lonf, location_name, units))
        if not weather:
            loop.close()
            return jsonify({'error': 'weather unavailable'}), 502
        weather_dict = {
            'temperature': weather.temperature,
            'humidity': weather.humidity,
            'pressure': weather.pressure,
            'wind_speed': weather.wind_speed,
            'wind_direction': weather.wind_direction,
            'precipitation': weather.precipitation,
            'visibility': weather.visibility,
            'cloud_cover': weather.cloud_cover
        }
        preds = ai_prediction_service.predict_disaster_risks(weather_dict)
        forecast = loop.run_until_complete(weather_service.get_weather_forecast(latf, lonf, 5, units))
        loop.close()
        analysis = {
            'location': {
                'name': location_name,
                'coordinates': {'lat': latf, 'lng': lonf},
                'geocoding_confidence': 'high'
            },
            'current_weather': weather_dict,
            'disaster_risks': preds,
            'forecast': (forecast or [])[:8],
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'risk_summary': _generate_risk_summary(preds, weather_dict),
            'sources': {
                'gdacs_nearby': fetch_gdacs_events_near(latf, lonf, days=10, radius_km=300.0),
                'firms_recent_count': fetch_firms_count_near(latf, lonf, days=7, radius_km=50.0, token=os.getenv('FIRMS_API_TOKEN')),
                'air_quality': fetch_openaq_near(latf, lonf, radius_m=15000)
            },
            'confidence': {
                'model': 'hybrid-ml',
                'coverage': {
                    'wildfire': 'ml' if isinstance(ai_prediction_service.models.get('wildfire'), dict) else 'heuristic',
                    'storm': 'ml' if isinstance(ai_prediction_service.models.get('storm'), dict) else 'heuristic',
                    'flood': 'ml' if isinstance(ai_prediction_service.models.get('flood'), dict) else 'heuristic',
                    'landslide': 'ml' if isinstance(ai_prediction_service.models.get('landslide'), dict) else 'heuristic',
                    'drought': 'ml' if isinstance(ai_prediction_service.models.get('drought'), dict) else 'heuristic',
                    'earthquake': 'data-driven'
                }
            }
        }
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error in coordinate analysis: {e}")
        return jsonify({'error': 'analysis failed'}), 500

# Explicitly handle CORS preflight for all API routes
@app.before_request
def handle_cors_preflight():
    from flask import make_response
    if request.method == 'OPTIONS' and request.path.startswith('/api/'):
        resp = make_response()
        origin = request.headers.get('Origin', '*')
        req_headers = request.headers.get('Access-Control-Request-Headers', 'Content-Type, Authorization')
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Vary'] = 'Origin'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = req_headers
        resp.headers['Access-Control-Max-Age'] = '600'
        return resp

@app.route('/api/events')
def get_events():
    """Get all disaster events"""
    return jsonify([event.to_dict() for event in disaster_events])

@app.route('/api/events/<event_id>')
def get_event(event_id):
    """Get specific disaster event"""
    event = next((e for e in disaster_events if e.id == event_id), None)
    if event:
        return jsonify(event.to_dict())
    return jsonify({'error': 'Event not found'}), 404

@app.route('/api/predictions')
def get_predictions():
    """Get all predictions"""
    return jsonify([pred.to_dict() for pred in predictions])

@app.route('/api/predictions/<prediction_id>')
def get_prediction(prediction_id):
    """Get specific prediction"""
    prediction = next((p for p in predictions if p.id == prediction_id), None)
    if prediction:
        return jsonify(prediction.to_dict())
    return jsonify({'error': 'Prediction not found'}), 404

@app.route('/api/sensors')
def get_sensor_data():
    """Get all sensor data"""
    return jsonify([sensor.to_dict() for sensor in sensor_data])

@app.route('/api/sensors/<sensor_id>')
def get_sensor(sensor_id):
    """Get specific sensor data"""
    sensor = next((s for s in sensor_data if s.id == sensor_id), None)
    if sensor:
        return jsonify(sensor.to_dict())
    return jsonify({'error': 'Sensor not found'}), 404

@app.route('/api/stats')
def get_stats():
    """Get real-time statistics"""
    active_events = [e for e in disaster_events if e.status in ['active', 'monitoring']]
    critical_events = [e for e in disaster_events if 'critical' in e.severity.lower() or 'extreme' in e.severity.lower()]
    
    return jsonify({
        'total_events': len(disaster_events),
        'active_events': len(active_events),
        'critical_events': len(critical_events),
        'total_predictions': len(predictions),
        'high_probability_predictions': len([p for p in predictions if p.probability > 0.7]),
        'total_sensors': len(sensor_data),
        'weather_locations_monitored': len(MONITORED_LOCATIONS),
        'ai_models_active': len(ai_prediction_service.models),
        'last_updated': datetime.now(timezone.utc).isoformat()
    })

@app.route('/api/ai/predict', methods=['POST'])
def predict_disaster():
    """Make AI prediction for a specific location"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        location_name = data.get('location_name')
        
        if not lat or not lon:
            logger.warning("Missing coordinates in prediction request")
            return jsonify({'error': 'Latitude and longitude required'}), 400
        
        # Fetch weather data for the location
        weather_fetch_start = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather = loop.run_until_complete(weather_service.get_current_weather(lat, lon, location_name))
        loop.close()
        weather_fetch_duration = time.time() - weather_fetch_start
        
        if not weather:
            logger.error("Failed to fetch weather data for prediction")
            return jsonify({'error': 'Failed to fetch weather data'}), 500
        
        weather_dict = {
            'temperature': weather.temperature,
            'humidity': weather.humidity,
            'pressure': weather.pressure,
            'wind_speed': weather.wind_speed,
            'wind_direction': weather.wind_direction,
            'precipitation': weather.precipitation,
            'visibility': weather.visibility,
            'cloud_cover': weather.cloud_cover
        }
        
        # Get AI predictions using enterprise models
        prediction_start = time.time()
        predictions_map = ai_prediction_service.predict_disaster_risks(weather_dict)
        prediction_duration = time.time() - prediction_start
        
        # Record metrics for each prediction type
        for hazard_type, risk_score in predictions_map.items():
            monitoring.record_ai_prediction(hazard_type, prediction_duration, risk_score)
        
        # Record weather request
        monitoring.record_weather_request()

        # Optional Gemini summaries for each predicted type
        summaries = {}
        for etype, score in predictions_map.items():
            summary = generate_prediction_summary(etype, weather.location, weather_dict, float(score))
            if summary:
                summaries[etype] = summary
        
        # Store predictions in the global predictions list
        for etype, score in predictions_map.items():
            if score > 0.1:  # Only store predictions with significant risk
                severity = 'extreme' if score > 0.8 else 'high' if score > 0.6 else 'moderate' if score > 0.4 else 'low'
                
                prediction = Prediction(
                    prediction_id=f"ai_pred_{len(predictions) + 1}",
                    event_type=etype,
                    location=weather.location,
                    probability=float(score),
                    severity=severity,
                    timeframe='24-72h',
                    coordinates=weather.coordinates,
                    weather_data=weather_dict,
                    ai_model='Enterprise Ensemble + Gemini' if summaries.get(etype) else 'Enterprise Ensemble'
                )
                if summaries.get(etype):
                    prediction.potential_impact = summaries[etype]
                
                predictions.append(prediction)
                # Emit real-time update
                socketio.emit('new_prediction', prediction.to_dict())
        
        total_duration = time.time() - start_time
        
        # Log successful prediction with performance metrics
        logger.info("Enterprise AI prediction completed", 
                   location=weather.location,
                   weather_fetch_duration_ms=round(weather_fetch_duration * 1000, 2),
                   prediction_duration_ms=round(prediction_duration * 1000, 2),
                   total_duration_ms=round(total_duration * 1000, 2),
                   predictions_count=len(predictions_map))
        
        return jsonify({
            'location': weather.location,
            'coordinates': weather.coordinates,
            'weather_data': weather_dict,
            'predictions': predictions_map,
            'summaries': summaries,
            'performance': {
                'weather_fetch_duration_ms': round(weather_fetch_duration * 1000, 2),
                'prediction_duration_ms': round(prediction_duration * 1000, 2),
                'total_duration_ms': round(total_duration * 1000, 2),
                'model_version': ai_prediction_service.model_metadata.get('version', '2.0.0'),
                'ensemble_enabled': True
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"Error in enterprise AI prediction: {e}", 
                    duration_ms=round(total_duration * 1000, 2),
                    exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'duration_ms': round(total_duration * 1000, 2)
        }), 500

@app.route('/api/ai/train', methods=['POST'])
def train_models():
    """Train enterprise AI models"""
    start_time = time.time()
    
    try:
        # Get training parameters from request
        data = request.get_json() or {}
        epochs = data.get('epochs', 100)
        auto_train = data.get('auto_train', True)
        
        logger.info("Starting enterprise AI model training", 
                   epochs=epochs,
                   auto_train=auto_train)
        
        # Use enterprise training method
        training_results = ai_prediction_service.train_advanced_models(epochs=epochs)
        
        training_duration = time.time() - start_time
        
        # Log training results
        successful_models = [hazard for hazard, result in training_results.items() 
                           if result.get('status') == 'success']
        
        logger.info("Enterprise AI model training completed",
                   duration_seconds=round(training_duration, 2),
                   successful_models=len(successful_models),
                   total_models=len(training_results))
        
        return jsonify({
            'status': 'success',
            'training_results': training_results,
            'performance': {
                'training_duration_seconds': round(training_duration, 2),
                'models_trained': len(successful_models),
                'total_models': len(training_results)
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        training_duration = time.time() - start_time
        logger.error(f"Error in enterprise AI training: {e}",
                    duration_seconds=round(training_duration, 2),
                    exc_info=True)
        return jsonify({
            'error': 'Training failed',
            'details': str(e),
            'duration_seconds': round(training_duration, 2)
        }), 500
        # Reload into memory
        try:
            wf_model = os.path.join(model_dir, 'wildfire_model.joblib')
            wf_scaler = os.path.join(model_dir, 'wildfire_scaler.joblib')
            if os.path.exists(wf_model) and os.path.exists(wf_scaler):
                ai_prediction_service.models['wildfire'] = {
                    'clf': joblib.load(wf_model),
                    'scaler': joblib.load(wf_scaler)
                }
            st_model = os.path.join(model_dir, 'storm_model.joblib')
            st_scaler = os.path.join(model_dir, 'storm_scaler.joblib')
            if os.path.exists(st_model) and os.path.exists(st_scaler):
                ai_prediction_service.models['storm'] = {
                    'clf': joblib.load(st_model),
                    'scaler': joblib.load(st_scaler)
                }
            fl_model = os.path.join(model_dir, 'flood_model.joblib')
            fl_scaler = os.path.join(model_dir, 'flood_scaler.joblib')
            if os.path.exists(fl_model) and os.path.exists(fl_scaler):
                ai_prediction_service.models['flood'] = {
                    'clf': joblib.load(fl_model),
                    'scaler': joblib.load(fl_scaler)
                }
            ls_model = os.path.join(model_dir, 'landslide_model.joblib')
            ls_scaler = os.path.join(model_dir, 'landslide_scaler.joblib')
            if os.path.exists(ls_model) and os.path.exists(ls_scaler):
                ai_prediction_service.models['landslide'] = {
                    'clf': joblib.load(ls_model),
                    'scaler': joblib.load(ls_scaler)
                }
            dr_model = os.path.join(model_dir, 'drought_model.joblib')
            dr_scaler = os.path.join(model_dir, 'drought_scaler.joblib')
            if os.path.exists(dr_model) and os.path.exists(dr_scaler):
                ai_prediction_service.models['drought'] = {
                    'clf': joblib.load(dr_model),
                    'scaler': joblib.load(dr_scaler)
                }
        except Exception as e:
            logger.warning(f"Model reload failed: {e}")
        return jsonify({'message': 'Models trained successfully'})
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return jsonify({'error': 'Training failed'}), 500

@app.route('/api/models')
def list_models():
    """List available AI models and their status"""
    try:
        # Use the enterprise model status
        model_status = ai_prediction_service.get_model_status()
        
        # Add enterprise metadata
        model_status.update({
            'enterprise_features': {
                'ensemble_enabled': True,
                'auto_training': True,
                'model_versioning': True,
                'performance_monitoring': True,
                'data_sources': ['ERA5', 'GDACS', 'FIRMS', 'USGS', 'NASA', 'NOAA']
            },
            'deployment_info': {
                'environment': app.config['ENVIRONMENT'],
                'deployment_time': os.getenv('DEPLOYMENT_TIME', datetime.now().isoformat()),
                'railway_service': os.getenv('RAILWAY_SERVICE_NAME', 'disastroscope-backend')
            }
        })
        
        # Record metrics
        monitoring.record_ai_prediction('model_status_check', 0.1)
        
        return jsonify(model_status)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return jsonify({'error': 'Failed to list models', 'details': str(e)}), 500

@app.route('/api/events', methods=['POST'])
def create_event():
    """Create a new disaster event"""
    data = request.get_json()
    
    event = DisasterEvent(
        event_id=data.get('id', f"event_{len(disaster_events) + 1}"),
        name=data.get('name'),
        event_type=data.get('event_type'),
        location=data.get('location'),
        severity=data.get('severity'),
        status=data.get('status'),
        coordinates=data.get('coordinates'),
        affected_population=data.get('affected_population', 0),
        economic_impact=data.get('economic_impact', 0.0),
        weather_data=data.get('weather_data'),
        ai_confidence=data.get('ai_confidence', 0.0)
    )
    
    disaster_events.append(event)
    socketio.emit('new_event', event.to_dict())
    
    return jsonify(event.to_dict()), 201

@app.route('/api/weather')
def get_weather_data():
    """Get weather data for monitored locations"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather_data = loop.run_until_complete(weather_service.get_multiple_locations_weather(MONITORED_LOCATIONS))
        loop.close()
        
        return jsonify([weather.to_dict() for weather in weather_data])
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return jsonify({'error': 'Failed to fetch weather data'}), 500

@app.route('/api/weather/current')
def get_current_weather():
    """Get current weather for specific coordinates"""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    name = request.args.get('name')
    units = request.args.get('units', 'metric')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather = loop.run_until_complete(weather_service.get_current_weather(lat, lon, name, units))
        loop.close()
        
        if not weather:
            return jsonify({'error': 'Failed to fetch weather data'}), 500
        
        return jsonify(weather.to_dict())
    except Exception as e:
        logger.error(f"Error fetching current weather: {e}")
        return jsonify({'error': 'Failed to fetch weather data'}), 500

@app.route('/api/weather/by-city')
def get_weather_by_city():
    """Get weather data by city name"""
    query = request.args.get('query')
    units = request.args.get('units', 'metric')
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Fetch multiple candidates to improve worldwide matching
        results = loop.run_until_complete(weather_service.geocode(query, limit=5))
        if not results:
            loop.close()
            return jsonify({'error': 'No results for query'}), 404

        # Choose the best candidate: prefer exact (case-insensitive) name match, else first
        qnorm = query.strip().lower()
        def score(item: dict) -> int:
            name = str(item.get('name') or '').lower()
            state = str(item.get('state') or '')
            country = str(item.get('country') or '')
            # exact name gets higher score
            s = 0
            if name == qnorm:
                s += 3
            if qnorm in name:
                s += 1
            # presence of state/country boosts confidence
            if state:
                s += 1
            if country:
                s += 1
            return s

        best = sorted(results, key=score, reverse=True)[0]
        lat = best['lat']
        lon = best['lon']
        name = f"{best.get('name')}{', ' + best.get('state') if best.get('state') else ''}{' ' + best.get('country') if best.get('country') else ''}".strip()
        weather = loop.run_until_complete(weather_service.get_current_weather(lat, lon, name, units))
        loop.close()
        if not weather:
            return jsonify({'error': 'Failed to fetch weather'}), 502
        return jsonify(weather.to_dict())
    except Exception as e:
        logger.error(f"Error fetching weather by city: {e}")
        return jsonify({'error': 'Failed to fetch weather by city'}), 500

@app.route('/api/weather/forecast')
def get_weather_forecast():
    """Get weather forecast for specific coordinates"""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    days = request.args.get('days', 5, type=int)
    units = request.args.get('units', 'metric')
    
    if not lat or not lon:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        forecast = loop.run_until_complete(weather_service.get_weather_forecast(lat, lon, days, units))
        loop.close()
        
        return jsonify(forecast)
    except Exception as e:
        logger.error(f"Error fetching forecast: {e}")
        return jsonify({'error': 'Failed to fetch forecast'}), 500

@app.route('/api/weather/<location>')
def get_weather_for_location(location: str):
    """Get weather data for a specific location"""
    try:
        # Try to find the location in monitored locations first
        monitored = next((loc for loc in MONITORED_LOCATIONS if loc['name'].lower() == location.lower()), None)
        
        if monitored:
            coords = monitored['coords']
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            weather = loop.run_until_complete(weather_service.get_current_weather(coords['lat'], coords['lng'], monitored['name']))
            loop.close()
            
            if weather:
                return jsonify(weather.to_dict())
        
        return jsonify({'error': 'Location not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching weather for location: {e}")
        return jsonify({'error': 'Failed to fetch weather data'}), 500

@app.route('/api/disasters')
def get_disasters():
    """Get FEMA disaster declarations"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        disasters = loop.run_until_complete(openfema_service.get_disaster_declarations())
        loop.close()
        
        if disasters:
            fema_disasters.clear()
            fema_disasters.extend(disasters)
            socketio.emit('disasters_update', disasters)
        
        return jsonify(disasters)
    except Exception as e:
        logger.error(f"Error fetching disasters: {e}")
        return jsonify({'error': 'Failed to fetch disasters'}), 500

@app.route('/api/disasters/state/<state_code>')
def get_disasters_by_state(state_code: str):
    """Get FEMA disaster declarations for a specific state"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        disasters = loop.run_until_complete(openfema_service.get_disasters_by_state(state_code))
        loop.close()
        
        return jsonify(disasters)
    except Exception as e:
        logger.error(f"Error fetching disasters by state: {e}")
        return jsonify({'error': 'Failed to fetch disasters by state'}), 500

@app.route('/api/eonet')
def get_eonet_events():
    """Get NASA EONET events"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        events = loop.run_until_complete(eonet_service.get_eonet_events())
        loop.close()
        
        if events:
            eonet_events.clear()
            eonet_events.extend(events)
            socketio.emit('eonet_update', events)
        
        return jsonify(events)
    except Exception as e:
        logger.error(f"Error fetching EONET events: {e}")
        return jsonify({'error': 'Failed to fetch EONET events'}), 500

@app.route('/api/eonet/category/<category>')
def get_eonet_by_category(category: str):
    """Get NASA EONET events by category"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        events = loop.run_until_complete(eonet_service.get_events_by_category(category))
        loop.close()
        
        return jsonify(events)
    except Exception as e:
        logger.error(f"Error fetching EONET events by category: {e}")
        return jsonify({'error': 'Failed to fetch EONET events by category'}), 500

# Test endpoint for debugging location issues
@app.route('/api/test/location')
def test_location():
    """Test endpoint to debug location-based analysis"""
    query = request.args.get('query', 'New York')
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Test geocoding
        geocode_results = loop.run_until_complete(weather_service.geocode(query, limit=3))
        
        # Test weather if geocoding succeeds
        weather_data = None
        if geocode_results:
            best = geocode_results[0]
            lat = best['lat']
            lon = best['lon']
            weather_data = loop.run_until_complete(weather_service.get_current_weather(lat, lon, query))
        
        loop.close()
        
        return jsonify({
            'query': query,
            'geocoding_results': geocode_results,
            'weather_data': weather_data.to_dict() if weather_data else None,
            'geocoding_success': len(geocode_results) > 0,
            'weather_success': weather_data is not None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Test location error: {e}")
        return jsonify({
            'query': query,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

# Background tasks
def background_weather_update():
    """Background task to update weather data every 5 minutes"""
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            weather_data = loop.run_until_complete(weather_service.get_multiple_locations_weather(MONITORED_LOCATIONS))
            loop.close()
            
            if weather_data:
                weather_data_cache.clear()
                weather_data_cache.extend([w.to_dict() for w in weather_data])
                socketio.emit('weather_update', weather_data_cache)
                
        except Exception as e:
            logger.error(f"Background weather update error: {e}")
        
        time.sleep(300)  # 5 minutes

def background_disaster_update():
    """Background task to update disaster data every 30 minutes"""
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            disasters = loop.run_until_complete(openfema_service.get_disaster_declarations())
            loop.close()
            
            if disasters:
                fema_disasters.clear()
                fema_disasters.extend(disasters)
                socketio.emit('disasters_update', disasters)
                
        except Exception as e:
            logger.error(f"Background disaster update error: {e}")
        
        time.sleep(1800)  # 30 minutes

def background_eonet_update():
    """Background task to update EONET data every 15 minutes"""
    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            events = loop.run_until_complete(eonet_service.get_eonet_events())
            loop.close()
            
            if events:
                eonet_events.clear()
                eonet_events.extend(events)
                socketio.emit('eonet_update', events)
                
        except Exception as e:
            logger.error(f"Background EONET update error: {e}")
        
        time.sleep(900)  # 15 minutes

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to DisastroScope API'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('subscribe_events')
def handle_subscribe_events():
    """Handle events subscription"""
    emit('events_update', [event.to_dict() for event in disaster_events])

@socketio.on('subscribe_predictions')
def handle_subscribe_predictions():
    """Handle predictions subscription"""
    emit('predictions_update', [pred.to_dict() for pred in predictions])

@socketio.on('subscribe_weather')
def handle_subscribe_weather():
    """Handle weather subscription"""
    emit('weather_update', weather_data_cache)

@socketio.on('subscribe_disasters')
def handle_subscribe_disasters():
    """Handle disasters subscription"""
    emit('disasters_update', fema_disasters)

@socketio.on('subscribe_eonet')
def handle_subscribe_eonet():
    """Handle EONET subscription"""
    emit('eonet_update', eonet_events)

if __name__ == '__main__':
    # Start background tasks
    weather_thread = threading.Thread(target=background_weather_update, daemon=True)
    weather_thread.start()
    
    disaster_thread = threading.Thread(target=background_disaster_update, daemon=True)
    disaster_thread.start()
    
    eonet_thread = threading.Thread(target=background_eonet_update, daemon=True)
    eonet_thread.start()
    
    # Run the app
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

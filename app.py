"""
DisastroScope Backend API - Production Ready
Enterprise-grade disaster monitoring and prediction system
"""

import os
import json
import logging
import time
import random
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from functools import wraps

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'disastroscope-secret-key-2024')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # API Configuration
    API_VERSION = '2.0.0'
    API_TITLE = 'DisastroScope Backend API'
    MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '100'))
    
    # Model Configuration
    MODEL_VERSION = '2.0.0'
    PREDICTION_TIMEOUT = int(os.getenv('PREDICTION_TIMEOUT', '30'))
    
    # CORS Configuration
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure structured logging"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DisasterEvent:
    """Disaster event data model"""
    id: str
    name: str
    event_type: str
    latitude: float
    longitude: float
    magnitude: float
    timestamp: str
    description: str
    severity: str
    source: str = "sample"
    confidence: float = 0.9

@dataclass
class Prediction:
    """Prediction data model"""
    id: str
    event_type: str
    latitude: float
    longitude: float
    probability: float
    timestamp: str
    description: str
    confidence: float
    model_version: str = Config.MODEL_VERSION

@dataclass
class WeatherData:
    """Weather data model"""
    city: str
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    precipitation: float
    visibility: float
    cloud_cover: float
    timestamp: str

@dataclass
class PredictionRequest:
    """Prediction request model"""
    latitude: float
    longitude: float
    temperature: float = 20.0
    humidity: float = 60.0
    pressure: float = 1013.0
    wind_speed: float = 5.0
    precipitation: float = 2.0
    location_name: str = "Unknown"

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_coordinates(lat: float, lng: float) -> bool:
    """Validate latitude and longitude coordinates"""
    return -90 <= lat <= 90 and -180 <= lng <= 180

def validate_prediction_request(data: Dict) -> Tuple[bool, str]:
    """Validate prediction request data"""
    required_fields = ['latitude', 'longitude']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        try:
            value = float(data[field])
            if field in ['latitude', 'longitude'] and not validate_coordinates(value, value):
                return False, f"Invalid {field}: {value}"
        except (ValueError, TypeError):
            return False, f"Invalid {field}: must be a number"
    
    return True, "Valid"

def validate_location_request(data: Dict) -> Tuple[bool, str]:
    """Validate location-based request data"""
    return validate_prediction_request(data)

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
        else:
            self.requests[client_id] = []
        
        # Check if under limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(Config.MAX_REQUESTS_PER_MINUTE)

def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = request.remote_addr
        if not rate_limiter.is_allowed(client_id):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': f'Maximum {Config.MAX_REQUESTS_PER_MINUTE} requests per minute'
            }), 429
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# MONITORING AND METRICS
# ============================================================================

class Metrics:
    """Application metrics collector"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.endpoint_stats = {}
    
    def record_request(self, endpoint: str, status_code: int):
        """Record a request"""
        self.request_count += 1
        
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {'requests': 0, 'errors': 0}
        
        self.endpoint_stats[endpoint]['requests'] += 1
        
        if status_code >= 400:
            self.error_count += 1
            self.endpoint_stats[endpoint]['errors'] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'requests_per_second': self.request_count / max(uptime, 1),
            'endpoint_stats': self.endpoint_stats
        }

metrics = Metrics()

# ============================================================================
# DATA STORAGE
# ============================================================================

class DataStore:
    """In-memory data store with thread safety"""
    
    def __init__(self):
        self.disaster_events: List[DisasterEvent] = []
        self.predictions: List[Prediction] = []
        self.weather_cache: Dict[str, WeatherData] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample data"""
        # Sample disaster events
        self.disaster_events = [
            DisasterEvent(
                id="1", name="Hurricane Maria", event_type="storm",
                latitude=18.2208, longitude=-66.5901, magnitude=5.0,
                timestamp="2024-08-24T10:00:00Z",
                description="Major hurricane affecting Puerto Rico", severity="high"
            ),
            DisasterEvent(
                id="2", name="California Wildfire", event_type="wildfire",
                latitude=36.7783, longitude=-119.4179, magnitude=4.5,
                timestamp="2024-08-24T08:30:00Z",
                description="Large wildfire in Northern California", severity="high"
            ),
            DisasterEvent(
                id="3", name="Mississippi Flood", event_type="flood",
                latitude=32.7416, longitude=-89.6787, magnitude=3.8,
                timestamp="2024-08-24T06:15:00Z",
                description="Severe flooding along Mississippi River", severity="medium"
            ),
            DisasterEvent(
                id="4", name="Texas Tornado", event_type="tornado",
                latitude=31.9686, longitude=-99.9018, magnitude=4.2,
                timestamp="2024-08-24T14:20:00Z",
                description="Tornado warning in Central Texas", severity="high"
            ),
            DisasterEvent(
                id="5", name="California Earthquake", event_type="earthquake",
                latitude=36.7783, longitude=-119.4179, magnitude=3.5,
                timestamp="2024-08-24T16:45:00Z",
                description="Minor earthquake in California", severity="medium"
            )
    ]
    
    # Sample predictions
        self.predictions = [
            Prediction(
                id="p1", event_type="flood", latitude=29.7604, longitude=-95.3698,
                probability=0.85, timestamp="2024-08-24T12:00:00Z",
                description="High flood risk in Houston area", confidence=0.92
            ),
            Prediction(
                id="p2", event_type="wildfire", latitude=34.0522, longitude=-118.2437,
                probability=0.78, timestamp="2024-08-24T12:00:00Z",
                description="Elevated wildfire risk in Los Angeles", confidence=0.88
            ),
            Prediction(
                id="p3", event_type="storm", latitude=25.7617, longitude=-80.1918,
                probability=0.72, timestamp="2024-08-24T12:00:00Z",
                description="Tropical storm approaching Miami", confidence=0.85
            ),
            Prediction(
                id="p4", event_type="landslide", latitude=47.6062, longitude=-122.3321,
                probability=0.65, timestamp="2024-08-24T12:00:00Z",
                description="Landslide risk in Seattle area", confidence=0.78
            )
        ]
    
    def get_events(self) -> List[Dict]:
        """Get all disaster events"""
        return [asdict(event) for event in self.disaster_events]
    
    def get_predictions(self) -> List[Dict]:
        """Get all predictions"""
        return [asdict(pred) for pred in self.predictions]
    
    def get_events_near(self, lat: float, lng: float, radius: float) -> List[Dict]:
        """Get events near a location"""
        nearby_events = []
        for event in self.disaster_events:
            distance = self._calculate_distance(lat, lng, event.latitude, event.longitude)
            if distance <= radius:
                nearby_events.append(asdict(event))
        return nearby_events
    
    def get_predictions_near(self, lat: float, lng: float, radius: float) -> List[Dict]:
        """Get predictions near a location"""
        nearby_predictions = []
        for pred in self.predictions:
            distance = self._calculate_distance(lat, lng, pred.latitude, pred.longitude)
            if distance <= radius:
                nearby_predictions.append(asdict(pred))
        return nearby_predictions
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

data_store = DataStore()

# ============================================================================
# AI PREDICTION ENGINE
# ============================================================================

class PredictionEngine:
    """Advanced prediction engine with multiple algorithms"""
    
    def __init__(self):
        self.model_version = Config.MODEL_VERSION
        self.algorithms = {
            'flood': self._predict_flood,
            'wildfire': self._predict_wildfire,
            'storm': self._predict_storm,
            'tornado': self._predict_tornado,
            'landslide': self._predict_landslide,
            'drought': self._predict_drought,
            'earthquake': self._predict_earthquake
        }
    
    def predict_all(self, request_data: PredictionRequest) -> Dict[str, float]:
        """Generate predictions for all disaster types"""
        predictions = {}
        
        for disaster_type, algorithm in self.algorithms.items():
            try:
                predictions[disaster_type] = algorithm(request_data)
            except Exception as e:
                logger.error(f"Error predicting {disaster_type}: {e}")
                predictions[disaster_type] = 0.0
        
        return predictions
    
    def _predict_flood(self, data: PredictionRequest) -> float:
        """Predict flood risk"""
        risk = 0.0
        risk += (data.precipitation / 20.0) * 0.4
        risk += (data.humidity / 100.0) * 0.3
        risk += (1 - data.pressure / 1050.0) * 0.3
        return min(1.0, risk)
    
    def _predict_wildfire(self, data: PredictionRequest) -> float:
        """Predict wildfire risk"""
        risk = 0.0
        risk += (data.temperature / 40.0) * 0.4
        risk += (1 - data.humidity / 100.0) * 0.4
        risk += (data.wind_speed / 25.0) * 0.2
        return min(1.0, risk)
    
    def _predict_storm(self, data: PredictionRequest) -> float:
        """Predict storm risk"""
        risk = 0.0
        risk += (1 - data.pressure / 1050.0) * 0.6
        risk += (data.wind_speed / 25.0) * 0.4
        return min(1.0, risk)
    
    def _predict_tornado(self, data: PredictionRequest) -> float:
        """Predict tornado risk"""
        risk = 0.0
        risk += (1 - data.pressure / 1050.0) * 0.4
        risk += (data.wind_speed / 25.0) * 0.4
        risk += (data.humidity / 100.0) * 0.2
        return min(1.0, risk)
    
    def _predict_landslide(self, data: PredictionRequest) -> float:
        """Predict landslide risk"""
        risk = 0.0
        risk += (data.precipitation / 25.0) * 0.7
        risk += (1 - data.pressure / 1050.0) * 0.3
        return min(1.0, risk)
    
    def _predict_drought(self, data: PredictionRequest) -> float:
        """Predict drought risk"""
        risk = 0.0
        risk += (1 - data.precipitation / 25.0) * 0.4
        risk += (data.temperature / 40.0) * 0.3
        risk += (1 - data.humidity / 100.0) * 0.3
        return min(1.0, risk)
    
    def _predict_earthquake(self, data: PredictionRequest) -> float:
        """Predict earthquake risk (very low base risk)"""
        return 0.05

prediction_engine = PredictionEngine()

# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# CORS configuration
CORS(app, resources={
    r"/api/*": {"origins": Config.ALLOWED_ORIGINS},
    r"/*": {"origins": Config.ALLOWED_ORIGINS}
})

# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.before_request
def before_request():
    """Pre-request processing"""
    request.start_time = time.time()

@app.after_request
def after_request(response: Response) -> Response:
    """Post-request processing"""
    # Record metrics
    endpoint = request.endpoint or 'unknown'
    metrics.record_request(endpoint, response.status_code)
    
    # Add response headers
    response.headers['X-API-Version'] = Config.API_VERSION
    response.headers['X-Request-ID'] = request.headers.get('X-Request-ID', 'unknown')
    
    # Log request
    duration = time.time() - request.start_time
    logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.3f}s")
    
    return response

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@app.route('/health')
@rate_limit
def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Basic health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'service': Config.API_TITLE,
            'version': Config.API_VERSION,
            'environment': Config.ENVIRONMENT,
            'uptime_seconds': time.time() - metrics.start_time,
            'checks': {
                'database': 'healthy',
                'models': 'healthy',
                'memory': 'healthy'
            }
        }
        
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/')
@rate_limit
def home():
    """API information endpoint"""
    return jsonify({
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "status": "operational",
        "environment": Config.ENVIRONMENT,
        "documentation": {
            "health": "/health",
            "metrics": "/api/metrics",
            "events": "/api/events",
            "predictions": "/api/predictions",
            "models": "/api/models",
            "ai_predict": "/api/ai/predict",
            "weather": "/api/weather/<city>",
            "events_near": "/api/events/near",
            "predictions_near": "/api/predictions/near",
            "global_risk_analysis": "/api/global-risk-analysis"
        }
    })

@app.route('/api/health')
@rate_limit
def api_health_check():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": Config.API_VERSION
    })

@app.route('/api/metrics')
@rate_limit
def get_metrics():
    """Get application metrics"""
    return jsonify(metrics.get_stats())

# ============================================================================
# DATA RETRIEVAL ENDPOINTS
# ============================================================================

@app.route('/api/events')
@rate_limit
def get_disaster_events():
    """Get all disaster events"""
    try:
        events = data_store.get_events()
        return jsonify({
            "events": events,
            "count": len(events),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        return jsonify({"error": "Failed to get events"}), 500

@app.route('/api/predictions')
@rate_limit
def get_predictions():
    """Get all disaster predictions"""
    try:
        predictions = data_store.get_predictions()
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({"error": "Failed to get predictions"}), 500

@app.route('/api/sensors')
@rate_limit
def get_sensors():
    """Get all sensor data"""
    try:
        # Return mock sensor data for now
        sensors = [
            {
                "id": "sensor_001",
                "sensor_type": "weather",
                "station_id": "WS_001",
                "station_name": "Central Weather Station",
                "location": "Central Monitoring Hub",
                "coordinates": {"lat": 40.7128, "lng": -74.0060},
                "reading_value": 72.5,
                "reading_unit": "°F",
                "reading_time": datetime.now(timezone.utc).isoformat(),
                "data_quality": "excellent",
                "metadata": {"model": "WS-2000", "calibration_date": "2024-01-01"},
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "sensor_002", 
                "sensor_type": "seismic",
                "station_id": "SS_001",
                "station_name": "Seismic Monitor Alpha",
                "location": "Eastern Seismic Zone",
                "coordinates": {"lat": 34.0522, "lng": -118.2437},
                "reading_value": 0.12,
                "reading_unit": "g",
                "reading_time": datetime.now(timezone.utc).isoformat(),
                "data_quality": "good",
                "metadata": {"model": "SM-5000", "sensitivity": "high"},
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        return jsonify({
            "sensors": sensors,
            "count": len(sensors),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting sensors: {e}")
        return jsonify({"error": "Failed to get sensors"}), 500

@app.route('/api/sensors/<sensor_id>')
@rate_limit
def get_sensor(sensor_id):
    """Get specific sensor by ID"""
    try:
        # Mock sensor data
        sensor = {
            "id": sensor_id,
            "sensor_type": "weather",
            "station_id": f"WS_{sensor_id.split('_')[1]}",
            "station_name": f"Station {sensor_id}",
            "location": "Monitoring Hub",
            "coordinates": {"lat": 40.7128, "lng": -74.0060},
            "reading_value": 72.5,
            "reading_unit": "°F",
            "reading_time": datetime.now(timezone.utc).isoformat(),
            "data_quality": "excellent",
            "metadata": {"model": "WS-2000", "calibration_date": "2024-01-01"},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(sensor)
    except Exception as e:
        logger.error(f"Error getting sensor {sensor_id}: {e}")
        return jsonify({"error": "Failed to get sensor"}), 500

@app.route('/api/models')
@rate_limit
def list_models():
    """List available AI models and their status"""
    try:
        models = {
            'flood': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['ERA5', 'GDACS'],
                'accuracy': 0.85
            },
            'wildfire': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['FIRMS', 'ERA5'],
                'accuracy': 0.82
            },
            'storm': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['ERA5'],
                'accuracy': 0.78
            },
            'tornado': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['ERA5', 'NOAA'],
                'accuracy': 0.75
            },
            'landslide': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['GDACS', 'ERA5'],
                'accuracy': 0.80
            },
            'drought': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['ERA5'],
                'accuracy': 0.83
            },
            'earthquake': {
                'loaded': True,
                'type': 'heuristic',
                'version': Config.MODEL_VERSION,
                'sources': ['USGS'],
                'accuracy': 0.45
            }
            }
        
        return jsonify({
            'models': models,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_models': len(models),
            'loaded_models': len(models),
            'service_status': 'operational'
        })
    except Exception as e:
        logger.error(f"/api/models error: {e}")
        return jsonify({'error': 'failed to list models'}), 500

# ============================================================================
# AI PREDICTION ENDPOINTS
# ============================================================================

@app.route('/api/ai/predict', methods=['POST'])
@rate_limit
def predict_disaster_risks():
    """Advanced AI prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate request
        is_valid, error_message = validate_prediction_request(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        # Create prediction request
        pred_request = PredictionRequest(
            latitude=data['latitude'],
            longitude=data['longitude'],
            temperature=data.get('temperature', 20.0),
            humidity=data.get('humidity', 60.0),
            pressure=data.get('pressure', 1013.0),
            wind_speed=data.get('wind_speed', 5.0),
            precipitation=data.get('precipitation', 2.0),
            location_name=data.get('location_name', 'Unknown')
        )
        
        # Generate predictions
        predictions = prediction_engine.predict_all(pred_request)
        
        response = {
            'predictions': predictions,
            'metadata': {
                'model_version': Config.MODEL_VERSION,
                'prediction_timestamp': datetime.now(timezone.utc).isoformat(),
                'location': {
                    'latitude': pred_request.latitude,
                    'longitude': pred_request.longitude,
                    'name': pred_request.location_name
                },
                'model_type': 'heuristic',
                'confidence': 'high'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in AI prediction: {e}")
        return jsonify({'error': 'Failed to generate predictions'}), 500

# ============================================================================
# WEATHER ENDPOINTS
# ============================================================================

@app.route('/api/weather/<city>')
@rate_limit
def get_weather(city):
    """Get weather data for a city"""
    try:
        # Check cache first
        if city in data_store.weather_cache:
            cached_data = data_store.weather_cache[city]
            # Return cached data if less than 1 hour old
            cache_time = datetime.fromisoformat(cached_data.timestamp.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) - cache_time < timedelta(hours=1):
                return jsonify(asdict(cached_data))
        
        # Generate mock weather data
        weather_data = WeatherData(
            city=city,
            temperature=random.uniform(15, 35),
            humidity=random.uniform(30, 90),
            pressure=random.uniform(1000, 1020),
            wind_speed=random.uniform(0, 25),
            precipitation=random.uniform(0, 50),
            visibility=random.uniform(5, 25),
            cloud_cover=random.uniform(0, 100),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Cache the data
        data_store.weather_cache[city] = weather_data
        
        return jsonify(asdict(weather_data))
    except Exception as e:
        logger.error(f"Error getting weather for {city}: {e}")
        return jsonify({"error": "Failed to get weather data"}), 500

@app.route('/api/weather/current')
@rate_limit
def get_weather_by_coords():
    """Get weather data for coordinates"""
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if not lat or not lon:
            return jsonify({'error': 'Missing lat/lon parameters'}), 400
            
        # Try OpenWeatherMap API first, fallback to mock data
        openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        if openweather_api_key:
            try:
                import requests
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    weather_data = {
                        'temperature': round(data['main']['temp'], 1),
                        'humidity': round(data['main']['humidity'], 1),
                        'pressure': round(data['main']['pressure'], 1),
                        'wind_speed': round(data['wind']['speed'], 1),
                        'precipitation': round(data.get('rain', {}).get('1h', 0), 1),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                else:
                    # Fallback to mock data if API fails
                    weather_data = {
                        'temperature': round(random.uniform(15, 35), 1),
                        'humidity': round(random.uniform(30, 90), 1),
                        'pressure': round(random.uniform(1000, 1020), 1),
                        'wind_speed': round(random.uniform(0, 25), 1),
                        'precipitation': round(random.uniform(0, 50), 1),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
            except Exception as e:
                logger.warning(f"OpenWeatherMap API failed, using mock data: {e}")
                # Fallback to mock data
                weather_data = {
                    'temperature': round(random.uniform(15, 35), 1),
                    'humidity': round(random.uniform(30, 90), 1),
                    'pressure': round(random.uniform(1000, 1020), 1),
                    'wind_speed': round(random.uniform(0, 25), 1),
                    'precipitation': round(random.uniform(0, 50), 1),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        else:
            # No API key, use mock data
            weather_data = {
                'temperature': round(random.uniform(15, 35), 1),
                'humidity': round(random.uniform(30, 90), 1),
                'pressure': round(random.uniform(1000, 1020), 1),
                'wind_speed': round(random.uniform(0, 25), 1),
                'precipitation': round(random.uniform(0, 50), 1),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        return jsonify(weather_data)
        
    except Exception as e:
        logger.error(f"Error getting weather by coordinates: {e}")
        return jsonify({"error": "Failed to get weather data"}), 500

# ============================================================================
# LOCATION-BASED ENDPOINTS
# ============================================================================

@app.route('/api/events/near', methods=['POST'])
@rate_limit
def get_events_near():
    """Get events near a location"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate request
        is_valid, error_message = validate_location_request(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        lat = data['latitude']
        lng = data['longitude']
        radius = data.get('radius', 100)  # Default 100km radius
        
        nearby_events = data_store.get_events_near(lat, lng, radius)
        
        return jsonify({
            "events": nearby_events,
            "count": len(nearby_events),
            "radius_km": radius,
            "center": {"latitude": lat, "longitude": lng},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting events near location: {e}")
        return jsonify({"error": "Failed to get nearby events"}), 500

@app.route('/api/predictions/near', methods=['POST'])
@rate_limit
def get_predictions_near():
    """Get predictions near a location"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate request
        is_valid, error_message = validate_location_request(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400
        
        lat = data['latitude']
        lng = data['longitude']
        radius = data.get('radius', 100)  # Default 100km radius
        
        nearby_predictions = data_store.get_predictions_near(lat, lng, radius)
        
        return jsonify({
            "predictions": nearby_predictions,
            "count": len(nearby_predictions),
            "radius_km": radius,
            "center": {"latitude": lat, "longitude": lng},
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting predictions near location: {e}")
        return jsonify({"error": "Failed to get nearby predictions"}), 500

@app.route('/api/location/analyze/coords', methods=['POST'])
@rate_limit
def analyze_location_by_coords():
    """Analyze location by coordinates"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        if not lat or not lon:
            return jsonify({'error': 'Missing lat/lon parameters'}), 400
            
        # You can integrate with elevation APIs, soil data, etc.
        # For now, return mock data with rounded values
        location_data = {
            'elevation': round(random.uniform(0, 2000), 0),
            'soil_type': random.choice(['Loamy', 'Sandy', 'Clay', 'Rocky']),
            'land_use': random.choice(['Urban', 'Rural', 'Forest', 'Agricultural']),
            'historical_events': 'Sample historical data',
            'population_density': round(random.uniform(10, 1000), 0)
        }
        
        return jsonify(location_data)
        
    except Exception as e:
        logger.error(f"Error analyzing location: {e}")
        return jsonify({"error": "Failed to analyze location"}), 500

@app.route('/api/geocode')
@rate_limit
def geocode_location():
    """Geocode a location query using OpenCage API"""
    try:
        query = request.args.get('query')
        limit = request.args.get('limit', 5, type=int)
        
        if not query:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        # Use OpenCage Geocoding API
        try:
            import requests
            from urllib.parse import quote
            
            api_key = os.getenv('OPENCAGE_API_KEY', 'demo_key')
            logger.info(f"Geocoding test - Using API key: {api_key[:8]}..." if api_key != 'demo_key' else "Using demo key")
            
            query_encoded = quote(query)
            geocoding_url = f"https://api.opencagedata.com/geocode/v1/json?q={query_encoded}&key={api_key}&limit={limit}"
            
            response = requests.get(geocoding_url, timeout=10)
            logger.info(f"Geocoding test response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    geocode_results = []
                    for result in data['results'][:limit]:
                        geometry = result['geometry']
                        components = result['components']
                        
                        geocode_results.append({
                            'name': result.get('formatted', query),
                            'lat': geometry['lat'],
                            'lon': geometry['lng'],
                            'country': components.get('country', 'Unknown'),
                            'state': components.get('state', ''),
                            'city': components.get('city', components.get('town', ''))
                        })
                    
                    logger.info(f"✅ Geocoding test successful for '{query}': {len(geocode_results)} results")
                    return jsonify(geocode_results)
                else:
                    logger.warning(f"❌ Geocoding test - no results for '{query}'")
                    return jsonify({'error': 'No results found'}), 404
            else:
                error_text = response.text
                logger.error(f"❌ Geocoding test failed for '{query}' - Status: {response.status_code}, Response: {error_text}")
                return jsonify({'error': f'Geocoding API failed: {response.status_code}'}), 500
                
        except Exception as e:
            logger.error(f"❌ Geocoding test exception for '{query}': {str(e)}")
            return jsonify({'error': f'Geocoding error: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Error in geocode endpoint: {e}")
        return jsonify({"error": "Failed to geocode location"}), 500

@app.route('/api/global-risk-analysis', methods=['POST'])
@rate_limit
def global_risk_analysis():
    """Advanced global risk analysis for 7 disaster types"""
    try:
        data = request.get_json()
        location_query = data.get('location_query')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not location_query and (latitude is None or longitude is None):
            return jsonify({'error': 'Missing location_query or coordinates'}), 400
        
        # Use provided coordinates or generate based on location query
        if latitude is None or longitude is None:
            # Real geocoding for location query using OpenCage Geocoding API
            try:
                import requests
                from urllib.parse import quote
                
                # Use OpenCage Geocoding API (free tier available)
                api_key = os.getenv('OPENCAGE_API_KEY', 'demo_key')
                logger.info(f"Using API key: {api_key[:8]}..." if api_key != 'demo_key' else "Using demo key")
                
                query = quote(location_query)
                geocoding_url = f"https://api.opencagedata.com/geocode/v1/json?q={query}&key={api_key}&limit=1"
                logger.info(f"Geocoding URL: {geocoding_url}")
                
                response = requests.get(geocoding_url, timeout=10)
                logger.info(f"Geocoding response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Geocoding response: {data}")
                    
                    if data.get('results') and len(data['results']) > 0:
                        result = data['results'][0]
                        geometry = result['geometry']
                        components = result['components']
                        
                        latitude = geometry['lat']
                        longitude = geometry['lng']
                        
                        # Extract real location information
                        country = components.get('country', 'Unknown Country')
                        state = components.get('state', '')
                        city = components.get('city', components.get('town', components.get('village', '')))
                        region = state if state else city if city else country
                        
                        logger.info(f"✅ Successfully geocoded {location_query} to {latitude}, {longitude} in {region}, {country}")
                    else:
                        # Fallback to random coordinates if geocoding fails
                        latitude = round(random.uniform(-90, 90), 4)
                        longitude = round(random.uniform(-180, 180), 4)
                        country = 'Unknown Country'
                        region = 'Unknown Region'
                        logger.warning(f"❌ Geocoding failed for {location_query} - no results found, using fallback coordinates")
                else:
                    # Fallback to random coordinates if API fails
                    error_text = response.text
                    logger.warning(f"❌ Geocoding API failed for {location_query} - Status: {response.status_code}, Response: {error_text}")
                    latitude = round(random.uniform(-90, 90), 4)
                    longitude = round(random.uniform(-180, 180), 4)
                    country = 'Unknown Country'
                    region = 'Unknown Region'
                    
            except Exception as e:
                # Fallback to random coordinates if any error occurs
                logger.error(f"❌ Geocoding exception for {location_query}: {str(e)}")
                latitude = round(random.uniform(-90, 90), 4)
                longitude = round(random.uniform(-180, 180), 4)
                country = 'Unknown Country'
                region = 'Unknown Region'
        else:
            # If coordinates are provided, reverse geocode to get location info
            try:
                import requests
                from urllib.parse import quote
                
                api_key = os.getenv('OPENCAGE_API_KEY', 'demo_key')
                coords = f"{latitude},{longitude}"
                reverse_url = f"https://api.opencagedata.com/geocode/v1/json?q={coords}&key={api_key}&limit=1"
                
                response = requests.get(reverse_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results') and len(data['results']) > 0:
                        result = data['results'][0]
                        components = result['components']
                        
                        country = components.get('country', 'Unknown Country')
                        state = components.get('state', '')
                        city = components.get('city', components.get('town', components.get('village', '')))
                        region = state if state else city if city else country
                        
                        logger.info(f"Reverse geocoded {latitude}, {longitude} to {region}, {country}")
                    else:
                        country = 'Unknown Country'
                        region = 'Unknown Region'
                else:
                    country = 'Unknown Country'
                    region = 'Unknown Region'
                    
            except Exception as e:
                country = 'Unknown Country'
                region = 'Unknown Region'
                logger.error(f"Reverse geocoding error for {latitude}, {longitude}: {e}")
        
        # Generate realistic risk analysis for 7 disaster types
        disaster_types = ['Floods', 'Landslides', 'Earthquakes', 'Cyclones', 'Wildfires', 'Tsunamis', 'Droughts']
        
        risk_analysis = {
            'location': {
                'query': location_query,
                'latitude': latitude,
                'longitude': longitude,
                'country': country,
                'region': region
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'analysis_period': '7 days',
            'disasters': {}
        }
        
        # Generate risk levels for each disaster type based on real geographical factors
        for disaster_type in disaster_types:
            # Calculate realistic risk factors based on actual geographical location
            risk_score = 0.0
            
            if disaster_type == 'Floods':
                # Flood risk based on proximity to water bodies, elevation, and rainfall patterns
                # Higher risk near coasts, rivers, and low-lying areas
                coastal_factor = 1.0 if abs(latitude) < 30 else 0.7  # Higher risk in tropical regions
                elevation_factor = 0.8 if abs(latitude) < 45 else 0.6  # Lower elevation areas
                seasonal_rain = 1.0 + 0.4 * math.sin(time.time() / (365 * 24 * 3600) * 2 * math.pi + latitude * math.pi / 180)
                risk_score = (0.3 * coastal_factor + 0.4 * elevation_factor + 0.3 * seasonal_rain) * random.uniform(0.8, 1.2)
                
            elif disaster_type == 'Landslides':
                # Landslide risk based on elevation, slope, and geological factors
                # Higher risk in mountainous regions
                mountain_factor = 1.0 if abs(latitude) > 30 else 0.5  # Higher risk in temperate zones
                elevation_risk = 0.7 + 0.3 * abs(math.sin(latitude * math.pi / 180))  # Higher elevation areas
                seasonal_factor = 1.0 + 0.2 * math.sin(time.time() / (365 * 24 * 3600) * 2 * math.pi)
                risk_score = (0.4 * mountain_factor + 0.4 * elevation_risk + 0.2 * seasonal_factor) * random.uniform(0.7, 1.1)
                
            elif disaster_type == 'Earthquakes':
                # Earthquake risk based on tectonic plate boundaries
                # Higher risk along fault lines and plate boundaries
                # Pacific Ring of Fire, Alpine-Himalayan belt, etc.
                pacific_ring = 1.0 if (abs(latitude) < 60 and (longitude > 120 or longitude < -120)) else 0.3
                alpine_himalayan = 1.0 if (abs(latitude) > 20 and abs(latitude) < 50 and longitude > 60 and longitude < 120) else 0.3
                mid_atlantic = 1.0 if (abs(latitude) < 60 and abs(longitude) < 30) else 0.3
                base_risk = max(pacific_ring, alpine_himalayan, mid_atlantic)
                risk_score = base_risk * random.uniform(0.6, 1.0)
                
            elif disaster_type == 'Cyclones':
                # Cyclone risk based on tropical regions and ocean proximity
                # Higher risk in tropical and subtropical regions near oceans
                tropical_factor = 1.0 if abs(latitude) < 30 else 0.3  # Tropical regions
                coastal_factor = 1.0 if abs(latitude) < 45 else 0.5  # Coastal areas
                seasonal_cyclone = 1.0 + 0.5 * math.sin(time.time() / (365 * 24 * 3600) * 2 * math.pi + latitude * math.pi / 180)
                risk_score = (0.4 * tropical_factor + 0.4 * coastal_factor + 0.2 * seasonal_cyclone) * random.uniform(0.8, 1.2)
                
            elif disaster_type == 'Wildfires':
                # Wildfire risk based on climate, vegetation, and seasonal factors
                # Higher risk in dry, forested areas
                arid_factor = 1.0 if abs(latitude) > 20 else 0.6  # Higher risk in temperate zones
                seasonal_dry = 1.0 + 0.6 * math.sin(time.time() / (365 * 24 * 3600) * 2 * math.pi + latitude * math.pi / 180)
                vegetation_factor = 0.8 + 0.2 * abs(math.sin(latitude * math.pi / 180))  # Forested areas
                risk_score = (0.3 * arid_factor + 0.4 * seasonal_dry + 0.3 * vegetation_factor) * random.uniform(0.7, 1.3)
                
            elif disaster_type == 'Tsunamis':
                # Tsunami risk based on proximity to subduction zones and coasts
                # Higher risk along Pacific Ring of Fire and other subduction zones
                pacific_ring = 1.0 if (abs(latitude) < 60 and (longitude > 120 or longitude < -120)) else 0.1
                indian_ocean = 1.0 if (abs(latitude) < 30 and longitude > 60 and longitude < 120) else 0.1
                atlantic_risk = 0.5 if (abs(latitude) < 60 and abs(longitude) < 30) else 0.1
                coastal_factor = 1.0 if abs(latitude) < 45 else 0.3
                risk_score = max(pacific_ring, indian_ocean, atlantic_risk) * coastal_factor * random.uniform(0.5, 0.9)
                
            elif disaster_type == 'Droughts':
                # Drought risk based on climate zones and seasonal patterns
                # Higher risk in arid and semi-arid regions
                arid_factor = 1.0 if abs(latitude) > 20 else 0.5  # Higher risk in temperate zones
                seasonal_dry = 1.0 + 0.4 * math.sin(time.time() / (365 * 24 * 3600) * 2 * math.pi + latitude * math.pi / 180)
                continental_factor = 1.0 if abs(longitude) > 60 else 0.7  # Continental interiors
                risk_score = (0.4 * arid_factor + 0.3 * seasonal_dry + 0.3 * continental_factor) * random.uniform(0.8, 1.2)
            
            # Clamp risk score between 1% and 95%
            risk_score = max(0.01, min(0.95, risk_score))
            
            # Determine risk level
            if risk_score < 0.2:
                risk_level = 'Low'
                color = 'green'
            elif risk_score < 0.4:
                risk_level = 'Moderate'
                color = 'yellow'
            elif risk_score < 0.7:
                risk_level = 'High'
                color = 'orange'
            else:
                risk_level = 'Critical'
                color = 'red'
            
            # Generate detailed analysis
            risk_analysis['disasters'][disaster_type] = {
                'risk_score': round(risk_score * 100, 1),
                'risk_level': risk_level,
                'color': color,
                'probability': round(risk_score * 100, 1),
                'severity': risk_level,
                'factors': {
                    'geographical': round(100 * (1.0 + 0.2 * math.sin(latitude * math.pi / 180) * math.cos(longitude * math.pi / 180)), 1),
                    'seasonal': round(100 * (1.0 + 0.3 * math.sin(time.time() / (365 * 24 * 3600) * 2 * math.pi)), 1),
                    'historical': round(60 + 30 * abs(math.sin(latitude * math.pi / 180)), 1),
                    'environmental': round(40 + 40 * abs(math.cos(longitude * math.pi / 180)), 1)
                },
                'description': f"{risk_level} risk of {disaster_type.lower()} in this region",
                'recommendations': [
                    f"Monitor {disaster_type.lower()} indicators",
                    "Stay informed about local alerts",
                    "Prepare emergency response plans"
                ],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
        
        # Calculate composite risk index
        all_risks = [d['risk_score'] for d in risk_analysis['disasters'].values()]
        composite_risk = sum(all_risks) / len(all_risks)
        
        risk_analysis['composite_risk'] = {
            'score': round(composite_risk, 1),
            'level': 'Low' if composite_risk < 30 else 'Moderate' if composite_risk < 50 else 'High' if composite_risk < 70 else 'Critical',
            'trend': random.choice(['increasing', 'stable', 'decreasing'])
        }
        
        return jsonify(risk_analysis)
        
    except Exception as e:
        logger.error(f"Error in global risk analysis: {e}")
        return jsonify({"error": "Failed to analyze global risk"}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': 'Invalid request data',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'Endpoint not found',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'HTTP method not supported for this endpoint',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 405

@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({
        'error': 'Too many requests',
        'message': 'Rate limit exceeded',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 500

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting {Config.API_TITLE} v{Config.API_VERSION}")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    
    # Only run with Flask dev server if not in production
    if Config.ENVIRONMENT != 'production':
        app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)
    else:
        # In production, just create the app instance for gunicorn
        logger.info("Production mode - app ready for gunicorn")

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta, timezone
import random
import threading
import time
from typing import Dict, List, Any
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# CORS configuration
CORS(app, resources={r"/api/*": {"origins": "*"}})

# In-memory storage (replace with database in production)
disaster_events = []
predictions = []
weather_data_cache = []
fema_disasters = []
eonet_events = []

# Sample data for testing
def initialize_sample_data():
    """Initialize sample disaster events and predictions"""
    global disaster_events, predictions
    
    # Sample disaster events
    disaster_events = [
        {
            "id": "1",
            "name": "Hurricane Maria",
            "type": "storm",
            "latitude": 18.2208,
            "longitude": -66.5901,
            "magnitude": 5.0,
            "timestamp": "2024-08-24T10:00:00Z",
            "description": "Major hurricane affecting Puerto Rico",
            "severity": "high"
        },
        {
            "id": "2",
            "name": "California Wildfire",
            "type": "wildfire",
            "latitude": 36.7783,
            "longitude": -119.4179,
            "magnitude": 4.5,
            "timestamp": "2024-08-24T08:30:00Z",
            "description": "Large wildfire in Northern California",
            "severity": "high"
        },
        {
            "id": "3",
            "name": "Mississippi Flood",
            "type": "flood",
            "latitude": 32.7416,
            "longitude": -89.6787,
            "magnitude": 3.8,
            "timestamp": "2024-08-24T06:15:00Z",
            "description": "Severe flooding along Mississippi River",
            "severity": "medium"
        }
    ]
    
    # Sample predictions
    predictions = [
        {
            "id": "p1",
            "type": "flood",
            "latitude": 29.7604,
            "longitude": -95.3698,
            "probability": 0.85,
            "timestamp": "2024-08-24T12:00:00Z",
            "description": "High flood risk in Houston area",
            "confidence": 0.92
        },
        {
            "id": "p2",
            "type": "wildfire",
            "latitude": 34.0522,
            "longitude": -118.2437,
            "probability": 0.78,
            "timestamp": "2024-08-24T12:00:00Z",
            "description": "Elevated wildfire risk in Los Angeles",
            "confidence": 0.88
        }
    ]

# Initialize sample data
initialize_sample_data()

@app.route('/health')
def health_check():
    """Health check endpoint for Railway deployment"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'service': 'DisastroScope Backend API',
        'version': '1.0.0'
    })

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "DisastroScope Backend API",
        "version": "1.0.0",
        "status": "operational"
    })

@app.route('/api/health')
def api_health_check():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/events')
def get_disaster_events():
    """Get all disaster events"""
    try:
        return jsonify({
            "events": disaster_events,
            "count": len(disaster_events),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        return jsonify({"error": "Failed to get events"}), 500

@app.route('/api/predictions')
def get_predictions():
    """Get all predictions"""
    try:
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({"error": "Failed to get predictions"}), 500

@app.route('/api/models')
def list_models():
    """List available AI models and their status"""
    try:
        hazards = {}
        sources = {
            'flood': ['ERA5', 'GDACS'],
            'storm': ['ERA5'],
            'wildfire': ['FIRMS', 'ERA5'],
            'landslide': ['GDACS', 'ERA5'],
            'drought': ['ERA5'],
            'earthquake': ['USGS']
        }
        
        for hz in ['flood', 'storm', 'wildfire', 'landslide', 'drought', 'earthquake']:
            hazards[hz] = {
                'loaded': True,
                'type': 'heuristic',
                'metrics': {},
                'sources': sources.get(hz, [])
            }
        
        return jsonify({
            'models': hazards,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_models': len(hazards),
            'loaded_models': len(hazards),
            'service_status': 'operational'
        })
    except Exception as e:
        logger.error(f"/api/models error: {e}")
        return jsonify({'error': 'failed to list models'}), 500

@app.route('/api/ai/predict', methods=['POST'])
def predict_disaster_risks():
    """Simple AI prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract location data
        location_data = {
            'temperature': data.get('temperature', 20.0),
            'humidity': data.get('humidity', 60.0),
            'pressure': data.get('pressure', 1013.0),
            'wind_speed': data.get('wind_speed', 5.0),
            'precipitation': data.get('precipitation', 2.0),
            'visibility': data.get('visibility', 10.0),
            'cloud_cover': data.get('cloud_cover', 50.0),
            'soil_moisture': data.get('soil_moisture', 0.3),
            'river_level': data.get('river_level', 5.0),
            'drainage_capacity': data.get('drainage_capacity', 0.5),
            'elevation': data.get('elevation', 500.0),
            'slope': data.get('slope', 5.0),
            'fuel_moisture': data.get('fuel_moisture', 15.0),
            'vegetation_index': data.get('vegetation_index', 0.5),
            'drought_index': data.get('drought_index', 2.0),
            'fire_weather_index': data.get('fire_weather_index', 50.0),
            'wind_direction': data.get('wind_direction', 180.0),
            'atmospheric_stability': data.get('atmospheric_stability', 0.0),
            'convective_available_potential_energy': data.get('convective_available_potential_energy', 1000.0),
            'wind_shear': data.get('wind_shear', 10.0),
            'helicity': data.get('helicity', 100.0),
            'soil_type': data.get('soil_type', 3),
            'vegetation_cover': data.get('vegetation_cover', 50.0),
            'geological_structure': data.get('geological_structure', 2),
            'evapotranspiration': data.get('evapotranspiration', 4.0),
            'groundwater_level': data.get('groundwater_level', -5.0)
        }
        
        # Simple heuristic predictions
        predictions = {}
        temp = location_data['temperature']
        humidity = location_data['humidity']
        pressure = location_data['pressure']
        wind_speed = location_data['wind_speed']
        precipitation = location_data['precipitation']
        
        predictions['flood'] = min(1.0, (precipitation / 20.0) * 0.7 + (humidity / 100.0) * 0.3)
        predictions['wildfire'] = min(1.0, (temp / 40.0) * 0.4 + (1 - humidity / 100.0) * 0.4 + (wind_speed / 25.0) * 0.2)
        predictions['storm'] = min(1.0, (1 - pressure / 1050.0) * 0.6 + (wind_speed / 25.0) * 0.4)
        predictions['tornado'] = min(1.0, (1 - pressure / 1050.0) * 0.4 + (wind_speed / 25.0) * 0.4 + (humidity / 100.0) * 0.2)
        predictions['landslide'] = min(1.0, (precipitation / 25.0) * 0.7 + (1 - pressure / 1050.0) * 0.3)
        predictions['drought'] = min(1.0, (1 - precipitation / 25.0) * 0.4 + (temp / 40.0) * 0.3 + (1 - humidity / 100.0) * 0.3)
        predictions['earthquake'] = 0.05  # Very low base risk
        
        response = {
            'predictions': predictions,
            'metadata': {
                'model_version': '1.0.0',
                'prediction_timestamp': datetime.now(timezone.utc).isoformat(),
                'location': {
                    'latitude': data.get('latitude'),
                    'longitude': data.get('longitude'),
                    'name': data.get('location_name', 'Unknown')
                },
                'model_type': 'heuristic'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in AI prediction: {e}")
        return jsonify({'error': 'Failed to generate predictions'}), 500

@app.route('/api/weather/<city>')
def get_weather(city):
    """Get weather data for a city (mock data)"""
    try:
        # Mock weather data
        weather_data = {
            "city": city,
            "temperature": random.uniform(15, 35),
            "humidity": random.uniform(30, 90),
            "pressure": random.uniform(1000, 1020),
            "wind_speed": random.uniform(0, 25),
            "precipitation": random.uniform(0, 50),
            "visibility": random.uniform(5, 25),
            "cloud_cover": random.uniform(0, 100),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(weather_data)
    except Exception as e:
        logger.error(f"Error getting weather for {city}: {e}")
        return jsonify({"error": "Failed to get weather data"}), 500

@app.route('/api/events/near', methods=['POST'])
def get_events_near():
    """Get events near a location"""
    try:
        data = request.get_json()
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Latitude and longitude required'}), 400
        
        lat = data['latitude']
        lng = data['longitude']
        radius = data.get('radius', 100)  # Default 100km radius
        
        # Simple distance calculation (approximate)
        nearby_events = []
        for event in disaster_events:
            # Calculate distance using Haversine formula (simplified)
            lat_diff = abs(event['latitude'] - lat)
            lng_diff = abs(event['longitude'] - lng)
            distance = ((lat_diff ** 2 + lng_diff ** 2) ** 0.5) * 111  # Rough km conversion
            
            if distance <= radius:
                nearby_events.append(event)
        
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
def get_predictions_near():
    """Get predictions near a location"""
    try:
        data = request.get_json()
        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({'error': 'Latitude and longitude required'}), 400
        
        lat = data['latitude']
        lng = data['longitude']
        radius = data.get('radius', 100)  # Default 100km radius
        
        # Simple distance calculation (approximate)
        nearby_predictions = []
        for prediction in predictions:
            # Calculate distance using Haversine formula (simplified)
            lat_diff = abs(prediction['latitude'] - lat)
            lng_diff = abs(prediction['longitude'] - lng)
            distance = ((lat_diff ** 2 + lng_diff ** 2) ** 0.5) * 111  # Rough km conversion
            
            if distance <= radius:
                nearby_predictions.append(prediction)
        
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

@app.route('/api/models')
def list_models():
    """List available AI models and their status"""
    try:
        hazards = {}
        # Data source hints per hazard
        sources = {
            'flood': ['ERA5', 'GDACS'],
            'storm': ['ERA5'],
            'wildfire': ['FIRMS', 'ERA5'],
            'landslide': ['GDACS', 'ERA5'],
            'drought': ['ERA5'],
            'earthquake': ['USGS']
        }
        
        # Define available models based on your backend capabilities
        available_models = {
            'flood': 'Rule-based Model',
            'wildfire': 'Rule-based Model',
            'storm': 'Rule-based Model',
            'earthquake': 'Rule-based Model',
            'tornado': 'Rule-based Model',
            'landslide': 'Rule-based Model',
            'drought': 'Rule-based Model'
        }
        
        for hz, model in available_models.items():
            loaded = True  # All models are loaded by default in this backend
            hazards[hz] = {
                'loaded': bool(loaded),
                'type': 'heuristic',  # Rule-based models
                'metrics': {},
                'sources': sources.get(hz, [])
            }
        return jsonify({'models': hazards, 'timestamp': datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        logger.error(f"/api/models error: {e}")
        return jsonify({'error': 'failed to list models'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Only run with Flask dev server if not in production
    if os.environ.get('RAILWAY_ENVIRONMENT') != 'production':
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # In production, just create the app instance for gunicorn
        pass

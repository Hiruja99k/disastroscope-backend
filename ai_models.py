import logging
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from datetime import datetime, timedelta
import json
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import aiohttp
import asyncio


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisasterPredictionService:
    """Simple service for managing disaster prediction models"""
    
    def __init__(self):
        self.models = {
            'flood': 'Rule-based Model',
            'wildfire': 'Rule-based Model',
            'storm': 'Rule-based Model',
            'earthquake': 'Rule-based Model',
            'tornado': 'Rule-based Model',
            'landslide': 'Rule-based Model',
            'drought': 'Rule-based Model'
        }
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Try load wildfire model if exists
        try:
            wf_model = os.path.join(self.model_dir, 'wildfire_model.joblib')
            wf_scaler = os.path.join(self.model_dir, 'wildfire_scaler.joblib')
            if os.path.exists(wf_model) and os.path.exists(wf_scaler):
                self.models['wildfire'] = {
                    'clf': joblib.load(wf_model),
                    'scaler': joblib.load(wf_scaler)
                }
        except Exception as e:
            logger.warning(f"Failed to load wildfire model: {e}")
    
    def predict_disaster_risks(self, weather_data: Dict) -> Dict[str, float]:
        """Predict disaster risks using simple heuristic rules"""
        predictions = {}
        
        try:
            # Extract weather parameters
            temp = weather_data.get('temperature', 20)
            humidity = weather_data.get('humidity', 50)
            pressure = weather_data.get('pressure', 1013)
            wind_speed = weather_data.get('wind_speed', 5)
            wind_direction = weather_data.get('wind_direction', 180)
            precipitation = weather_data.get('precipitation', 0)
            visibility = weather_data.get('visibility', 10)
            cloud_cover = weather_data.get('cloud_cover', 50)
            
            # Simple heuristic predictions
            predictions['flood'] = min(1.0, (precipitation / 20.0) * 0.7 + (humidity / 100.0) * 0.3)
            predictions['wildfire'] = min(1.0, (temp / 40.0) * 0.4 + (1 - humidity / 100.0) * 0.4 + (wind_speed / 25.0) * 0.2)
            predictions['storm'] = min(1.0, (1 - pressure / 1050.0) * 0.6 + (wind_speed / 25.0) * 0.4)
            predictions['tornado'] = min(1.0, (1 - pressure / 1050.0) * 0.4 + (wind_speed / 25.0) * 0.4 + (humidity / 100.0) * 0.2)
            predictions['landslide'] = min(1.0, (precipitation / 25.0) * 0.7 + (1 - pressure / 1050.0) * 0.3)
            predictions['drought'] = min(1.0, (1 - precipitation / 25.0) * 0.4 + (temp / 40.0) * 0.3 + (1 - humidity / 100.0) * 0.3)
            predictions['earthquake'] = 0.05  # Very low base risk
            
            logger.info(f"Generated predictions for {len(predictions)} hazard types")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_disaster_risks: {e}")
            # Return safe defaults
            return {
                'flood': 0.1,
                'wildfire': 0.1,
                'storm': 0.1,
                'tornado': 0.1,
                'landslide': 0.1,
                'drought': 0.1,
                'earthquake': 0.05
            }

# Initialize the service
ai_prediction_service = DisasterPredictionService()

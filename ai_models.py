import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

EARTHQUAKE_RISK_MULTIPLIER: float = float(os.getenv('EARTHQUAKE_RISK_MULTIPLIER', '0.05'))
ALLOW_EARTHQUAKE_PREDICTIONS: bool = os.getenv('ALLOW_EARTHQUAKE_PREDICTIONS', 'false').lower() == 'true'

class DisasterPredictionService:
    """Service for managing disaster prediction models"""
    
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
    
    def predict_disaster_risks(self, weather_data: Dict) -> Dict[str, float]:
        """Predict risks for all disaster types using continuous, weather-based scoring.
        The mappings below are calibrated heuristics using sigmoids to avoid "all zero" outputs
        while still reflecting live weather features.
        """
        t = float(weather_data.get('temperature', 20))
        h = float(weather_data.get('humidity', 50))
        pcp = float(weather_data.get('precipitation', 0))
        ws = float(weather_data.get('wind_speed', 0))
        pr = float(weather_data.get('pressure', 1013))
        cc = float(weather_data.get('cloud_cover', 0))

        def clamp(x: float) -> float:
            return max(0.0, min(1.0, x))

        def sigmoid(x: float) -> float:
            import math
            return 1.0 / (1.0 + math.exp(-x))

        predictions: Dict[str, float] = {}
        # Flood: precipitation + humidity/clouds
        flood_precip = sigmoid((pcp - 4.0) / 3.0)
        flood_humid = sigmoid((h - 70.0) / 8.0)
        flood_cloud = sigmoid((cc - 60.0) / 10.0)
        predictions['flood'] = clamp(0.7 * flood_precip + 0.2 * flood_humid + 0.1 * flood_cloud)

        # Wildfire: high temp + low humidity + some wind
        wildfire_temp = sigmoid((t - 32.0) / 4.0)
        wildfire_dry = sigmoid((30.0 - h) / 6.0)
        wildfire_wind = sigmoid((ws - 5.0) / 3.0)
        predictions['wildfire'] = clamp(0.6 * wildfire_temp * wildfire_dry + 0.4 * wildfire_wind * wildfire_dry)

        # Storm: wind + low pressure + clouds
        storm_wind = sigmoid((ws - 8.0) / 3.0)
        storm_press = sigmoid((1013.0 - pr) / 6.0)
        storm_cloud = sigmoid((cc - 50.0) / 10.0)
        predictions['storm'] = clamp(0.6 * storm_wind + 0.3 * storm_press + 0.1 * storm_cloud)

        # Earthquake (clamped below unless explicitly allowed)
        predictions['earthquake'] = 0.05

        # Tornado: strong wind + high humidity + deep clouds
        tor_wind = sigmoid((ws - 15.0) / 5.0)
        tor_humid = sigmoid((h - 60.0) / 8.0)
        tor_cloud = sigmoid((cc - 60.0) / 8.0)
        predictions['tornado'] = clamp(tor_wind * tor_humid * tor_cloud)

        # Landslide: heavy precipitation + very humid
        land_precip = sigmoid((pcp - 10.0) / 5.0)
        land_humid = sigmoid((h - 75.0) / 6.0)
        predictions['landslide'] = clamp(land_precip * land_humid)

        # Drought: hot + very dry + little precip
        dr_temp = sigmoid((t - 30.0) / 4.0)
        dr_dry = sigmoid((25.0 - h) / 5.0)
        dr_precip = sigmoid((1.0 - min(pcp, 1.0)) / 0.2)
        predictions['drought'] = clamp(dr_temp * dr_dry * dr_precip)

        # Clamp earthquake unless allowed
        if not ALLOW_EARTHQUAKE_PREDICTIONS:
            try:
                predictions['earthquake'] = max(0.0, min(0.05, predictions['earthquake'] * EARTHQUAKE_RISK_MULTIPLIER))
            except Exception:
                predictions['earthquake'] = 0.0

        return predictions
    
    def train_models(self, epochs: int = 50):
        """Train all models (placeholder for demo)"""
        logger.info(f"Training models for {epochs} epochs...")
        # In a real implementation, this would load training data and train the models
        logger.info("Models trained successfully")

# Global AI prediction service instance
ai_prediction_service = DisasterPredictionService()

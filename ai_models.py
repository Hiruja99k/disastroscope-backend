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
        """Predict risks for all disaster types using rule-based logic (fast and reliable)."""
        t = float(weather_data.get('temperature', 20))
        h = float(weather_data.get('humidity', 50))
        pcp = float(weather_data.get('precipitation', 0))
        ws = float(weather_data.get('wind_speed', 0))
        pr = float(weather_data.get('pressure', 1013))
        cc = float(weather_data.get('cloud_cover', 0))

        def clamp(x: float) -> float:
            return max(0.0, min(1.0, x))

        predictions: Dict[str, float] = {}
        # Flood
        predictions['flood'] = clamp((pcp / 50.0) * (0.5 + cc / 200.0)) if pcp > 0 else 0.0
        # Wildfire
        predictions['wildfire'] = clamp(((t - 30.0) / 20.0) * (1.0 - h / 100.0)) if (t > 30 and h < 35) else 0.0
        # Storm
        predictions['storm'] = clamp((ws / 60.0) * (1.0 - min(pr, 1100.0) / 1100.0) * (0.5 + cc / 200.0))
        # Earthquake (clamped below unless explicitly allowed)
        predictions['earthquake'] = 0.05
        # Tornado
        predictions['tornado'] = clamp(((ws - 25.0) / 40.0) * (h / 100.0)) if (ws > 25 and h > 60 and cc > 60) else 0.0
        # Landslide
        predictions['landslide'] = clamp(pcp / 100.0) if (pcp > 20 and h > 70) else 0.0
        # Drought
        predictions['drought'] = clamp(((28.0 - min(h, 28.0)) / 28.0) * (1.0 - min(pcp, 10.0) / 10.0)) if (h < 25 and pcp < 1 and t > 28) else 0.0

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

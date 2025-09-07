#!/usr/bin/env python3
"""
Enhanced AI Models Training Script
Trains the advanced machine learning models for disaster prediction
"""

import os
import sys
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ai_models import enhanced_ai_prediction_service

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_models_training.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main training function"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("ENHANCED AI MODELS TRAINING STARTED")
    logger.info("=" * 60)
    
    try:
        # Check if models already exist
        model_path = os.path.join(os.path.dirname(__file__), "enhanced_models")
        if os.path.exists(model_path) and os.listdir(model_path):
            logger.info("Enhanced models already exist. Retraining...")
        
        # Train enhanced models
        logger.info("Starting enhanced model training...")
        start_time = datetime.now()
        
        enhanced_ai_prediction_service.train_enhanced_models(
            epochs=100,  # More epochs for better accuracy
            batch_size=64
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"Enhanced model training completed in {training_duration}")
        
        # Display model performance
        performance = enhanced_ai_prediction_service.get_model_performance()
        logger.info("Model Performance Summary:")
        for disaster_type, metrics in performance.items():
            logger.info(f"  {disaster_type}:")
            for model_name, model_metrics in metrics.items():
                logger.info(f"    {model_name}: R² = {model_metrics.get('r2', 0):.4f}, MSE = {model_metrics.get('mse', 0):.4f}")
        
        # Display feature importance
        feature_importance = enhanced_ai_prediction_service.get_feature_importance()
        logger.info("Feature Importance Summary:")
        for disaster_type, importance in feature_importance.items():
            if importance:
                logger.info(f"  {disaster_type}: Top features available")
        
        logger.info("=" * 60)
        logger.info("ENHANCED AI MODELS TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("=" * 60)
        logger.error("ENHANCED AI MODELS TRAINING FAILED")
        logger.error("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

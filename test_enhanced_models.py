#!/usr/bin/env python3
"""
Enhanced AI Models Test Script
Tests the enhanced machine learning models for disaster prediction
"""

import os
import sys
import logging
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ai_models import enhanced_ai_prediction_service

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_enhanced_predictions():
    """Test enhanced predictions with sample data"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("ENHANCED AI MODELS TESTING STARTED")
    logger.info("=" * 60)
    
    # Test cases with different locations and conditions
    test_cases = [
        {
            "name": "San Francisco, CA (Earthquake Prone)",
            "weather_data": {
                "temperature": 18.0,
                "humidity": 65.0,
                "pressure": 1013.0,
                "wind_speed": 12.0,
                "wind_direction": 270.0,
                "precipitation": 0.0,
                "visibility": 15.0,
                "cloud_cover": 30.0,
                "uv_index": 6.0,
                "dew_point": 12.0,
                "heat_index": 18.0,
                "wind_chill": 18.0,
                "precipitation_intensity": 0.0,
                "atmospheric_stability": 0.7,
                "moisture_content": 0.65
            },
            "geospatial_data": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "elevation": 52.0,
                "slope": 8.0,
                "aspect": 180.0,
                "soil_type": 2,
                "land_use": 3,
                "distance_to_water": 5.0,
                "distance_to_fault": 15.0,
                "population_density": 8000.0,
                "infrastructure_density": 0.9,
                "historical_events": 25,
                "tectonic_zone": 1,  # Ring of Fire
                "climate_zone": 1,
                "vegetation_index": 0.4,
                "urbanization_level": 0.95
            }
        },
        {
            "name": "Miami, FL (Flood Prone)",
            "weather_data": {
                "temperature": 28.0,
                "humidity": 85.0,
                "pressure": 1005.0,
                "wind_speed": 8.0,
                "wind_direction": 180.0,
                "precipitation": 45.0,
                "visibility": 8.0,
                "cloud_cover": 90.0,
                "uv_index": 8.0,
                "dew_point": 25.0,
                "heat_index": 32.0,
                "wind_chill": 28.0,
                "precipitation_intensity": 15.0,
                "atmospheric_stability": 0.3,
                "moisture_content": 0.85
            },
            "geospatial_data": {
                "latitude": 25.7617,
                "longitude": -80.1918,
                "elevation": 2.0,
                "slope": 1.0,
                "aspect": 0.0,
                "soil_type": 3,
                "land_use": 3,
                "distance_to_water": 1.0,
                "distance_to_fault": 200.0,
                "population_density": 5000.0,
                "infrastructure_density": 0.8,
                "historical_events": 15,
                "tectonic_zone": 0,
                "climate_zone": 0,
                "vegetation_index": 0.7,
                "urbanization_level": 0.9
            }
        },
        {
            "name": "Seattle, WA (Landslide Prone)",
            "weather_data": {
                "temperature": 12.0,
                "humidity": 90.0,
                "pressure": 1008.0,
                "wind_speed": 15.0,
                "wind_direction": 225.0,
                "precipitation": 60.0,
                "visibility": 5.0,
                "cloud_cover": 95.0,
                "uv_index": 2.0,
                "dew_point": 11.0,
                "heat_index": 12.0,
                "wind_chill": 8.0,
                "precipitation_intensity": 20.0,
                "atmospheric_stability": 0.2,
                "moisture_content": 0.90
            },
            "geospatial_data": {
                "latitude": 47.6062,
                "longitude": -122.3321,
                "elevation": 50.0,
                "slope": 25.0,
                "aspect": 270.0,
                "soil_type": 2,
                "land_use": 2,
                "distance_to_water": 3.0,
                "distance_to_fault": 50.0,
                "population_density": 3000.0,
                "infrastructure_density": 0.7,
                "historical_events": 20,
                "tectonic_zone": 1,
                "climate_zone": 1,
                "vegetation_index": 0.8,
                "urbanization_level": 0.8
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest Case {i}: {test_case['name']}")
        logger.info("-" * 40)
        
        try:
            # Get enhanced predictions
            predictions = enhanced_ai_prediction_service.predict_enhanced_risks(
                test_case['weather_data'],
                test_case['geospatial_data']
            )
            
            # Display results
            logger.info("Enhanced Predictions:")
            for disaster_type, risk_level in predictions.items():
                risk_category = "LOW" if risk_level < 0.3 else "MEDIUM" if risk_level < 0.7 else "HIGH"
                logger.info(f"  {disaster_type.capitalize()}: {risk_level:.3f} ({risk_category})")
            
            # Store results
            results.append({
                "test_case": test_case['name'],
                "predictions": predictions,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in test case {i}: {e}")
            results.append({
                "test_case": test_case['name'],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Save results to file
    results_file = "enhanced_models_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nTest results saved to {results_file}")
    
    # Performance summary
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED AI MODELS TESTING COMPLETED")
    logger.info("=" * 60)
    
    # Display model performance if available
    try:
        performance = enhanced_ai_prediction_service.get_model_performance()
        if performance:
            logger.info("Model Performance Summary:")
            for disaster_type, metrics in performance.items():
                logger.info(f"  {disaster_type}:")
                for model_name, model_metrics in metrics.items():
                    r2 = model_metrics.get('r2', 0)
                    mse = model_metrics.get('mse', 0)
                    logger.info(f"    {model_name}: R² = {r2:.4f}, MSE = {mse:.4f}")
    except Exception as e:
        logger.warning(f"Could not retrieve model performance: {e}")
    
    return results

if __name__ == "__main__":
    results = test_enhanced_predictions()
    print(f"\nTesting completed. {len(results)} test cases processed.")

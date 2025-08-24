#!/usr/bin/env python3
"""
Test script to verify DisastroScope Backend functionality
Run this to identify specific issues with the backend
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from ai_models import ai_prediction_service
        print("✅ AI models imported successfully")
    except Exception as e:
        print(f"❌ AI models import failed: {e}")
        return False
    
    try:
        from weather_service import weather_service
        print("✅ Weather service imported successfully")
    except Exception as e:
        print(f"❌ Weather service import failed: {e}")
        return False
    
    try:
        from advanced_monitoring import advanced_monitoring
        print("✅ Monitoring imported successfully")
    except Exception as e:
        print(f"❌ Monitoring import failed: {e}")
        return False
    
    return True

def test_ai_models():
    """Test AI model functionality"""
    print("\n🤖 Testing AI Models...")
    
    try:
        from ai_models import ai_prediction_service
        
        # Check model status
        status = ai_prediction_service.get_model_status()
        print(f"✅ Model status: {status['overall_health']}")
        print(f"✅ Model version: {status['version']}")
        
        # Check if models are loaded
        models_loaded = sum(1 for info in status['models'].values() if info['models_loaded'])
        total_models = len(status['models'])
        print(f"✅ Models loaded: {models_loaded}/{total_models}")
        
        # Test prediction with sample data
        sample_weather = {
            'temperature': 25.0,
            'humidity': 60.0,
            'pressure': 1013.0,
            'wind_speed': 10.0,
            'wind_direction': 180.0,
            'precipitation': 0.0,
            'visibility': 10.0,
            'cloud_cover': 30.0
        }
        
        predictions = ai_prediction_service.predict_disaster_risks(sample_weather)
        print(f"✅ Predictions generated: {len(predictions)} hazard types")
        
        for hazard, risk in predictions.items():
            print(f"   {hazard}: {risk:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weather_service():
    """Test weather service functionality"""
    print("\n🌤️ Testing Weather Service...")
    
    try:
        from weather_service import weather_service
        
        # Test geocoding
        print("Testing geocoding...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(weather_service.geocode("New York", limit=3))
        if results:
            print(f"✅ Geocoding successful: {len(results)} results")
            for result in results[:2]:
                print(f"   {result.get('name')}, {result.get('state')}, {result.get('country')}")
        else:
            print("❌ Geocoding failed")
            return False
        
        # Test weather fetch
        print("Testing weather fetch...")
        lat, lon = results[0]['lat'], results[0]['lon']
        weather = loop.run_until_complete(weather_service.get_current_weather(lat, lon, "New York"))
        
        if weather:
            print(f"✅ Weather fetch successful: {weather.temperature}°C, {weather.humidity}% humidity")
        else:
            print("❌ Weather fetch failed")
            return False
        
        loop.close()
        return True
        
    except Exception as e:
        print(f"❌ Weather service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\n🎯 Testing Prediction Endpoint...")
    
    try:
        from ai_models import ai_prediction_service
        
        # Test with the coordinates from your dashboard
        test_coords = {
            'lat': 7.0725,
            'lon': 80.0011
        }
        
        # Create sample weather data
        sample_weather = {
            'temperature': 28.0,  # Typical for Sri Lanka
            'humidity': 75.0,
            'pressure': 1010.0,
            'wind_speed': 8.0,
            'wind_direction': 90.0,
            'precipitation': 2.0,
            'visibility': 15.0,
            'cloud_cover': 40.0
        }
        
        print(f"Testing with coordinates: {test_coords}")
        print(f"Sample weather data: {sample_weather}")
        
        # Test prediction
        predictions = ai_prediction_service.predict_disaster_risks(sample_weather)
        
        if predictions:
            print("✅ Predictions successful!")
            for hazard, risk in predictions.items():
                print(f"   {hazard}: {risk:.3f}")
        else:
            print("❌ No predictions generated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 DisastroScope Backend Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("AI Models", test_ai_models),
        ("Weather Service", test_weather_service),
        ("Prediction Endpoint", test_prediction_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n🔧 Recommended fixes:")
        if not any(name == "AI Models" and result for name, result in results):
            print("   - AI models need to be trained")
            print("   - Set AI_AUTO_TRAIN_ON_STARTUP=true")
        if not any(name == "Weather Service" and result for name, result in results):
            print("   - Weather service has issues")
            print("   - Check API keys and network connectivity")

if __name__ == "__main__":
    main()

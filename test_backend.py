#!/usr/bin/env python3
"""
Test script for DisastroScope Backend
Run this to verify all endpoints are working correctly
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"  # Change this to your Railway URL when testing production

def test_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
            
        if response.status_code == expected_status:
            print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            return True
        else:
            print(f"❌ {method} {endpoint} - Expected: {expected_status}, Got: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing DisastroScope Backend...")
    print("=" * 50)
    
    tests = [
        # Health endpoints
        ("/health", "GET"),
        ("/api/health", "GET"),
        ("/", "GET"),
        
        # Data endpoints
        ("/api/events", "GET"),
        ("/api/predictions", "GET"),
        ("/api/models", "GET"),
        
        # Weather endpoint
        ("/api/weather/NewYork", "GET"),
        
        # Location-based endpoints
        ("/api/events/near", "POST", {"latitude": 40.7128, "longitude": -74.0060, "radius": 100}),
        ("/api/predictions/near", "POST", {"latitude": 40.7128, "longitude": -74.0060, "radius": 100}),
        
        # AI prediction endpoint
        ("/api/ai/predict", "POST", {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "temperature": 25.0,
            "humidity": 70.0,
            "pressure": 1013.0,
            "wind_speed": 10.0,
            "precipitation": 5.0
        }),
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if len(test) == 2:
            endpoint, method = test
            data = None
        else:
            endpoint, method, data = test
            
        if test_endpoint(endpoint, method, data):
            passed += 1
            
        time.sleep(0.1)  # Small delay between requests
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()

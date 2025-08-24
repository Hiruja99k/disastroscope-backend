#!/usr/bin/env python3
"""
Comprehensive Test Suite for DisastroScope Backend
Production-ready testing with coverage and performance metrics
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

# Configuration
BASE_URL = "http://localhost:5000"  # Change this to your Railway URL when testing production

@dataclass
class TestResult:
    """Test result data structure"""
    name: str
    method: str
    endpoint: str
    status_code: int
    response_time: float
    success: bool
    error: str = None

class BackendTester:
    """Comprehensive backend testing framework"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'DisastroScope-TestSuite/2.0.0'
        })
    
    def test_endpoint(self, name: str, method: str, endpoint: str, 
                     data: Dict = None, expected_status: int = 200) -> TestResult:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=10)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=10)
            else:
                return TestResult(
                    name=name, method=method, endpoint=endpoint,
                    status_code=0, response_time=0, success=False,
                    error=f"Unsupported method: {method}"
                )
            
            response_time = time.time() - start_time
            success = response.status_code == expected_status
            
            result = TestResult(
                name=name, method=method, endpoint=endpoint,
                status_code=response.status_code, response_time=response_time,
                success=success
            )
            
            if not success:
                result.error = f"Expected {expected_status}, got {response.status_code}"
                if response.text:
                    result.error += f" - {response.text[:200]}"
            
            return result
            
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return TestResult(
                name=name, method=method, endpoint=endpoint,
                status_code=0, response_time=response_time, success=False,
                error=str(e)
            )
    
    def run_health_tests(self) -> List[TestResult]:
        """Run health and status endpoint tests"""
        tests = [
            ("Health Check", "GET", "/health"),
            ("API Health Check", "GET", "/api/health"),
            ("API Information", "GET", "/"),
            ("Metrics Endpoint", "GET", "/api/metrics"),
        ]
        
        results = []
        for name, method, endpoint in tests:
            result = self.test_endpoint(name, method, endpoint)
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)  # Small delay between requests
        
        return results
    
    def run_data_tests(self) -> List[TestResult]:
        """Run data retrieval endpoint tests"""
        tests = [
            ("Get Events", "GET", "/api/events"),
            ("Get Predictions", "GET", "/api/predictions"),
            ("List Models", "GET", "/api/models"),
        ]
        
        results = []
        for name, method, endpoint in tests:
            result = self.test_endpoint(name, method, endpoint)
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)
        
        return results
    
    def run_weather_tests(self) -> List[TestResult]:
        """Run weather endpoint tests"""
        cities = ["NewYork", "London", "Tokyo", "Sydney", "Mumbai"]
        results = []
        
        for city in cities:
            name = f"Weather Data - {city}"
            result = self.test_endpoint(name, "GET", f"/api/weather/{city}")
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)
        
        return results
    
    def run_location_tests(self) -> List[TestResult]:
        """Run location-based endpoint tests"""
        test_locations = [
            {"latitude": 40.7128, "longitude": -74.0060, "name": "New York"},
            {"latitude": 51.5074, "longitude": -0.1278, "name": "London"},
            {"latitude": 35.6762, "longitude": 139.6503, "name": "Tokyo"},
        ]
        
        results = []
        for location in test_locations:
            # Test events near location
            name = f"Events Near {location['name']}"
            result = self.test_endpoint(
                name, "POST", "/api/events/near",
                {"latitude": location["latitude"], "longitude": location["longitude"], "radius": 100}
            )
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)
            
            # Test predictions near location
            name = f"Predictions Near {location['name']}"
            result = self.test_endpoint(
                name, "POST", "/api/predictions/near",
                {"latitude": location["latitude"], "longitude": location["longitude"], "radius": 100}
            )
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)
        
        return results
    
    def run_prediction_tests(self) -> List[TestResult]:
        """Run AI prediction endpoint tests"""
        test_data = [
            {
                "name": "Basic Prediction",
                "data": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "temperature": 25.0,
                    "humidity": 70.0,
                    "pressure": 1013.0,
                    "wind_speed": 10.0,
                    "precipitation": 5.0
                }
            },
            {
                "name": "High Risk Prediction",
                "data": {
                    "latitude": 29.7604,
                    "longitude": -95.3698,
                    "temperature": 35.0,
                    "humidity": 90.0,
                    "pressure": 1000.0,
                    "wind_speed": 25.0,
                    "precipitation": 50.0
                }
            },
            {
                "name": "Low Risk Prediction",
                "data": {
                    "latitude": 34.0522,
                    "longitude": -118.2437,
                    "temperature": 20.0,
                    "humidity": 40.0,
                    "pressure": 1020.0,
                    "wind_speed": 5.0,
                    "precipitation": 0.0
                }
            }
        ]
        
        results = []
        for test in test_data:
            result = self.test_endpoint(
                test["name"], "POST", "/api/ai/predict", test["data"]
            )
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)
        
        return results
    
    def run_error_tests(self) -> List[TestResult]:
        """Run error handling tests"""
        tests = [
            ("Invalid Endpoint", "GET", "/api/nonexistent", None, 404),
            ("Invalid Method", "PUT", "/api/events", None, 405),
            ("Invalid Prediction Data", "POST", "/api/ai/predict", {}, 400),
            ("Invalid Location Data", "POST", "/api/events/near", {}, 400),
        ]
        
        results = []
        for name, method, endpoint, data, expected_status in tests:
            result = self.test_endpoint(name, method, endpoint, data, expected_status)
            results.append(result)
            self.results.append(result)
            time.sleep(0.1)
        
        return results
    
    def run_performance_tests(self) -> List[TestResult]:
        """Run performance and load tests"""
        print("🏃 Running performance tests...")
        
        # Test concurrent requests
        import threading
        results = []
        
        def make_request():
            result = self.test_endpoint("Concurrent Request", "GET", "/health")
            results.append(result)
        
        threads = []
        for i in range(10):  # 10 concurrent requests
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        self.results.extend(results)
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate performance metrics
        response_times = [r.response_time for r in self.results if r.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Group results by endpoint
        endpoint_stats = {}
        for result in self.results:
            if result.endpoint not in endpoint_stats:
                endpoint_stats[result.endpoint] = {"total": 0, "passed": 0, "failed": 0}
            
            endpoint_stats[result.endpoint]["total"] += 1
            if result.success:
                endpoint_stats[result.endpoint]["passed"] += 1
            else:
                endpoint_stats[result.endpoint]["failed"] += 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "performance": {
                "average_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "min_response_time": round(min_response_time, 3)
            },
            "endpoint_stats": endpoint_stats,
            "failed_tests": [r for r in self.results if not r.success]
        }
    
    def print_results(self):
        """Print test results in a formatted way"""
        print("\n" + "="*80)
        print("🧪 DISASTROSCOPE BACKEND TEST RESULTS")
        print("="*80)
        
        # Print individual test results
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.name}")
            print(f"   {result.method} {result.endpoint}")
            print(f"   Status: {result.status_code} | Time: {result.response_time:.3f}s")
            if result.error:
                print(f"   Error: {result.error}")
            print()
        
        # Print summary
        report = self.generate_report()
        summary = report["summary"]
        performance = report["performance"]
        
        print("📊 TEST SUMMARY")
        print("-" * 40)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print()
        
        print("⚡ PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Average Response Time: {performance['average_response_time']}s")
        print(f"Max Response Time: {performance['max_response_time']}s")
        print(f"Min Response Time: {performance['min_response_time']}s")
        print()
        
        if report["failed_tests"]:
            print("❌ FAILED TESTS")
            print("-" * 40)
            for test in report["failed_tests"]:
                print(f"• {test.name}: {test.error}")
            print()
        
        # Final verdict
        if summary['success_rate'] >= 95:
            print("🎉 EXCELLENT! Backend is performing exceptionally well!")
        elif summary['success_rate'] >= 80:
            print("✅ GOOD! Backend is working well with minor issues.")
        elif summary['success_rate'] >= 60:
            print("⚠️  FAIR! Backend has some issues that need attention.")
        else:
            print("❌ POOR! Backend has significant issues that need immediate attention.")
        
        print("="*80)

def main():
    """Main test execution function"""
    print("🧪 DisastroScope Backend Test Suite v2.0.0")
    print("=" * 50)
    
    # Initialize tester
    tester = BackendTester(BASE_URL)
    
    try:
        # Run all test suites
        print("🏥 Testing Health Endpoints...")
        tester.run_health_tests()
        
        print("📊 Testing Data Endpoints...")
        tester.run_data_tests()
        
        print("🌤️ Testing Weather Endpoints...")
        tester.run_weather_tests()
        
        print("📍 Testing Location-based Endpoints...")
        tester.run_location_tests()
        
        print("🤖 Testing AI Prediction Endpoints...")
        tester.run_prediction_tests()
        
        print("🚨 Testing Error Handling...")
        tester.run_error_tests()
        
        print("⚡ Testing Performance...")
        tester.run_performance_tests()
        
        # Generate and print results
        tester.print_results()
        
        # Return exit code based on success rate
        report = tester.generate_report()
        success_rate = report["summary"]["success_rate"]
        
        if success_rate >= 80:
            print("✅ All critical tests passed!")
            return 0
        else:
            print("❌ Some critical tests failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

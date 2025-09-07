#!/usr/bin/env python3
"""
Comprehensive Integration Test Script
Tests all enhanced features: Railway, Tinybird, Firebase, and AI models
"""

import os
import sys
import json
import logging
import requests
import time
from datetime import datetime, timezone
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('integration_test.log')
        ]
    )
    return logging.getLogger(__name__)

class IntegrationTester:
    """Comprehensive integration tester for all enhanced features"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.logger = setup_logging()
        self.test_results = {}
        
    def test_basic_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            self.logger.info("Testing basic API connectivity...")
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                self.logger.info("✅ Basic API connectivity: PASSED")
                self.test_results['basic_connectivity'] = True
                return True
            else:
                self.logger.error(f"❌ Basic API connectivity: FAILED (Status: {response.status_code})")
                self.test_results['basic_connectivity'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Basic API connectivity: FAILED ({e})")
            self.test_results['basic_connectivity'] = False
            return False
    
    def test_integration_status(self) -> bool:
        """Test integration status endpoint"""
        try:
            self.logger.info("Testing integration status...")
            response = requests.get(f"{self.base_url}/api/integrations/status", timeout=10)
            
            if response.status_code == 200:
                status_data = response.json()
                self.logger.info("✅ Integration status endpoint: PASSED")
                self.logger.info(f"   Enhanced AI Available: {status_data.get('enhanced_ai_available', False)}")
                self.logger.info(f"   Enhanced Tinybird Available: {status_data.get('enhanced_tinybird_available', False)}")
                self.logger.info(f"   AI Model Manager Available: {status_data.get('ai_model_manager_available', False)}")
                self.logger.info(f"   Smart Notifications Available: {status_data.get('smart_notifications_available', False)}")
                
                self.test_results['integration_status'] = True
                self.test_results['status_data'] = status_data
                return True
            else:
                self.logger.error(f"❌ Integration status: FAILED (Status: {response.status_code})")
                self.test_results['integration_status'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Integration status: FAILED ({e})")
            self.test_results['integration_status'] = False
            return False
    
    def test_enhanced_predictions(self) -> bool:
        """Test enhanced AI predictions"""
        try:
            self.logger.info("Testing enhanced AI predictions...")
            
            test_data = {
                "user_id": "test_user_123",
                "location_query": "San Francisco, CA",
                "latitude": 37.7749,
                "longitude": -122.4194
            }
            
            response = requests.post(
                f"{self.base_url}/api/ai/predict-enhanced",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                predictions = prediction_data.get('predictions', {})
                metadata = prediction_data.get('metadata', {})
                
                self.logger.info("✅ Enhanced AI predictions: PASSED")
                self.logger.info(f"   Model Type: {metadata.get('model_type', 'unknown')}")
                self.logger.info(f"   Personalized: {metadata.get('personalized', False)}")
                self.logger.info(f"   Prediction ID: {metadata.get('prediction_id', 'none')}")
                
                # Log prediction results
                for disaster_type, risk_level in predictions.items():
                    self.logger.info(f"   {disaster_type.capitalize()}: {risk_level:.3f}")
                
                self.test_results['enhanced_predictions'] = True
                self.test_results['prediction_data'] = prediction_data
                return True
            else:
                self.logger.error(f"❌ Enhanced AI predictions: FAILED (Status: {response.status_code})")
                self.logger.error(f"   Response: {response.text}")
                self.test_results['enhanced_predictions'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced AI predictions: FAILED ({e})")
            self.test_results['enhanced_predictions'] = False
            return False
    
    def test_prediction_feedback(self) -> bool:
        """Test prediction feedback system"""
        try:
            self.logger.info("Testing prediction feedback system...")
            
            # First get a prediction to get a prediction ID
            if 'prediction_data' not in self.test_results:
                self.logger.warning("No prediction data available for feedback test")
                return False
            
            prediction_id = self.test_results['prediction_data']['metadata'].get('prediction_id')
            if not prediction_id:
                self.logger.warning("No prediction ID available for feedback test")
                return False
            
            feedback_data = {
                "prediction_id": prediction_id,
                "user_id": "test_user_123",
                "accuracy_rating": 0.85,
                "feedback": "Prediction was accurate for the location"
            }
            
            response = requests.post(
                f"{self.base_url}/api/ai/feedback",
                json=feedback_data,
                timeout=10
            )
            
            if response.status_code == 200:
                feedback_result = response.json()
                self.logger.info("✅ Prediction feedback: PASSED")
                self.logger.info(f"   Success: {feedback_result.get('success', False)}")
                self.logger.info(f"   Message: {feedback_result.get('message', '')}")
                
                self.test_results['prediction_feedback'] = True
                return True
            else:
                self.logger.error(f"❌ Prediction feedback: FAILED (Status: {response.status_code})")
                self.logger.error(f"   Response: {response.text}")
                self.test_results['prediction_feedback'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Prediction feedback: FAILED ({e})")
            self.test_results['prediction_feedback'] = False
            return False
    
    def test_ai_insights(self) -> bool:
        """Test AI insights endpoint"""
        try:
            self.logger.info("Testing AI insights...")
            response = requests.get(f"{self.base_url}/api/ai/insights", timeout=15)
            
            if response.status_code == 200:
                insights_data = response.json()
                insights = insights_data.get('insights', {})
                
                self.logger.info("✅ AI insights: PASSED")
                self.logger.info(f"   Model Manager Available: {'model_manager' in insights}")
                self.logger.info(f"   Enhanced Models Available: {'enhanced_models' in insights}")
                self.logger.info(f"   Prediction Analytics Available: {'prediction_analytics' in insights}")
                self.logger.info(f"   Notification System Available: {'notification_system' in insights}")
                
                self.test_results['ai_insights'] = True
                self.test_results['insights_data'] = insights_data
                return True
            else:
                self.logger.error(f"❌ AI insights: FAILED (Status: {response.status_code})")
                self.test_results['ai_insights'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ AI insights: FAILED ({e})")
            self.test_results['ai_insights'] = False
            return False
    
    def test_notification_preferences(self) -> bool:
        """Test notification preferences system"""
        try:
            self.logger.info("Testing notification preferences...")
            
            preferences_data = {
                "user_id": "test_user_123",
                "email_enabled": True,
                "push_enabled": True,
                "sms_enabled": False,
                "alert_frequency": "immediate",
                "risk_threshold": 0.6,
                "disaster_types": ["flood", "earthquake", "landslide"],
                "quiet_hours": {"start": "22:00", "end": "07:00"},
                "location_radius": 25.0
            }
            
            response = requests.post(
                f"{self.base_url}/api/notifications/preferences",
                json=preferences_data,
                timeout=10
            )
            
            if response.status_code == 200:
                preferences_result = response.json()
                self.logger.info("✅ Notification preferences: PASSED")
                self.logger.info(f"   Success: {preferences_result.get('success', False)}")
                
                self.test_results['notification_preferences'] = True
                return True
            else:
                self.logger.error(f"❌ Notification preferences: FAILED (Status: {response.status_code})")
                self.logger.error(f"   Response: {response.text}")
                self.test_results['notification_preferences'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Notification preferences: FAILED ({e})")
            self.test_results['notification_preferences'] = False
            return False
    
    def test_user_notifications(self) -> bool:
        """Test user notifications endpoint"""
        try:
            self.logger.info("Testing user notifications...")
            response = requests.get(f"{self.base_url}/api/notifications/user/test_user_123", timeout=10)
            
            if response.status_code == 200:
                notifications_data = response.json()
                notifications = notifications_data.get('notifications', [])
                
                self.logger.info("✅ User notifications: PASSED")
                self.logger.info(f"   Notifications count: {len(notifications)}")
                
                self.test_results['user_notifications'] = True
                return True
            else:
                self.logger.error(f"❌ User notifications: FAILED (Status: {response.status_code})")
                self.test_results['user_notifications'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ User notifications: FAILED ({e})")
            self.test_results['user_notifications'] = False
            return False
    
    def test_user_behavior_analytics(self) -> bool:
        """Test user behavior analytics"""
        try:
            self.logger.info("Testing user behavior analytics...")
            response = requests.get(f"{self.base_url}/api/analytics/user-behavior/test_user_123", timeout=10)
            
            if response.status_code == 200:
                analytics_data = response.json()
                self.logger.info("✅ User behavior analytics: PASSED")
                self.logger.info(f"   User ID: {analytics_data.get('user_id', 'unknown')}")
                self.logger.info(f"   Behavior Profile Available: {analytics_data.get('behavior_profile') is not None}")
                self.logger.info(f"   Prediction Analytics Available: {analytics_data.get('prediction_analytics') is not None}")
                
                self.test_results['user_behavior_analytics'] = True
                return True
            else:
                self.logger.error(f"❌ User behavior analytics: FAILED (Status: {response.status_code})")
                self.test_results['user_behavior_analytics'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ User behavior analytics: FAILED ({e})")
            self.test_results['user_behavior_analytics'] = False
            return False
    
    def test_risk_trend_analytics(self) -> bool:
        """Test risk trend analytics"""
        try:
            self.logger.info("Testing risk trend analytics...")
            
            params = {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "days": 30
            }
            
            response = requests.get(
                f"{self.base_url}/api/analytics/risk-trends",
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                trends_data = response.json()
                self.logger.info("✅ Risk trend analytics: PASSED")
                self.logger.info(f"   Location: {trends_data.get('location', {})}")
                self.logger.info(f"   Analysis Period: {trends_data.get('analysis_period_days', 0)} days")
                self.logger.info(f"   Risk Trends Available: {trends_data.get('risk_trends') is not None}")
                self.logger.info(f"   Historical Events: {len(trends_data.get('historical_events', []))}")
                
                self.test_results['risk_trend_analytics'] = True
                return True
            else:
                self.logger.error(f"❌ Risk trend analytics: FAILED (Status: {response.status_code})")
                self.test_results['risk_trend_analytics'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Risk trend analytics: FAILED ({e})")
            self.test_results['risk_trend_analytics'] = False
            return False
    
    def test_global_risk_analysis(self) -> bool:
        """Test global risk analysis endpoint"""
        try:
            self.logger.info("Testing global risk analysis...")
            
            test_data = {
                "location_query": "Tokyo, Japan",
                "latitude": 35.6762,
                "longitude": 139.6503
            }
            
            response = requests.post(
                f"{self.base_url}/api/global-risk-analysis",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                analysis_data = response.json()
                disasters = analysis_data.get('disasters', {})
                
                self.logger.info("✅ Global risk analysis: PASSED")
                self.logger.info(f"   Location: {analysis_data.get('location', {})}")
                self.logger.info(f"   Disasters analyzed: {len(disasters)}")
                
                for disaster_type, disaster_data in disasters.items():
                    risk_level = disaster_data.get('risk_level', 0)
                    confidence = disaster_data.get('confidence', 0)
                    self.logger.info(f"   {disaster_type}: Risk {risk_level:.3f}, Confidence {confidence:.3f}")
                
                self.test_results['global_risk_analysis'] = True
                return True
            else:
                self.logger.error(f"❌ Global risk analysis: FAILED (Status: {response.status_code})")
                self.test_results['global_risk_analysis'] = False
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Global risk analysis: FAILED ({e})")
            self.test_results['global_risk_analysis'] = False
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE INTEGRATION TESTS")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            ("Basic Connectivity", self.test_basic_connectivity),
            ("Integration Status", self.test_integration_status),
            ("Enhanced Predictions", self.test_enhanced_predictions),
            ("Prediction Feedback", self.test_prediction_feedback),
            ("AI Insights", self.test_ai_insights),
            ("Notification Preferences", self.test_notification_preferences),
            ("User Notifications", self.test_user_notifications),
            ("User Behavior Analytics", self.test_user_behavior_analytics),
            ("Risk Trend Analytics", self.test_risk_trend_analytics),
            ("Global Risk Analysis", self.test_global_risk_analysis)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            self.logger.info(f"\n--- Running {test_name} Test ---")
            try:
                if test_func():
                    passed_tests += 1
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.logger.error(f"❌ {test_name} test crashed: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("INTEGRATION TEST SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        self.logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        self.logger.info(f"Total Duration: {duration:.2f} seconds")
        
        # Detailed results
        self.logger.info("\nDetailed Results:")
        for test_name, test_func in tests:
            test_key = test_name.lower().replace(" ", "_")
            status = "✅ PASSED" if self.test_results.get(test_key, False) else "❌ FAILED"
            self.logger.info(f"  {test_name}: {status}")
        
        # Save results to file
        results_file = "integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'passed_tests': passed_tests,
                    'total_tests': total_tests,
                    'success_rate': (passed_tests/total_tests)*100,
                    'duration_seconds': duration,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        self.logger.info(f"\nDetailed results saved to {results_file}")
        
        return {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'duration': duration,
            'results': self.test_results
        }

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive integration tests')
    parser.add_argument('--url', default='http://localhost:5000', help='Base URL for the API')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = IntegrationTester(args.url)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['success_rate'] >= 80:
        print(f"\n🎉 Integration tests completed successfully! ({results['success_rate']:.1f}% pass rate)")
        sys.exit(0)
    else:
        print(f"\n⚠️  Integration tests completed with issues ({results['success_rate']:.1f}% pass rate)")
        sys.exit(1)

if __name__ == "__main__":
    main()

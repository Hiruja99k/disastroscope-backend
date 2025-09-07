"""
Enhanced Tinybird Service for Real-Time AI Model Integration
Advanced data streaming, analytics, and machine learning features
"""

import os
import json
import logging
import requests
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class PredictionEvent:
    """Enhanced prediction event with full context"""
    id: str
    user_id: str
    event_type: str
    latitude: float
    longitude: float
    probability: float
    confidence: float
    model_version: str
    location_name: str
    weather_data: Dict[str, Any]
    geospatial_data: Dict[str, Any]
    timestamp: str
    prediction_accuracy: Optional[float] = None
    user_feedback: Optional[str] = None
    actual_outcome: Optional[bool] = None

@dataclass
class UserBehavior:
    """User behavior tracking for personalized models"""
    user_id: str
    location_preferences: List[Tuple[float, float]]
    risk_sensitivity: float
    alert_frequency: str
    interaction_patterns: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    last_updated: str

@dataclass
class HistoricalEvent:
    """Historical disaster event for model training"""
    id: str
    event_type: str
    severity: float
    latitude: float
    longitude: float
    timestamp: str
    description: str
    source: str
    weather_conditions: Dict[str, Any]
    geospatial_context: Dict[str, Any]
    casualties: Optional[int] = None
    economic_impact: Optional[float] = None
    verified: bool = False

class EnhancedTinybirdService:
    """Enhanced Tinybird service with AI model integration"""
    
    def __init__(self):
        self.base_url = os.getenv('TINYBIRD_API_URL', 'https://cloud.tinybird.co/gcp/europe-west3/DisastroScope')
        self.token = os.getenv('TINYBIRD_TOKEN', '')
        self.workspace_id = os.getenv('TINYBIRD_WORKSPACE_ID', '')
        self.initialized = bool(self.token and self.workspace_id)
        
        # Data source names
        self.data_sources = {
            'predictions': 'disaster_predictions',
            'user_behavior': 'user_behavior',
            'historical_events': 'historical_events',
            'weather_stream': 'weather_stream',
            'user_feedback': 'user_feedback',
            'model_performance': 'model_performance',
            'real_time_metrics': 'real_time_metrics'
        }
        
        if not self.initialized:
            logger.warning("Enhanced Tinybird not configured. Advanced features will be disabled.")
        else:
            logger.info("Enhanced Tinybird service initialized with AI integration")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Tinybird API requests"""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to Tinybird API with enhanced error handling"""
        if not self.initialized:
            logger.warning("Enhanced Tinybird not initialized. Request skipped.")
            return None
        
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data, timeout=30)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Enhanced Tinybird API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Enhanced Tinybird request error: {e}")
            return None
    
    # Enhanced Prediction Tracking
    def log_prediction_event(self, prediction: PredictionEvent) -> bool:
        """Log detailed prediction event for model improvement"""
        event_data = {
            'name': 'disaster_prediction_enhanced',
            'data': asdict(prediction)
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        if result:
            logger.info(f"Logged enhanced prediction event: {prediction.id}")
            return True
        return False
    
    def update_prediction_accuracy(self, prediction_id: str, accuracy: float, user_feedback: str = None) -> bool:
        """Update prediction accuracy for model learning"""
        update_data = {
            'name': 'prediction_accuracy_update',
            'data': {
                'prediction_id': prediction_id,
                'accuracy': accuracy,
                'user_feedback': user_feedback,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self._make_request('POST', '/v0/events', update_data)
        return result is not None
    
    def get_prediction_analytics(self, user_id: str = None, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive prediction analytics"""
        params = {'days': days}
        if user_id:
            params['user_id'] = user_id
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/prediction_analytics.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    # User Behavior Learning
    def track_user_behavior(self, behavior: UserBehavior) -> bool:
        """Track user behavior for personalized models"""
        event_data = {
            'name': 'user_behavior_update',
            'data': asdict(behavior)
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        return result is not None
    
    def get_user_behavior_profile(self, user_id: str) -> Optional[UserBehavior]:
        """Get user behavior profile for personalized predictions"""
        endpoint = f"/v0/pipes/user_behavior_profile.json?user_id={user_id}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result and result['data']:
            data = result['data'][0]
            return UserBehavior(**data)
        return None
    
    def update_user_risk_sensitivity(self, user_id: str, sensitivity: float) -> bool:
        """Update user's risk sensitivity based on feedback"""
        update_data = {
            'name': 'risk_sensitivity_update',
            'data': {
                'user_id': user_id,
                'risk_sensitivity': sensitivity,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self._make_request('POST', '/v0/events', update_data)
        return result is not None
    
    # Historical Event Management
    def log_historical_event(self, event: HistoricalEvent) -> bool:
        """Log historical disaster event for model training"""
        event_data = {
            'name': 'historical_disaster_event',
            'data': asdict(event)
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        if result:
            logger.info(f"Logged historical event: {event.id}")
            return True
        return False
    
    def get_historical_events(self, event_type: str = None, days: int = 365) -> List[HistoricalEvent]:
        """Get historical events for model training"""
        params = {'days': days}
        if event_type:
            params['event_type'] = event_type
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/historical_events.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return [HistoricalEvent(**event) for event in result['data']]
        return []
    
    def get_location_risk_history(self, latitude: float, longitude: float, radius_km: float = 50) -> List[HistoricalEvent]:
        """Get historical events near a specific location"""
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'radius_km': radius_km
        }
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/location_risk_history.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return [HistoricalEvent(**event) for event in result['data']]
        return []
    
    # Real-Time Weather Streams
    def stream_weather_data(self, weather_data: Dict[str, Any]) -> bool:
        """Stream real-time weather data for model updates"""
        event_data = {
            'name': 'weather_stream',
            'data': {
                **weather_data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        return result is not None
    
    def get_weather_trends(self, location: Tuple[float, float], hours: int = 24) -> Dict[str, Any]:
        """Get weather trends for a location"""
        params = {
            'latitude': location[0],
            'longitude': location[1],
            'hours': hours
        }
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/weather_trends.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    # User Feedback and Crowdsourcing
    def log_user_feedback(self, user_id: str, prediction_id: str, feedback: str, accuracy_rating: float) -> bool:
        """Log user feedback for model improvement"""
        feedback_data = {
            'name': 'user_feedback',
            'data': {
                'user_id': user_id,
                'prediction_id': prediction_id,
                'feedback': feedback,
                'accuracy_rating': accuracy_rating,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self._make_request('POST', '/v0/events', feedback_data)
        return result is not None
    
    def get_user_feedback_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get user feedback analytics for model improvement"""
        endpoint = f"/v0/pipes/user_feedback_analytics.json?days={days}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    def get_community_risk_perception(self, location: Tuple[float, float], radius_km: float = 10) -> Dict[str, Any]:
        """Get community risk perception for a location"""
        params = {
            'latitude': location[0],
            'longitude': location[1],
            'radius_km': radius_km
        }
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/community_risk_perception.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    # Model Performance Monitoring
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], timestamp: str = None) -> bool:
        """Log model performance metrics"""
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        performance_data = {
            'name': 'model_performance',
            'data': {
                'model_name': model_name,
                'timestamp': timestamp,
                **metrics
            }
        }
        
        result = self._make_request('POST', '/v0/events', performance_data)
        return result is not None
    
    def get_model_performance_trends(self, model_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get model performance trends over time"""
        params = {
            'model_name': model_name,
            'days': days
        }
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/model_performance_trends.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data']
        return []
    
    def get_retraining_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for model retraining"""
        endpoint = "/v0/pipes/retraining_recommendations.json"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    # Advanced Analytics
    def get_risk_trend_analysis(self, location: Tuple[float, float], days: int = 90) -> Dict[str, Any]:
        """Get comprehensive risk trend analysis for a location"""
        params = {
            'latitude': location[0],
            'longitude': location[1],
            'days': days
        }
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/risk_trend_analysis.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    def get_predictive_insights(self, user_id: str = None) -> Dict[str, Any]:
        """Get predictive insights for users or system-wide"""
        params = {}
        if user_id:
            params['user_id'] = user_id
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/predictive_insights.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics"""
        endpoint = "/v0/pipes/system_health_metrics.json"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    # Real-Time Streaming
    async def subscribe_to_weather_updates(self, callback, location: Tuple[float, float] = None):
        """Subscribe to real-time weather updates"""
        # This would typically use WebSockets or Server-Sent Events
        # For now, we'll implement a polling mechanism
        while True:
            try:
                if location:
                    weather_data = self.get_weather_trends(location, hours=1)
                else:
                    weather_data = self.get_realtime_metrics()
                
                if weather_data:
                    await callback(weather_data)
                
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Weather subscription error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def subscribe_to_risk_alerts(self, callback, user_id: str = None):
        """Subscribe to real-time risk alerts"""
        while True:
            try:
                # Get latest risk predictions
                if user_id:
                    analytics = self.get_prediction_analytics(user_id, days=1)
                else:
                    analytics = self.get_prediction_analytics(days=1)
                
                if analytics and analytics.get('high_risk_events', 0) > 0:
                    await callback(analytics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Risk alert subscription error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    # Data Export for Model Training
    def export_training_data(self, event_type: str = None, days: int = 365) -> pd.DataFrame:
        """Export training data for model retraining"""
        events = self.get_historical_events(event_type, days)
        
        if not events:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for event in events:
            row = {
                'event_type': event.event_type,
                'severity': event.severity,
                'latitude': event.latitude,
                'longitude': event.longitude,
                'timestamp': event.timestamp,
                'verified': event.verified,
                **event.weather_conditions,
                **event.geospatial_context
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_user_behavior_data(self, days: int = 90) -> pd.DataFrame:
        """Export user behavior data for personalization"""
        # This would query user behavior data
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with detailed status"""
        if not self.initialized:
            return {
                'status': 'disabled',
                'message': 'Enhanced Tinybird not configured',
                'initialized': False,
                'features': {
                    'predictions': False,
                    'user_behavior': False,
                    'historical_events': False,
                    'weather_streams': False,
                    'feedback': False,
                    'performance_monitoring': False
                }
            }
        
        try:
            # Test basic connectivity
            result = self._make_request('GET', '/v0/workspaces')
            
            if result:
                return {
                    'status': 'healthy',
                    'message': 'Enhanced Tinybird service is operational',
                    'initialized': True,
                    'workspace_id': self.workspace_id,
                    'features': {
                        'predictions': True,
                        'user_behavior': True,
                        'historical_events': True,
                        'weather_streams': True,
                        'feedback': True,
                        'performance_monitoring': True
                    },
                    'data_sources': self.data_sources
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Enhanced Tinybird API request failed',
                    'initialized': True,
                    'features': {
                        'predictions': False,
                        'user_behavior': False,
                        'historical_events': False,
                        'weather_streams': False,
                        'feedback': False,
                        'performance_monitoring': False
                    }
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Enhanced Tinybird health check failed: {str(e)}',
                'initialized': True,
                'features': {
                    'predictions': False,
                    'user_behavior': False,
                    'historical_events': False,
                    'weather_streams': False,
                    'feedback': False,
                    'performance_monitoring': False
                }
            }
    
    def is_initialized(self) -> bool:
        """Check if Enhanced Tinybird is initialized"""
        return self.initialized

# Global Enhanced Tinybird service instance
enhanced_tinybird_service = EnhancedTinybirdService()

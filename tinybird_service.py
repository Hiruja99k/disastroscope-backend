"""
Tinybird API Integration for DisastroScope Backend
Handles real-time data streaming and analytics
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class TinybirdService:
    """Tinybird API service for real-time data and analytics"""
    
    def __init__(self):
        self.base_url = os.getenv('TINYBIRD_API_URL', 'https://api.tinybird.co')
        self.token = os.getenv('TINYBIRD_TOKEN', '')
        self.workspace_id = os.getenv('TINYBIRD_WORKSPACE_ID', '')
        self.initialized = bool(self.token and self.workspace_id)
        
        if not self.initialized:
            logger.warning("Tinybird not configured. Tinybird features will be disabled.")
        else:
            logger.info("Tinybird service initialized")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Tinybird API requests"""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to Tinybird API"""
        if not self.initialized:
            logger.warning("Tinybird not initialized. Request skipped.")
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
            logger.error(f"Tinybird API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Tinybird request error: {e}")
            return None
    
    # User Management Methods
    def create_user_event(self, user_data: Dict[str, Any]) -> bool:
        """Create a user event in Tinybird"""
        event_data = {
            'name': 'user_created',
            'data': {
                'uid': user_data.get('uid'),
                'email': user_data.get('email'),
                'display_name': user_data.get('display_name', ''),
                'photo_url': user_data.get('photo_url', ''),
                'email_verified': user_data.get('email_verified', False),
                'created_at': user_data.get('created_at', datetime.now(timezone.utc).isoformat()),
                'last_login_at': user_data.get('last_login_at', datetime.now(timezone.utc).isoformat()),
                'theme': user_data.get('theme', 'light'),
                'notifications': user_data.get('notifications', True),
                'language': user_data.get('language', 'en')
            }
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        return result is not None
    
    def update_user_event(self, uid: str, updates: Dict[str, Any]) -> bool:
        """Update user event in Tinybird"""
        event_data = {
            'name': 'user_updated',
            'data': {
                'uid': uid,
                'updated_at': datetime.now(timezone.utc).isoformat(),
                **updates
            }
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        return result is not None
    
    def track_user_login(self, uid: str, email: str) -> bool:
        """Track user login event"""
        event_data = {
            'name': 'user_login',
            'data': {
                'uid': uid,
                'email': email,
                'login_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        return result is not None
    
    def delete_user_event(self, uid: str) -> bool:
        """Delete user event in Tinybird"""
        event_data = {
            'name': 'user_deleted',
            'data': {
                'uid': uid,
                'deleted_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
        result = self._make_request('POST', '/v0/events', event_data)
        return result is not None
    
    # Disaster Data Methods
    def create_disaster_event(self, event_data: Dict[str, Any]) -> bool:
        """Create a disaster event in Tinybird"""
        event = {
            'name': 'disaster_event',
            'data': {
                'id': event_data.get('id'),
                'type': event_data.get('type'),
                'severity': event_data.get('severity'),
                'latitude': event_data.get('latitude'),
                'longitude': event_data.get('longitude'),
                'timestamp': event_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'description': event_data.get('description', ''),
                'user_id': event_data.get('user_id', ''),
                'magnitude': event_data.get('magnitude', 0),
                'source': event_data.get('source', 'api'),
                'confidence': event_data.get('confidence', 0.5)
            }
        }
        
        result = self._make_request('POST', '/v0/events', event)
        return result is not None
    
    def get_disaster_events(self, limit: int = 100, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get disaster events from Tinybird"""
        params = {'limit': limit}
        if filters:
            params.update(filters)
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/disaster_events.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data']
        return []
    
    def get_user_analytics(self, uid: str) -> Dict[str, Any]:
        """Get user analytics from Tinybird"""
        endpoint = f"/v0/pipes/user_analytics.json?uid={uid}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    def get_disaster_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get disaster analytics from Tinybird"""
        params = {}
        if filters:
            params.update(filters)
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = f"/v0/pipes/disaster_analytics.json?{query_params}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    # Real-time Data Methods
    def create_prediction_event(self, prediction_data: Dict[str, Any]) -> bool:
        """Create a prediction event in Tinybird"""
        event = {
            'name': 'disaster_prediction',
            'data': {
                'id': prediction_data.get('id'),
                'event_type': prediction_data.get('event_type'),
                'latitude': prediction_data.get('latitude'),
                'longitude': prediction_data.get('longitude'),
                'probability': prediction_data.get('probability'),
                'confidence': prediction_data.get('confidence'),
                'timestamp': prediction_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'model_version': prediction_data.get('model_version', '2.0.0'),
                'user_id': prediction_data.get('user_id', ''),
                'location_name': prediction_data.get('location_name', '')
            }
        }
        
        result = self._make_request('POST', '/v0/events', event)
        return result is not None
    
    def create_weather_event(self, weather_data: Dict[str, Any]) -> bool:
        """Create a weather event in Tinybird"""
        event = {
            'name': 'weather_data',
            'data': {
                'location': weather_data.get('location'),
                'latitude': weather_data.get('latitude'),
                'longitude': weather_data.get('longitude'),
                'temperature': weather_data.get('temperature'),
                'humidity': weather_data.get('humidity'),
                'pressure': weather_data.get('pressure'),
                'wind_speed': weather_data.get('wind_speed'),
                'precipitation': weather_data.get('precipitation'),
                'timestamp': weather_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'user_id': weather_data.get('user_id', '')
            }
        }
        
        result = self._make_request('POST', '/v0/events', event)
        return result is not None
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics from Tinybird"""
        endpoint = "/v0/pipes/realtime_metrics.json"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data'][0] if result['data'] else {}
        return {}
    
    def get_user_activity(self, uid: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get user activity from Tinybird"""
        endpoint = f"/v0/pipes/user_activity.json?uid={uid}&days={days}"
        
        result = self._make_request('GET', endpoint)
        if result and 'data' in result:
            return result['data']
        return []
    
    # Health Check
    def health_check(self) -> Dict[str, Any]:
        """Check Tinybird service health"""
        if not self.initialized:
            return {
                'status': 'disabled',
                'message': 'Tinybird not configured',
                'initialized': False
            }
        
        try:
            # Try to make a simple request to check connectivity
            result = self._make_request('GET', '/v0/workspaces')
            
            if result:
                return {
                    'status': 'healthy',
                    'message': 'Tinybird service is operational',
                    'initialized': True,
                    'workspace_id': self.workspace_id
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Tinybird API request failed',
                    'initialized': True
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Tinybird health check failed: {str(e)}',
                'initialized': True
            }
    
    def is_initialized(self) -> bool:
        """Check if Tinybird is initialized"""
        return self.initialized

# Global Tinybird service instance
tinybird_service = TinybirdService()

"""
Smart Notification System with Behavioral Triggers
Intelligent alerting based on user behavior, risk patterns, and real-time data
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time

from enhanced_tinybird_service import enhanced_tinybird_service
from ai_model_manager import ai_model_manager

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationType(Enum):
    """Types of notifications"""
    RISK_ALERT = "risk_alert"
    WEATHER_UPDATE = "weather_update"
    SAFETY_TIP = "safety_tip"
    EVACUATION_WARNING = "evacuation_warning"
    SYSTEM_UPDATE = "system_update"
    PERSONALIZED_INSIGHT = "personalized_insight"

@dataclass
class NotificationRule:
    """Notification rule configuration"""
    id: str
    user_id: str
    disaster_type: str
    alert_level: AlertLevel
    conditions: Dict[str, Any]
    cooldown_minutes: int
    enabled: bool
    created_at: str
    last_triggered: Optional[str] = None

@dataclass
class SmartNotification:
    """Smart notification with context and personalization"""
    id: str
    user_id: str
    type: NotificationType
    alert_level: AlertLevel
    title: str
    message: str
    disaster_type: str
    location: Dict[str, float]
    risk_level: float
    confidence: float
    personalized_content: Dict[str, Any]
    action_required: bool
    timestamp: str
    expires_at: Optional[str] = None

@dataclass
class UserNotificationPreferences:
    """User notification preferences"""
    user_id: str
    email_enabled: bool
    push_enabled: bool
    sms_enabled: bool
    alert_frequency: str  # immediate, hourly, daily
    risk_threshold: float
    disaster_types: List[str]
    quiet_hours: Dict[str, str]  # start_time, end_time
    location_radius: float  # km
    last_updated: str

class SmartNotificationSystem:
    """Advanced notification system with behavioral intelligence"""
    
    def __init__(self):
        self.notification_rules = {}
        self.user_preferences = {}
        self.active_notifications = {}
        self.notification_queue = []
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Notification templates
        self.templates = self._load_notification_templates()
        
        # Start monitoring
        self._start_monitoring()
    
    def _load_notification_templates(self) -> Dict[str, Dict[str, str]]:
        """Load notification templates for different scenarios"""
        return {
            'flood_high_risk': {
                'title': '🌊 High Flood Risk Alert',
                'message': 'Flood risk is elevated in your area. Consider moving to higher ground and avoid flood-prone areas.',
                'action_required': True
            },
            'earthquake_high_risk': {
                'title': '🌍 Earthquake Risk Alert',
                'message': 'Seismic activity risk is high. Secure loose items and identify safe areas in your building.',
                'action_required': True
            },
            'landslide_high_risk': {
                'title': '🏔️ Landslide Risk Alert',
                'message': 'Landslide risk is elevated due to weather conditions. Avoid steep slopes and unstable terrain.',
                'action_required': True
            },
            'weather_update': {
                'title': '🌤️ Weather Update',
                'message': 'Weather conditions are changing in your area. Stay informed about potential risks.',
                'action_required': False
            },
            'safety_tip': {
                'title': '💡 Safety Tip',
                'message': 'Based on current conditions, here are some safety recommendations for your area.',
                'action_required': False
            },
            'evacuation_warning': {
                'title': '🚨 EVACUATION WARNING',
                'message': 'IMMEDIATE EVACUATION RECOMMENDED. Leave the area immediately and follow emergency routes.',
                'action_required': True
            }
        }
    
    def _start_monitoring(self):
        """Start the notification monitoring thread"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.is_monitoring = True
            logger.info("Smart notification monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop for notifications"""
        while self.is_monitoring:
            try:
                # Check for new risk alerts
                self._check_risk_alerts()
                
                # Process notification queue
                self._process_notification_queue()
                
                # Update user preferences
                self._update_user_preferences()
                
                # Clean up expired notifications
                self._cleanup_expired_notifications()
                
                # Sleep for 30 seconds before next check
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in notification monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _check_risk_alerts(self):
        """Check for new risk alerts that require notifications"""
        try:
            # Get recent high-risk predictions
            risk_analytics = enhanced_tinybird_service.get_prediction_analytics(days=1)
            
            if risk_analytics and risk_analytics.get('high_risk_events', 0) > 0:
                # Process high-risk events
                self._process_high_risk_events(risk_analytics)
                
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
    
    def _process_high_risk_events(self, risk_analytics: Dict[str, Any]):
        """Process high-risk events and generate notifications"""
        try:
            # Get users in affected areas
            affected_users = self._get_affected_users(risk_analytics)
            
            for user_id in affected_users:
                # Check if user should receive notification
                if self._should_notify_user(user_id, risk_analytics):
                    # Generate personalized notification
                    notification = self._generate_personalized_notification(user_id, risk_analytics)
                    
                    if notification:
                        self._queue_notification(notification)
                        
        except Exception as e:
            logger.error(f"Error processing high-risk events: {e}")
    
    def _get_affected_users(self, risk_analytics: Dict[str, Any]) -> List[str]:
        """Get users in affected areas"""
        try:
            # This would typically query user locations from Tinybird
            # For now, return a mock list
            return ['user_1', 'user_2', 'user_3']
            
        except Exception as e:
            logger.error(f"Error getting affected users: {e}")
            return []
    
    def _should_notify_user(self, user_id: str, risk_data: Dict[str, Any]) -> bool:
        """Check if user should receive notification based on preferences and rules"""
        try:
            # Get user preferences
            preferences = self._get_user_preferences(user_id)
            if not preferences:
                return True  # Default to notify if no preferences
            
            # Check if notifications are enabled
            if not (preferences.email_enabled or preferences.push_enabled or preferences.sms_enabled):
                return False
            
            # Check risk threshold
            max_risk = max(risk_data.get('risk_levels', {}).values()) if risk_data.get('risk_levels') else 0.0
            if max_risk < preferences.risk_threshold:
                return False
            
            # Check quiet hours
            if self._is_quiet_hours(preferences.quiet_hours):
                return max_risk > 0.8  # Only notify for critical risks during quiet hours
            
            # Check cooldown for this user
            if self._is_in_cooldown(user_id, risk_data.get('disaster_type', 'general')):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification eligibility: {e}")
            return True  # Default to notify on error
    
    def _get_user_preferences(self, user_id: str) -> Optional[UserNotificationPreferences]:
        """Get user notification preferences"""
        try:
            if user_id in self.user_preferences:
                return self.user_preferences[user_id]
            
            # Try to get from Tinybird
            user_behavior = enhanced_tinybird_service.get_user_behavior_profile(user_id)
            if user_behavior:
                preferences = UserNotificationPreferences(
                    user_id=user_id,
                    email_enabled=True,
                    push_enabled=True,
                    sms_enabled=False,
                    alert_frequency='immediate',
                    risk_threshold=0.5,
                    disaster_types=['flood', 'earthquake', 'landslide'],
                    quiet_hours={'start': '22:00', 'end': '07:00'},
                    location_radius=50.0,
                    last_updated=datetime.now(timezone.utc).isoformat()
                )
                
                self.user_preferences[user_id] = preferences
                return preferences
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return None
    
    def _is_quiet_hours(self, quiet_hours: Dict[str, str]) -> bool:
        """Check if current time is within quiet hours"""
        try:
            if not quiet_hours:
                return False
            
            now = datetime.now(timezone.utc)
            current_time = now.strftime('%H:%M')
            
            start_time = quiet_hours.get('start', '22:00')
            end_time = quiet_hours.get('end', '07:00')
            
            # Handle overnight quiet hours
            if start_time > end_time:
                return current_time >= start_time or current_time <= end_time
            else:
                return start_time <= current_time <= end_time
                
        except Exception as e:
            logger.error(f"Error checking quiet hours: {e}")
            return False
    
    def _is_in_cooldown(self, user_id: str, disaster_type: str) -> bool:
        """Check if user is in notification cooldown"""
        try:
            # Check notification rules for cooldown
            for rule_id, rule in self.notification_rules.items():
                if rule.user_id == user_id and rule.disaster_type == disaster_type:
                    if rule.last_triggered:
                        last_triggered = datetime.fromisoformat(rule.last_triggered.replace('Z', '+00:00'))
                        cooldown_end = last_triggered + timedelta(minutes=rule.cooldown_minutes)
                        
                        if datetime.now(timezone.utc) < cooldown_end:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    def _generate_personalized_notification(self, user_id: str, risk_data: Dict[str, Any]) -> Optional[SmartNotification]:
        """Generate personalized notification for user"""
        try:
            # Get user behavior for personalization
            user_behavior = enhanced_tinybird_service.get_user_behavior_profile(user_id)
            
            # Determine disaster type and risk level
            disaster_type = risk_data.get('primary_disaster_type', 'general')
            risk_level = risk_data.get('max_risk_level', 0.5)
            
            # Determine alert level
            if risk_level >= 0.9:
                alert_level = AlertLevel.CRITICAL
            elif risk_level >= 0.7:
                alert_level = AlertLevel.HIGH
            elif risk_level >= 0.5:
                alert_level = AlertLevel.MEDIUM
            else:
                alert_level = AlertLevel.LOW
            
            # Get template
            template_key = f"{disaster_type}_{alert_level.value}_risk"
            if template_key not in self.templates:
                template_key = 'weather_update'
            
            template = self.templates[template_key]
            
            # Personalize content
            personalized_content = self._personalize_content(user_behavior, risk_data, template)
            
            # Create notification
            notification = SmartNotification(
                id=f"notif_{int(time.time())}_{user_id}",
                user_id=user_id,
                type=NotificationType.RISK_ALERT,
                alert_level=alert_level,
                title=personalized_content['title'],
                message=personalized_content['message'],
                disaster_type=disaster_type,
                location=risk_data.get('location', {'latitude': 0.0, 'longitude': 0.0}),
                risk_level=risk_level,
                confidence=risk_data.get('confidence', 0.85),
                personalized_content=personalized_content,
                action_required=template.get('action_required', False),
                timestamp=datetime.now(timezone.utc).isoformat(),
                expires_at=(datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
            )
            
            return notification
            
        except Exception as e:
            logger.error(f"Error generating personalized notification: {e}")
            return None
    
    def _personalize_content(self, user_behavior: Optional[Any], risk_data: Dict[str, Any], template: Dict[str, str]) -> Dict[str, str]:
        """Personalize notification content based on user behavior"""
        try:
            personalized_content = template.copy()
            
            if user_behavior:
                # Adjust message based on user's risk sensitivity
                risk_sensitivity = getattr(user_behavior, 'risk_sensitivity', 1.0)
                
                if risk_sensitivity > 1.2:
                    # High sensitivity user - emphasize urgency
                    personalized_content['title'] = f"⚠️ URGENT: {personalized_content['title']}"
                    personalized_content['message'] = f"🚨 {personalized_content['message']} Take immediate action!"
                elif risk_sensitivity < 0.8:
                    # Low sensitivity user - more measured tone
                    personalized_content['message'] = f"ℹ️ {personalized_content['message']} Please stay informed."
            
            # Add location-specific information
            location = risk_data.get('location_name', 'your area')
            personalized_content['message'] = personalized_content['message'].replace('your area', location)
            
            # Add time-sensitive information
            current_time = datetime.now(timezone.utc).strftime('%H:%M UTC')
            personalized_content['message'] += f"\n\n⏰ Alert generated at {current_time}"
            
            return personalized_content
            
        except Exception as e:
            logger.error(f"Error personalizing content: {e}")
            return template
    
    def _queue_notification(self, notification: SmartNotification):
        """Queue notification for delivery"""
        try:
            self.notification_queue.append(notification)
            self.active_notifications[notification.id] = notification
            
            logger.info(f"Queued notification {notification.id} for user {notification.user_id}")
            
        except Exception as e:
            logger.error(f"Error queuing notification: {e}")
    
    def _process_notification_queue(self):
        """Process queued notifications"""
        try:
            while self.notification_queue:
                notification = self.notification_queue.pop(0)
                self._deliver_notification(notification)
                
        except Exception as e:
            logger.error(f"Error processing notification queue: {e}")
    
    def _deliver_notification(self, notification: SmartNotification):
        """Deliver notification to user"""
        try:
            # Get user preferences
            preferences = self._get_user_preferences(notification.user_id)
            
            if not preferences:
                logger.warning(f"No preferences found for user {notification.user_id}")
                return
            
            # Deliver via enabled channels
            if preferences.email_enabled:
                self._send_email_notification(notification)
            
            if preferences.push_enabled:
                self._send_push_notification(notification)
            
            if preferences.sms_enabled:
                self._send_sms_notification(notification)
            
            # Log notification delivery
            self._log_notification_delivery(notification)
            
            # Update notification rules cooldown
            self._update_notification_cooldown(notification)
            
            logger.info(f"Delivered notification {notification.id} to user {notification.user_id}")
            
        except Exception as e:
            logger.error(f"Error delivering notification: {e}")
    
    def _send_email_notification(self, notification: SmartNotification):
        """Send email notification"""
        try:
            # This would integrate with an email service
            logger.info(f"Email notification sent: {notification.title}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_push_notification(self, notification: SmartNotification):
        """Send push notification"""
        try:
            # This would integrate with Firebase Cloud Messaging
            logger.info(f"Push notification sent: {notification.title}")
            
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
    
    def _send_sms_notification(self, notification: SmartNotification):
        """Send SMS notification"""
        try:
            # This would integrate with an SMS service
            logger.info(f"SMS notification sent: {notification.title}")
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    def _log_notification_delivery(self, notification: SmartNotification):
        """Log notification delivery to Tinybird"""
        try:
            delivery_data = {
                'notification_id': notification.id,
                'user_id': notification.user_id,
                'type': notification.type.value,
                'alert_level': notification.alert_level.value,
                'disaster_type': notification.disaster_type,
                'delivered_at': datetime.now(timezone.utc).isoformat(),
                'action_required': notification.action_required
            }
            
            # Log to Tinybird
            enhanced_tinybird_service._make_request('POST', '/v0/events', {
                'name': 'notification_delivery',
                'data': delivery_data
            })
            
        except Exception as e:
            logger.error(f"Error logging notification delivery: {e}")
    
    def _update_notification_cooldown(self, notification: SmartNotification):
        """Update notification cooldown for user"""
        try:
            # Update cooldown for this disaster type
            for rule_id, rule in self.notification_rules.items():
                if rule.user_id == notification.user_id and rule.disaster_type == notification.disaster_type:
                    rule.last_triggered = notification.timestamp
                    break
            
        except Exception as e:
            logger.error(f"Error updating notification cooldown: {e}")
    
    def _update_user_preferences(self):
        """Update user preferences from Tinybird"""
        try:
            # This would periodically sync user preferences from Tinybird
            # For now, we'll implement a basic version
            pass
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    def _cleanup_expired_notifications(self):
        """Clean up expired notifications"""
        try:
            current_time = datetime.now(timezone.utc)
            expired_notifications = []
            
            for notification_id, notification in self.active_notifications.items():
                if notification.expires_at:
                    expires_at = datetime.fromisoformat(notification.expires_at.replace('Z', '+00:00'))
                    if current_time > expires_at:
                        expired_notifications.append(notification_id)
            
            for notification_id in expired_notifications:
                del self.active_notifications[notification_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up expired notifications: {e}")
    
    # Public API Methods
    def create_notification_rule(self, user_id: str, disaster_type: str, alert_level: AlertLevel, 
                                conditions: Dict[str, Any], cooldown_minutes: int = 60) -> str:
        """Create a new notification rule for a user"""
        try:
            rule_id = f"rule_{int(time.time())}_{user_id}_{disaster_type}"
            
            rule = NotificationRule(
                id=rule_id,
                user_id=user_id,
                disaster_type=disaster_type,
                alert_level=alert_level,
                conditions=conditions,
                cooldown_minutes=cooldown_minutes,
                enabled=True,
                created_at=datetime.now(timezone.utc).isoformat()
            )
            
            self.notification_rules[rule_id] = rule
            
            logger.info(f"Created notification rule {rule_id} for user {user_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Error creating notification rule: {e}")
            return None
    
    def update_user_preferences(self, user_id: str, preferences: UserNotificationPreferences):
        """Update user notification preferences"""
        try:
            self.user_preferences[user_id] = preferences
            
            # Log to Tinybird
            enhanced_tinybird_service._make_request('POST', '/v0/events', {
                'name': 'user_preferences_update',
                'data': {
                    'user_id': user_id,
                    'preferences': {
                        'email_enabled': preferences.email_enabled,
                        'push_enabled': preferences.push_enabled,
                        'sms_enabled': preferences.sms_enabled,
                        'alert_frequency': preferences.alert_frequency,
                        'risk_threshold': preferences.risk_threshold,
                        'disaster_types': preferences.disaster_types,
                        'quiet_hours': preferences.quiet_hours,
                        'location_radius': preferences.location_radius
                    },
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
            })
            
            logger.info(f"Updated notification preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    def get_user_notifications(self, user_id: str, limit: int = 10) -> List[SmartNotification]:
        """Get recent notifications for a user"""
        try:
            user_notifications = []
            
            for notification in self.active_notifications.values():
                if notification.user_id == user_id:
                    user_notifications.append(notification)
            
            # Sort by timestamp (newest first)
            user_notifications.sort(key=lambda x: x.timestamp, reverse=True)
            
            return user_notifications[:limit]
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get smart notification system status"""
        try:
            return {
                'monitoring_active': self.is_monitoring,
                'active_notifications': len(self.active_notifications),
                'queued_notifications': len(self.notification_queue),
                'notification_rules': len(self.notification_rules),
                'user_preferences': len(self.user_preferences),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
    
    def stop_monitoring(self):
        """Stop the notification monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        logger.info("Smart notification monitoring stopped")

# Global Smart Notification System instance
smart_notification_system = SmartNotificationSystem()

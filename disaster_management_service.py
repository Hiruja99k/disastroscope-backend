"""
Disaster Management Service
Handles CRUD operations for disaster reports and management
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from flask import request, jsonify

# Import Firebase and Tinybird services if available
try:
    from firebase_service import firebase_service
    from tinybird_service import tinybird_service
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DisasterReport:
    """Disaster report data structure"""
    id: str
    title: str
    type: str
    severity: str
    location: str
    coordinates: Dict[str, float]
    description: str
    status: str
    reported_by: str
    reported_at: str
    affected_people: int
    estimated_damage: str
    contact_info: Dict[str, str]
    updates: List[Dict[str, Any]]
    created_at: str
    updated_at: str

class DisasterManagementService:
    """Service for managing disaster reports and operations"""
    
    def __init__(self):
        self.disasters: Dict[str, DisasterReport] = {}
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample disaster data"""
        sample_disasters = [
            {
                "id": "1",
                "title": "California Wildfire Alert",
                "type": "Wildfire",
                "severity": "High",
                "location": "Northern California",
                "coordinates": {"lat": 37.7749, "lng": -122.4194},
                "description": "Active wildfire spreading rapidly due to high winds and dry conditions",
                "status": "Active",
                "reported_by": "John Smith",
                "reported_at": "2024-01-15T10:30:00Z",
                "affected_people": 1500,
                "estimated_damage": "$2.5M",
                "contact_info": {"phone": "+1-555-0123", "email": "emergency@california.gov"},
                "updates": [
                    {"timestamp": "2024-01-15T10:30:00Z", "message": "Fire reported", "author": "John Smith"},
                    {"timestamp": "2024-01-15T11:15:00Z", "message": "Evacuation orders issued", "author": "Emergency Services"}
                ],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T11:15:00Z"
            },
            {
                "id": "2",
                "title": "Flood Warning - Mississippi River",
                "type": "Flood",
                "severity": "Medium",
                "location": "Mississippi River Basin",
                "coordinates": {"lat": 29.7604, "lng": -95.3698},
                "description": "River levels rising above flood stage due to heavy rainfall",
                "status": "Warning",
                "reported_by": "Sarah Johnson",
                "reported_at": "2024-01-15T08:45:00Z",
                "affected_people": 800,
                "estimated_damage": "$1.2M",
                "contact_info": {"phone": "+1-555-0456", "email": "flood@mississippi.gov"},
                "updates": [
                    {"timestamp": "2024-01-15T08:45:00Z", "message": "Flood warning issued", "author": "Sarah Johnson"}
                ],
                "created_at": "2024-01-15T08:45:00Z",
                "updated_at": "2024-01-15T08:45:00Z"
            },
            {
                "id": "3",
                "title": "Earthquake - Pacific Northwest",
                "type": "Earthquake",
                "severity": "Low",
                "location": "Seattle, WA",
                "coordinates": {"lat": 47.6062, "lng": -122.3321},
                "description": "Minor earthquake detected, no significant damage reported",
                "status": "Resolved",
                "reported_by": "Mike Chen",
                "reported_at": "2024-01-14T22:15:00Z",
                "affected_people": 0,
                "estimated_damage": "$0",
                "contact_info": {"phone": "+1-555-0789", "email": "seismic@washington.gov"},
                "updates": [
                    {"timestamp": "2024-01-14T22:15:00Z", "message": "Earthquake detected", "author": "Mike Chen"},
                    {"timestamp": "2024-01-14T23:00:00Z", "message": "No damage confirmed", "author": "Emergency Services"}
                ],
                "created_at": "2024-01-14T22:15:00Z",
                "updated_at": "2024-01-14T23:00:00Z"
            }
        ]
        
        for disaster_data in sample_disasters:
            disaster = DisasterReport(**disaster_data)
            self.disasters[disaster.id] = disaster
    
    def get_all_disasters(self, limit: int = 100, offset: int = 0, 
                         status: Optional[str] = None, 
                         type: Optional[str] = None,
                         severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all disaster reports with optional filtering"""
        try:
            disasters = list(self.disasters.values())
            
            # Apply filters
            if status:
                disasters = [d for d in disasters if d.status.lower() == status.lower()]
            if type:
                disasters = [d for d in disasters if d.type.lower() == type.lower()]
            if severity:
                disasters = [d for d in disasters if d.severity.lower() == severity.lower()]
            
            # Sort by reported_at (newest first)
            disasters.sort(key=lambda x: x.reported_at, reverse=True)
            
            # Apply pagination
            disasters = disasters[offset:offset + limit]
            
            return [asdict(disaster) for disaster in disasters]
        except Exception as e:
            logger.error(f"Error getting disasters: {e}")
            return []
    
    def get_disaster_by_id(self, disaster_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific disaster report by ID"""
        try:
            disaster = self.disasters.get(disaster_id)
            return asdict(disaster) if disaster else None
        except Exception as e:
            logger.error(f"Error getting disaster {disaster_id}: {e}")
            return None
    
    def create_disaster(self, disaster_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new disaster report"""
        try:
            # Generate ID
            disaster_id = str(len(self.disasters) + 1)
            
            # Set timestamps
            now = datetime.now(timezone.utc).isoformat()
            
            # Create disaster report
            disaster = DisasterReport(
                id=disaster_id,
                title=disaster_data.get('title', ''),
                type=disaster_data.get('type', ''),
                severity=disaster_data.get('severity', 'Low'),
                location=disaster_data.get('location', ''),
                coordinates=disaster_data.get('coordinates', {'lat': 0, 'lng': 0}),
                description=disaster_data.get('description', ''),
                status=disaster_data.get('status', 'Active'),
                reported_by=disaster_data.get('reported_by', 'Anonymous'),
                reported_at=now,
                affected_people=disaster_data.get('affected_people', 0),
                estimated_damage=disaster_data.get('estimated_damage', '$0'),
                contact_info=disaster_data.get('contact_info', {'phone': '', 'email': ''}),
                updates=disaster_data.get('updates', []),
                created_at=now,
                updated_at=now
            )
            
            self.disasters[disaster_id] = disaster
            
            # Log to external services if available
            if INTEGRATIONS_AVAILABLE:
                self._log_to_external_services(disaster, 'create')
            
            return asdict(disaster)
        except Exception as e:
            logger.error(f"Error creating disaster: {e}")
            return None
    
    def update_disaster(self, disaster_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing disaster report"""
        try:
            disaster = self.disasters.get(disaster_id)
            if not disaster:
                return None
            
            # Update fields
            for key, value in update_data.items():
                if hasattr(disaster, key) and key not in ['id', 'created_at']:
                    setattr(disaster, key, value)
            
            # Update timestamp
            disaster.updated_at = datetime.now(timezone.utc).isoformat()
            
            # Log to external services if available
            if INTEGRATIONS_AVAILABLE:
                self._log_to_external_services(disaster, 'update')
            
            return asdict(disaster)
        except Exception as e:
            logger.error(f"Error updating disaster {disaster_id}: {e}")
            return None
    
    def delete_disaster(self, disaster_id: str) -> bool:
        """Delete a disaster report"""
        try:
            disaster = self.disasters.get(disaster_id)
            if not disaster:
                return False
            
            # Log to external services if available
            if INTEGRATIONS_AVAILABLE:
                self._log_to_external_services(disaster, 'delete')
            
            del self.disasters[disaster_id]
            return True
        except Exception as e:
            logger.error(f"Error deleting disaster {disaster_id}: {e}")
            return False
    
    def add_disaster_update(self, disaster_id: str, update_message: str, author: str) -> Optional[Dict[str, Any]]:
        """Add an update to a disaster report"""
        try:
            disaster = self.disasters.get(disaster_id)
            if not disaster:
                return None
            
            update = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": update_message,
                "author": author
            }
            
            disaster.updates.append(update)
            disaster.updated_at = datetime.now(timezone.utc).isoformat()
            
            return asdict(disaster)
        except Exception as e:
            logger.error(f"Error adding update to disaster {disaster_id}: {e}")
            return None
    
    def get_disaster_statistics(self) -> Dict[str, Any]:
        """Get disaster statistics"""
        try:
            disasters = list(self.disasters.values())
            
            stats = {
                "total_disasters": len(disasters),
                "active_disasters": len([d for d in disasters if d.status == "Active"]),
                "warning_disasters": len([d for d in disasters if d.status == "Warning"]),
                "resolved_disasters": len([d for d in disasters if d.status == "Resolved"]),
                "total_affected_people": sum(d.affected_people for d in disasters),
                "disaster_types": {},
                "severity_distribution": {},
                "recent_disasters": len([d for d in disasters if self._is_recent(d.reported_at)])
            }
            
            # Count by type
            for disaster in disasters:
                disaster_type = disaster.type
                stats["disaster_types"][disaster_type] = stats["disaster_types"].get(disaster_type, 0) + 1
            
            # Count by severity
            for disaster in disasters:
                severity = disaster.severity
                stats["severity_distribution"][severity] = stats["severity_distribution"].get(severity, 0) + 1
            
            return stats
        except Exception as e:
            logger.error(f"Error getting disaster statistics: {e}")
            return {}
    
    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Check if a timestamp is within the last N days"""
        try:
            disaster_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            return (now - disaster_time).days <= days
        except:
            return False
    
    def _log_to_external_services(self, disaster: DisasterReport, action: str):
        """Log disaster operations to external services"""
        try:
            if INTEGRATIONS_AVAILABLE:
                # Log to Firebase
                firebase_data = {
                    "disaster_id": disaster.id,
                    "action": action,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "disaster_data": asdict(disaster)
                }
                
                # Log to Tinybird
                tinybird_data = {
                    "event": f"disaster_{action}",
                    "disaster_id": disaster.id,
                    "disaster_type": disaster.type,
                    "severity": disaster.severity,
                    "status": disaster.status,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Note: In a real implementation, you would call the actual services
                logger.info(f"Logged disaster {action} to external services: {disaster.id}")
        except Exception as e:
            logger.error(f"Error logging to external services: {e}")

# Global service instance
disaster_management_service = DisasterManagementService()

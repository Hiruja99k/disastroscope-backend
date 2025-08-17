import aiohttp
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EONETEvent:
    """NASA EONET event data structure"""
    
    def __init__(self, event_id: str, title: str, category: str, coordinates: Dict[str, float],
                 start_date: str, end_date: str = None):
        self.id = event_id
        self.title = title
        self.category = category
        self.coordinates = coordinates
        self.start_date = start_date
        self.end_date = end_date
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'category': self.category,
            'coordinates': self.coordinates,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'created_at': self.created_at.isoformat()
        }

class EONETService:
    """Service for fetching NASA EONET events"""
    
    def __init__(self):
        self.base_url = "https://eonet.gsfc.nasa.gov/api/v3"
        self.session = None
    
    async def initialize(self):
        """Initialize the service"""
        pass
    
    async def get_eonet_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent NASA EONET events"""
        try:
            # For demo purposes, return sample data
            # In production, this would call the actual EONET API
            sample_events = [
                {
                    'id': 'DEMO001',
                    'title': 'Tropical Storm Activity',
                    'category': 'severe-storms',
                    'coordinates': {'lat': 25.7617, 'lng': -80.1918},
                    'start_date': '2024-01-15',
                    'end_date': '2024-01-20'
                },
                {
                    'id': 'DEMO002',
                    'title': 'Wildfire in Western Region',
                    'category': 'wildfires',
                    'coordinates': {'lat': 45.5152, 'lng': -122.6784},
                    'start_date': '2024-01-10',
                    'end_date': '2024-01-25'
                },
                {
                    'id': 'DEMO003',
                    'title': 'Flooding in Midwest',
                    'category': 'floods',
                    'coordinates': {'lat': 41.8781, 'lng': -87.6298},
                    'start_date': '2024-01-12',
                    'end_date': '2024-01-18'
                }
            ]
            
            return sample_events
            
        except Exception as e:
            logger.error(f"Error fetching EONET events: {e}")
            return []
    
    async def get_events_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get NASA EONET events by category"""
        try:
            all_events = await self.get_eonet_events()
            category_events = [e for e in all_events if e['category'] == category.lower()]
            return category_events
            
        except Exception as e:
            logger.error(f"Error fetching EONET events by category: {e}")
            return []

# Global EONET service instance
eonet_service = EONETService()

import aiohttp
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FEMADeclaration:
    """FEMA disaster declaration data structure"""
    
    def __init__(self, declaration_id: str, title: str, state: str, disaster_type: str, 
                 incident_begin_date: str, incident_end_date: str = None):
        self.id = declaration_id
        self.title = title
        self.state = state
        self.disaster_type = disaster_type
        self.incident_begin_date = incident_begin_date
        self.incident_end_date = incident_end_date
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'state': self.state,
            'disaster_type': self.disaster_type,
            'incident_begin_date': self.incident_begin_date,
            'incident_end_date': self.incident_end_date,
            'created_at': self.created_at.isoformat()
        }

class OpenFEMAService:
    """Service for fetching FEMA disaster declarations"""
    
    def __init__(self):
        self.base_url = "https://www.fema.gov/api/open/v1/DisasterDeclarations"
        self.session = None
    
    async def initialize(self):
        """Initialize the service"""
        pass
    
    async def get_disaster_declarations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent FEMA disaster declarations"""
        try:
            # For demo purposes, return sample data
            # In production, this would call the actual FEMA API
            sample_disasters = [
                {
                    'id': 'DEMO001',
                    'title': 'Severe Storm and Flooding',
                    'state': 'CA',
                    'disaster_type': 'Severe Storm(s)',
                    'incident_begin_date': '2024-01-15',
                    'incident_end_date': '2024-01-20'
                },
                {
                    'id': 'DEMO002',
                    'title': 'Wildfire Emergency',
                    'state': 'OR',
                    'disaster_type': 'Fire',
                    'incident_begin_date': '2024-01-10',
                    'incident_end_date': '2024-01-25'
                }
            ]
            
            return sample_disasters
            
        except Exception as e:
            logger.error(f"Error fetching FEMA disasters: {e}")
            return []
    
    async def get_disasters_by_state(self, state_code: str) -> List[Dict[str, Any]]:
        """Get FEMA disaster declarations for a specific state"""
        try:
            all_disasters = await self.get_disaster_declarations()
            state_disasters = [d for d in all_disasters if d['state'] == state_code.upper()]
            return state_disasters
            
        except Exception as e:
            logger.error(f"Error fetching disasters by state: {e}")
            return []

# Global OpenFEMA service instance
openfema_service = OpenFEMAService()

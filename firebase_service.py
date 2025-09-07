"""
Firebase Admin SDK Integration for DisastroScope Backend
Handles user authentication verification and user management
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timezone

try:
    import firebase_admin
    from firebase_admin import credentials, auth, exceptions
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase Admin SDK not available. Install with: pip install firebase-admin")

logger = logging.getLogger(__name__)

class FirebaseService:
    """Firebase Admin SDK service for user authentication and management"""
    
    def __init__(self):
        self.app = None
        self.initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        if not FIREBASE_AVAILABLE:
            logger.warning("Firebase Admin SDK not available")
            return
        
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
                self.initialized = True
                logger.info("Firebase Admin SDK already initialized")
                return
            
            # Get Firebase configuration from environment variables
            firebase_config = self._get_firebase_config()
            
            if not firebase_config:
                logger.warning("Firebase configuration not found. Firebase features will be disabled.")
                return
            
            # Initialize Firebase Admin SDK
            if isinstance(firebase_config, dict):
                # Use service account key from environment variable
                cred = credentials.Certificate(firebase_config)
            else:
                # Use service account key file path
                cred = credentials.Certificate(firebase_config)
            
            self.app = firebase_admin.initialize_app(cred)
            self.initialized = True
            logger.info("Firebase Admin SDK initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
            self.initialized = False
    
    def _get_firebase_config(self) -> Optional[Dict[str, Any]]:
        """Get Firebase configuration from environment variables"""
        # Option 1: Service account key as JSON string
        firebase_key_json = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
        if firebase_key_json:
            try:
                return json.loads(firebase_key_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid Firebase service account key JSON: {e}")
                return None
        
        # Option 2: Service account key file path
        firebase_key_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        if firebase_key_path and os.path.exists(firebase_key_path):
            return firebase_key_path
        
        # Option 3: Individual configuration values
        project_id = os.getenv('FIREBASE_PROJECT_ID')
        private_key = os.getenv('FIREBASE_PRIVATE_KEY')
        client_email = os.getenv('FIREBASE_CLIENT_EMAIL')
        
        if project_id and private_key and client_email:
            return {
                "type": "service_account",
                "project_id": project_id,
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
                "private_key": private_key.replace('\\n', '\n'),
                "client_email": client_email,
                "client_id": os.getenv('FIREBASE_CLIENT_ID', ''),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email}"
            }
        
        return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token and return user information"""
        if not self.initialized:
            logger.warning("Firebase not initialized. Token verification skipped.")
            return None
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(token)
            
            # Extract user information
            user_info = {
                'uid': decoded_token['uid'],
                'email': decoded_token.get('email'),
                'email_verified': decoded_token.get('email_verified', False),
                'name': decoded_token.get('name'),
                'picture': decoded_token.get('picture'),
                'firebase': decoded_token.get('firebase', {}),
                'iat': decoded_token.get('iat'),
                'exp': decoded_token.get('exp'),
                'auth_time': decoded_token.get('auth_time')
            }
            
            logger.info(f"Token verified for user: {user_info['uid']}")
            return user_info
            
        except exceptions.InvalidIdTokenError as e:
            logger.error(f"Invalid ID token: {e}")
            return None
        except exceptions.ExpiredIdTokenError as e:
            logger.error(f"Expired ID token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get user information by UID"""
        if not self.initialized:
            logger.warning("Firebase not initialized. User lookup skipped.")
            return None
        
        try:
            user_record = auth.get_user(uid)
            
            user_info = {
                'uid': user_record.uid,
                'email': user_record.email,
                'email_verified': user_record.email_verified,
                'display_name': user_record.display_name,
                'photo_url': user_record.photo_url,
                'phone_number': user_record.phone_number,
                'disabled': user_record.disabled,
                'metadata': {
                    'creation_timestamp': user_record.user_metadata.creation_timestamp,
                    'last_sign_in_timestamp': user_record.user_metadata.last_sign_in_timestamp,
                    'last_refresh_timestamp': user_record.user_metadata.last_refresh_timestamp
                },
                'custom_claims': user_record.custom_claims or {},
                'provider_data': [
                    {
                        'uid': provider.uid,
                        'email': provider.email,
                        'display_name': provider.display_name,
                        'photo_url': provider.photo_url,
                        'provider_id': provider.provider_id
                    }
                    for provider in user_record.provider_data
                ]
            }
            
            return user_info
            
        except exceptions.UserNotFoundError:
            logger.warning(f"User not found: {uid}")
            return None
        except Exception as e:
            logger.error(f"Failed to get user {uid}: {e}")
            return None
    
    def create_custom_token(self, uid: str, additional_claims: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a custom token for a user"""
        if not self.initialized:
            logger.warning("Firebase not initialized. Custom token creation skipped.")
            return None
        
        try:
            custom_token = auth.create_custom_token(uid, additional_claims)
            return custom_token.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to create custom token for {uid}: {e}")
            return None
    
    def set_custom_user_claims(self, uid: str, custom_claims: Dict[str, Any]) -> bool:
        """Set custom claims for a user"""
        if not self.initialized:
            logger.warning("Firebase not initialized. Custom claims setting skipped.")
            return False
        
        try:
            auth.set_custom_user_claims(uid, custom_claims)
            logger.info(f"Custom claims set for user: {uid}")
            return True
        except Exception as e:
            logger.error(f"Failed to set custom claims for {uid}: {e}")
            return False
    
    def delete_user(self, uid: str) -> bool:
        """Delete a user"""
        if not self.initialized:
            logger.warning("Firebase not initialized. User deletion skipped.")
            return False
        
        try:
            auth.delete_user(uid)
            logger.info(f"User deleted: {uid}")
            return True
        except exceptions.UserNotFoundError:
            logger.warning(f"User not found for deletion: {uid}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete user {uid}: {e}")
            return False
    
    def list_users(self, max_results: int = 1000, page_token: Optional[str] = None) -> Dict[str, Any]:
        """List users with pagination"""
        if not self.initialized:
            logger.warning("Firebase not initialized. User listing skipped.")
            return {'users': [], 'next_page_token': None}
        
        try:
            result = auth.list_users(max_results=max_results, page_token=page_token)
            
            users = []
            for user_record in result.users:
                user_info = {
                    'uid': user_record.uid,
                    'email': user_record.email,
                    'email_verified': user_record.email_verified,
                    'display_name': user_record.display_name,
                    'photo_url': user_record.photo_url,
                    'disabled': user_record.disabled,
                    'metadata': {
                        'creation_timestamp': user_record.user_metadata.creation_timestamp,
                        'last_sign_in_timestamp': user_record.user_metadata.last_sign_in_timestamp
                    }
                }
                users.append(user_info)
            
            return {
                'users': users,
                'next_page_token': result.next_page_token
            }
            
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return {'users': [], 'next_page_token': None}
    
    def is_initialized(self) -> bool:
        """Check if Firebase is initialized"""
        return self.initialized

# Global Firebase service instance
firebase_service = FirebaseService()

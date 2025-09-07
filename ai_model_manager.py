"""
AI Model Manager for Real-Time Learning and Personalization
Manages model training, retraining, and personalization using Tinybird data
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
import threading
import time

from enhanced_ai_models import enhanced_ai_prediction_service
from enhanced_tinybird_service import enhanced_tinybird_service, PredictionEvent, UserBehavior, HistoricalEvent

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: str
    training_samples: int
    validation_samples: int

@dataclass
class PersonalizedModel:
    """Personalized model for a specific user"""
    user_id: str
    model_type: str
    model_data: bytes  # Serialized model
    performance_metrics: Dict[str, float]
    last_trained: str
    training_samples: int
    user_preferences: Dict[str, Any]

class AIModelManager:
    """Advanced AI Model Manager with real-time learning capabilities"""
    
    def __init__(self):
        self.models = {}
        self.personalized_models = {}
        self.performance_tracker = {}
        self.retraining_queue = []
        self.learning_thread = None
        self.is_learning = False
        
        # Model configuration
        self.retraining_threshold = 0.1  # Retrain if accuracy drops by 10%
        self.min_training_samples = 1000
        self.personalization_threshold = 50  # Personalize after 50 user interactions
        
        # Initialize model manager
        self._initialize_models()
        self._start_learning_thread()
    
    def _initialize_models(self):
        """Initialize all models and performance tracking"""
        disaster_types = ['flood', 'earthquake', 'landslide', 'wildfire', 'storm', 'tornado', 'drought']
        
        for disaster_type in disaster_types:
            self.performance_tracker[disaster_type] = {
                'accuracy': 0.0,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'training_samples': 0,
                'validation_samples': 0,
                'retraining_needed': False
            }
        
        logger.info("AI Model Manager initialized")
    
    def _start_learning_thread(self):
        """Start the continuous learning thread"""
        if self.learning_thread is None or not self.learning_thread.is_alive():
            self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
            self.learning_thread.start()
            logger.info("Continuous learning thread started")
    
    def _continuous_learning_loop(self):
        """Continuous learning loop for model improvement"""
        self.is_learning = True
        
        while self.is_learning:
            try:
                # Check for retraining needs
                self._check_retraining_needs()
                
                # Process user feedback
                self._process_user_feedback()
                
                # Update personalized models
                self._update_personalized_models()
                
                # Log performance metrics
                self._log_performance_metrics()
                
                # Sleep for 5 minutes before next iteration
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _check_retraining_needs(self):
        """Check if models need retraining based on performance"""
        try:
            # Get latest performance data from Tinybird
            for disaster_type in self.performance_tracker.keys():
                performance_data = enhanced_tinybird_service.get_model_performance_trends(
                    f"{disaster_type}_model", days=7
                )
                
                if performance_data:
                    latest_accuracy = performance_data[-1].get('accuracy', 0.0)
                    current_accuracy = self.performance_tracker[disaster_type]['accuracy']
                    
                    # Check if accuracy has dropped significantly
                    if current_accuracy - latest_accuracy > self.retraining_threshold:
                        self.performance_tracker[disaster_type]['retraining_needed'] = True
                        logger.info(f"Retraining needed for {disaster_type} model: accuracy dropped from {current_accuracy:.3f} to {latest_accuracy:.3f}")
            
            # Process retraining queue
            self._process_retraining_queue()
            
        except Exception as e:
            logger.error(f"Error checking retraining needs: {e}")
    
    def _process_retraining_queue(self):
        """Process models that need retraining"""
        for disaster_type, needs_retraining in self.performance_tracker.items():
            if needs_retraining.get('retraining_needed', False):
                try:
                    logger.info(f"Retraining {disaster_type} model...")
                    
                    # Export training data from Tinybird
                    training_data = enhanced_tinybird_service.export_training_data(disaster_type, days=365)
                    
                    if len(training_data) >= self.min_training_samples:
                        # Retrain the model
                        self._retrain_model(disaster_type, training_data)
                        
                        # Update performance tracker
                        self.performance_tracker[disaster_type]['retraining_needed'] = False
                        self.performance_tracker[disaster_type]['last_updated'] = datetime.now(timezone.utc).isoformat()
                        
                        logger.info(f"Successfully retrained {disaster_type} model")
                    else:
                        logger.warning(f"Insufficient training data for {disaster_type}: {len(training_data)} samples")
                        
                except Exception as e:
                    logger.error(f"Error retraining {disaster_type} model: {e}")
    
    def _retrain_model(self, disaster_type: str, training_data: pd.DataFrame):
        """Retrain a specific model with new data"""
        try:
            # Prepare features and labels
            feature_columns = [col for col in training_data.columns if col not in ['event_type', 'severity', 'timestamp', 'verified']]
            X = training_data[feature_columns].values
            y = training_data['severity'].values
            
            # Retrain the model
            if hasattr(enhanced_ai_prediction_service.models, disaster_type):
                model = enhanced_ai_prediction_service.models[disaster_type]
                if hasattr(model, 'train'):
                    model.train(X, y)
                    
                    # Save the retrained model
                    enhanced_ai_prediction_service.save_enhanced_model(disaster_type)
                    
                    # Log retraining event
                    enhanced_tinybird_service.log_model_performance(
                        f"{disaster_type}_model",
                        {
                            'accuracy': 0.85,  # This would be calculated from validation
                            'training_samples': len(training_data),
                            'retraining_event': True
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
    
    def _process_user_feedback(self):
        """Process user feedback for model improvement"""
        try:
            # Get recent user feedback
            feedback_analytics = enhanced_tinybird_service.get_user_feedback_analytics(days=7)
            
            if feedback_analytics and feedback_analytics.get('total_feedback', 0) > 0:
                # Process feedback for each disaster type
                for disaster_type in self.performance_tracker.keys():
                    feedback_count = feedback_analytics.get(f'{disaster_type}_feedback', 0)
                    
                    if feedback_count > 0:
                        # Update model based on feedback
                        self._update_model_from_feedback(disaster_type, feedback_analytics)
                        
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
    
    def _update_model_from_feedback(self, disaster_type: str, feedback_analytics: Dict[str, Any]):
        """Update model based on user feedback"""
        try:
            # Get feedback accuracy for this disaster type
            feedback_key = f'{disaster_type}_accuracy'
            if feedback_key in feedback_analytics:
                feedback_accuracy = feedback_analytics[feedback_key]
                
                # If feedback accuracy is significantly different from model accuracy
                current_accuracy = self.performance_tracker[disaster_type]['accuracy']
                if abs(feedback_accuracy - current_accuracy) > 0.05:  # 5% difference
                    logger.info(f"Updating {disaster_type} model based on user feedback: {feedback_accuracy:.3f} vs {current_accuracy:.3f}")
                    
                    # Mark for retraining
                    self.performance_tracker[disaster_type]['retraining_needed'] = True
                    
        except Exception as e:
            logger.error(f"Error updating model from feedback: {e}")
    
    def _update_personalized_models(self):
        """Update personalized models for active users"""
        try:
            # Get active users from Tinybird
            # This would typically query user activity data
            # For now, we'll implement a basic version
            
            # Check if any users need personalized models
            for user_id, personalized_model in self.personalized_models.items():
                if self._should_update_personalized_model(user_id, personalized_model):
                    self._retrain_personalized_model(user_id, personalized_model)
                    
        except Exception as e:
            logger.error(f"Error updating personalized models: {e}")
    
    def _should_update_personalized_model(self, user_id: str, personalized_model: PersonalizedModel) -> bool:
        """Check if a personalized model should be updated"""
        try:
            # Get user behavior profile
            user_behavior = enhanced_tinybird_service.get_user_behavior_profile(user_id)
            
            if user_behavior:
                # Check if user has enough interactions for personalization
                interaction_count = len(user_behavior.feedback_history)
                
                if interaction_count >= self.personalization_threshold:
                    # Check if model is outdated
                    last_trained = datetime.fromisoformat(personalized_model.last_trained.replace('Z', '+00:00'))
                    days_since_training = (datetime.now(timezone.utc) - last_trained).days
                    
                    return days_since_training > 7  # Retrain weekly
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking personalized model update: {e}")
            return False
    
    def _retrain_personalized_model(self, user_id: str, personalized_model: PersonalizedModel):
        """Retrain a personalized model for a specific user"""
        try:
            # Get user-specific data
            user_behavior = enhanced_tinybird_service.get_user_behavior_profile(user_id)
            user_predictions = enhanced_tinybird_service.get_prediction_analytics(user_id, days=90)
            
            if user_behavior and user_predictions:
                # Create personalized model based on user preferences
                personalized_model = self._create_personalized_model(user_id, user_behavior, user_predictions)
                self.personalized_models[user_id] = personalized_model
                
                logger.info(f"Updated personalized model for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error retraining personalized model: {e}")
    
    def _create_personalized_model(self, user_id: str, user_behavior: UserBehavior, user_predictions: Dict[str, Any]) -> PersonalizedModel:
        """Create a personalized model for a user"""
        try:
            # Adjust model parameters based on user behavior
            risk_sensitivity = user_behavior.risk_sensitivity
            
            # Create a copy of the base model with adjusted parameters
            model_data = pickle.dumps({
                'risk_sensitivity': risk_sensitivity,
                'user_preferences': user_behavior.interaction_patterns,
                'base_model': 'enhanced_ai_prediction_service'
            })
            
            return PersonalizedModel(
                user_id=user_id,
                model_type='personalized',
                model_data=model_data,
                performance_metrics={
                    'accuracy': 0.85,  # This would be calculated
                    'personalization_score': risk_sensitivity
                },
                last_trained=datetime.now(timezone.utc).isoformat(),
                training_samples=len(user_behavior.feedback_history),
                user_preferences=user_behavior.interaction_patterns
            )
            
        except Exception as e:
            logger.error(f"Error creating personalized model: {e}")
            return None
    
    def _log_performance_metrics(self):
        """Log current performance metrics to Tinybird"""
        try:
            for disaster_type, metrics in self.performance_tracker.items():
                enhanced_tinybird_service.log_model_performance(
                    f"{disaster_type}_model",
                    {
                        'accuracy': metrics['accuracy'],
                        'training_samples': metrics['training_samples'],
                        'validation_samples': metrics['validation_samples'],
                        'retraining_needed': metrics['retraining_needed']
                    }
                )
                
        except Exception as e:
            logger.error(f"Error logging performance metrics: {e}")
    
    # Public API Methods
    def get_personalized_prediction(self, user_id: str, weather_data: Dict[str, Any], geospatial_data: Dict[str, Any]) -> Dict[str, float]:
        """Get personalized prediction for a user"""
        try:
            # Check if user has a personalized model
            if user_id in self.personalized_models:
                personalized_model = self.personalized_models[user_id]
                
                # Use personalized model for prediction
                return self._predict_with_personalized_model(personalized_model, weather_data, geospatial_data)
            else:
                # Use base model with user behavior adjustments
                base_predictions = enhanced_ai_prediction_service.predict_enhanced_risks(weather_data, geospatial_data)
                user_behavior = enhanced_tinybird_service.get_user_behavior_profile(user_id)
                
                if user_behavior:
                    return self._adjust_predictions_for_user(base_predictions, user_behavior)
                else:
                    return base_predictions
                    
        except Exception as e:
            logger.error(f"Error getting personalized prediction: {e}")
            # Fallback to base model
            return enhanced_ai_prediction_service.predict_enhanced_risks(weather_data, geospatial_data)
    
    def _predict_with_personalized_model(self, personalized_model: PersonalizedModel, weather_data: Dict[str, Any], geospatial_data: Dict[str, Any]) -> Dict[str, float]:
        """Make prediction using personalized model"""
        try:
            # Deserialize model data
            model_data = pickle.loads(personalized_model.model_data)
            risk_sensitivity = model_data.get('risk_sensitivity', 1.0)
            
            # Get base predictions
            base_predictions = enhanced_ai_prediction_service.predict_enhanced_risks(weather_data, geospatial_data)
            
            # Adjust predictions based on user's risk sensitivity
            adjusted_predictions = {}
            for disaster_type, risk in base_predictions.items():
                # Adjust risk based on user sensitivity
                adjusted_risk = min(1.0, risk * risk_sensitivity)
                adjusted_predictions[disaster_type] = adjusted_risk
            
            return adjusted_predictions
            
        except Exception as e:
            logger.error(f"Error with personalized prediction: {e}")
            return enhanced_ai_prediction_service.predict_enhanced_risks(weather_data, geospatial_data)
    
    def _adjust_predictions_for_user(self, base_predictions: Dict[str, float], user_behavior: UserBehavior) -> Dict[str, float]:
        """Adjust base predictions based on user behavior"""
        try:
            adjusted_predictions = {}
            risk_sensitivity = user_behavior.risk_sensitivity
            
            for disaster_type, risk in base_predictions.items():
                # Adjust based on user's risk sensitivity
                adjusted_risk = min(1.0, risk * risk_sensitivity)
                adjusted_predictions[disaster_type] = adjusted_risk
            
            return adjusted_predictions
            
        except Exception as e:
            logger.error(f"Error adjusting predictions for user: {e}")
            return base_predictions
    
    def log_prediction_with_context(self, user_id: str, prediction_data: Dict[str, Any], weather_data: Dict[str, Any], geospatial_data: Dict[str, Any]) -> str:
        """Log prediction with full context for learning"""
        try:
            # Create prediction event
            prediction_event = PredictionEvent(
                id=f"pred_{int(time.time())}_{user_id}",
                user_id=user_id,
                event_type=prediction_data.get('event_type', 'multi_disaster'),
                latitude=geospatial_data.get('latitude', 0.0),
                longitude=geospatial_data.get('longitude', 0.0),
                probability=max(prediction_data.values()) if prediction_data else 0.0,
                confidence=0.85,  # This would be calculated
                model_version='2.0.0',
                location_name=geospatial_data.get('location_name', 'Unknown'),
                weather_data=weather_data,
                geospatial_data=geospatial_data,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Log to Tinybird
            success = enhanced_tinybird_service.log_prediction_event(prediction_event)
            
            if success:
                logger.info(f"Logged prediction event: {prediction_event.id}")
                return prediction_event.id
            else:
                logger.error("Failed to log prediction event")
                return None
                
        except Exception as e:
            logger.error(f"Error logging prediction with context: {e}")
            return None
    
    def update_prediction_accuracy(self, prediction_id: str, accuracy: float, user_feedback: str = None):
        """Update prediction accuracy for learning"""
        try:
            success = enhanced_tinybird_service.update_prediction_accuracy(prediction_id, accuracy, user_feedback)
            
            if success:
                logger.info(f"Updated prediction accuracy for {prediction_id}: {accuracy:.3f}")
            else:
                logger.error(f"Failed to update prediction accuracy for {prediction_id}")
                
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")
    
    def get_model_insights(self) -> Dict[str, Any]:
        """Get comprehensive model insights"""
        try:
            insights = {
                'performance_summary': self.performance_tracker,
                'personalized_models_count': len(self.personalized_models),
                'retraining_queue': len([m for m in self.performance_tracker.values() if m.get('retraining_needed', False)]),
                'learning_active': self.is_learning,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Add Tinybird analytics
            system_health = enhanced_tinybird_service.get_system_health_metrics()
            if system_health:
                insights['system_health'] = system_health
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model insights: {e}")
            return {}
    
    def stop_learning(self):
        """Stop the continuous learning thread"""
        self.is_learning = False
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=10)
        logger.info("Continuous learning stopped")

# Global AI Model Manager instance
ai_model_manager = AIModelManager()

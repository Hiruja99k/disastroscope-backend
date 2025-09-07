"""
Enhanced AI Models for Disaster Prediction
Advanced machine learning models with improved accuracy and strictness
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pickle
import json
import math
from dataclasses import dataclass

# Advanced ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, using fallback models")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using basic models")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

logger = logging.getLogger(__name__)

# Configuration
EARTHQUAKE_RISK_MULTIPLIER = float(os.getenv('EARTHQUAKE_RISK_MULTIPLIER', '0.05'))
ALLOW_EARTHQUAKE_PREDICTIONS = os.getenv('ALLOW_EARTHQUAKE_PREDICTIONS', 'false').lower() == 'true'
MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', '3600'))  # 1 hour

@dataclass
class GeospatialFeatures:
    """Enhanced geospatial features for disaster prediction"""
    latitude: float
    longitude: float
    elevation: float
    slope: float
    aspect: float
    soil_type: str
    land_use: str
    distance_to_water: float
    distance_to_fault: float
    population_density: float
    infrastructure_density: float
    historical_events: int
    tectonic_zone: str
    climate_zone: str
    vegetation_index: float
    urbanization_level: float

@dataclass
class WeatherFeatures:
    """Enhanced weather features"""
    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    wind_direction: float
    precipitation: float
    visibility: float
    cloud_cover: float
    uv_index: float
    dew_point: float
    heat_index: float
    wind_chill: float
    precipitation_intensity: float
    atmospheric_stability: float
    moisture_content: float

class AdvancedNeuralNetwork(nn.Module):
    """Advanced neural network with attention mechanism and residual connections"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super(AdvancedNeuralNetwork, self).__init__()
        
        self.use_attention = use_attention
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_sizes[-1], 
                num_heads=4, 
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Residual connections
        self.residual_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            if hidden_sizes[i] == hidden_sizes[i + 1]:
                self.residual_layers.append(nn.Identity())
            else:
                self.residual_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
    
    def forward(self, x):
        # Store inputs for residual connections
        inputs = [x]
        
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            residual = x if i == 0 else inputs[-1]
            
            # Forward pass
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # Residual connection
            if i > 0 and i < len(self.residual_layers):
                residual = self.residual_layers[i-1](residual)
                x = x + residual
            
            inputs.append(x)
        
        # Attention mechanism
        if self.use_attention and len(inputs) > 1:
            # Reshape for attention
            x_reshaped = x.unsqueeze(1)  # Add sequence dimension
            attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            x = attended.squeeze(1)
        
        # Output
        x = torch.sigmoid(self.output_layer(x))
        return x

class EnsembleModel:
    """Advanced ensemble model combining multiple algorithms"""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def _create_models(self, input_size: int):
        """Create ensemble of models"""
        models = {}
        
        if SKLEARN_AVAILABLE:
            # Random Forest with optimized parameters
            models['random_forest'] = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Support Vector Regression
            models['svr'] = SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.01
            )
            
            # Neural Network
            models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        
        if XGBOOST_AVAILABLE:
            # XGBoost with optimized parameters
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        
        if TORCH_AVAILABLE:
            # PyTorch Neural Network
            models['pytorch_nn'] = AdvancedNeuralNetwork(
                input_size=input_size,
                hidden_sizes=[256, 128, 64],
                dropout_rate=0.3,
                use_attention=True
            )
        
        return models
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """Train ensemble models"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Create models
        self.models = self._create_models(X.shape[1])
        
        # Train each model
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name} for {self.model_type}")
                
                if name == 'pytorch_nn':
                    self._train_pytorch_model(model, X_train, y_train, X_val, y_val)
                else:
                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    self.scalers[name] = scaler
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_val_scaled)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    self.performance_metrics[name] = {
                        'mse': mse,
                        'r2': r2,
                        'mae': mean_absolute_error(y_val, y_pred)
                    }
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = model.feature_importances_
                    
                    logger.info(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Create voting ensemble
        if len(self.models) > 1:
            self._create_voting_ensemble()
    
    def _train_pytorch_model(self, model, X_train, y_train, X_val, y_val, epochs=100):
        """Train PyTorch neural network"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['pytorch_nn'] = scaler
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                scheduler.step(val_loss)
            
            model.train()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            r2 = r2_score(y_val, val_outputs.numpy())
            
        self.performance_metrics['pytorch_nn'] = {
            'mse': val_loss.item(),
            'r2': r2,
            'mae': mean_absolute_error(y_val, val_outputs.numpy())
        }
    
    def _create_voting_ensemble(self):
        """Create voting ensemble of best models"""
        if not SKLEARN_AVAILABLE:
            return
        
        # Select best models based on R² score
        best_models = []
        for name, metrics in self.performance_metrics.items():
            if metrics['r2'] > 0.7:  # Only include models with good performance
                best_models.append((name, self.models[name]))
        
        if len(best_models) > 1:
            self.models['ensemble'] = VotingRegressor(
                estimators=best_models,
                weights=[1.0] * len(best_models)  # Equal weights
            )
            
            # Train ensemble
            X_train, X_val, y_train, y_val = train_test_split(
                np.vstack([self.scalers[name].transform(X_train) for name, _ in best_models]),
                y_train, test_size=0.2, random_state=42
            )
            
            self.models['ensemble'].fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> float:
        """Make ensemble prediction"""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if name == 'ensemble':
                continue
                
            try:
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X.reshape(1, -1))
                else:
                    X_scaled = X.reshape(1, -1)
                
                if name == 'pytorch_nn':
                    model.eval()
                    with torch.no_grad():
                        pred = model(torch.FloatTensor(X_scaled)).item()
                else:
                    pred = model.predict(X_scaled)[0]
                
                predictions.append(pred)
                
                # Weight by model performance
                if name in self.performance_metrics:
                    weight = max(0.1, self.performance_metrics[name]['r2'])
                else:
                    weight = 1.0
                weights.append(weight)
                
            except Exception as e:
                logger.error(f"Error in {name} prediction: {e}")
                continue
        
        if not predictions:
            return 0.0
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        return np.average(predictions, weights=weights)

class EnhancedDisasterPredictionService:
    """Enhanced disaster prediction service with advanced ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.last_update = {}
        self.model_path = os.path.join(os.path.dirname(__file__), "enhanced_models")
        os.makedirs(self.model_path, exist_ok=True)
        
        # Feature engineering parameters
        self.feature_engineering_params = {
            'flood': {
                'precipitation_threshold': 20.0,
                'humidity_threshold': 70.0,
                'pressure_threshold': 1000.0,
                'elevation_weight': 0.3,
                'distance_to_water_weight': 0.4
            },
            'earthquake': {
                'tectonic_zone_weights': {
                    'ring_of_fire': 0.8,
                    'alpine_himalayan': 0.7,
                    'mid_atlantic': 0.6,
                    'east_african': 0.5,
                    'stable': 0.1
                },
                'distance_to_fault_weight': 0.6,
                'historical_events_weight': 0.4
            },
            'landslide': {
                'slope_threshold': 15.0,
                'precipitation_threshold': 25.0,
                'soil_type_weights': {
                    'clay': 0.8,
                    'silt': 0.6,
                    'sand': 0.3,
                    'rock': 0.1
                },
                'vegetation_index_weight': 0.3
            }
        }
        
        self.allow_earthquake_predictions = ALLOW_EARTHQUAKE_PREDICTIONS
        self.load_or_initialize_models()
    
    def load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        disaster_types = ['flood', 'earthquake', 'landslide', 'wildfire', 'storm', 'tornado', 'drought']
        
        for disaster_type in disaster_types:
            try:
                # Try to load existing model
                model_file = os.path.join(self.model_path, f"{disaster_type}_ensemble.pkl")
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        self.models[disaster_type] = pickle.load(f)
                    logger.info(f"Loaded enhanced {disaster_type} model")
                else:
                    # Initialize new model
                    self.models[disaster_type] = EnsembleModel(disaster_type)
                    logger.info(f"Initialized new {disaster_type} model")
                    
            except Exception as e:
                logger.error(f"Error loading {disaster_type} model: {e}")
                self.models[disaster_type] = EnsembleModel(disaster_type)
    
    def generate_enhanced_training_data(self, num_samples: int = 50000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Generate enhanced training data with geospatial and temporal features"""
        training_data = {}
        
        for disaster_type in ['flood', 'earthquake', 'landslide', 'wildfire', 'storm', 'tornado', 'drought']:
            features = []
            labels = []
            
            for _ in range(num_samples):
                # Generate realistic features
                feature_vector = self._generate_disaster_features(disaster_type)
                risk_score = self._calculate_enhanced_risk(disaster_type, feature_vector)
                
                features.append(feature_vector)
                labels.append(risk_score)
            
            training_data[disaster_type] = (np.array(features), np.array(labels))
            logger.info(f"Generated {num_samples} training samples for {disaster_type}")
        
        return training_data
    
    def _generate_disaster_features(self, disaster_type: str) -> List[float]:
        """Generate realistic feature vector for specific disaster type"""
        features = []
        
        # Weather features (common to all)
        features.extend([
            np.random.uniform(-20, 45),  # temperature
            np.random.uniform(10, 100),  # humidity
            np.random.uniform(900, 1100),  # pressure
            np.random.uniform(0, 60),  # wind_speed
            np.random.uniform(0, 360),  # wind_direction
            np.random.uniform(0, 100),  # precipitation
            np.random.uniform(0, 25),  # visibility
            np.random.uniform(0, 100),  # cloud_cover
            np.random.uniform(0, 12),  # uv_index
            np.random.uniform(-10, 30),  # dew_point
            np.random.uniform(-20, 50),  # heat_index
            np.random.uniform(-30, 20),  # wind_chill
            np.random.uniform(0, 50),  # precipitation_intensity
            np.random.uniform(0, 1),  # atmospheric_stability
            np.random.uniform(0, 1)  # moisture_content
        ])
        
        # Geospatial features
        features.extend([
            np.random.uniform(-90, 90),  # latitude
            np.random.uniform(-180, 180),  # longitude
            np.random.uniform(0, 4000),  # elevation
            np.random.uniform(0, 60),  # slope
            np.random.uniform(0, 360),  # aspect
            np.random.uniform(0, 10),  # soil_type (encoded)
            np.random.uniform(0, 10),  # land_use (encoded)
            np.random.uniform(0, 50),  # distance_to_water
            np.random.uniform(0, 100),  # distance_to_fault
            np.random.uniform(0, 10000),  # population_density
            np.random.uniform(0, 1),  # infrastructure_density
            np.random.uniform(0, 50),  # historical_events
            np.random.uniform(0, 5),  # tectonic_zone (encoded)
            np.random.uniform(0, 10),  # climate_zone (encoded)
            np.random.uniform(0, 1),  # vegetation_index
            np.random.uniform(0, 1)  # urbanization_level
        ])
        
        # Disaster-specific features
        if disaster_type == 'flood':
            features.extend([
                np.random.uniform(0, 100),  # river_level
                np.random.uniform(0, 1),  # drainage_efficiency
                np.random.uniform(0, 1)  # flood_control_infrastructure
            ])
        elif disaster_type == 'earthquake':
            features.extend([
                np.random.uniform(0, 1),  # seismic_activity_level
                np.random.uniform(0, 1),  # ground_acceleration
                np.random.uniform(0, 1)  # soil_liquefaction_potential
            ])
        elif disaster_type == 'landslide':
            features.extend([
                np.random.uniform(0, 1),  # soil_saturation
                np.random.uniform(0, 1),  # slope_stability_index
                np.random.uniform(0, 1)  # vegetation_cover
            ])
        
        return features
    
    def _calculate_enhanced_risk(self, disaster_type: str, features: List[float]) -> float:
        """Calculate enhanced risk score using multiple factors"""
        # Extract key features
        weather_features = features[:15]
        geo_features = features[15:31]
        
        temp, humidity, pressure, wind_speed, wind_dir, precip, visibility, cloud_cover = weather_features[:8]
        lat, lon, elevation, slope, aspect, soil_type, land_use, dist_water, dist_fault = geo_features[:9]
        pop_density, infra_density, hist_events, tectonic_zone, climate_zone, veg_index, urban_level = geo_features[9:16]
        
        if disaster_type == 'flood':
            # Enhanced flood risk calculation
            precip_risk = min(1.0, precip / 50.0)
            humidity_risk = min(1.0, humidity / 100.0)
            pressure_risk = min(1.0, (1013 - pressure) / 100.0)
            elevation_risk = min(1.0, (100 - elevation) / 100.0) if elevation < 100 else 0.1
            water_distance_risk = min(1.0, (50 - dist_water) / 50.0) if dist_water < 50 else 0.1
            
            # Weighted combination
            risk = (precip_risk * 0.3 + humidity_risk * 0.2 + pressure_risk * 0.2 + 
                   elevation_risk * 0.15 + water_distance_risk * 0.15)
            
        elif disaster_type == 'earthquake':
            # Enhanced earthquake risk calculation
            tectonic_risk = {
                0: 0.1,  # stable
                1: 0.8,  # ring of fire
                2: 0.7,  # alpine-himalayan
                3: 0.6,  # mid-atlantic
                4: 0.5   # east african
            }.get(int(tectonic_zone), 0.1)
            
            fault_distance_risk = min(1.0, (100 - dist_fault) / 100.0) if dist_fault < 100 else 0.1
            historical_risk = min(1.0, hist_events / 50.0)
            
            risk = (tectonic_risk * 0.5 + fault_distance_risk * 0.3 + historical_risk * 0.2)
            
        elif disaster_type == 'landslide':
            # Enhanced landslide risk calculation
            slope_risk = min(1.0, slope / 45.0)
            precip_risk = min(1.0, precip / 80.0)
            soil_risk = {
                0: 0.1,  # rock
                1: 0.3,  # sand
                2: 0.6,  # silt
                3: 0.8   # clay
            }.get(int(soil_type), 0.3)
            
            veg_risk = 1.0 - veg_index  # Lower vegetation = higher risk
            
            risk = (slope_risk * 0.4 + precip_risk * 0.3 + soil_risk * 0.2 + veg_risk * 0.1)
            
        else:
            # Fallback for other disaster types
            risk = min(1.0, (precip / 50.0) * 0.5 + (temp / 40.0) * 0.3 + (wind_speed / 30.0) * 0.2)
        
        return max(0.0, min(1.0, risk))
    
    def train_enhanced_models(self, epochs: int = 100, batch_size: int = 32):
        """Train enhanced models with advanced techniques"""
        logger.info("Starting enhanced model training...")
        
        # Generate training data
        training_data = self.generate_enhanced_training_data()
        
        for disaster_type, (features, labels) in training_data.items():
            try:
                logger.info(f"Training enhanced {disaster_type} model...")
                
                # Train ensemble model
                self.models[disaster_type].train(features, labels)
                
                # Store metadata
                self.model_metadata[disaster_type] = {
                    'training_samples': len(features),
                    'feature_count': features.shape[1],
                    'last_trained': datetime.now().isoformat(),
                    'performance': self.models[disaster_type].performance_metrics
                }
                
                # Save model
                self.save_enhanced_model(disaster_type)
                
                logger.info(f"Enhanced {disaster_type} model training completed")
                
            except Exception as e:
                logger.error(f"Error training enhanced {disaster_type} model: {e}")
                continue
        
        logger.info("Enhanced model training completed")
    
    def save_enhanced_model(self, disaster_type: str):
        """Save enhanced model to disk"""
        try:
            model_file = os.path.join(self.model_path, f"{disaster_type}_ensemble.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(self.models[disaster_type], f)
            
            # Save metadata
            metadata_file = os.path.join(self.model_path, f"{disaster_type}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.model_metadata.get(disaster_type, {}), f, indent=2)
            
            logger.info(f"Saved enhanced {disaster_type} model")
            
        except Exception as e:
            logger.error(f"Error saving enhanced {disaster_type} model: {e}")
    
    def predict_enhanced_risks(self, weather_data: Dict, geospatial_data: Dict = None) -> Dict[str, float]:
        """Make enhanced predictions with geospatial context"""
        predictions = {}
        
        # Default geospatial data if not provided
        if geospatial_data is None:
            geospatial_data = {
                'latitude': 0.0,
                'longitude': 0.0,
                'elevation': 100.0,
                'slope': 5.0,
                'aspect': 180.0,
                'soil_type': 2,
                'land_use': 3,
                'distance_to_water': 10.0,
                'distance_to_fault': 50.0,
                'population_density': 1000.0,
                'infrastructure_density': 0.5,
                'historical_events': 5,
                'tectonic_zone': 0,
                'climate_zone': 3,
                'vegetation_index': 0.6,
                'urbanization_level': 0.4
            }
        
        for disaster_type, model in self.models.items():
            try:
                # Prepare feature vector
                feature_vector = self._prepare_feature_vector(weather_data, geospatial_data, disaster_type)
                
                # Make prediction
                if hasattr(model, 'predict') and model.models:
                    prediction = model.predict(feature_vector)
                else:
                    # Fallback to rule-based
                    prediction = self._rule_based_enhanced_risk(disaster_type, weather_data, geospatial_data)
                
                predictions[disaster_type] = max(0.0, min(1.0, prediction))
                
            except Exception as e:
                logger.error(f"Error in enhanced {disaster_type} prediction: {e}")
                predictions[disaster_type] = self._rule_based_enhanced_risk(disaster_type, weather_data, geospatial_data)
        
        # Apply earthquake risk multiplier if needed
        if not self.allow_earthquake_predictions and 'earthquake' in predictions:
            predictions['earthquake'] = max(0.0, min(0.05, predictions['earthquake'] * EARTHQUAKE_RISK_MULTIPLIER))
        
        return predictions
    
    def _prepare_feature_vector(self, weather_data: Dict, geospatial_data: Dict, disaster_type: str) -> np.ndarray:
        """Prepare feature vector for prediction"""
        features = []
        
        # Weather features
        features.extend([
            weather_data.get('temperature', 20.0),
            weather_data.get('humidity', 50.0),
            weather_data.get('pressure', 1013.0),
            weather_data.get('wind_speed', 0.0),
            weather_data.get('wind_direction', 0.0),
            weather_data.get('precipitation', 0.0),
            weather_data.get('visibility', 10.0),
            weather_data.get('cloud_cover', 50.0),
            weather_data.get('uv_index', 5.0),
            weather_data.get('dew_point', 10.0),
            weather_data.get('heat_index', 20.0),
            weather_data.get('wind_chill', 20.0),
            weather_data.get('precipitation_intensity', 0.0),
            weather_data.get('atmospheric_stability', 0.5),
            weather_data.get('moisture_content', 0.5)
        ])
        
        # Geospatial features
        features.extend([
            geospatial_data.get('latitude', 0.0),
            geospatial_data.get('longitude', 0.0),
            geospatial_data.get('elevation', 100.0),
            geospatial_data.get('slope', 5.0),
            geospatial_data.get('aspect', 180.0),
            geospatial_data.get('soil_type', 2),
            geospatial_data.get('land_use', 3),
            geospatial_data.get('distance_to_water', 10.0),
            geospatial_data.get('distance_to_fault', 50.0),
            geospatial_data.get('population_density', 1000.0),
            geospatial_data.get('infrastructure_density', 0.5),
            geospatial_data.get('historical_events', 5),
            geospatial_data.get('tectonic_zone', 0),
            geospatial_data.get('climate_zone', 3),
            geospatial_data.get('vegetation_index', 0.6),
            geospatial_data.get('urbanization_level', 0.4)
        ])
        
        # Disaster-specific features
        if disaster_type == 'flood':
            features.extend([
                geospatial_data.get('river_level', 50.0),
                geospatial_data.get('drainage_efficiency', 0.5),
                geospatial_data.get('flood_control_infrastructure', 0.5)
            ])
        elif disaster_type == 'earthquake':
            features.extend([
                geospatial_data.get('seismic_activity_level', 0.1),
                geospatial_data.get('ground_acceleration', 0.1),
                geospatial_data.get('soil_liquefaction_potential', 0.1)
            ])
        elif disaster_type == 'landslide':
            features.extend([
                geospatial_data.get('soil_saturation', 0.3),
                geospatial_data.get('slope_stability_index', 0.7),
                geospatial_data.get('vegetation_cover', 0.6)
            ])
        
        return np.array(features)
    
    def _rule_based_enhanced_risk(self, disaster_type: str, weather_data: Dict, geospatial_data: Dict) -> float:
        """Enhanced rule-based fallback with geospatial context"""
        temp = weather_data.get('temperature', 20.0)
        humidity = weather_data.get('humidity', 50.0)
        precip = weather_data.get('precipitation', 0.0)
        wind_speed = weather_data.get('wind_speed', 0.0)
        pressure = weather_data.get('pressure', 1013.0)
        
        lat = geospatial_data.get('latitude', 0.0)
        elevation = geospatial_data.get('elevation', 100.0)
        slope = geospatial_data.get('slope', 5.0)
        dist_water = geospatial_data.get('distance_to_water', 10.0)
        dist_fault = geospatial_data.get('distance_to_fault', 50.0)
        tectonic_zone = geospatial_data.get('tectonic_zone', 0)
        
        if disaster_type == 'flood':
            # Enhanced flood risk with geospatial factors
            precip_risk = min(1.0, precip / 50.0)
            elevation_risk = min(1.0, (100 - elevation) / 100.0) if elevation < 100 else 0.1
            water_risk = min(1.0, (50 - dist_water) / 50.0) if dist_water < 50 else 0.1
            return (precip_risk * 0.5 + elevation_risk * 0.3 + water_risk * 0.2)
            
        elif disaster_type == 'earthquake':
            # Enhanced earthquake risk with tectonic context
            tectonic_risk = {0: 0.1, 1: 0.8, 2: 0.7, 3: 0.6, 4: 0.5}.get(tectonic_zone, 0.1)
            fault_risk = min(1.0, (100 - dist_fault) / 100.0) if dist_fault < 100 else 0.1
            return (tectonic_risk * 0.7 + fault_risk * 0.3)
            
        elif disaster_type == 'landslide':
            # Enhanced landslide risk with slope and soil factors
            slope_risk = min(1.0, slope / 45.0)
            precip_risk = min(1.0, precip / 80.0)
            return (slope_risk * 0.6 + precip_risk * 0.4)
            
        else:
            # Basic weather-based risk for other types
            return min(1.0, (precip / 50.0) * 0.4 + (temp / 40.0) * 0.3 + (wind_speed / 30.0) * 0.3)
    
    def get_model_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all models"""
        return {
            disaster_type: model.performance_metrics 
            for disaster_type, model in self.models.items()
            if hasattr(model, 'performance_metrics')
        }
    
    def get_feature_importance(self) -> Dict[str, Dict]:
        """Get feature importance for all models"""
        return {
            disaster_type: model.feature_importance 
            for disaster_type, model in self.models.items()
            if hasattr(model, 'feature_importance')
        }

# Global enhanced prediction service instance
enhanced_ai_prediction_service = EnhancedDisasterPredictionService()

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from datetime import datetime, timedelta
import json
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
# Try to import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. XGBoost models will be disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. LightGBM models will be disabled.")

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Deep learning models will be disabled.")

# Try to import Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter optimization will be disabled.")

from scipy import stats

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Model interpretability will be disabled.")

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Advanced Enterprise Configuration
EARTHQUAKE_RISK_MULTIPLIER: float = float(os.getenv('EARTHQUAKE_RISK_MULTIPLIER', '0.05'))
ALLOW_EARTHQUAKE_PREDICTIONS: bool = os.getenv('ALLOW_EARTHQUAKE_PREDICTIONS', 'false').lower() == 'true'
MODEL_VERSION = os.getenv('MODEL_VERSION', '3.0.0')
ENSEMBLE_ENABLED = os.getenv('ENSEMBLE_ENABLED', 'true').lower() == 'true'
DEEP_LEARNING_ENABLED = os.getenv('DEEP_LEARNING_ENABLED', 'true').lower() == 'true' and TENSORFLOW_AVAILABLE
HYPERPARAMETER_OPTIMIZATION = os.getenv('HYPERPARAMETER_OPTIMIZATION', 'true').lower() == 'true' and OPTUNA_AVAILABLE
FEATURE_ENGINEERING_ENABLED = os.getenv('FEATURE_ENGINEERING_ENABLED', 'true').lower() == 'true'

class AdvancedDisasterPredictionService:
    """Enterprise-level service for managing advanced disaster prediction models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.allow_earthquake_predictions = ALLOW_EARTHQUAKE_PREDICTIONS
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize model registry with advanced features
        self._initialize_advanced_model_registry()
        self.load_or_initialize_models()
        
        # Auto-train if needed
        try:
            need_training = len(self.scalers) < len(self.models) or not self._check_model_health()
            auto_train = os.getenv('AI_AUTO_TRAIN_ON_STARTUP', 'true').lower() == 'true'
            if need_training and auto_train:
                logger.info("Advanced AI: Auto-training deep learning models on startup...")
                self.train_advanced_models(epochs=int(os.getenv('AI_STARTUP_TRAIN_EPOCHS', '100')))
        except Exception as e:
            logger.error(f"Auto-train on startup failed: {e}")
    
    def _initialize_advanced_model_registry(self):
        """Initialize the advanced model registry with deep learning and ensemble features"""
        # Build available models list based on dependencies
        available_models = ['random_forest', 'gradient_boosting']
        if XGBOOST_AVAILABLE:
            available_models.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            available_models.append('lightgbm')
        if TENSORFLOW_AVAILABLE:
            available_models.extend(['deep_neural_network', 'lstm', 'conv1d'])
        
        self.model_registry = {
            'flood': {
                'type': 'deep_ensemble',
                'models': available_models,
                'features': ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation', 'visibility', 'cloud_cover', 
                           'soil_moisture', 'river_level', 'drainage_capacity', 'elevation', 'slope'],
                'advanced_features': ['precipitation_intensity', 'flood_index', 'water_accumulation_rate', 'drainage_efficiency'],
                'data_sources': ['ERA5', 'GDACS', 'NASA_SRTM', 'USGS_Hydro', 'NOAA_Rivers'],
                'update_frequency': 'hourly',
                'ensemble_weight': 0.8
            },
            'wildfire': {
                'type': 'deep_ensemble',
                'models': available_models,
                'features': ['temperature', 'humidity', 'wind_speed', 'precipitation', 'visibility', 'fuel_moisture',
                           'vegetation_index', 'drought_index', 'fire_weather_index'],
                'advanced_features': ['fire_danger_index', 'ignition_probability', 'spread_rate', 'fuel_load'],
                'data_sources': ['FIRMS', 'ERA5', 'MODIS', 'LANDSAT', 'NOAA_Fire_Weather'],
                'update_frequency': 'hourly',
                'ensemble_weight': 0.85
            },
            'storm': {
                'type': 'deep_ensemble',
                'models': available_models,
                'features': ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'cloud_cover',
                           'atmospheric_stability', 'convective_available_potential_energy', 'wind_shear'],
                'advanced_features': ['storm_intensity', 'lightning_probability', 'hail_probability', 'tornado_probability'],
                'data_sources': ['ERA5', 'NOAA', 'NASA', 'GOES_Satellite', 'NEXRAD_Radar'],
                'update_frequency': 'hourly',
                'ensemble_weight': 0.9
            },
            'earthquake': {
                'type': 'advanced_heuristic',
                'models': ['rule_based', 'statistical', 'deep_neural_network'],
                'features': ['seismic_activity', 'fault_lines', 'historical_events', 'stress_accumulation',
                           'ground_deformation', 'gravity_anomalies'],
                'advanced_features': ['seismic_hazard', 'rupture_probability', 'magnitude_estimate'],
                'data_sources': ['USGS', 'IRIS', 'GPS_Deformation', 'InSAR', 'Gravity_Satellites'],
                'update_frequency': 'real_time',
                'ensemble_weight': 0.3
            },
            'tornado': {
                'type': 'deep_ensemble',
                'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'deep_neural_network', 'lstm'],
                'features': ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure', 'cloud_cover',
                           'convective_available_potential_energy', 'wind_shear', 'helicity'],
                'advanced_features': ['tornado_probability', 'supercell_detection', 'mesocyclone_strength'],
                'data_sources': ['ERA5', 'NOAA', 'SPC', 'NEXRAD_Radar', 'GOES_Satellite'],
                'update_frequency': 'hourly',
                'ensemble_weight': 0.95
            },
            'landslide': {
                'type': 'deep_ensemble',
                'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'deep_neural_network'],
                'features': ['precipitation', 'temperature', 'humidity', 'pressure', 'slope', 'elevation',
                           'soil_type', 'vegetation_cover', 'geological_structure'],
                'advanced_features': ['slope_stability_index', 'rainfall_threshold', 'ground_saturation'],
                'data_sources': ['GDACS', 'ERA5', 'NASA_SRTM', 'USGS_Geology', 'Soil_Maps'],
                'update_frequency': 'daily',
                'ensemble_weight': 0.75
            },
            'drought': {
                'type': 'deep_ensemble',
                'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'deep_neural_network', 'lstm'],
                'features': ['temperature', 'precipitation', 'humidity', 'pressure', 'soil_moisture',
                           'vegetation_index', 'evapotranspiration', 'groundwater_level'],
                'advanced_features': ['drought_severity_index', 'water_deficit', 'vegetation_stress'],
                'data_sources': ['ERA5', 'NASA', 'USDA', 'GRACE_Satellite', 'MODIS'],
                'update_frequency': 'weekly',
                'ensemble_weight': 0.8
            }
        }
    
    def _check_model_health(self) -> bool:
        """Check if all models are healthy and up-to-date"""
        try:
            for hazard_type, registry in self.model_registry.items():
                if hazard_type not in self.models:
                    return False
                if hazard_type not in self.scalers:
                    return False
            return True
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False
    
    def create_advanced_features(self, data: pd.DataFrame, hazard_type: str) -> pd.DataFrame:
        """Create advanced engineered features for better prediction accuracy"""
        df = data.copy()
        
        # Common advanced features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
            df['comfort_index'] = self._calculate_comfort_index(df['temperature'], df['humidity'])
        
        if 'wind_speed' in df.columns:
            df['wind_power'] = df['wind_speed'] ** 3  # Wind power is proportional to cube of speed
            df['wind_category'] = pd.cut(df['wind_speed'], bins=[0, 10, 20, 30, 50, 100], labels=[1, 2, 3, 4, 5])
        
        if 'precipitation' in df.columns:
            df['precipitation_intensity'] = df['precipitation'].rolling(window=3).mean()
            df['precipitation_accumulation'] = df['precipitation'].rolling(window=24).sum()
        
        # Hazard-specific advanced features
        if hazard_type == 'flood':
            df = self._create_flood_features(df)
        elif hazard_type == 'wildfire':
            df = self._create_wildfire_features(df)
        elif hazard_type == 'storm':
            df = self._create_storm_features(df)
        elif hazard_type == 'tornado':
            df = self._create_tornado_features(df)
        elif hazard_type == 'landslide':
            df = self._create_landslide_features(df)
        elif hazard_type == 'drought':
            df = self._create_drought_features(df)
        
        return df
    
    def _create_flood_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create flood-specific advanced features"""
        if 'precipitation' in df.columns:
            df['flood_index'] = df['precipitation'] * (df.get('humidity', 50) / 100) * (df.get('temperature', 20) / 20)
            df['water_accumulation_rate'] = df['precipitation'].diff().fillna(0)
        
        if 'elevation' in df.columns and 'slope' in df.columns:
            df['drainage_efficiency'] = 1 / (1 + df['slope']) * (1 / (1 + df['elevation'] / 1000))
        
        return df
    
    def _create_wildfire_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create wildfire-specific advanced features"""
        if 'temperature' in df.columns and 'humidity' in df.columns and 'wind_speed' in df.columns:
            df['fire_danger_index'] = (df['temperature'] * df['wind_speed']) / (df['humidity'] + 1)
            df['ignition_probability'] = np.exp(-df['humidity'] / 30) * (df['temperature'] / 40)
        
        if 'fuel_moisture' in df.columns:
            df['fuel_load'] = 1 / (df['fuel_moisture'] + 0.1)
        
        return df
    
    def _create_storm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create storm-specific advanced features"""
        if 'temperature' in df.columns and 'humidity' in df.columns and 'pressure' in df.columns:
            df['storm_intensity'] = (df['humidity'] * df['temperature']) / (df['pressure'] + 900)
            df['lightning_probability'] = df['humidity'] * (df['temperature'] - 20) / 100
        
        return df
    
    def _create_tornado_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tornado-specific advanced features"""
        if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
            df['wind_shear'] = df['wind_speed'].diff().abs()
            df['helicity'] = df['wind_speed'] * np.sin(np.radians(df['wind_direction']))
        
        if 'temperature' in df.columns and 'humidity' in df.columns and 'pressure' in df.columns:
            df['supercell_probability'] = (df['humidity'] * df['temperature']) / (df['pressure'] + 900) * df.get('wind_shear', 0)
        
        return df
    
    def _create_landslide_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create landslide-specific advanced features"""
        if 'precipitation' in df.columns and 'slope' in df.columns:
            df['slope_stability_index'] = 1 / (1 + df['precipitation'] * df['slope'] / 100)
            df['rainfall_threshold'] = df['precipitation'].rolling(window=7).sum()
        
        return df
    
    def _create_drought_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create drought-specific advanced features"""
        if 'precipitation' in df.columns and 'temperature' in df.columns:
            df['drought_severity_index'] = (df['temperature'] - df['precipitation']) / 10
            df['water_deficit'] = df['precipitation'].rolling(window=30).sum() - df['temperature'].rolling(window=30).mean()
        
        return df
    
    def _generate_advanced_features(self, weather_data: Dict) -> Dict[str, float]:
        """Generate advanced features for better prediction accuracy"""
        features = weather_data.copy()
        
        # Add derived features
        if 'temperature' in features and 'humidity' in features:
            features['heat_index'] = self._calculate_heat_index(features['temperature'], features['humidity'])
        
        if 'wind_speed' in features:
            features['wind_power'] = features['wind_speed'] ** 3
        
        if 'precipitation' in features:
            features['precipitation_intensity'] = self._categorize_precipitation_intensity(features['precipitation'])
        
        return features
    
    def _calculate_heat_index(self, temp: float, humidity: float) -> float:
        """Calculate heat index using NOAA formula"""
        if temp < 80:
            return temp
        hi = 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (humidity * 0.094))
        return min(hi, temp + humidity * 0.1)
    
    def _calculate_comfort_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate comfort index"""
        return (temp + humidity) / 2
    
    def _categorize_precipitation_intensity(self, precipitation: float) -> float:
        """Categorize precipitation intensity"""
        if precipitation < 0.1:
            return 0.0
        elif precipitation < 2.5:
            return 0.3
        elif precipitation < 7.5:
            return 0.6
        else:
            return 1.0
    
    def _generate_synthetic_training_data(self, hazard_type: str, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate high-quality synthetic training data for each hazard type"""
        np.random.seed(42)
        
        # Generate base features
        temperatures = np.random.uniform(-20, 45, num_samples)
        humidities = np.random.uniform(10, 100, num_samples)
        pressures = np.random.uniform(950, 1050, num_samples)
        wind_speeds = np.random.uniform(0, 30, num_samples)
        wind_directions = np.random.uniform(0, 360, num_samples)
        precipitations = np.random.uniform(0, 50, num_samples)
        visibilities = np.random.uniform(0, 20, num_samples)
        cloud_covers = np.random.uniform(0, 100, num_samples)
        
        # Generate advanced features
        heat_indices = np.array([self._calculate_heat_index(t, h) for t, h in zip(temperatures, humidities)])
        wind_powers = wind_speeds ** 3
        precip_intensities = np.array([self._categorize_precipitation_intensity(p) for p in precipitations])
        
        # Create feature matrix
        features = np.column_stack([
            temperatures, humidities, pressures, wind_speeds, wind_directions,
            precipitations, visibilities, cloud_covers, heat_indices, wind_powers, precip_intensities
        ])
        
        # Generate labels based on hazard-specific rules
        labels = np.zeros(num_samples)
        
        if hazard_type == 'flood':
            flood_risk = (precipitations / 50.0) * 0.4 + (1 - visibilities / 20.0) * 0.3 + (humidities / 100.0) * 0.3
            labels = (flood_risk > 0.6).astype(int)
        elif hazard_type == 'wildfire':
            wildfire_risk = (temperatures / 45.0) * 0.4 + (1 - humidities / 100.0) * 0.4 + (wind_speeds / 30.0) * 0.2
            labels = (wildfire_risk > 0.7).astype(int)
        elif hazard_type == 'storm':
            storm_risk = (1 - pressures / 1050.0) * 0.4 + (wind_speeds / 30.0) * 0.4 + (cloud_covers / 100.0) * 0.2
            labels = (storm_risk > 0.6).astype(int)
        elif hazard_type == 'tornado':
            tornado_risk = (1 - pressures / 1050.0) * 0.3 + (wind_speeds / 30.0) * 0.4 + (humidities / 100.0) * 0.3
            labels = (tornado_risk > 0.8).astype(int)
        elif hazard_type == 'landslide':
            landslide_risk = (precipitations / 50.0) * 0.6 + (1 - pressures / 1050.0) * 0.4
            labels = (landslide_risk > 0.7).astype(int)
        elif hazard_type == 'drought':
            drought_risk = (1 - precipitations / 50.0) * 0.4 + (temperatures / 45.0) * 0.3 + (1 - humidities / 100.0) * 0.3
            labels = (drought_risk > 0.7).astype(int)
        
        return features, labels
    
    def train_advanced_models(self, epochs: int = 100) -> Dict[str, Any]:
        """Train advanced models with deep learning and ensemble methods"""
        results = {}
        
        for hazard_type in self.model_registry.keys():
            try:
                logger.info(f"Training advanced models for {hazard_type}...")
                
                # Generate synthetic data with advanced features
                synthetic_data = self._generate_advanced_synthetic_data(hazard_type, n_samples=10000)
                
                # Create advanced features
                synthetic_data = self.create_advanced_features(synthetic_data, hazard_type)
                
                # Prepare features and target
                feature_columns = self.model_registry[hazard_type]['features'] + self.model_registry[hazard_type]['advanced_features']
                available_features = [col for col in feature_columns if col in synthetic_data.columns]
                
                X = synthetic_data[available_features].fillna(0)
                y = synthetic_data['disaster_occurred'].astype(int)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Scale features
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train ensemble models
                ensemble_models = {}
                
                # Random Forest
                if HYPERPARAMETER_OPTIMIZATION:
                    rf_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'random_forest')
                    rf_model = RandomForestClassifier(**rf_params, random_state=42)
                else:
                    rf_model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
                    rf_model.fit(X_train_scaled, y_train)
                    ensemble_models['random_forest'] = rf_model
                
                # XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        if HYPERPARAMETER_OPTIMIZATION:
                            xgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgboost')
                            xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
                        else:
                            xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
                        xgb_model.fit(X_train_scaled, y_train)
                        ensemble_models['xgboost'] = xgb_model
                    except Exception as e:
                        logger.warning(f"XGBoost training failed for {hazard_type}: {e}")
                
                # LightGBM
                if LIGHTGBM_AVAILABLE:
                    try:
                        if HYPERPARAMETER_OPTIMIZATION:
                            lgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'lightgbm')
                            lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
                        else:
                            lgb_model = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=42)
                        lgb_model.fit(X_train_scaled, y_train)
                        ensemble_models['lightgbm'] = lgb_model
                    except Exception as e:
                        logger.warning(f"LightGBM training failed for {hazard_type}: {e}")
                
                # Deep Neural Network
                if DEEP_LEARNING_ENABLED:
                    try:
                        dnn_model = self.create_deep_neural_network(X_train_scaled.shape[1])
                        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
                        
                        dnn_model.fit(
                            X_train_scaled, y_train,
                            epochs=epochs,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping, reduce_lr],
                            verbose=0
                        )
                        ensemble_models['deep_neural_network'] = dnn_model
                    except Exception as e:
                        logger.warning(f"Deep learning training failed for {hazard_type}: {e}")
                
                # Create voting ensemble
                if len(ensemble_models) > 1:
                    estimators = [(name, model) for name, model in ensemble_models.items()]
                    ensemble = VotingClassifier(estimators=estimators, voting='soft')
                    ensemble.fit(X_train_scaled, y_train)
                    ensemble_models['ensemble'] = ensemble
                
                # Evaluate models
                model_performance = {}
                for name, model in ensemble_models.items():
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        y_pred = model.predict(X_test_scaled)
                    else:
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = y_pred
                    
                    accuracy = (y_pred == y_test).mean()
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                    auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
                    
                    model_performance[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc
                    }
                
                # Store models and scaler
                self.models[hazard_type] = ensemble_models
                self.scalers[hazard_type] = scaler
                self.model_performance[hazard_type] = model_performance
                
                # Calculate feature importance
                if 'random_forest' in ensemble_models:
                    self.feature_importance[hazard_type] = dict(zip(
                        available_features,
                        ensemble_models['random_forest'].feature_importances_
                    ))
                
                results[hazard_type] = {
                    'status': 'success',
                    'models_trained': list(ensemble_models.keys()),
                    'performance': model_performance,
                    'features_used': available_features
                }
                
                logger.info(f"Advanced training completed for {hazard_type}")
                
            except Exception as e:
                logger.error(f"Advanced training failed for {hazard_type}: {e}")
                results[hazard_type] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def _ensemble_predict(self, models: Dict, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions using weighted voting"""
        if not ENSEMBLE_ENABLED or len(models) == 1:
            model_name = list(models.keys())[0]
            return models[model_name].predict(X)
        
        # Weighted ensemble prediction
        predictions = []
        weights = []
        
        for model_name, model in models.items():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
            
            if model_name == 'random_forest':
                weights.append(0.4)
            elif model_name == 'gradient_boosting':
                weights.append(0.35)
            elif model_name == 'neural_network':
                weights.append(0.25)
            else:
                weights.append(0.1)
        
        weights = np.array(weights) / sum(weights)
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_disaster_risks(self, weather_data: Dict) -> Dict[str, float]:
        """Predict disaster risks using advanced ensemble models"""
        predictions = {}
        
        try:
            # Generate advanced features
            advanced_features = self._generate_advanced_features(weather_data)
            
            # Convert to feature vector
            feature_vector = self._extract_feature_vector(advanced_features)
            
            for hazard_type, registry in self.model_registry.items():
                if hazard_type == 'earthquake' and not self.allow_earthquake_predictions:
                    predictions[hazard_type] = 0.0
                    continue
                
                try:
                    if (hazard_type in self.models and hazard_type in self.scalers and 
                        self.models[hazard_type] and self.scalers[hazard_type]):
                        # Use ensemble model
                        models = self.models[hazard_type]
                        scaler = self.scalers[hazard_type]
                        
                        # Scale features
                        features_scaled = scaler.transform([feature_vector])
                        
                        # Make ensemble prediction
                        risk_probability = self._ensemble_predict_proba(models, features_scaled)
                        predictions[hazard_type] = float(risk_probability)
                        logger.debug(f"Ensemble prediction for {hazard_type}: {risk_probability:.3f}")
                        
                    else:
                        # Fallback to heuristic
                        risk_probability = self._heuristic_prediction(hazard_type, advanced_features)
                        predictions[hazard_type] = float(risk_probability)
                        logger.debug(f"Heuristic prediction for {hazard_type}: {risk_probability:.3f}")
                        
                except Exception as e:
                    logger.error(f"Prediction failed for {hazard_type}: {e}")
                    # Emergency fallback to basic heuristic
                    try:
                        risk_probability = self._heuristic_prediction(hazard_type, weather_data)
                        predictions[hazard_type] = float(risk_probability)
                        logger.info(f"Emergency heuristic fallback for {hazard_type}: {risk_probability:.3f}")
                    except Exception as fallback_error:
                        logger.error(f"Even heuristic fallback failed for {hazard_type}: {fallback_error}")
                        # Last resort: return a safe default
                        predictions[hazard_type] = 0.1
            
            # Ensure we have predictions for all hazard types
            if not predictions:
                logger.warning("No predictions generated, using default heuristic values")
                for hazard_type in self.model_registry.keys():
                    if hazard_type != 'earthquake' or self.allow_earthquake_predictions:
                        predictions[hazard_type] = self._heuristic_prediction(hazard_type, weather_data)
            
            logger.info(f"Generated predictions for {len(predictions)} hazard types")
            return predictions
            
        except Exception as e:
            logger.error(f"Critical error in predict_disaster_risks: {e}")
            # Emergency fallback - return basic heuristic predictions
            emergency_predictions = {}
            for hazard_type in self.model_registry.keys():
                if hazard_type != 'earthquake' or self.allow_earthquake_predictions:
                    try:
                        emergency_predictions[hazard_type] = self._heuristic_prediction(hazard_type, weather_data)
                    except Exception:
                        emergency_predictions[hazard_type] = 0.1
            
            logger.info(f"Emergency fallback predictions generated: {len(emergency_predictions)} hazard types")
            return emergency_predictions
    
    def _extract_feature_vector(self, features: Dict) -> List[float]:
        """Extract feature vector in the correct order"""
        feature_order = [
            'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
            'precipitation', 'visibility', 'cloud_cover', 'heat_index', 'wind_power', 'precipitation_intensity'
        ]
        
        feature_vector = []
        for feature in feature_order:
            feature_vector.append(features.get(feature, 0.0))
        
        return feature_vector
    
    def _ensemble_predict_proba(self, models: Dict, X: np.ndarray) -> float:
        """Get ensemble probability prediction"""
        if not ENSEMBLE_ENABLED or len(models) == 1:
            model_name = list(models.keys())[0]
            return models[model_name].predict_proba(X)[0, 1]
        
        # Weighted ensemble probability
        probabilities = []
        weights = []
        
        for model_name, model in models.items():
            prob = model.predict_proba(X)[0, 1]
            probabilities.append(prob)
            
            if model_name == 'random_forest':
                weights.append(0.4)
            elif model_name == 'gradient_boosting':
                weights.append(0.35)
            elif model_name == 'neural_network':
                weights.append(0.25)
            else:
                weights.append(0.1)
        
        weights = np.array(weights) / sum(weights)
        ensemble_prob = sum(p * w for p, w in zip(probabilities, weights))
        
        return ensemble_prob
    
    def _heuristic_prediction(self, hazard_type: str, features: Dict) -> float:
        """Fallback heuristic prediction"""
        if hazard_type == 'flood':
            precip = features.get('precipitation', 0)
            humidity = features.get('humidity', 50)
            return min(1.0, (precip / 20.0) * 0.7 + (humidity / 100.0) * 0.3)
        elif hazard_type == 'wildfire':
            temp = features.get('temperature', 20)
            humidity = features.get('humidity', 50)
            wind_speed = features.get('wind_speed', 5)
            return min(1.0, (temp / 40.0) * 0.4 + (1 - humidity / 100.0) * 0.4 + (wind_speed / 25.0) * 0.2)
        elif hazard_type == 'storm':
            pressure = features.get('pressure', 1013)
            wind_speed = features.get('wind_speed', 5)
            return min(1.0, (1 - pressure / 1050.0) * 0.6 + (wind_speed / 25.0) * 0.4)
        elif hazard_type == 'tornado':
            pressure = features.get('pressure', 1013)
            wind_speed = features.get('wind_speed', 5)
            humidity = features.get('humidity', 50)
            return min(1.0, (1 - pressure / 1050.0) * 0.4 + (wind_speed / 25.0) * 0.4 + (humidity / 100.0) * 0.2)
        elif hazard_type == 'landslide':
            precip = features.get('precipitation', 0)
            pressure = features.get('pressure', 1013)
            return min(1.0, (precip / 25.0) * 0.7 + (1 - pressure / 1050.0) * 0.3)
        elif hazard_type == 'drought':
            precip = features.get('precipitation', 0)
            temp = features.get('temperature', 20)
            humidity = features.get('humidity', 50)
            return min(1.0, (1 - precip / 25.0) * 0.4 + (temp / 40.0) * 0.3 + (1 - humidity / 100.0) * 0.3)
        else:
            return 0.0
    
    def _save_models(self, hazard_type: str, models: Dict, scaler: StandardScaler):
        """Save models and metadata to disk"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Save models
            for model_name, model in models.items():
                model_path = os.path.join(self.model_dir, f'{hazard_type}_{model_name}_model.joblib')
                joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f'{hazard_type}_scaler.joblib')
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata_path = os.path.join(self.model_dir, f'{hazard_type}_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata[hazard_type], f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save models for {hazard_type}: {e}")
    
    def load_or_initialize_models(self):
        """Load existing models or initialize new ones"""
        try:
            for hazard_type in self.model_registry.keys():
                if hazard_type == 'earthquake' and not self.allow_earthquake_predictions:
                    continue
                
                # Try to load existing models
                models = {}
                scaler = None
                
                for model_name in self.model_registry[hazard_type]['models']:
                    model_path = os.path.join(self.model_dir, f'{hazard_type}_{model_name}_model.joblib')
                    if os.path.exists(model_path):
                        models[model_name] = joblib.load(model_path)
                
                scaler_path = os.path.join(self.model_dir, f'{hazard_type}_scaler.joblib')
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                
                metadata_path = os.path.join(self.model_dir, f'{hazard_type}_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.model_metadata[hazard_type] = json.load(f)
                
                if models and scaler:
                    self.models[hazard_type] = models
                    self.scalers[hazard_type] = scaler
                    logger.info(f"Loaded existing models for {hazard_type}")
                else:
                    logger.info(f"No existing models found for {hazard_type}, will train new ones")
                    
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status information"""
        status = {
            'version': MODEL_VERSION,
            'ensemble_enabled': ENSEMBLE_ENABLED,
            'models': {},
            'overall_health': 'healthy',
            'last_updated': datetime.now().isoformat()
        }
        
        for hazard_type, registry in self.model_registry.items():
            model_info = {
                'type': registry['type'],
                'models_loaded': hazard_type in self.models,
                'scaler_loaded': hazard_type in self.scalers,
                'data_sources': registry['data_sources'],
                'update_frequency': registry['update_frequency']
            }
            
            if hazard_type in self.model_metadata:
                model_info.update(self.model_metadata[hazard_type])
            
            status['models'][hazard_type] = model_info
            
            if not model_info['models_loaded'] or not model_info['scaler_loaded']:
                status['overall_health'] = 'degraded'
        
        return status

    def _generate_advanced_synthetic_data(self, hazard_type: str, n_samples: int = 10000) -> pd.DataFrame:
        """Generate advanced synthetic data with more realistic patterns"""
        np.random.seed(42)
        
        # Base weather parameters
        data = {
            'temperature': np.random.normal(20, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'pressure': np.random.normal(1013, 20, n_samples),
            'wind_speed': np.random.exponential(5, n_samples),
            'precipitation': np.random.exponential(2, n_samples),
            'visibility': np.random.uniform(5, 20, n_samples),
            'cloud_cover': np.random.uniform(0, 100, n_samples)
        }
        
        # Hazard-specific parameters
        if hazard_type == 'flood':
            data.update({
                'soil_moisture': np.random.uniform(0.1, 0.9, n_samples),
                'river_level': np.random.uniform(0, 10, n_samples),
                'drainage_capacity': np.random.uniform(0.1, 1.0, n_samples),
                'elevation': np.random.uniform(0, 1000, n_samples),
                'slope': np.random.uniform(0, 30, n_samples)
            })
        elif hazard_type == 'wildfire':
            data.update({
                'fuel_moisture': np.random.uniform(5, 30, n_samples),
                'vegetation_index': np.random.uniform(0.1, 0.8, n_samples),
                'drought_index': np.random.uniform(0, 5, n_samples),
                'fire_weather_index': np.random.uniform(0, 100, n_samples)
            })
        elif hazard_type == 'storm':
            data.update({
                'wind_direction': np.random.uniform(0, 360, n_samples),
                'atmospheric_stability': np.random.uniform(-5, 5, n_samples),
                'convective_available_potential_energy': np.random.uniform(0, 3000, n_samples),
                'wind_shear': np.random.uniform(0, 50, n_samples)
            })
        elif hazard_type == 'tornado':
            data.update({
                'wind_direction': np.random.uniform(0, 360, n_samples),
                'convective_available_potential_energy': np.random.uniform(0, 3000, n_samples),
                'wind_shear': np.random.uniform(0, 50, n_samples),
                'helicity': np.random.uniform(0, 500, n_samples)
            })
        elif hazard_type == 'landslide':
            data.update({
                'slope': np.random.uniform(0, 45, n_samples),
                'elevation': np.random.uniform(0, 2000, n_samples),
                'soil_type': np.random.randint(1, 6, n_samples),
                'vegetation_cover': np.random.uniform(0, 100, n_samples),
                'geological_structure': np.random.randint(1, 4, n_samples)
            })
        elif hazard_type == 'drought':
            data.update({
                'soil_moisture': np.random.uniform(0.05, 0.4, n_samples),
                'vegetation_index': np.random.uniform(0.1, 0.6, n_samples),
                'evapotranspiration': np.random.uniform(1, 8, n_samples),
                'groundwater_level': np.random.uniform(-10, 0, n_samples)
            })
        
        df = pd.DataFrame(data)
        
        # Create disaster occurrence based on hazard-specific conditions
        disaster_occurred = np.zeros(n_samples, dtype=bool)
        
        if hazard_type == 'flood':
            disaster_occurred = (
                (df['precipitation'] > 10) & 
                (df['humidity'] > 70) & 
                (df['soil_moisture'] > 0.7)
            )
        elif hazard_type == 'wildfire':
            disaster_occurred = (
                (df['temperature'] > 30) & 
                (df['humidity'] < 40) & 
                (df['fuel_moisture'] < 15) &
                (df['wind_speed'] > 10)
            )
        elif hazard_type == 'storm':
            disaster_occurred = (
                (df['wind_speed'] > 20) & 
                (df['pressure'] < 1000) & 
                (df['convective_available_potential_energy'] > 1000)
            )
        elif hazard_type == 'tornado':
            disaster_occurred = (
                (df['wind_speed'] > 25) & 
                (df['wind_shear'] > 20) & 
                (df['convective_available_potential_energy'] > 1500)
            )
        elif hazard_type == 'landslide':
            disaster_occurred = (
                (df['precipitation'] > 15) & 
                (df['slope'] > 20) & 
                (df['soil_moisture'] > 0.8)
            )
        elif hazard_type == 'drought':
            disaster_occurred = (
                (df['precipitation'] < 2) & 
                (df['temperature'] > 25) & 
                (df['soil_moisture'] < 0.2)
            )
        
        df['disaster_occurred'] = disaster_occurred
        
        return df
    
    def create_deep_neural_network(self, input_dim: int, num_classes: int = 2):
        """Create a deep neural network for disaster prediction"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Deep learning models are disabled.")
        
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE:
            # Return default parameters if Optuna is not available
            if model_type == 'random_forest':
                return {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}
            elif model_type == 'xgboost':
                return {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8}
            elif model_type == 'lightgbm':
                return {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 50}
            else:
                return {}
        
        def objective(trial):
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(**params, random_state=42)
            
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                model = xgb.XGBClassifier(**params, random_state=42)
            
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100)
                }
                model = lgb.LGBMClassifier(**params, random_state=42)
            
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        return self.model_performance
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance for all models"""
        return self.feature_importance

# Initialize the enterprise service
ai_prediction_service = AdvancedDisasterPredictionService()

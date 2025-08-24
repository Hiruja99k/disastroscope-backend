"""
Enterprise Configuration for DisastroScope Backend
Provides centralized configuration management with environment-specific settings.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = True
    opentelemetry_enabled: bool = True
    jaeger_host: str = "localhost"
    jaeger_port: int = 6831
    metrics_interval: int = 30
    health_check_interval: int = 60

@dataclass
class AIConfig:
    """AI model configuration"""
    ensemble_enabled: bool = True
    auto_training: bool = True
    model_version: str = "2.0.0"
    training_epochs: int = 100
    prediction_threshold: float = 0.1
    max_prediction_time: float = 5.0

@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str
    allowed_origins: List[str]
    api_key_required: bool = False
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600

@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""
    worker_processes: int = 4
    worker_threads: int = 2
    max_connections: int = 1000
    request_timeout: int = 30
    response_timeout: int = 60
    cache_ttl: int = 300

@dataclass
class DataSourceConfig:
    """Data source configuration"""
    weather_api_key: str
    firms_api_token: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openfema_api_key: Optional[str] = None
    eonet_api_key: Optional[str] = None

class EnterpriseConfig:
    """Enterprise configuration manager"""
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'production')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Database configuration
        self.database = DatabaseConfig(
            url=os.getenv('DATABASE_URL', 'sqlite:///disastroscope.db'),
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600'))
        )
        
        # Redis configuration
        self.redis = RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            password=os.getenv('REDIS_PASSWORD'),
            db=int(os.getenv('REDIS_DB', '0')),
            max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '20'))
        )
        
        # Monitoring configuration
        self.monitoring = MonitoringConfig(
            prometheus_enabled=os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true',
            opentelemetry_enabled=os.getenv('OPENTELEMETRY_ENABLED', 'true').lower() == 'true',
            jaeger_host=os.getenv('JAEGER_HOST', 'localhost'),
            jaeger_port=int(os.getenv('JAEGER_PORT', '6831')),
            metrics_interval=int(os.getenv('METRICS_INTERVAL', '30')),
            health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))
        )
        
        # AI configuration
        self.ai = AIConfig(
            ensemble_enabled=os.getenv('ENSEMBLE_ENABLED', 'true').lower() == 'true',
            auto_training=os.getenv('AI_AUTO_TRAIN_ON_STARTUP', 'true').lower() == 'true',
            model_version=os.getenv('MODEL_VERSION', '2.0.0'),
            training_epochs=int(os.getenv('AI_STARTUP_TRAIN_EPOCHS', '100')),
            prediction_threshold=float(os.getenv('PREDICTION_THRESHOLD', '0.1')),
            max_prediction_time=float(os.getenv('MAX_PREDICTION_TIME', '5.0'))
        )
        
        # Security configuration
        self.security = SecurityConfig(
            secret_key=os.getenv('SECRET_KEY', 'your-secret-key-here'),
            allowed_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
            api_key_required=os.getenv('API_KEY_REQUIRED', 'false').lower() == 'true',
            rate_limit_enabled=os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            rate_limit_requests=int(os.getenv('RATE_LIMIT_REQUESTS', '100')),
            rate_limit_window=int(os.getenv('RATE_LIMIT_WINDOW', '3600'))
        )
        
        # Performance configuration
        self.performance = PerformanceConfig(
            worker_processes=int(os.getenv('WORKER_PROCESSES', '4')),
            worker_threads=int(os.getenv('WORKER_THREADS', '2')),
            max_connections=int(os.getenv('MAX_CONNECTIONS', '1000')),
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30')),
            response_timeout=int(os.getenv('RESPONSE_TIMEOUT', '60')),
            cache_ttl=int(os.getenv('CACHE_TTL', '300'))
        )
        
        # Data source configuration
        self.data_sources = DataSourceConfig(
            weather_api_key=os.getenv('WEATHER_API_KEY', ''),
            firms_api_token=os.getenv('FIRMS_API_TOKEN'),
            gemini_api_key=os.getenv('GEMINI_API_KEY'),
            openfema_api_key=os.getenv('OPENFEMA_API_KEY'),
            eonet_api_key=os.getenv('EONET_API_KEY')
        )
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask configuration dictionary"""
        return {
            'SECRET_KEY': self.security.secret_key,
            'ENVIRONMENT': self.environment,
            'DEBUG': self.debug,
            'TESTING': self.environment == 'testing',
            'DATABASE_URL': self.database.url,
            'REDIS_URL': f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}",
            'PROMETHEUS_ENABLED': self.monitoring.prometheus_enabled,
            'OPENTELEMETRY_ENABLED': self.monitoring.opentelemetry_enabled,
            'ENSEMBLE_ENABLED': self.ai.ensemble_enabled,
            'AUTO_TRAINING': self.ai.auto_training,
            'MODEL_VERSION': self.ai.model_version,
            'RATE_LIMIT_ENABLED': self.security.rate_limit_enabled,
            'RATE_LIMIT_REQUESTS': self.security.rate_limit_requests,
            'RATE_LIMIT_WINDOW': self.security.rate_limit_window,
            'CACHE_TTL': self.performance.cache_ttl,
            'REQUEST_TIMEOUT': self.performance.request_timeout,
            'RESPONSE_TIMEOUT': self.performance.response_timeout
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            'origins': self.security.allowed_origins,
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allow_headers': ['Content-Type', 'Authorization', 'X-API-Key'],
            'max_age': 600
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            'prometheus_enabled': self.monitoring.prometheus_enabled,
            'opentelemetry_enabled': self.monitoring.opentelemetry_enabled,
            'jaeger_host': self.monitoring.jaeger_host,
            'jaeger_port': self.monitoring.jaeger_port,
            'metrics_interval': self.monitoring.metrics_interval,
            'health_check_interval': self.monitoring.health_check_interval
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration"""
        return {
            'ensemble_enabled': self.ai.ensemble_enabled,
            'auto_training': self.ai.auto_training,
            'model_version': self.ai.model_version,
            'training_epochs': self.ai.training_epochs,
            'prediction_threshold': self.ai.prediction_threshold,
            'max_prediction_time': self.ai.max_prediction_time
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Required environment variables
        if not self.security.secret_key or self.security.secret_key == 'your-secret-key-here':
            errors.append("SECRET_KEY must be set to a secure value")
        
        if not self.data_sources.weather_api_key:
            errors.append("WEATHER_API_KEY is required")
        
        # Validate database URL
        if not self.database.url:
            errors.append("DATABASE_URL is required")
        
        # Validate Redis configuration
        if not self.redis.host:
            errors.append("REDIS_HOST is required")
        
        # Validate performance settings
        if self.performance.worker_processes < 1:
            errors.append("WORKER_PROCESSES must be at least 1")
        
        if self.performance.worker_threads < 1:
            errors.append("WORKER_THREADS must be at least 1")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'database': {
                'url': self.database.url,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'pool_recycle': self.database.pool_recycle
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db,
                'max_connections': self.redis.max_connections
            },
            'monitoring': {
                'prometheus_enabled': self.monitoring.prometheus_enabled,
                'opentelemetry_enabled': self.monitoring.opentelemetry_enabled,
                'jaeger_host': self.monitoring.jaeger_host,
                'jaeger_port': self.monitoring.jaeger_port,
                'metrics_interval': self.monitoring.metrics_interval,
                'health_check_interval': self.monitoring.health_check_interval
            },
            'ai': {
                'ensemble_enabled': self.ai.ensemble_enabled,
                'auto_training': self.ai.auto_training,
                'model_version': self.ai.model_version,
                'training_epochs': self.ai.training_epochs,
                'prediction_threshold': self.ai.prediction_threshold,
                'max_prediction_time': self.ai.max_prediction_time
            },
            'security': {
                'allowed_origins': self.security.allowed_origins,
                'api_key_required': self.security.api_key_required,
                'rate_limit_enabled': self.security.rate_limit_enabled,
                'rate_limit_requests': self.security.rate_limit_requests,
                'rate_limit_window': self.security.rate_limit_window
            },
            'performance': {
                'worker_processes': self.performance.worker_processes,
                'worker_threads': self.performance.worker_threads,
                'max_connections': self.performance.max_connections,
                'request_timeout': self.performance.request_timeout,
                'response_timeout': self.performance.response_timeout,
                'cache_ttl': self.performance.cache_ttl
            },
            'data_sources': {
                'weather_api_key': '***' if self.data_sources.weather_api_key else None,
                'firms_api_token': '***' if self.data_sources.firms_api_token else None,
                'gemini_api_key': '***' if self.data_sources.gemini_api_key else None,
                'openfema_api_key': '***' if self.data_sources.openfema_api_key else None,
                'eonet_api_key': '***' if self.data_sources.eonet_api_key else None
            }
        }

# Global configuration instance
config = EnterpriseConfig()

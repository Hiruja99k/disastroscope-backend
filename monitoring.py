"""
Enterprise Monitoring and Observability System
Provides comprehensive monitoring, metrics, and health checks for the DisastroScope backend.
"""

import logging
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import threading
from collections import defaultdict, deque
import asyncio

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: Prometheus client not available. Metrics will be disabled.")

# OpenTelemetry for distributed tracing
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    print("Warning: OpenTelemetry not available. Tracing will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    active_connections: int
    requests_per_second: float
    average_response_time: float
    error_rate: float
    ai_predictions_made: int
    weather_requests: int
    disaster_events_processed: int
    timestamp: datetime

@dataclass
class ModelMetrics:
    """AI model performance metrics"""
    model_name: str
    prediction_accuracy: float
    inference_time_ms: float
    training_samples: int
    last_updated: datetime
    status: str

class EnterpriseMonitoring:
    """Enterprise-level monitoring and observability system"""
    
    def __init__(self, app=None):
        self.app = app
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metric points
        self.health_checks = {}
        self.alerts = []
        self.monitoring_enabled = True
        self.metrics_interval = 30  # seconds
        
        # Initialize Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        # Initialize OpenTelemetry
        if OPENTELEMETRY_AVAILABLE:
            self._init_opentelemetry()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize monitoring with Flask app"""
        self.app = app
        
        # Add monitoring routes
        app.add_url_rule('/metrics', 'metrics', self._metrics_endpoint)
        app.add_url_rule('/health', 'health', self._health_endpoint)
        app.add_url_rule('/monitoring/status', 'monitoring_status', self._monitoring_status)
        
        # Add middleware for request tracking
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        
        # Initialize OpenTelemetry instrumentation
        if OPENTELEMETRY_AVAILABLE:
            FlaskInstrumentor().instrument_app(app)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter('disastroscope_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('disastroscope_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
        
        # AI model metrics
        self.ai_prediction_counter = Counter('disastroscope_ai_predictions_total', 'Total AI predictions', ['model_type'])
        self.ai_prediction_duration = Histogram('disastroscope_ai_prediction_duration_seconds', 'AI prediction duration', ['model_type'])
        
        # System metrics
        self.cpu_gauge = Gauge('disastroscope_cpu_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('disastroscope_memory_percent', 'Memory usage percentage')
        self.disk_gauge = Gauge('disastroscope_disk_percent', 'Disk usage percentage')
        
        # Business metrics
        self.active_connections_gauge = Gauge('disastroscope_active_connections', 'Active WebSocket connections')
        self.weather_requests_counter = Counter('disastroscope_weather_requests_total', 'Total weather API requests')
        self.disaster_events_counter = Counter('disastroscope_disaster_events_total', 'Total disaster events processed')
    
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing"""
        # Set up trace provider
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Set up Jaeger exporter (for distributed tracing)
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv('JAEGER_HOST', 'localhost'),
            agent_port=int(os.getenv('JAEGER_PORT', 6831)),
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Set up metrics
        metric_reader = PeriodicExportingMetricReader(
            jaeger_exporter,
            export_interval_millis=5000
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
    
    def _before_request(self):
        """Track request start time"""
        request.start_time = time.time()
    
    def _after_request(self, response):
        """Track request completion"""
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            
            # Record Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.request_counter.labels(
                    method=request.method,
                    endpoint=request.endpoint,
                    status=response.status_code
                ).inc()
                
                self.request_duration.labels(
                    method=request.method,
                    endpoint=request.endpoint
                ).observe(duration)
        
        return response
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                
                # Store metrics
                self.metrics_history.append({
                    'system': asdict(system_metrics),
                    'application': asdict(app_metrics),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update Prometheus gauges
                if PROMETHEUS_AVAILABLE:
                    self.cpu_gauge.set(system_metrics.cpu_percent)
                    self.memory_gauge.set(system_metrics.memory_percent)
                    self.disk_gauge.set(system_metrics.disk_usage_percent)
                
                # Check for alerts
                self._check_alerts(system_metrics, app_metrics)
                
                # Sleep for next collection
                time.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            
            # Network usage
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                disk_usage_percent=disk_usage_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                disk_usage_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                timestamp=datetime.now()
            )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            # Calculate requests per second from recent history
            recent_requests = [m for m in self.metrics_history if 
                             (datetime.now() - datetime.fromisoformat(m['timestamp'])).seconds < 60]
            
            requests_per_second = len(recent_requests) / 60.0 if recent_requests else 0.0
            
            # Calculate average response time
            response_times = [m.get('response_time', 0) for m in recent_requests]
            average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Calculate error rate
            error_requests = [m for m in recent_requests if m.get('status_code', 200) >= 400]
            error_rate = len(error_requests) / len(recent_requests) if recent_requests else 0.0
            
            return ApplicationMetrics(
                active_connections=self._get_active_connections(),
                requests_per_second=requests_per_second,
                average_response_time=average_response_time,
                error_rate=error_rate,
                ai_predictions_made=self._get_ai_predictions_count(),
                weather_requests=self._get_weather_requests_count(),
                disaster_events_processed=self._get_disaster_events_count(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                active_connections=0,
                requests_per_second=0.0,
                average_response_time=0.0,
                error_rate=0.0,
                ai_predictions_made=0,
                weather_requests=0,
                disaster_events_processed=0,
                timestamp=datetime.now()
            )
    
    def _get_active_connections(self) -> int:
        """Get number of active WebSocket connections"""
        # This would need to be implemented based on your WebSocket setup
        return 0
    
    def _get_ai_predictions_count(self) -> int:
        """Get count of AI predictions made"""
        # This would need to be implemented based on your AI service
        return 0
    
    def _get_weather_requests_count(self) -> int:
        """Get count of weather API requests"""
        # This would need to be implemented based on your weather service
        return 0
    
    def _get_disaster_events_count(self) -> int:
        """Get count of disaster events processed"""
        # This would need to be implemented based on your event processing
        return 0
    
    def _check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # System alerts
        if system_metrics.cpu_percent > 80:
            alerts.append({
                'type': 'system',
                'severity': 'warning',
                'message': f'High CPU usage: {system_metrics.cpu_percent:.1f}%',
                'timestamp': datetime.now().isoformat()
            })
        
        if system_metrics.memory_percent > 85:
            alerts.append({
                'type': 'system',
                'severity': 'critical',
                'message': f'High memory usage: {system_metrics.memory_percent:.1f}%',
                'timestamp': datetime.now().isoformat()
            })
        
        if system_metrics.disk_usage_percent > 90:
            alerts.append({
                'type': 'system',
                'severity': 'critical',
                'message': f'High disk usage: {system_metrics.disk_usage_percent:.1f}%',
                'timestamp': datetime.now().isoformat()
            })
        
        # Application alerts
        if app_metrics.error_rate > 0.05:  # 5% error rate
            alerts.append({
                'type': 'application',
                'severity': 'warning',
                'message': f'High error rate: {app_metrics.error_rate:.2%}',
                'timestamp': datetime.now().isoformat()
            })
        
        if app_metrics.average_response_time > 2.0:  # 2 seconds
            alerts.append({
                'type': 'application',
                'severity': 'warning',
                'message': f'Slow response time: {app_metrics.average_response_time:.2f}s',
                'timestamp': datetime.now().isoformat()
            })
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts 
                      if datetime.fromisoformat(alert['timestamp']) > cutoff_time]
        
        # Log critical alerts
        for alert in alerts:
            if alert['severity'] == 'critical':
                logger.critical(f"CRITICAL ALERT: {alert['message']}")
            elif alert['severity'] == 'warning':
                logger.warning(f"WARNING: {alert['message']}")
    
    def _metrics_endpoint(self):
        """Prometheus metrics endpoint"""
        if not PROMETHEUS_AVAILABLE:
            return "Prometheus metrics not available", 503
        
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    def _health_endpoint(self):
        """Health check endpoint"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'checks': {}
        }
        
        # System health checks
        try:
            system_metrics = self._collect_system_metrics()
            health_status['checks']['system'] = {
                'cpu_usage': system_metrics.cpu_percent,
                'memory_usage': system_metrics.memory_percent,
                'disk_usage': system_metrics.disk_usage_percent,
                'status': 'healthy' if system_metrics.cpu_percent < 90 and 
                         system_metrics.memory_percent < 90 and 
                         system_metrics.disk_usage_percent < 95 else 'degraded'
            }
        except Exception as e:
            health_status['checks']['system'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Application health checks
        try:
            app_metrics = self._collect_application_metrics()
            health_status['checks']['application'] = {
                'requests_per_second': app_metrics.requests_per_second,
                'error_rate': app_metrics.error_rate,
                'average_response_time': app_metrics.average_response_time,
                'status': 'healthy' if app_metrics.error_rate < 0.1 and 
                         app_metrics.average_response_time < 5.0 else 'degraded'
            }
        except Exception as e:
            health_status['checks']['application'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Overall status
        all_healthy = all(check['status'] == 'healthy' 
                         for check in health_status['checks'].values())
        health_status['status'] = 'healthy' if all_healthy else 'degraded'
        
        return health_status, 200 if all_healthy else 503
    
    def _monitoring_status(self):
        """Get comprehensive monitoring status"""
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'opentelemetry_available': OPENTELEMETRY_AVAILABLE,
            'metrics_history_size': len(self.metrics_history),
            'active_alerts': len([a for a in self.alerts if a['severity'] == 'critical']),
            'last_metrics_collection': self.metrics_history[-1]['timestamp'] if self.metrics_history else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def record_ai_prediction(self, model_type: str, duration: float, accuracy: float = None):
        """Record AI prediction metrics"""
        if PROMETHEUS_AVAILABLE:
            self.ai_prediction_counter.labels(model_type=model_type).inc()
            self.ai_prediction_duration.labels(model_type=model_type).observe(duration)
    
    def record_weather_request(self):
        """Record weather API request"""
        if PROMETHEUS_AVAILABLE:
            self.weather_requests_counter.inc()
    
    def record_disaster_event(self):
        """Record disaster event processing"""
        if PROMETHEUS_AVAILABLE:
            self.disaster_events_counter.inc()
    
    def set_active_connections(self, count: int):
        """Set active WebSocket connections count"""
        if PROMETHEUS_AVAILABLE:
            self.active_connections_gauge.set(count)

# Global monitoring instance
monitoring = EnterpriseMonitoring()

import logging
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import structlog
from flask import Flask, jsonify, request
import json
import os

logger = structlog.get_logger()

class AdvancedMonitoringSystem:
    """Enterprise-level monitoring and observability system"""
    
    def __init__(self):
        self.metrics = {
            'system': {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'disk_usage': 0.0,
                'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
                'uptime': 0.0
            },
            'application': {
                'requests_per_second': 0.0,
                'error_rate': 0.0,
                'response_time_avg': 0.0,
                'active_connections': 0,
                'total_requests': 0,
                'total_errors': 0
            },
            'ai_models': {
                'predictions_made': 0,
                'training_jobs': 0,
                'model_accuracy': {},
                'prediction_latency': 0.0
            },
            'external_services': {
                'weather_api_calls': 0,
                'disaster_api_calls': 0,
                'api_response_times': {},
                'api_error_rates': {}
            }
        }
        
        self.health_checks = {
            'database': 'healthy',
            'ai_models': 'healthy',
            'external_apis': 'healthy',
            'memory': 'healthy',
            'disk': 'healthy'
        }
        
        self.alerts = []
        self.start_time = time.time()
        self.monitoring_thread = None
        self.is_running = False
        
        # Start monitoring thread
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.is_running:
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitoring_thread.start()
            logger.info("Advanced monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Advanced monitoring system stopped")
    
    def _monitor_system(self):
        """Background thread for system monitoring"""
        while self.is_running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Update health checks
                self._update_health_checks()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU usage
            self.metrics['system']['cpu_usage'] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['system']['memory_usage'] = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics['system']['disk_usage'] = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            self.metrics['system']['network_io'] = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
            # Uptime
            self.metrics['system']['uptime'] = time.time() - self.start_time
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _update_health_checks(self):
        """Update health check status"""
        try:
            # Memory health check
            if self.metrics['system']['memory_usage'] > 90:
                self.health_checks['memory'] = 'critical'
            elif self.metrics['system']['memory_usage'] > 80:
                self.health_checks['memory'] = 'warning'
            else:
                self.health_checks['memory'] = 'healthy'
            
            # Disk health check
            if self.metrics['system']['disk_usage'] > 90:
                self.health_checks['disk'] = 'critical'
            elif self.metrics['system']['disk_usage'] > 80:
                self.health_checks['disk'] = 'warning'
            else:
                self.health_checks['disk'] = 'healthy'
            
            # AI models health check
            if self.metrics['ai_models']['predictions_made'] == 0:
                self.health_checks['ai_models'] = 'warning'
            else:
                self.health_checks['ai_models'] = 'healthy'
            
            # External APIs health check
            total_errors = sum(self.metrics['external_services']['api_error_rates'].values())
            total_calls = sum(self.metrics['external_services']['api_response_times'].values())
            
            if total_calls > 0 and (total_errors / total_calls) > 0.1:
                self.health_checks['external_apis'] = 'critical'
            elif total_calls > 0 and (total_errors / total_calls) > 0.05:
                self.health_checks['external_apis'] = 'warning'
            else:
                self.health_checks['external_apis'] = 'healthy'
                
        except Exception as e:
            logger.error(f"Error updating health checks: {e}")
    
    def _check_alerts(self):
        """Check for alert conditions"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Memory alert
            if self.metrics['system']['memory_usage'] > 90:
                self._create_alert('CRITICAL', 'High memory usage', 
                                 f"Memory usage is {self.metrics['system']['memory_usage']:.1f}%")
            
            # Disk alert
            if self.metrics['system']['disk_usage'] > 90:
                self._create_alert('CRITICAL', 'High disk usage', 
                                 f"Disk usage is {self.metrics['system']['disk_usage']:.1f}%")
            
            # Error rate alert
            if self.metrics['application']['error_rate'] > 0.1:
                self._create_alert('WARNING', 'High error rate', 
                                 f"Error rate is {self.metrics['application']['error_rate']:.2f}")
            
            # AI model alert
            if self.metrics['ai_models']['predictions_made'] == 0:
                self._create_alert('WARNING', 'No AI predictions', 
                                 "AI models have not made any predictions recently")
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _create_alert(self, severity: str, title: str, message: str):
        """Create a new alert"""
        alert = {
            'id': f"alert_{len(self.alerts) + 1}",
            'severity': severity,
            'title': title,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'acknowledged': False
        }
        
        # Keep only last 100 alerts
        if len(self.alerts) >= 100:
            self.alerts.pop(0)
        
        self.alerts.append(alert)
        logger.warning(f"Alert created: {severity} - {title}: {message}")
    
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """Record an API request"""
        try:
            # Update request metrics
            self.metrics['application']['total_requests'] += 1
            
            if status_code >= 400:
                self.metrics['application']['total_errors'] += 1
            
            # Calculate error rate
            if self.metrics['application']['total_requests'] > 0:
                self.metrics['application']['error_rate'] = (
                    self.metrics['application']['total_errors'] / 
                    self.metrics['application']['total_requests']
                )
            
            # Update response time average
            current_avg = self.metrics['application']['response_time_avg']
            total_requests = self.metrics['application']['total_requests']
            self.metrics['application']['response_time_avg'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            # Update requests per second (simple rolling average)
            self.metrics['application']['requests_per_second'] = (
                self.metrics['application']['total_requests'] / 
                (time.time() - self.start_time)
            )
            
        except Exception as e:
            logger.error(f"Error recording request: {e}")
    
    def record_ai_prediction(self, model_type: str, latency: float, accuracy: Optional[float] = None):
        """Record an AI prediction"""
        try:
            self.metrics['ai_models']['predictions_made'] += 1
            self.metrics['ai_models']['prediction_latency'] = latency
            
            if accuracy is not None:
                self.metrics['ai_models']['model_accuracy'][model_type] = accuracy
            
        except Exception as e:
            logger.error(f"Error recording AI prediction: {e}")
    
    def record_external_api_call(self, service: str, response_time: float, success: bool):
        """Record an external API call"""
        try:
            if success:
                self.metrics['external_services'][f'{service}_api_calls'] += 1
            else:
                self.metrics['external_services']['api_error_rates'][service] = (
                    self.metrics['external_services']['api_error_rates'].get(service, 0) + 1
                )
            
            self.metrics['external_services']['api_response_times'][service] = response_time
            
        except Exception as e:
            logger.error(f"Error recording external API call: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        return {
            'metrics': self.metrics,
            'health_checks': self.health_checks,
            'alerts': self.alerts,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        overall_health = 'healthy'
        
        # Check if any component is critical
        for status in self.health_checks.values():
            if status == 'critical':
                overall_health = 'critical'
                break
            elif status == 'warning' and overall_health == 'healthy':
                overall_health = 'warning'
        
        return {
            'status': overall_health,
            'checks': self.health_checks,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now(timezone.utc).isoformat()
                break
    
    def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00')) > cutoff_time
        ]

# Initialize the monitoring system
advanced_monitoring = AdvancedMonitoringSystem()

def init_app(app: Flask):
    """Initialize monitoring with Flask app"""
    
    @app.route('/metrics')
    def get_metrics():
        """Get all monitoring metrics"""
        return jsonify(advanced_monitoring.get_metrics())
    
    @app.route('/health')
    def get_health():
        """Get health status"""
        return jsonify(advanced_monitoring.get_health_status())
    
    @app.route('/alerts')
    def get_alerts():
        """Get all alerts"""
        return jsonify({
            'alerts': advanced_monitoring.alerts,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    @app.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
    def acknowledge_alert(alert_id):
        """Acknowledge an alert"""
        advanced_monitoring.acknowledge_alert(alert_id)
        return jsonify({'status': 'acknowledged'})
    
    @app.before_request
    def before_request():
        """Record request start time"""
        request.start_time = time.time()
    
    @app.after_request
    def after_request(response):
        """Record request metrics"""
        if hasattr(request, 'start_time'):
            response_time = time.time() - request.start_time
            advanced_monitoring.record_request(
                request.endpoint or 'unknown',
                response_time,
                response.status_code
            )
        return response
    
    logger.info("Advanced monitoring initialized with Flask app")

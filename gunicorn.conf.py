# Gunicorn configuration file for DisastroScope Backend
# Production-ready configuration with monitoring and optimization

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes - optimized for Railway
workers = 1  # Single worker for Railway's resource constraints
worker_class = 'sync'  # Explicitly use sync workers
worker_connections = 1000
timeout = 120
keepalive = 2

# Performance optimization
max_requests = 1000
max_requests_jitter = 50
preload_app = False

# Logging configuration
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'disastroscope-backend'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (not needed for Railway)
keyfile = None
certfile = None

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Worker lifecycle hooks
def when_ready(server):
    """Called just after the server is started"""
    server.log.info("🚀 DisastroScope Backend is ready and accepting connections")

def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT"""
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info(f"Spawning worker (pid: {worker.pid})")

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def post_worker_init(worker):
    """Called just after a worker has initialized the application"""
    worker.log.info(f"Worker initialized (pid: {worker.pid})")

def worker_exit(server, worker):
    """Called when a worker exits"""
    server.log.info(f"Worker exited (pid: {worker.pid})")

def on_exit(server):
    """Called just before the server exits"""
    server.log.info("DisastroScope Backend shutting down")

# Health check configuration
def health_check(env, start_response):
    """Custom health check for load balancers"""
    status = '200 OK'
    response_headers = [('Content-Type', 'application/json')]
    start_response(status, response_headers)
    return [b'{"status":"healthy"}']

# Request processing hooks
def pre_request(worker, req):
    """Called before processing each request"""
    worker.log.info(f"Processing request: {req.method} {req.path}")

def post_request(worker, req, environ, resp):
    """Called after processing each request"""
    worker.log.info(f"Completed request: {req.method} {req.path} - Status: {resp.status}")

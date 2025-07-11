"""
Comprehensive Monitoring and Metrics Service
"""

import time
import threading
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import psutil
import asyncio
from functools import wraps
import statistics

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'tags': self.tags or {}
        }

@dataclass
class APIMetrics:
    """API endpoint metrics"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    user_agent: str = None
    ip_address: str = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

class MonitoringService:
    """
    Comprehensive system monitoring and metrics collection
    """
    
    def __init__(self):
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # Keep last 1000 points
        self.api_metrics = deque(maxlen=5000)  # Keep last 5000 API calls
        
        # System metrics
        self.system_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'disk_usage': deque(maxlen=100),
            'network_io': deque(maxlen=100),
            'process_count': deque(maxlen=100)
        }
        
        # Application metrics
        self.app_metrics = {
            'requests_per_minute': deque(maxlen=100),
            'average_response_time': deque(maxlen=100),
            'error_rate': deque(maxlen=100),
            'active_connections': deque(maxlen=100),
            'cache_hit_rate': deque(maxlen=100)
        }
        
        # Performance counters
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_thread = None
        self.collection_interval = 30  # seconds
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,  # seconds
            'error_rate': 5.0,     # percentage
        }
        
        # Start monitoring
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("✅ Monitoring service started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 Monitoring service stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Check thresholds and alerts
                self._check_alert_thresholds()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {str(e)}")
                time.sleep(60)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            now = datetime.now()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_usage'].append(MetricPoint(now, cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics['memory_usage'].append(MetricPoint(now, memory.percent))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_metrics['disk_usage'].append(MetricPoint(now, disk_percent))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.system_metrics['network_io'].append(MetricPoint(
                now, 
                net_io.bytes_sent + net_io.bytes_recv,
                {'sent': str(net_io.bytes_sent), 'recv': str(net_io.bytes_recv)}
            ))
            
            # Process count
            process_count = len(psutil.pids())
            self.system_metrics['process_count'].append(MetricPoint(now, process_count))
            
        except Exception as e:
            logger.error(f"❌ System metrics collection error: {str(e)}")
    
    def _collect_application_metrics(self):
        """Collect application-level metrics"""
        try:
            now = datetime.now()
            
            # Calculate requests per minute
            recent_api_calls = [
                call for call in self.api_metrics 
                if now - call.timestamp < timedelta(minutes=1)
            ]
            rpm = len(recent_api_calls)
            self.app_metrics['requests_per_minute'].append(MetricPoint(now, rpm))
            
            # Calculate average response time
            if recent_api_calls:
                avg_response_time = statistics.mean([call.response_time for call in recent_api_calls])
                self.app_metrics['average_response_time'].append(MetricPoint(now, avg_response_time))
            
            # Calculate error rate
            error_calls = [call for call in recent_api_calls if call.status_code >= 400]
            error_rate = (len(error_calls) / len(recent_api_calls) * 100) if recent_api_calls else 0
            self.app_metrics['error_rate'].append(MetricPoint(now, error_rate))
            
            # Active connections (approximation)
            active_connections = self.gauges.get('active_connections', 0)
            self.app_metrics['active_connections'].append(MetricPoint(now, active_connections))
            
            # Cache hit rate (if available)
            cache_hits = self.counters.get('cache_hits', 0)
            cache_misses = self.counters.get('cache_misses', 0)
            total_cache_requests = cache_hits + cache_misses
            cache_hit_rate = (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
            self.app_metrics['cache_hit_rate'].append(MetricPoint(now, cache_hit_rate))
            
        except Exception as e:
            logger.error(f"❌ Application metrics collection error: {str(e)}")
    
    def _check_alert_thresholds(self):
        """Check metrics against alert thresholds"""
        try:
            # Check CPU usage
            if self.system_metrics['cpu_usage']:
                latest_cpu = self.system_metrics['cpu_usage'][-1].value
                if latest_cpu > self.alert_thresholds['cpu_usage']:
                    self._trigger_metric_alert('cpu_usage', latest_cpu, self.alert_thresholds['cpu_usage'])
            
            # Check memory usage
            if self.system_metrics['memory_usage']:
                latest_memory = self.system_metrics['memory_usage'][-1].value
                if latest_memory > self.alert_thresholds['memory_usage']:
                    self._trigger_metric_alert('memory_usage', latest_memory, self.alert_thresholds['memory_usage'])
            
            # Check response time
            if self.app_metrics['average_response_time']:
                latest_response_time = self.app_metrics['average_response_time'][-1].value
                if latest_response_time > self.alert_thresholds['response_time']:
                    self._trigger_metric_alert('response_time', latest_response_time, self.alert_thresholds['response_time'])
            
            # Check error rate
            if self.app_metrics['error_rate']:
                latest_error_rate = self.app_metrics['error_rate'][-1].value
                if latest_error_rate > self.alert_thresholds['error_rate']:
                    self._trigger_metric_alert('error_rate', latest_error_rate, self.alert_thresholds['error_rate'])
            
        except Exception as e:
            logger.error(f"❌ Alert threshold check error: {str(e)}")
    
    def _trigger_metric_alert(self, metric_name: str, current_value: float, threshold: float):
        """Trigger alert for metric threshold breach"""
        logger.warning(f"🚨 METRIC ALERT: {metric_name} = {current_value:.2f} (threshold: {threshold:.2f})")
    
    def _cleanup_old_data(self):
        """Clean up old metric data"""
        try:
            # Clean API metrics older than 1 hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.api_metrics = deque([
                call for call in self.api_metrics 
                if call.timestamp > cutoff_time
            ], maxlen=5000)
            
            # Reset counters periodically (every hour)
            if datetime.now().minute == 0:
                self.counters.clear()
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {str(e)}")
    
    def record_api_call(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, user_agent: str = None, ip_address: str = None):
        """Record API call metrics"""
        try:
            api_metric = APIMetrics(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time,
                timestamp=datetime.now(),
                user_agent=user_agent,
                ip_address=ip_address
            )
            self.api_metrics.append(api_metric)
            
            # Update counters
            self.counters['total_requests'] += 1
            if status_code >= 400:
                self.counters['error_requests'] += 1
            else:
                self.counters['success_requests'] += 1
                
        except Exception as e:
            logger.error(f"❌ API call recording error: {str(e)}")
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        try:
            key = f"{name}_{hash(str(tags))}" if tags else name
            self.counters[key] += value
        except Exception as e:
            logger.error(f"❌ Counter increment error: {str(e)}")
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        try:
            key = f"{name}_{hash(str(tags))}" if tags else name
            self.gauges[key] = value
        except Exception as e:
            logger.error(f"❌ Gauge set error: {str(e)}")
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram metric"""
        try:
            key = f"{name}_{hash(str(tags))}" if tags else name
            self.histograms[key].append(value)
            
            # Keep only last 1000 values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
                
        except Exception as e:
            logger.error(f"❌ Histogram recording error: {str(e)}")
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': {},
                'application_metrics': {},
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_stats': {}
            }
            
            # System metrics summary
            for metric_name, metric_data in self.system_metrics.items():
                if metric_data:
                    values = [point.value for point in metric_data]
                    summary['system_metrics'][metric_name] = {
                        'current': values[-1],
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            # Application metrics summary
            for metric_name, metric_data in self.app_metrics.items():
                if metric_data:
                    values = [point.value for point in metric_data]
                    summary['application_metrics'][metric_name] = {
                        'current': values[-1],
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            # Histogram statistics
            for hist_name, hist_data in self.histograms.items():
                if hist_data:
                    summary['histogram_stats'][hist_name] = {
                        'count': len(hist_data),
                        'mean': statistics.mean(hist_data),
                        'median': statistics.median(hist_data),
                        'min': min(hist_data),
                        'max': max(hist_data),
                        'p95': self._percentile(hist_data, 95),
                        'p99': self._percentile(hist_data, 99)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Metrics summary error: {str(e)}")
            return {'error': str(e)}
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        try:
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * percentile / 100
            f = int(k)
            c = k - f
            if f == len(sorted_data) - 1:
                return sorted_data[f]
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
        except:
            return 0.0
    
    def get_health_status(self) -> Dict:
        """Get overall system health status"""
        try:
            status = {
                'overall_status': 'healthy',
                'components': {},
                'alerts': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Check system metrics
            if self.system_metrics['cpu_usage']:
                cpu_usage = self.system_metrics['cpu_usage'][-1].value
                if cpu_usage > 90:
                    status['components']['cpu'] = 'critical'
                    status['overall_status'] = 'critical'
                elif cpu_usage > 80:
                    status['components']['cpu'] = 'warning'
                    if status['overall_status'] == 'healthy':
                        status['overall_status'] = 'warning'
                else:
                    status['components']['cpu'] = 'healthy'
            
            if self.system_metrics['memory_usage']:
                memory_usage = self.system_metrics['memory_usage'][-1].value
                if memory_usage > 95:
                    status['components']['memory'] = 'critical'
                    status['overall_status'] = 'critical'
                elif memory_usage > 85:
                    status['components']['memory'] = 'warning'
                    if status['overall_status'] == 'healthy':
                        status['overall_status'] = 'warning'
                else:
                    status['components']['memory'] = 'healthy'
            
            # Check application metrics
            if self.app_metrics['error_rate']:
                error_rate = self.app_metrics['error_rate'][-1].value
                if error_rate > 10:
                    status['components']['api'] = 'critical'
                    status['overall_status'] = 'critical'
                elif error_rate > 5:
                    status['components']['api'] = 'warning'
                    if status['overall_status'] == 'healthy':
                        status['overall_status'] = 'warning'
                else:
                    status['components']['api'] = 'healthy'
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Health status error: {str(e)}")
            return {
                'overall_status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Decorator for automatic API monitoring
def monitor_api_call(func):
    """Decorator to automatically monitor API calls"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            response_time = time.time() - start_time
            
            # Record successful API call
            monitoring_service.record_api_call(
                endpoint=func.__name__,
                method='ASYNC',
                status_code=200,
                response_time=response_time
            )
            
            return result
        except Exception as e:
            response_time = time.time() - start_time
            
            # Record failed API call
            monitoring_service.record_api_call(
                endpoint=func.__name__,
                method='ASYNC',
                status_code=500,
                response_time=response_time
            )
            
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            
            # Record successful API call
            monitoring_service.record_api_call(
                endpoint=func.__name__,
                method='SYNC',
                status_code=200,
                response_time=response_time
            )
            
            return result
        except Exception as e:
            response_time = time.time() - start_time
            
            # Record failed API call
            monitoring_service.record_api_call(
                endpoint=func.__name__,
                method='SYNC',
                status_code=500,
                response_time=response_time
            )
            
            raise
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# Global monitoring service instance
monitoring_service = MonitoringService() 
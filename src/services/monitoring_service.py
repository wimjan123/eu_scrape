"""Comprehensive monitoring service for EU Parliament scraper."""

import time
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass

from ..core.logging import get_logger
from ..core.metrics import metrics_collector
from ..core.circuit_breaker import circuit_registry
from ..services.progress_tracker import ProgressTracker

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    timestamp: str


class MonitoringService:
    """Comprehensive monitoring service for system health and performance."""
    
    def __init__(self, progress_tracker: ProgressTracker = None):
        """
        Initialize monitoring service.
        
        Args:
            progress_tracker: Optional progress tracker instance
        """
        self.progress_tracker = progress_tracker
        self.start_time = datetime.utcnow()
        self.monitoring_data_file = Path("data/monitoring/monitoring.json")
        
        # Ensure monitoring directory exists
        self.monitoring_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize system monitoring
        self.process = psutil.Process()
        self.initial_network_stats = psutil.net_io_counters()
        
        logger.info("Monitoring service initialized")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network metrics
            current_network = psutil.net_io_counters()
            network_sent_mb = (current_network.bytes_sent - self.initial_network_stats.bytes_sent) / (1024**2)
            network_recv_mb = (current_network.bytes_recv - self.initial_network_stats.bytes_recv) / (1024**2)
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=round(memory_used_gb, 2),
                memory_total_gb=round(memory_total_gb, 2),
                disk_percent=disk.percent,
                disk_used_gb=round(disk_used_gb, 2),
                disk_total_gb=round(disk_total_gb, 2),
                network_sent_mb=round(network_sent_mb, 2),
                network_recv_mb=round(network_recv_mb, 2),
                process_count=process_count,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            raise
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics."""
        try:
            # Get process metrics
            process_info = {
                'pid': self.process.pid,
                'cpu_percent': self.process.cpu_percent(),
                'memory_mb': round(self.process.memory_info().rss / (1024**2), 2),
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'create_time': datetime.fromtimestamp(self.process.create_time()).isoformat(),
                'status': self.process.status()
            }
            
            # Get API metrics
            api_metrics = metrics_collector.get_all_metrics()
            
            # Get circuit breaker states
            circuit_states = circuit_registry.get_all_states()
            
            # Get progress metrics if available
            progress_metrics = {}
            if self.progress_tracker:
                progress_metrics = self.progress_tracker.get_progress_summary()
            
            return {
                'process': process_info,
                'apis': api_metrics,
                'circuit_breakers': circuit_states,
                'progress': progress_metrics,
                'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to collect application metrics", error=str(e))
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            system_metrics = self.get_system_metrics()
            app_metrics = self.get_application_metrics()
            
            # Determine health status
            health_issues = []
            overall_status = "healthy"
            
            # Check system resources
            if system_metrics.cpu_percent > 80:
                health_issues.append("High CPU usage")
            if system_metrics.memory_percent > 85:
                health_issues.append("High memory usage")
            if system_metrics.disk_percent > 90:
                health_issues.append("Low disk space")
            
            # Check application health
            process_memory_mb = app_metrics['process']['memory_mb']
            if process_memory_mb > 1000:  # 1GB
                health_issues.append("High process memory usage")
            
            # Check API health
            for service, metrics in app_metrics['apis'].items():
                if metrics and metrics['success_rate'] < 90:
                    health_issues.append(f"Low success rate for {service}")
            
            # Check circuit breakers
            for service, state in app_metrics['circuit_breakers'].items():
                if not state['is_healthy']:
                    health_issues.append(f"Circuit breaker open for {service}")
            
            # Determine overall status
            if health_issues:
                overall_status = "degraded" if len(health_issues) <= 2 else "unhealthy"
            
            return {
                'status': overall_status,
                'issues': health_issues,
                'system_metrics': system_metrics.__dict__,
                'application_metrics': app_metrics,
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get health status", error=str(e))
            return {
                'status': 'unhealthy',
                'issues': [f"Health check failed: {str(e)}"],
                'last_check': datetime.utcnow().isoformat()
            }
    
    def log_performance_alert(self, metric_name: str, current_value: float, 
                            threshold: float, severity: str = "warning") -> None:
        """Log performance alert when thresholds are exceeded."""
        alert_data = {
            'alert_type': 'performance',
            'metric': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if severity == "critical":
            logger.error("Performance alert", **alert_data)
        elif severity == "warning":
            logger.warning("Performance alert", **alert_data)
        else:
            logger.info("Performance alert", **alert_data)
    
    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """Check performance thresholds and generate alerts."""
        alerts = []
        
        try:
            system_metrics = self.get_system_metrics()
            app_metrics = self.get_application_metrics()
            
            # Define thresholds
            thresholds = {
                'cpu_percent': {'warning': 70, 'critical': 90},
                'memory_percent': {'warning': 80, 'critical': 95},
                'disk_percent': {'warning': 85, 'critical': 95},
                'process_memory_mb': {'warning': 500, 'critical': 1000}
            }
            
            # Check system thresholds
            for metric_name, levels in thresholds.items():
                if metric_name == 'process_memory_mb':
                    current_value = app_metrics['process']['memory_mb']
                else:
                    current_value = getattr(system_metrics, metric_name)
                
                severity = None
                if current_value >= levels['critical']:
                    severity = 'critical'
                elif current_value >= levels['warning']:
                    severity = 'warning'
                
                if severity:
                    alert = {
                        'metric': metric_name,
                        'current_value': current_value,
                        'threshold': levels[severity],
                        'severity': severity,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    alerts.append(alert)
                    self.log_performance_alert(metric_name, current_value, levels[severity], severity)
            
            # Check API success rates
            for service, metrics in app_metrics['apis'].items():
                if metrics and metrics['success_rate'] < 90:
                    severity = 'critical' if metrics['success_rate'] < 70 else 'warning'
                    alert = {
                        'metric': f'{service}_success_rate',
                        'current_value': metrics['success_rate'],
                        'threshold': 90,
                        'severity': severity,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    alerts.append(alert)
                    self.log_performance_alert(f'{service}_success_rate', 
                                             metrics['success_rate'], 90, severity)
            
            return alerts
            
        except Exception as e:
            logger.error("Failed to check performance thresholds", error=str(e))
            return []
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        try:
            health_status = self.get_health_status()
            alerts = self.check_performance_thresholds()
            
            report = {
                'report_type': 'monitoring',
                'generated_at': datetime.utcnow().isoformat(),
                'uptime_hours': round((datetime.utcnow() - self.start_time).total_seconds() / 3600, 2),
                'health_status': health_status,
                'performance_alerts': alerts,
                'summary': {
                    'overall_status': health_status['status'],
                    'total_issues': len(health_status['issues']),
                    'active_alerts': len(alerts),
                    'services_monitored': len(health_status['application_metrics']['apis']),
                    'circuit_breakers': len(health_status['application_metrics']['circuit_breakers'])
                }
            }
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate monitoring report", error=str(e))
            return {
                'report_type': 'monitoring',
                'generated_at': datetime.utcnow().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    def save_monitoring_snapshot(self) -> None:
        """Save current monitoring data to file."""
        try:
            report = self.generate_monitoring_report()
            
            # Load existing data
            monitoring_data = []
            if self.monitoring_data_file.exists():
                with open(self.monitoring_data_file, 'r') as f:
                    monitoring_data = json.load(f)
            
            # Add new report
            monitoring_data.append(report)
            
            # Keep only last 100 reports
            if len(monitoring_data) > 100:
                monitoring_data = monitoring_data[-100:]
            
            # Save updated data
            with open(self.monitoring_data_file, 'w') as f:
                json.dump(monitoring_data, f, indent=2, default=str)
            
            logger.debug("Monitoring snapshot saved", report_count=len(monitoring_data))
            
        except Exception as e:
            logger.error("Failed to save monitoring snapshot", error=str(e))
    
    def get_monitoring_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get monitoring history for specified hours."""
        try:
            if not self.monitoring_data_file.exists():
                return []
            
            with open(self.monitoring_data_file, 'r') as f:
                monitoring_data = json.load(f)
            
            # Filter by time range
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            filtered_data = []
            for report in monitoring_data:
                report_time = datetime.fromisoformat(report['generated_at'])
                if report_time >= cutoff_time:
                    filtered_data.append(report)
            
            return filtered_data
            
        except Exception as e:
            logger.error("Failed to get monitoring history", error=str(e))
            return []
    
    def cleanup_old_monitoring_data(self, days: int = 7) -> None:
        """Clean up monitoring data older than specified days."""
        try:
            history = self.get_monitoring_history(hours=days * 24)
            
            if history:
                with open(self.monitoring_data_file, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
                
                logger.info("Old monitoring data cleaned up", days=days, retained_reports=len(history))
            
        except Exception as e:
            logger.error("Failed to cleanup monitoring data", error=str(e))
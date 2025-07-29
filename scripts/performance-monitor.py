#!/usr/bin/env python3
"""Continuous Performance Monitoring System.

This script provides real-time performance monitoring with:
- Live performance metrics collection
- Automated alerting for performance degradation
- Integration with Prometheus and Grafana
- Performance dashboard updates
"""

import time
import json
import requests
import threading
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import statistics
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    timestamp: str
    description: str


class PerformanceMonitor:
    """Continuous performance monitoring system."""
    
    def __init__(self, config_file: str = "performance-monitor-config.json"):
        """Initialize the performance monitor.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.running = False
        self.metrics_cache = {}
        self.alert_history = []
        
        # Prometheus integration
        self.prometheus_url = self.config.get('prometheus_url', 'http://localhost:9090')
        self.pushgateway_url = self.config.get('pushgateway_url', 'http://localhost:9091')
        
        # Alerting configuration
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.alert_cooldown = self.config.get('alert_cooldown_minutes', 10)
        
        # Monitoring intervals
        self.monitor_interval = self.config.get('monitor_interval_seconds', 30)
        self.alert_check_interval = self.config.get('alert_check_interval_seconds', 60)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file."""
        default_config = {
            "prometheus_url": "http://localhost:9090",
            "pushgateway_url": "http://localhost:9091",
            "monitor_interval_seconds": 30,
            "alert_check_interval_seconds": 60,
            "alert_cooldown_minutes": 10,
            "alert_thresholds": {
                "response_time_p95": {"threshold": 2.0, "severity": "warning"},
                "response_time_p99": {"threshold": 5.0, "severity": "critical"},
                "error_rate": {"threshold": 0.05, "severity": "critical"},
                "memory_usage_gb": {"threshold": 2.0, "severity": "warning"},
                "cpu_utilization": {"threshold": 80.0, "severity": "warning"}
            },
            "metrics_to_monitor": [
                "crewai:http_request_duration:p95",
                "crewai:http_request_duration:p99",
                "crewai:http_error_rate",
                "crewai:app_memory_usage_gb",
                "crewai:cpu_utilization"
            ]
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                # Create default config file
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default configuration file: {config_file}")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return default_config
    
    def start(self):
        """Start the performance monitoring system."""
        logger.info("Starting performance monitor...")
        self.running = True
        
        # Start monitoring threads
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        
        monitor_thread.start()
        alert_thread.start()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Performance monitor started successfully")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the performance monitoring system."""
        logger.info("Stopping performance monitor...")
        self.running = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _alert_loop(self):
        """Alert checking loop."""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(self.alert_check_interval)
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _collect_metrics(self):
        """Collect performance metrics from Prometheus."""
        try:
            for metric_name in self.config['metrics_to_monitor']:
                value = self._query_prometheus(metric_name)
                if value is not None:
                    self.metrics_cache[metric_name] = {
                        'value': value,
                        'timestamp': datetime.now().isoformat()
                    }
                    logger.debug(f"Collected metric {metric_name}: {value}")
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    def _query_prometheus(self, query: str) -> Optional[float]:
        """Query Prometheus for a specific metric."""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                # Get the most recent value
                result = data['data']['result'][0]
                return float(result['value'][1])
            else:
                logger.warning(f"No data returned for query: {query}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to parse Prometheus response: {e}")
            return None
    
    def _check_alerts(self):
        """Check metrics against alert thresholds."""
        current_time = datetime.now()
        
        for metric_name, metric_data in self.metrics_cache.items():
            # Map metric names to alert configurations
            alert_config = None
            for alert_name, config in self.alert_thresholds.items():
                if alert_name in metric_name or metric_name.endswith(alert_name):
                    alert_config = config
                    break
            
            if not alert_config:
                continue
            
            value = metric_data['value']
            threshold = alert_config['threshold']
            severity = alert_config['severity']
            
            # Check if threshold is exceeded
            if value > threshold:
                # Check if we've already alerted recently (cooldown)
                if not self._is_in_cooldown(metric_name, current_time):
                    alert = PerformanceAlert(
                        metric_name=metric_name,
                        current_value=value,
                        threshold=threshold,
                        severity=severity,
                        timestamp=current_time.isoformat(),
                        description=f"{metric_name} exceeded threshold: {value:.3f} > {threshold:.3f}"
                    )
                    
                    self._send_alert(alert)
                    self.alert_history.append(alert)
    
    def _is_in_cooldown(self, metric_name: str, current_time: datetime) -> bool:
        """Check if a metric is in alert cooldown period."""
        cooldown_delta = timedelta(minutes=self.alert_cooldown)
        
        # Check recent alerts for this metric
        for alert in reversed(self.alert_history):
            alert_time = datetime.fromisoformat(alert.timestamp)
            if (alert.metric_name == metric_name and 
                current_time - alert_time < cooldown_delta):
                return True
        
        return False
    
    def _send_alert(self, alert: PerformanceAlert):
        """Send performance alert."""
        logger.warning(f"PERFORMANCE ALERT: {alert.description}")
        
        # Send to multiple channels
        self._send_alert_to_slack(alert)
        self._send_alert_to_webhook(alert)
        self._create_prometheus_alert_metric(alert)
    
    def _send_alert_to_slack(self, alert: PerformanceAlert):
        """Send alert to Slack (if configured)."""
        slack_webhook = self.config.get('slack_webhook_url')
        if not slack_webhook:
            return
        
        try:
            severity_emoji = {
                'critical': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️'
            }
            
            message = {
                "text": f"{severity_emoji.get(alert.severity, '🔔')} Performance Alert",
                "attachments": [
                    {
                        "color": "danger" if alert.severity == "critical" else "warning",
                        "fields": [
                            {
                                "title": "Metric",
                                "value": alert.metric_name,
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert.current_value:.3f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold:.3f}",
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.upper(),
                                "short": True
                            }
                        ],
                        "footer": "CrewAI Performance Monitor",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(slack_webhook, json=message, timeout=10)
            response.raise_for_status()
            logger.info(f"Alert sent to Slack: {alert.metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_alert_to_webhook(self, alert: PerformanceAlert):
        """Send alert to custom webhook (if configured)."""
        webhook_url = self.config.get('alert_webhook_url')
        if not webhook_url:
            return
        
        try:
            payload = asdict(alert)
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Alert sent to webhook: {alert.metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _create_prometheus_alert_metric(self, alert: PerformanceAlert):
        """Create a Prometheus metric for the alert."""
        try:
            metric_name = alert.metric_name.replace(':', '_').replace('-', '_')
            prometheus_metric = (
                f'performance_alert_triggered{{metric="{metric_name}",severity="{alert.severity}"}} 1\n'
            )
            
            # Push to Pushgateway
            response = requests.post(
                f"{self.pushgateway_url}/metrics/job/performance-monitor/instance/alerts",
                data=prometheus_metric,
                headers={'Content-Type': 'text/plain'},
                timeout=10
            )
            response.raise_for_status()
            logger.debug(f"Alert metric pushed to Prometheus: {metric_name}")
            
        except Exception as e:
            logger.error(f"Failed to push alert metric to Prometheus: {e}")
    
    def get_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'running': self.running,
            'metrics_count': len(self.metrics_cache),
            'recent_alerts': len([
                a for a in self.alert_history
                if datetime.now() - datetime.fromisoformat(a.timestamp) < timedelta(hours=24)
            ]),
            'last_update': max(
                [m['timestamp'] for m in self.metrics_cache.values()],
                default="Never"
            ),
            'config': {
                'monitor_interval': self.monitor_interval,
                'alert_check_interval': self.alert_check_interval,
                'alert_cooldown_minutes': self.alert_cooldown
            }
        }
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of current metrics."""
        summary = {}
        for metric_name, metric_data in self.metrics_cache.items():
            summary[metric_name] = {
                'current_value': metric_data['value'],
                'timestamp': metric_data['timestamp'],
                'threshold': None,
                'status': 'unknown'
            }
            
            # Add threshold information if available
            for alert_name, config in self.alert_thresholds.items():
                if alert_name in metric_name or metric_name.endswith(alert_name):
                    threshold = config['threshold']
                    summary[metric_name]['threshold'] = threshold
                    summary[metric_name]['status'] = (
                        'critical' if metric_data['value'] > threshold else 'ok'
                    )
                    break
        
        return summary


def main():
    """Main entry point for the performance monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Performance Monitor")
    parser.add_argument("--config", default="performance-monitor-config.json",
                       help="Configuration file path")
    parser.add_argument("--status", action="store_true",
                       help="Show current monitoring status and exit")
    parser.add_argument("--metrics", action="store_true",
                       help="Show current metrics summary and exit")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.config)
    
    if args.status:
        status = monitor.get_status()
        print(json.dumps(status, indent=2))
        return 0
    
    if args.metrics:
        # Collect metrics once
        monitor._collect_metrics()
        metrics = monitor.get_metrics_summary()
        print(json.dumps(metrics, indent=2))
        return 0
    
    # Start continuous monitoring
    try:
        monitor.start()
        return 0
    except Exception as e:
        logger.error(f"Performance monitor failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
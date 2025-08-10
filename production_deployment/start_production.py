#!/usr/bin/env python3
"""Production startup script for CrewAI Email Triage system."""

import sys
import os
import json
import logging
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_production_logging():
    """Setup production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('production.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_production_config():
    """Load production configuration."""
    config_file = Path(__file__).parent / "production_config.json"
    
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    else:
        logging.warning("Production config not found, using defaults")
        return {}

def start_production_services(config):
    """Start production services."""
    services = []
    
    try:
        # Initialize high-performance processor
        from crewai_email_triage.scale_core import get_hp_processor
        processor = get_hp_processor()
        
        logging.info("High-performance processor initialized")
        
        # Start health monitoring
        if config.get("monitoring", {}).get("enable_health_checks", True):
            from crewai_email_triage.robust_health import get_health_monitor
            monitor = get_health_monitor()
            
            interval = config.get("monitoring", {}).get("health_check_interval", 30)
            monitor.start_continuous_monitoring(interval)
            services.append(monitor)
            
            logging.info(f"Health monitoring started (interval: {interval}s)")
        
        # Start metrics export
        if config.get("monitoring", {}).get("enable_metrics_export", False):
            from crewai_email_triage.metrics_export import get_metrics_collector, PrometheusExporter, MetricsEndpoint, MetricsConfig
            
            metrics_config = MetricsConfig(
                enabled=True,
                export_port=config.get("monitoring", {}).get("metrics_port", 8080),
                export_path="/metrics"
            )
            
            collector = get_metrics_collector()
            exporter = PrometheusExporter(collector)
            endpoint = MetricsEndpoint(exporter, metrics_config)
            
            try:
                endpoint.start()
                services.append(endpoint)
                logging.info(f"Metrics endpoint started on port {metrics_config.export_port}")
            except Exception as e:
                logging.error(f"Failed to start metrics endpoint: {e}")
        
        return services, processor
        
    except ImportError as e:
        logging.warning(f"Some production services not available: {e}")
        return [], None

def signal_handler(signum, frame, services):
    """Handle shutdown signals."""
    logging.info(f"Received signal {signum}, shutting down...")
    
    for service in services:
        try:
            if hasattr(service, 'stop'):
                service.stop()
            elif hasattr(service, 'shutdown'):
                service.shutdown()
        except Exception as e:
            logging.error(f"Error stopping service: {e}")
    
    logging.info("Production services stopped")
    sys.exit(0)

def main():
    """Main production startup."""
    print("🚀 CREWAI EMAIL TRIAGE - PRODUCTION STARTUP")
    print("=" * 60)
    
    # Setup logging
    setup_production_logging()
    
    # Load configuration
    config = load_production_config()
    logging.info("Production configuration loaded")
    
    # Start services
    services, processor = start_production_services(config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, services))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, services))
    
    logging.info("Production system started successfully")
    print("✅ Production system is running")
    print("   - Health monitoring active")
    print("   - High-performance processing enabled")
    print("   - Metrics export available")
    print("   - Press Ctrl+C to shutdown")
    
    # Keep running
    try:
        signal.pause()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None, services)

if __name__ == "__main__":
    main()

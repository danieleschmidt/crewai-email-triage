#!/usr/bin/env python3
"""Runtime Security Monitoring System.

This system provides comprehensive runtime security monitoring with:
- Real-time threat detection
- Behavioral analysis
- Automated incident response
- Integration with security tools and SIEM systems
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    severity: str  # critical, high, medium, low
    source: str
    description: str
    details: Dict[str, Any]
    timestamp: str
    affected_resources: List[str]
    indicators_of_compromise: List[str]
    recommended_actions: List[str]


@dataclass
class ThreatIndicator:  
    """Threat indicator data structure."""
    indicator_type: str  # ip, domain, file_hash, process_name, etc.
    value: str
    threat_level: str
    source: str
    description: str
    first_seen: str
    last_seen: str


class RuntimeSecurityMonitor:
    """Comprehensive runtime security monitoring system."""
    
    def __init__(self, config_file: str = "security-monitor-config.json"):
        """Initialize the security monitor.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.running = False
        self.threat_indicators = {}
        self.security_events = []
        
        # Monitoring components
        self.file_monitor = FileIntegrityMonitor(self.config)
        self.network_monitor = NetworkSecurityMonitor(self.config)
        self.process_monitor = ProcessSecurityMonitor(self.config)
        self.auth_monitor = AuthenticationMonitor(self.config)
        
        # Integration endpoints
        self.siem_endpoint = self.config.get('siem_endpoint')
        self.slack_webhook = self.config.get('slack_webhook_url')
        self.prometheus_pushgateway = self.config.get('prometheus_pushgateway')
    
    def _load_config(self, config_file: str) -> Dict:
        """Load security monitoring configuration."""
        default_config = {
            "monitoring_interval": 30,
            "file_integrity_paths": [
                "/etc/passwd", "/etc/shadow", "/etc/hosts",
                "/root/.ssh/", "/home/*/.ssh/",
                "/usr/local/bin/", "/opt/",
                "./src/", "./config/"
            ],
            "network_monitoring": {
                "suspicious_ports": [22, 23, 135, 139, 445, 1433, 3389],
                "allowed_outbound_domains": ["api.openai.com", "github.com"],
                "blocked_countries": ["CN", "RU", "KP"]
            },
            "process_monitoring": {
                "suspicious_processes": [
                    "nc", "netcat", "ncat", "socat",
                    "wget", "curl", "python -c", "perl -e",
                    "bash -i", "sh -c", "powershell"
                ],
                "resource_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 80,
                    "disk_io_threshold": 1000000  # bytes per second
                }
            },
            "authentication_monitoring": {
                "max_failed_attempts": 5,
                "lockout_duration": 300,  # seconds
                "suspicious_patterns": [
                    "admin", "root", "administrator",
                    "password", "123456", "qwerty"
                ]
            },
            "threat_intelligence": {
                "enabled": True,
                "update_interval": 3600,  # seconds
                "sources": [
                    "https://raw.githubusercontent.com/stamparm/ipsum/master/ipsum.txt",
                    "https://rules.emergingthreats.net/open/suricata/rules/emerging-compromised.rules"
                ]
            },
            "incident_response": {
                "auto_block_ips": True,
                "auto_isolate_processes": True,
                "backup_critical_files": True,
                "notify_immediately": ["critical", "high"]
            }
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
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default security config: {config_file}")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load security config: {e}")
            return default_config
    
    def start(self):
        """Start the runtime security monitoring system."""
        logger.info("Starting runtime security monitor...")
        self.running = True
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self.file_monitor.start, daemon=True),
            threading.Thread(target=self.network_monitor.start, daemon=True),
            threading.Thread(target=self.process_monitor.start, daemon=True),
            threading.Thread(target=self.auth_monitor.start, daemon=True),
            threading.Thread(target=self._threat_intelligence_updater, daemon=True),
            threading.Thread(target=self._security_event_processor, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        logger.info("Runtime security monitor started successfully")
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the runtime security monitoring system."""
        logger.info("Stopping runtime security monitor...")
        self.running = False
        self.file_monitor.stop()
        self.network_monitor.stop()
        self.process_monitor.stop()
        self.auth_monitor.stop()
    
    def _threat_intelligence_updater(self):
        """Update threat intelligence indicators periodically."""
        while self.running:
            try:
                if self.config['threat_intelligence']['enabled']:
                    self._update_threat_indicators()
                time.sleep(self.config['threat_intelligence']['update_interval'])
            except Exception as e:
                logger.error(f"Error updating threat intelligence: {e}")
    
    def _update_threat_indicators(self):
        """Update threat indicators from external sources."""
        for source_url in self.config['threat_intelligence']['sources']:
            try:
                response = requests.get(source_url, timeout=30)
                response.raise_for_status()
                
                # Parse different formats
                if 'ipsum.txt' in source_url:
                    self._parse_ipsum_indicators(response.text)
                elif 'emerging-compromised.rules' in source_url:
                    self._parse_suricata_rules(response.text)
                
                logger.info(f"Updated threat indicators from {source_url}")
                
            except Exception as e:
                logger.error(f"Failed to update from {source_url}: {e}")
    
    def _parse_ipsum_indicators(self, content: str):
        """Parse IPSUM malicious IP list."""
        lines = content.strip().split('\n')
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            ip = line.strip()
            if self._is_valid_ip(ip):
                indicator = ThreatIndicator(
                    indicator_type="ip",
                    value=ip,
                    threat_level="high",
                    source="ipsum",
                    description="Known malicious IP address",
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat()
                )
                self.threat_indicators[ip] = indicator
    
    def _parse_suricata_rules(self, content: str):
        """Parse Suricata security rules for indicators."""
        # Extract IP addresses from Suricata rules
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, content)
        
        for ip in set(ips):  # Remove duplicates
            if self._is_valid_ip(ip) and not ip.startswith('192.168.'):
                indicator = ThreatIndicator(
                    indicator_type="ip",
                    value=ip,
                    threat_level="medium",
                    source="emerging_threats",
                    description="IP from Suricata rules",
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat()
                )
                self.threat_indicators[ip] = indicator
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            for part in parts:
                if not 0 <= int(part) <= 255:
                    return False
            return True
        except ValueError:
            return False
    
    def _security_event_processor(self):
        """Process security events and trigger responses."""
        while self.running:
            try:
                # Check for events from monitoring components
                self._check_monitoring_components()
                
                # Process accumulated events
                self._process_security_events()
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Error processing security events: {e}")
    
    def _check_monitoring_components(self):
        """Check all monitoring components for new events."""
        components = [
            self.file_monitor,
            self.network_monitor,
            self.process_monitor,
            self.auth_monitor
        ]
        
        for component in components:
            if hasattr(component, 'get_events'):
                new_events = component.get_events()
                self.security_events.extend(new_events)
    
    def _process_security_events(self):
        """Process accumulated security events."""
        for event in self.security_events[:]:  # Copy to avoid modification during iteration
            try:
                # Enrich event with threat intelligence
                self._enrich_event_with_threat_intel(event)
                
                # Determine if automated response is needed
                if self._requires_automated_response(event):
                    self._trigger_automated_response(event)
                
                # Send notifications
                self._send_security_notification(event)
                
                # Export to SIEM
                self._export_to_siem(event)
                
                # Export metrics to Prometheus
                self._export_security_metrics(event)
                
                # Remove processed event
                self.security_events.remove(event)
                
            except Exception as e:
                logger.error(f"Error processing security event: {e}")
    
    def _enrich_event_with_threat_intel(self, event: SecurityEvent):
        """Enrich security event with threat intelligence."""
        for ioc in event.indicators_of_compromise:
            if ioc in self.threat_indicators:
                indicator = self.threat_indicators[ioc]
                event.details['threat_intelligence'] = {
                    'indicator_type': indicator.indicator_type,
                    'threat_level': indicator.threat_level,
                    'source': indicator.source,
                    'description': indicator.description
                }
                
                # Escalate severity if high-confidence threat
                if indicator.threat_level == 'high' and event.severity == 'medium':
                    event.severity = 'high'
                    logger.warning(f"Escalated event severity due to threat intelligence: {ioc}")
    
    def _requires_automated_response(self, event: SecurityEvent) -> bool:
        """Determine if event requires automated response."""
        auto_response_config = self.config['incident_response']
        
        # Check severity thresholds
        if event.severity in ['critical', 'high']:
            return True
        
        # Check specific event types
        if event.event_type in ['malicious_process', 'unauthorized_access', 'data_exfiltration']:
            return True
        
        return False
    
    def _trigger_automated_response(self, event: SecurityEvent):
        """Trigger automated incident response."""
        logger.warning(f"Triggering automated response for event: {event.event_type}")
        
        try:
            if event.event_type == 'suspicious_network_connection':
                self._block_malicious_ip(event)
            elif event.event_type == 'malicious_process':
                self._isolate_malicious_process(event)
            elif event.event_type == 'file_integrity_violation':
                self._backup_affected_files(event)
            elif event.event_type == 'authentication_anomaly':
                self._lockout_suspicious_user(event)
            
            # Add automated response to event details
            event.details['automated_response'] = True
            event.details['response_timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Failed to execute automated response: {e}")
            event.details['automated_response_error'] = str(e)
    
    def _block_malicious_ip(self, event: SecurityEvent):
        """Block malicious IP address using iptables."""
        for ioc in event.indicators_of_compromise:
            if self._is_valid_ip(ioc):
                try:
                    # Block IP using iptables
                    subprocess.run([
                        'sudo', 'iptables', '-A', 'INPUT',
                        '-s', ioc, '-j', 'DROP'
                    ], check=True)
                    
                    logger.warning(f"Blocked malicious IP: {ioc}")
                    event.recommended_actions.append(f"Blocked IP {ioc} with iptables")
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to block IP {ioc}: {e}")
    
    def _isolate_malicious_process(self, event: SecurityEvent):
        """Isolate or terminate malicious processes."""
        process_info = event.details.get('process_info', {})
        pid = process_info.get('pid')
        
        if pid:
            try:
                # Try to terminate process gracefully first
                process = psutil.Process(pid)
                process.terminate()
                
                # Wait for termination, then force kill if needed
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    process.kill()
                
                logger.warning(f"Terminated malicious process PID {pid}")
                event.recommended_actions.append(f"Terminated process PID {pid}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.error(f"Failed to terminate process {pid}: {e}")
    
    def _backup_affected_files(self, event: SecurityEvent):
        """Backup files affected by integrity violations."""
        affected_files = event.affected_resources
        backup_dir = f"/tmp/security-backup-{int(time.time())}"
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_path in affected_files:
                if os.path.exists(file_path):
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    subprocess.run(['cp', file_path, backup_path], check=True)
            
            logger.info(f"Backed up affected files to {backup_dir}")
            event.recommended_actions.append(f"Backed up files to {backup_dir}")
            
        except Exception as e:
            logger.error(f"Failed to backup affected files: {e}")
    
    def _lockout_suspicious_user(self, event: SecurityEvent):
        """Lockout user accounts showing suspicious behavior."""
        user_info = event.details.get('user_info', {})
        username = user_info.get('username')
        
        if username and username != 'root':  # Don't lock out root
            try:
                # Lock user account
                subprocess.run(['sudo', 'usermod', '-L', username], check=True)
                
                logger.warning(f"Locked suspicious user account: {username}")
                event.recommended_actions.append(f"Locked user account {username}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to lock user {username}: {e}")
    
    def _send_security_notification(self, event: SecurityEvent):
        """Send security notification through configured channels."""
        if event.severity in self.config['incident_response']['notify_immediately']:
            self._send_immediate_notification(event)
        else:
            self._send_standard_notification(event)
    
    def _send_immediate_notification(self, event: SecurityEvent):
        """Send immediate notification for critical events."""
        if self.slack_webhook:
            self._send_slack_alert(event, urgent=True)
        
        # Could add SMS, email, or other immediate notification methods here
        logger.critical(f"IMMEDIATE SECURITY ALERT: {event.description}")
    
    def _send_standard_notification(self, event: SecurityEvent):
        """Send standard notification for non-critical events."""
        if self.slack_webhook:
            self._send_slack_alert(event, urgent=False)
        
        logger.warning(f"Security event: {event.description}")
    
    def _send_slack_alert(self, event: SecurityEvent, urgent: bool = False):
        """Send security alert to Slack."""
        try:
            severity_colors = {
                'critical': '#FF0000',
                'high': '#FF6600',
                'medium': '#FFCC00',
                'low': '#00FF00'
            }
            
            severity_emojis = {
                'critical': '🚨',
                'high': '⚠️',
                'medium': '⚡',
                'low': 'ℹ️'
            }
            
            message = {
                "text": f"{severity_emojis.get(event.severity, '🔔')} Security Alert",
                "attachments": [
                    {
                        "color": severity_colors.get(event.severity, '#808080'),
                        "fields": [
                            {
                                "title": "Event Type",
                                "value": event.event_type,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": event.severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": event.source,
                                "short": True
                            },
                            {
                                "title": "Description",
                                "value": event.description,
                                "short": False
                            }
                        ],
                        "footer": "CrewAI Security Monitor",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            if urgent:
                message["text"] = f"🚨 URGENT {message['text']}"
            
            response = requests.post(self.slack_webhook, json=message, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _export_to_siem(self, event: SecurityEvent):
        """Export security event to SIEM system."""
        if not self.siem_endpoint:
            return
        
        try:
            siem_event = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "severity": event.severity,
                "source": event.source,
                "description": event.description,
                "details": event.details,
                "affected_resources": event.affected_resources,
                "indicators_of_compromise": event.indicators_of_compromise,
                "recommended_actions": event.recommended_actions
            }
            
            response = requests.post(
                self.siem_endpoint,
                json=siem_event,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            logger.debug(f"Exported event to SIEM: {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to export to SIEM: {e}")
    
    def _export_security_metrics(self, event: SecurityEvent):
        """Export security metrics to Prometheus."""
        if not self.prometheus_pushgateway:
            return
        
        try:
            # Create metrics for the event
            metrics = [
                f'security_events_total{{type="{event.event_type}",severity="{event.severity}",source="{event.source}"}} 1',
                f'security_event_timestamp{{type="{event.event_type}"}} {int(datetime.now().timestamp())}'
            ]
            
            # Add IOC count metric
            if event.indicators_of_compromise:
                metrics.append(
                    f'security_iocs_detected{{type="{event.event_type}"}} {len(event.indicators_of_compromise)}'
                )
            
            metrics_data = '\n'.join(metrics)
            
            response = requests.post(
                f"{self.prometheus_pushgateway}/metrics/job/security-monitor/instance/{event.source}",
                data=metrics_data,
                headers={'Content-Type': 'text/plain'},
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to export security metrics: {e}")


class FileIntegrityMonitor:
    """File integrity monitoring component."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.file_hashes = {}
        self.events = []
    
    def start(self):
        """Start file integrity monitoring."""
        self.running = True
        self._initialize_file_hashes()
        
        while self.running:
            try:
                self._check_file_integrity()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"File integrity monitoring error: {e}")
    
    def stop(self):
        """Stop file integrity monitoring."""
        self.running = False
    
    def get_events(self) -> List[SecurityEvent]:
        """Get accumulated security events."""
        events = self.events[:]
        self.events.clear()
        return events
    
    def _initialize_file_hashes(self):
        """Initialize file hash database."""
        for path_pattern in self.config['file_integrity_paths']:
            try:
                import glob
                for file_path in glob.glob(path_pattern, recursive=True):
                    if os.path.isfile(file_path):
                        file_hash = self._calculate_file_hash(file_path)
                        if file_hash:
                            self.file_hashes[file_path] = file_hash
            except Exception as e:
                logger.error(f"Failed to initialize hashes for {path_pattern}: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate SHA-256 hash of file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None
    
    def _check_file_integrity(self):
        """Check file integrity against stored hashes."""
        for file_path, expected_hash in self.file_hashes.items():
            try:
                current_hash = self._calculate_file_hash(file_path)
                if current_hash and current_hash != expected_hash:
                    # File has been modified
                    event = SecurityEvent(
                        event_type="file_integrity_violation",
                        severity="high",
                        source="file_integrity_monitor",
                        description=f"File integrity violation detected: {file_path}",
                        details={
                            "file_path": file_path,
                            "expected_hash": expected_hash,
                            "current_hash": current_hash,
                            "modification_time": os.path.getmtime(file_path)
                        },
                        timestamp=datetime.now().isoformat(),
                        affected_resources=[file_path],
                        indicators_of_compromise=[file_path],
                        recommended_actions=[
                            f"Investigate changes to {file_path}",
                            "Restore from backup if unauthorized",
                            "Review system access logs"
                        ]
                    )
                    
                    self.events.append(event)
                    
                    # Update hash for continued monitoring
                    self.file_hashes[file_path] = current_hash
                    
            except Exception as e:
                logger.error(f"Error checking integrity of {file_path}: {e}")


class NetworkSecurityMonitor:
    """Network security monitoring component."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.events = []
    
    def start(self):
        """Start network security monitoring."""
        self.running = True
        
        while self.running:
            try:
                self._monitor_network_connections()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
    
    def stop(self):
        """Stop network security monitoring."""
        self.running = False
    
    def get_events(self) -> List[SecurityEvent]:
        """Get accumulated security events."""
        events = self.events[:]
        self.events.clear()
        return events
    
    def _monitor_network_connections(self):
        """Monitor network connections for suspicious activity."""
        try:
            connections = psutil.net_connections(kind='inet')
            
            for conn in connections:
                if conn.raddr:  # Remote address exists
                    remote_ip = conn.raddr.ip
                    remote_port = conn.raddr.port
                    
                    # Check for suspicious ports
                    if remote_port in self.config['network_monitoring']['suspicious_ports']:
                        self._create_suspicious_connection_event(conn)
                    
                    # Check against threat intelligence (would be passed from main monitor)
                    # This is a simplified check - in practice, would integrate with main monitor
                    
        except Exception as e:
            logger.error(f"Failed to monitor network connections: {e}")
    
    def _create_suspicious_connection_event(self, connection):
        """Create security event for suspicious network connection."""
        event = SecurityEvent(
            event_type="suspicious_network_connection",
            severity="medium",
            source="network_monitor",
            description=f"Suspicious network connection to {connection.raddr.ip}:{connection.raddr.port}",
            details={
                "local_address": f"{connection.laddr.ip}:{connection.laddr.port}" if connection.laddr else "unknown",
                "remote_address": f"{connection.raddr.ip}:{connection.raddr.port}",
                "protocol": connection.type.name,
                "status": connection.status,
                "pid": connection.pid
            },
            timestamp=datetime.now().isoformat(),
            affected_resources=[f"{connection.laddr.ip}:{connection.laddr.port}"],
            indicators_of_compromise=[connection.raddr.ip],
            recommended_actions=[
                f"Investigate connection to {connection.raddr.ip}:{connection.raddr.port}",
                "Check process making the connection",
                "Consider blocking the IP if malicious"
            ]
        )
        
        self.events.append(event)


class ProcessSecurityMonitor:
    """Process security monitoring component."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.events = []
        self.known_processes = set()
    
    def start(self):
        """Start process security monitoring."""
        self.running = True
        
        # Initialize known processes
        self._initialize_known_processes()
        
        while self.running:
            try:
                self._monitor_processes()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
    
    def stop(self):
        """Stop process security monitoring."""
        self.running = False
    
    def get_events(self) -> List[SecurityEvent]:
        """Get accumulated security events."""
        events = self.events[:]
        self.events.clear()
        return events
    
    def _initialize_known_processes(self):
        """Initialize set of known processes."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['cmdline']:
                        cmdline = ' '.join(proc_info['cmdline'])
                        self.known_processes.add(cmdline)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Failed to initialize known processes: {e}")
    
    def _monitor_processes(self):
        """Monitor running processes for suspicious activity."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    
                    # Check for suspicious process names/commands
                    if self._is_suspicious_process(proc_info):
                        self._create_suspicious_process_event(proc_info)
                    
                    # Check for resource abuse
                    if self._is_resource_abusive(proc_info):
                        self._create_resource_abuse_event(proc_info)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to monitor processes: {e}")
    
    def _is_suspicious_process(self, proc_info: Dict) -> bool:
        """Check if process is suspicious based on name/command."""
        suspicious_patterns = self.config['process_monitoring']['suspicious_processes']
        
        name = proc_info.get('name', '').lower()
        cmdline = ' '.join(proc_info.get('cmdline', [])).lower()
        
        for pattern in suspicious_patterns:
            if pattern.lower() in name or pattern.lower() in cmdline:
                return True
        
        return False
    
    def _is_resource_abusive(self, proc_info: Dict) -> bool:
        """Check if process is consuming excessive resources."""
        thresholds = self.config['process_monitoring']['resource_thresholds']
        
        cpu_percent = proc_info.get('cpu_percent', 0)
        memory_percent = proc_info.get('memory_percent', 0)
        
        return (cpu_percent > thresholds['cpu_percent'] or 
                memory_percent > thresholds['memory_percent'])
    
    def _create_suspicious_process_event(self, proc_info: Dict):
        """Create security event for suspicious process."""
        event = SecurityEvent(
            event_type="malicious_process",
            severity="high",
            source="process_monitor",
            description=f"Suspicious process detected: {proc_info.get('name', 'unknown')}",
            details={
                "process_info": proc_info,
                "detection_reason": "Matches suspicious process pattern"
            },
            timestamp=datetime.now().isoformat(),
            affected_resources=[f"PID:{proc_info.get('pid', 'unknown')}"],
            indicators_of_compromise=[proc_info.get('name', 'unknown')],
            recommended_actions=[
                "Investigate process activity",
                "Consider terminating if malicious",
                "Check for persistence mechanisms"
            ]
        )
        
        self.events.append(event)
    
    def _create_resource_abuse_event(self, proc_info: Dict):
        """Create security event for resource-abusive process."""
        event = SecurityEvent(
            event_type="resource_abuse",
            severity="medium",
            source="process_monitor",
            description=f"Process consuming excessive resources: {proc_info.get('name', 'unknown')}",
            details={
                "process_info": proc_info,
                "cpu_percent": proc_info.get('cpu_percent', 0),
                "memory_percent": proc_info.get('memory_percent', 0)
            },
            timestamp=datetime.now().isoformat(),
            affected_resources=[f"PID:{proc_info.get('pid', 'unknown')}"],
            indicators_of_compromise=[],
            recommended_actions=[
                "Investigate high resource usage",
                "Check if process is legitimate",
                "Consider resource limits"
            ]
        )
        
        self.events.append(event)


class AuthenticationMonitor:
    """Authentication security monitoring component."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.events = []
        self.failed_attempts = {}
    
    def start(self):
        """Start authentication monitoring."""
        self.running = True
        
        while self.running:
            try:
                self._monitor_authentication()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Authentication monitoring error: {e}")
    
    def stop(self):
        """Stop authentication monitoring."""
        self.running = False
    
    def get_events(self) -> List[SecurityEvent]:
        """Get accumulated security events."""
        events = self.events[:]
        self.events.clear()
        return events
    
    def _monitor_authentication(self):
        """Monitor authentication events from system logs."""
        try:
            # In a real implementation, this would parse /var/log/auth.log
            # or integrate with system authentication logs
            # For now, this is a placeholder implementation
            
            # Check for excessive failed attempts
            self._check_failed_attempts()
            
        except Exception as e:
            logger.error(f"Failed to monitor authentication: {e}")
    
    def _check_failed_attempts(self):
        """Check for excessive failed authentication attempts."""
        # This is a simplified implementation
        # In practice, would parse actual auth logs
        
        current_time = datetime.now()
        max_attempts = self.config['authentication_monitoring']['max_failed_attempts']
        
        # Clean up old entries
        expired_time = current_time - timedelta(minutes=5)
        for ip in list(self.failed_attempts.keys()):
            self.failed_attempts[ip] = [
                attempt_time for attempt_time in self.failed_attempts[ip]
                if attempt_time > expired_time
            ]
            if not self.failed_attempts[ip]:
                del self.failed_attempts[ip]
        
        # Check for IPs exceeding threshold
        for ip, attempts in self.failed_attempts.items():
            if len(attempts) >= max_attempts:
                self._create_auth_anomaly_event(ip, len(attempts))
    
    def _create_auth_anomaly_event(self, ip: str, attempt_count: int):
        """Create security event for authentication anomaly."""
        event = SecurityEvent(
            event_type="authentication_anomaly",
            severity="high",
            source="auth_monitor",
            description=f"Excessive failed authentication attempts from {ip}",
            details={
                "source_ip": ip,
                "failed_attempts": attempt_count,
                "time_window": "5 minutes"
            },
            timestamp=datetime.now().isoformat(),
            affected_resources=["authentication_system"],
            indicators_of_compromise=[ip],
            recommended_actions=[
                f"Block IP address {ip}",
                "Investigate source of attacks",
                "Review authentication logs"
            ]
        )
        
        self.events.append(event)


def main():
    """Main entry point for runtime security monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Runtime Security Monitor")
    parser.add_argument("--config", default="security-monitor-config.json",
                       help="Configuration file path")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon")
    
    args = parser.parse_args()
    
    monitor = RuntimeSecurityMonitor(args.config)
    
    if args.daemon:
        # In a real implementation, would properly daemonize
        logger.info("Running as daemon (simplified)")
    
    try:
        monitor.start()
        return 0
    except Exception as e:
        logger.error(f"Runtime security monitor failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
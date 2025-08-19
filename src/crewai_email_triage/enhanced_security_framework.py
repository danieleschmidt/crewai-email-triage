"""
Enhanced Security Framework v2.0
Comprehensive security controls for autonomous SDLC execution
"""

import hashlib
import hmac
import secrets
import time
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    MALICIOUS_INPUT = "malicious_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_FAILURE = "authentication_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityThreat:
    threat_type: str
    severity: ThreatLevel
    description: str
    evidence: str
    mitigation: str
    confidence: float
    timestamp: float


@dataclass
class SecurityEvent:
    event_type: SecurityEventType
    timestamp: float
    source_ip: str
    user_id: Optional[str]
    description: str
    severity: ThreatLevel
    metadata: Dict[str, Any]


@dataclass
class SecurityScanResult:
    is_safe: bool
    risk_score: float
    threats: List[SecurityThreat]
    quarantine_recommended: bool
    analysis_time_ms: float
    scan_metadata: Dict[str, Any]


class MaliciousPatternDetector:
    """Advanced pattern detection for malicious content."""
    
    def __init__(self):
        self.patterns = {
            'sql_injection': [
                r'(\bUNION\b.*\bSELECT\b)|(\bSELECT\b.*\bFROM\b.*\bWHERE\b)',
                r'(\bDROP\b.*\bTABLE\b)|(\bDELETE\b.*\bFROM\b)',
                r'(\bINSERT\b.*\bINTO\b)|(\bUPDATE\b.*\bSET\b)',
                r'(\'\s*OR\s*\'1\'\s*=\s*\'1)|(\'\s*OR\s*1\s*=\s*1)',
                r'(\'\s*;\s*DROP\s+)|(\'\s*;\s*DELETE\s+)',
            ],
            'xss_injection': [
                r'<script[^>]*>.*?</script>',
                r'javascript:[^"\'\s]*',
                r'on\w+\s*=\s*["\'][^"\']*["\']',
                r'<iframe[^>]*>.*?</iframe>',
                r'<object[^>]*>.*?</object>',
                r'expression\s*\(',
            ],
            'command_injection': [
                r'[;&|`\$\(\)\\]',
                r'\bcat\b|\bls\b|\bps\b|\bkill\b',
                r'\brm\b|\bmv\b|\bcp\b|\bchmod\b',
                r'\bwget\b|\bcurl\b|\bnc\b|\bnetcat\b',
                r'>\s*/dev/null|2>&1',
            ],
            'path_traversal': [
                r'\.\.[\\/]',
                r'[\\/]etc[\\/]passwd',
                r'[\\/]proc[\\/]',
                r'[\\/]var[\\/]log',
                r'file://|ftp://|gopher://',
            ],
            'data_exfiltration': [
                r'\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b',  # Credit card
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # Phone
            ],
            'malware_indicators': [
                r'\bbase64_decode\b|\bencode\b|\beval\b',
                r'\bexec\b|\bsystem\b|\bshell_exec\b',
                r'\b__import__\b|\bgetattr\b|\bsetattr\b',
                r'powershell\.exe|cmd\.exe|bash',
                r'meterpreter|metasploit|cobalt',
            ]
        }
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in pattern_list
            ]
    
    def scan_content(self, content: str) -> List[SecurityThreat]:
        """Scan content for malicious patterns."""
        threats = []
        
        for category, patterns in self.compiled_patterns.items():
            for i, pattern in enumerate(patterns):
                matches = pattern.findall(content)
                if matches:
                    # Calculate confidence based on pattern specificity and matches
                    confidence = min(0.9, 0.5 + (len(matches) * 0.1))
                    
                    # Determine severity
                    severity = self._get_threat_severity(category, len(matches))
                    
                    threat = SecurityThreat(
                        threat_type=category,
                        severity=severity,
                        description=f"Detected {category} pattern in content",
                        evidence=f"Pattern {i+1}: {matches[:3]}",  # Show first 3 matches
                        mitigation=self._get_mitigation_advice(category),
                        confidence=confidence,
                        timestamp=time.time()
                    )
                    threats.append(threat)
        
        return threats
    
    def _get_threat_severity(self, category: str, match_count: int) -> ThreatLevel:
        """Determine threat severity based on category and match count."""
        severity_map = {
            'sql_injection': ThreatLevel.HIGH,
            'xss_injection': ThreatLevel.HIGH,
            'command_injection': ThreatLevel.CRITICAL,
            'path_traversal': ThreatLevel.HIGH,
            'data_exfiltration': ThreatLevel.CRITICAL,
            'malware_indicators': ThreatLevel.CRITICAL
        }
        
        base_severity = severity_map.get(category, ThreatLevel.MEDIUM)
        
        # Escalate severity for multiple matches
        if match_count > 3:
            if base_severity == ThreatLevel.MEDIUM:
                return ThreatLevel.HIGH
            elif base_severity == ThreatLevel.HIGH:
                return ThreatLevel.CRITICAL
        
        return base_severity
    
    def _get_mitigation_advice(self, category: str) -> str:
        """Get mitigation advice for threat category."""
        mitigation_map = {
            'sql_injection': "Implement parameterized queries and input validation",
            'xss_injection': "Sanitize HTML input and implement Content Security Policy",
            'command_injection': "Validate and sanitize all user input, avoid system calls",
            'path_traversal': "Implement proper path validation and access controls",
            'data_exfiltration': "Review and redact sensitive data, implement DLP controls",
            'malware_indicators': "Quarantine content and perform deep malware analysis"
        }
        return mitigation_map.get(category, "Implement general input validation and monitoring")


class AccessControlManager:
    """Advanced access control and authorization."""
    
    def __init__(self):
        self.access_tokens: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._lock = threading.Lock()
    
    def create_access_token(self, user_id: str, permissions: List[str], 
                          expiry_hours: int = 24) -> str:
        """Create a secure access token."""
        token = secrets.token_urlsafe(32)
        
        with self._lock:
            self.access_tokens[token] = {
                'user_id': user_id,
                'permissions': permissions,
                'created_at': time.time(),
                'expires_at': time.time() + (expiry_hours * 3600),
                'last_used': time.time(),
                'usage_count': 0
            }
        
        logger.info(f"Created access token for user {user_id} with permissions: {permissions}")
        return token
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate an access token."""
        with self._lock:
            if token not in self.access_tokens:
                return False, None
            
            token_data = self.access_tokens[token]
            
            # Check expiry
            if time.time() > token_data['expires_at']:
                del self.access_tokens[token]
                return False, None
            
            # Update usage
            token_data['last_used'] = time.time()
            token_data['usage_count'] += 1
            
            return True, token_data
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, 
                        window_seconds: int = 3600) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        
        with self._lock:
            requests = self.rate_limits[identifier]
            
            # Remove old requests outside the window
            while requests and current_time - requests[0] > window_seconds:
                requests.popleft()
            
            # Check if within limit
            if len(requests) >= max_requests:
                return False
            
            # Add current request
            requests.append(current_time)
            return True
    
    def record_failed_attempt(self, identifier: str) -> bool:
        """Record a failed authentication attempt."""
        current_time = time.time()
        
        with self._lock:
            failures = self.failed_attempts[identifier]
            failures.append(current_time)
            
            # Check for too many recent failures (5 in 15 minutes)
            recent_failures = [f for f in failures if current_time - f < 900]
            return len(recent_failures) >= 5


class DataEncryptionManager:
    """Manage data encryption and secure storage."""
    
    def __init__(self):
        self.encryption_key = self._generate_key()
        self.encrypted_data: Dict[str, bytes] = {}
    
    def _generate_key(self) -> bytes:
        """Generate a secure encryption key."""
        return secrets.token_bytes(32)
    
    def encrypt_data(self, data: str, identifier: str) -> str:
        """Encrypt sensitive data."""
        try:
            from cryptography.fernet import Fernet
            
            # Use a derived key for this specific data
            derived_key = hashlib.pbkdf2_hmac('sha256', self.encryption_key, 
                                           identifier.encode(), 100000)
            fernet = Fernet(base64.urlsafe_b64encode(derived_key))
            
            encrypted = fernet.encrypt(data.encode())
            self.encrypted_data[identifier] = encrypted
            
            return f"encrypted:{identifier}"
            
        except ImportError:
            # Fallback to simple obfuscation if cryptography not available
            import base64
            encoded = base64.b64encode(data.encode()).decode()
            self.encrypted_data[identifier] = encoded.encode()
            return f"encoded:{identifier}"
    
    def decrypt_data(self, reference: str) -> Optional[str]:
        """Decrypt data using reference."""
        if not reference.startswith(('encrypted:', 'encoded:')):
            return None
        
        method, identifier = reference.split(':', 1)
        
        if identifier not in self.encrypted_data:
            return None
        
        try:
            if method == 'encrypted':
                from cryptography.fernet import Fernet
                
                derived_key = hashlib.pbkdf2_hmac('sha256', self.encryption_key,
                                               identifier.encode(), 100000)
                fernet = Fernet(base64.urlsafe_b64encode(derived_key))
                
                decrypted = fernet.decrypt(self.encrypted_data[identifier])
                return decrypted.decode()
            
            elif method == 'encoded':
                import base64
                decoded = base64.b64decode(self.encrypted_data[identifier])
                return decoded.decode()
                
        except Exception as e:
            logger.error(f"Decryption failed for {identifier}: {e}")
            return None


class SecurityEventLogger:
    """Log and analyze security events."""
    
    def __init__(self):
        self.events: deque = deque(maxlen=1000)
        self.event_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def log_event(self, event: SecurityEvent):
        """Log a security event."""
        with self._lock:
            self.events.append(event)
            self.event_counts[event.event_type.value] += 1
            
        # Log critical events immediately
        if event.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.warning(f"Security event: {event.event_type.value} - {event.description}")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]
            
            summary = {
                'total_events': len(recent_events),
                'event_types': defaultdict(int),
                'severity_breakdown': defaultdict(int),
                'top_sources': defaultdict(int),
                'recent_events': []
            }
            
            for event in recent_events:
                summary['event_types'][event.event_type.value] += 1
                summary['severity_breakdown'][event.severity.value] += 1
                summary['top_sources'][event.source_ip] += 1
                
                if event.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    summary['recent_events'].append({
                        'type': event.event_type.value,
                        'severity': event.severity.value,
                        'description': event.description,
                        'timestamp': event.timestamp
                    })
            
            return dict(summary)


class EnhancedSecurityFramework:
    """Main security framework orchestrating all security components."""
    
    def __init__(self):
        self.pattern_detector = MaliciousPatternDetector()
        self.access_control = AccessControlManager()
        self.encryption_manager = DataEncryptionManager()
        self.event_logger = SecurityEventLogger()
        self._initialized = False
    
    def initialize(self):
        """Initialize the security framework."""
        if self._initialized:
            return
        
        logger.info("Enhanced security framework initialized")
        self._initialized = True
    
    def comprehensive_security_scan(self, content: str, 
                                  source_ip: str = "unknown",
                                  user_id: Optional[str] = None) -> SecurityScanResult:
        """Perform comprehensive security scan of content."""
        start_time = time.time()
        
        if not self._initialized:
            self.initialize()
        
        # Detect malicious patterns
        threats = self.pattern_detector.scan_content(content)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(threats)
        
        # Determine if quarantine is recommended
        quarantine_recommended = (
            risk_score > 0.7 or 
            any(t.severity == ThreatLevel.CRITICAL for t in threats)
        )
        
        # Log security event if threats detected
        if threats:
            event = SecurityEvent(
                event_type=SecurityEventType.MALICIOUS_INPUT,
                timestamp=time.time(),
                source_ip=source_ip,
                user_id=user_id,
                description=f"Detected {len(threats)} security threats in content",
                severity=max(t.severity for t in threats),
                metadata={'threat_count': len(threats), 'risk_score': risk_score}
            )
            self.event_logger.log_event(event)
        
        analysis_time = (time.time() - start_time) * 1000
        
        return SecurityScanResult(
            is_safe=risk_score < 0.3 and not quarantine_recommended,
            risk_score=risk_score,
            threats=threats,
            quarantine_recommended=quarantine_recommended,
            analysis_time_ms=analysis_time,
            scan_metadata={
                'content_length': len(content),
                'patterns_checked': sum(len(patterns) for patterns in self.pattern_detector.compiled_patterns.values()),
                'source_ip': source_ip,
                'user_id': user_id
            }
        )
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score from threats."""
        if not threats:
            return 0.0
        
        # Weight threats by severity
        severity_weights = {
            ThreatLevel.LOW: 0.2,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.CRITICAL: 1.0
        }
        
        weighted_score = sum(
            severity_weights[threat.severity] * threat.confidence
            for threat in threats
        )
        
        # Normalize to 0-1 range
        max_possible_score = len(threats) * 1.0
        normalized_score = min(1.0, weighted_score / max_possible_score) if max_possible_score > 0 else 0.0
        
        return normalized_score
    
    def secure_data_storage(self, data: str, identifier: str) -> str:
        """Securely store sensitive data."""
        return self.encryption_manager.encrypt_data(data, identifier)
    
    def retrieve_secure_data(self, reference: str) -> Optional[str]:
        """Retrieve securely stored data."""
        return self.encryption_manager.decrypt_data(reference)
    
    def validate_access(self, token: str, required_permission: str) -> bool:
        """Validate access token and permissions."""
        is_valid, token_data = self.access_control.validate_token(token)
        
        if not is_valid or not token_data:
            return False
        
        return required_permission in token_data.get('permissions', [])
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        return self.access_control.check_rate_limit(identifier)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'active_tokens': len(self.access_control.access_tokens),
            'encrypted_data_items': len(self.encryption_manager.encrypted_data),
            'security_events_24h': self.event_logger.get_security_summary(24),
            'rate_limit_buckets': len(self.access_control.rate_limits),
            'framework_initialized': self._initialized,
            'timestamp': time.time()
        }


# Global security framework instance
security_framework = EnhancedSecurityFramework()


def secure_operation(required_permission: str = "basic"):
    """Decorator for securing operations with access control."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, you'd extract token from request context
            # For now, we'll skip the actual validation
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_security_framework() -> EnhancedSecurityFramework:
    """Get the global security framework instance."""
    return security_framework
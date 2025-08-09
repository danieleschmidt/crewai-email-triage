"""Advanced security enhancements for email triage system."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse
import ipaddress

from .logging_utils import get_logger
from .metrics_export import get_metrics_collector

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    
    threat_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    evidence: str
    mitigation: str
    confidence: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))


@dataclass
class SecurityAnalysisResult:
    """Result of comprehensive security analysis."""
    
    is_safe: bool
    risk_score: float  # 0.0 (safe) to 1.0 (maximum risk)
    threats: List[SecurityThreat] = field(default_factory=list)
    quarantine_recommended: bool = False
    safe_content: str = ""
    analysis_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_safe": self.is_safe,
            "risk_score": self.risk_score,
            "threats": [
                {
                    "type": threat.threat_type,
                    "severity": threat.severity,
                    "description": threat.description,
                    "evidence": threat.evidence,
                    "mitigation": threat.mitigation,
                    "confidence": threat.confidence,
                }
                for threat in self.threats
            ],
            "quarantine_recommended": self.quarantine_recommended,
            "analysis_time_ms": self.analysis_time_ms,
        }


class AdvancedSecurityScanner:
    """Advanced security scanner with threat detection capabilities."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the security scanner."""
        self.config = config or {}
        self.threat_signatures = self._load_threat_signatures()
        self.malicious_domains = self._load_malicious_domains()
        self.suspicious_patterns = self._load_suspicious_patterns()
        
        # Security thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
        
        logger.info("AdvancedSecurityScanner initialized")
    
    def scan_content(self, content: str, metadata: Optional[Dict] = None) -> SecurityAnalysisResult:
        """Perform comprehensive security analysis of email content."""
        
        start_time = time.perf_counter()
        threats = []
        safe_content = content
        
        try:
            # Phishing detection
            phishing_threats = self._detect_phishing(content)
            threats.extend(phishing_threats)
            
            # Malware indicators
            malware_threats = self._detect_malware_indicators(content)
            threats.extend(malware_threats)
            
            # Social engineering detection
            social_threats = self._detect_social_engineering(content)
            threats.extend(social_threats)
            
            # Spam detection
            spam_threats = self._detect_spam(content)
            threats.extend(spam_threats)
            
            # Data exfiltration attempts
            exfil_threats = self._detect_data_exfiltration(content)
            threats.extend(exfil_threats)
            
            # URL analysis
            url_threats = self._analyze_urls(content)
            threats.extend(url_threats)
            
            # Attachment analysis (if metadata provided)
            if metadata and 'attachments' in metadata:
                attachment_threats = self._analyze_attachments(metadata['attachments'])
                threats.extend(attachment_threats)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(threats)
            
            # Determine if content is safe
            is_safe = risk_score < self.risk_thresholds['medium']
            
            # Recommend quarantine for high-risk emails
            quarantine_recommended = risk_score >= self.risk_thresholds['high']
            
            # Sanitize content if threats detected
            if threats:
                safe_content = self._sanitize_content(content, threats)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            result = SecurityAnalysisResult(
                is_safe=is_safe,
                risk_score=risk_score,
                threats=threats,
                quarantine_recommended=quarantine_recommended,
                safe_content=safe_content,
                analysis_time_ms=processing_time
            )
            
            # Update metrics
            _metrics_collector.increment_counter("security_scans")
            _metrics_collector.record_histogram("security_scan_time_ms", processing_time)
            _metrics_collector.set_gauge("security_risk_score", risk_score)
            
            if threats:
                _metrics_collector.increment_counter("security_threats_detected", len(threats))
                for threat in threats:
                    _metrics_collector.increment_counter(f"security_threat_{threat.threat_type}")
            
            logger.info("Security scan completed",
                       extra={
                           'risk_score': risk_score,
                           'threats_count': len(threats),
                           'is_safe': is_safe,
                           'processing_time_ms': processing_time
                       })
            
            return result
            
        except Exception as e:
            _metrics_collector.increment_counter("security_scan_errors")
            logger.error("Security scan failed", extra={'error': str(e)})
            
            # Return safe result on error
            return SecurityAnalysisResult(
                is_safe=True,
                risk_score=0.0,
                safe_content=content,
                analysis_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def _load_threat_signatures(self) -> Dict[str, List[re.Pattern]]:
        """Load threat signature patterns."""
        return {
            'phishing': [
                re.compile(r'verify.*account.*immediately', re.IGNORECASE),
                re.compile(r'suspend.*account.*click', re.IGNORECASE),
                re.compile(r'confirm.*identity.*urgent', re.IGNORECASE),
                re.compile(r'update.*payment.*expire', re.IGNORECASE),
                re.compile(r'click.*here.*avoid', re.IGNORECASE),
            ],
            'malware': [
                re.compile(r'download.*attachment.*important', re.IGNORECASE),
                re.compile(r'install.*software.*required', re.IGNORECASE),
                re.compile(r'run.*executable.*update', re.IGNORECASE),
                re.compile(r'enable.*macros.*document', re.IGNORECASE),
            ],
            'social_engineering': [
                re.compile(r'ceo.*urgent.*transfer', re.IGNORECASE),
                re.compile(r'wire.*transfer.*immediately', re.IGNORECASE),
                re.compile(r'confidential.*invoice.*payment', re.IGNORECASE),
                re.compile(r'emergency.*funds.*transfer', re.IGNORECASE),
            ],
            'spam': [
                re.compile(r'make.*money.*fast', re.IGNORECASE),
                re.compile(r'win.*lottery.*claim', re.IGNORECASE),
                re.compile(r'work.*from.*home.*guaranteed', re.IGNORECASE),
                re.compile(r'free.*trial.*limited.*time', re.IGNORECASE),
            ]
        }
    
    def _load_malicious_domains(self) -> Set[str]:
        """Load known malicious domains."""
        return {
            'phishing-example.com',
            'fake-bank.com',
            'suspicious-site.org',
            'malware-host.net',
        }
    
    def _load_suspicious_patterns(self) -> List[re.Pattern]:
        """Load suspicious content patterns."""
        return [
            re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\s+[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}){5,}'),  # Email harvesting
            re.compile(r'(?:password|pwd|pin|ssn|social):\s*\w+', re.IGNORECASE),  # Credential theft
            re.compile(r'credit\s*card.*\d{4}.*\d{4}.*\d{4}.*\d{4}', re.IGNORECASE),  # Credit card numbers
            re.compile(r'bitcoin|btc|cryptocurrency.*wallet', re.IGNORECASE),  # Crypto scams
        ]
    
    def _detect_phishing(self, content: str) -> List[SecurityThreat]:
        """Detect phishing attempts."""
        threats = []
        
        for pattern in self.threat_signatures['phishing']:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="phishing",
                    severity="high",
                    description="Potential phishing attempt detected",
                    evidence=pattern.pattern,
                    mitigation="Do not click links or provide credentials",
                    confidence=0.8
                )
                threats.append(threat)
        
        # Check for credential harvesting
        cred_pattern = re.compile(r'(?:login|password|username).*(?:verify|confirm|update)', re.IGNORECASE)
        if cred_pattern.search(content):
            threat = SecurityThreat(
                threat_type="credential_harvesting",
                severity="high",
                description="Credential harvesting attempt detected",
                evidence="Request to verify/update login credentials",
                mitigation="Verify request through official channels",
                confidence=0.7
            )
            threats.append(threat)
        
        return threats
    
    def _detect_malware_indicators(self, content: str) -> List[SecurityThreat]:
        """Detect malware distribution indicators."""
        threats = []
        
        for pattern in self.threat_signatures['malware']:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="malware",
                    severity="critical",
                    description="Potential malware distribution detected",
                    evidence=pattern.pattern,
                    mitigation="Do not download or execute attachments",
                    confidence=0.9
                )
                threats.append(threat)
        
        # Check for suspicious file extensions in text
        malicious_extensions = re.compile(r'\.(exe|scr|pif|bat|cmd|com|vbs|js|jar)(?:\s|$)', re.IGNORECASE)
        if malicious_extensions.search(content):
            threat = SecurityThreat(
                threat_type="suspicious_attachment",
                severity="high",
                description="Suspicious file extension mentioned",
                evidence="References potentially malicious file types",
                mitigation="Scan files with antivirus before opening",
                confidence=0.6
            )
            threats.append(threat)
        
        return threats
    
    def _detect_social_engineering(self, content: str) -> List[SecurityThreat]:
        """Detect social engineering attempts."""
        threats = []
        
        for pattern in self.threat_signatures['social_engineering']:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="social_engineering",
                    severity="high",
                    description="Social engineering attempt detected",
                    evidence=pattern.pattern,
                    mitigation="Verify request through independent communication",
                    confidence=0.8
                )
                threats.append(threat)
        
        # Business Email Compromise (BEC) indicators
        bec_pattern = re.compile(r'(?:ceo|cfo|president|director).*(?:urgent|confidential|wire|transfer)', re.IGNORECASE)
        if bec_pattern.search(content):
            threat = SecurityThreat(
                threat_type="business_email_compromise",
                severity="critical",
                description="Business Email Compromise attempt detected",
                evidence="Executive impersonation with financial request",
                mitigation="Verify through direct communication with executive",
                confidence=0.85
            )
            threats.append(threat)
        
        return threats
    
    def _detect_spam(self, content: str) -> List[SecurityThreat]:
        """Detect spam content."""
        threats = []
        
        for pattern in self.threat_signatures['spam']:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="spam",
                    severity="low",
                    description="Spam content detected",
                    evidence=pattern.pattern,
                    mitigation="Mark as spam and delete",
                    confidence=0.6
                )
                threats.append(threat)
        
        # Check for excessive promotional language
        promo_words = ['free', 'win', 'prize', 'offer', 'deal', 'discount', 'save', 'money']
        promo_count = sum(1 for word in promo_words if word.lower() in content.lower())
        
        if promo_count >= 4:
            threat = SecurityThreat(
                threat_type="promotional_spam",
                severity="low",
                description="Excessive promotional language detected",
                evidence=f"Contains {promo_count} promotional keywords",
                mitigation="Review for legitimate promotional content",
                confidence=0.5
            )
            threats.append(threat)
        
        return threats
    
    def _detect_data_exfiltration(self, content: str) -> List[SecurityThreat]:
        """Detect potential data exfiltration attempts."""
        threats = []
        
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                threat = SecurityThreat(
                    threat_type="data_exfiltration",
                    severity="medium",
                    description="Potential data exfiltration attempt",
                    evidence="Contains suspicious data patterns",
                    mitigation="Review content for sensitive information",
                    confidence=0.6
                )
                threats.append(threat)
                break  # Only report once for this category
        
        return threats
    
    def _analyze_urls(self, content: str) -> List[SecurityThreat]:
        """Analyze URLs in content for threats."""
        threats = []
        
        # Extract URLs
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        urls = url_pattern.findall(content)
        
        for url in urls:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                
                # Check against malicious domains
                if domain in self.malicious_domains:
                    threat = SecurityThreat(
                        threat_type="malicious_url",
                        severity="critical",
                        description="Known malicious domain detected",
                        evidence=f"URL: {url}",
                        mitigation="Do not click the link",
                        confidence=0.95
                    )
                    threats.append(threat)
                
                # Check for suspicious URL characteristics
                if self._is_suspicious_url(url, parsed_url):
                    threat = SecurityThreat(
                        threat_type="suspicious_url",
                        severity="medium",
                        description="Suspicious URL characteristics detected",
                        evidence=f"URL: {url}",
                        mitigation="Verify URL legitimacy before clicking",
                        confidence=0.7
                    )
                    threats.append(threat)
                
                # Check for URL shorteners (potential hiding mechanism)
                shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly']
                if any(shortener in domain for shortener in shorteners):
                    threat = SecurityThreat(
                        threat_type="url_shortener",
                        severity="low",
                        description="URL shortener detected",
                        evidence=f"Shortened URL: {url}",
                        mitigation="Expand URL to verify destination",
                        confidence=0.5
                    )
                    threats.append(threat)
                    
            except Exception as e:
                logger.debug(f"Error analyzing URL {url}: {e}")
        
        return threats
    
    def _is_suspicious_url(self, url: str, parsed_url) -> bool:
        """Check if URL has suspicious characteristics."""
        
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        # Very long URLs (potential obfuscation)
        if len(url) > 200:
            return True
        
        # Suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.pw']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            return True
        
        # IP addresses instead of domains
        try:
            ipaddress.ip_address(domain)
            return True  # Using IP address directly is suspicious
        except ValueError:
            pass
        
        # Many subdomains (potential subdomain abuse)
        if domain.count('.') > 3:
            return True
        
        # Suspicious path patterns
        if any(pattern in path for pattern in ['/phish', '/fake', '/scam', '/malware']):
            return True
        
        return False
    
    def _analyze_attachments(self, attachments: List[Dict]) -> List[SecurityThreat]:
        """Analyze email attachments for threats."""
        threats = []
        
        for attachment in attachments:
            filename = attachment.get('filename', '').lower()
            content_type = attachment.get('content_type', '').lower()
            size = attachment.get('size', 0)
            
            # Check for dangerous file extensions
            dangerous_extensions = [
                '.exe', '.scr', '.pif', '.bat', '.cmd', '.com', '.vbs', '.js',
                '.jar', '.zip', '.rar', '.7z', '.doc', '.docm', '.xls', '.xlsm'
            ]
            
            if any(filename.endswith(ext) for ext in dangerous_extensions):
                severity = "critical" if filename.endswith(('.exe', '.scr', '.bat', '.vbs')) else "high"
                threat = SecurityThreat(
                    threat_type="dangerous_attachment",
                    severity=severity,
                    description="Potentially dangerous attachment detected",
                    evidence=f"File: {filename}",
                    mitigation="Scan with antivirus before opening",
                    confidence=0.8
                )
                threats.append(threat)
            
            # Check for suspiciously large files
            if size > 50 * 1024 * 1024:  # 50MB
                threat = SecurityThreat(
                    threat_type="large_attachment",
                    severity="low",
                    description="Unusually large attachment",
                    evidence=f"File: {filename}, Size: {size/1024/1024:.1f}MB",
                    mitigation="Verify file legitimacy",
                    confidence=0.4
                )
                threats.append(threat)
            
            # Check for double extensions (common obfuscation technique)
            if filename.count('.') >= 2:
                parts = filename.split('.')
                if len(parts) >= 3 and parts[-2] in ['pdf', 'doc', 'txt', 'jpg']:
                    threat = SecurityThreat(
                        threat_type="double_extension",
                        severity="high",
                        description="Double file extension detected (obfuscation)",
                        evidence=f"File: {filename}",
                        mitigation="File may be disguised executable",
                        confidence=0.85
                    )
                    threats.append(threat)
        
        return threats
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score based on detected threats."""
        
        if not threats:
            return 0.0
        
        # Weight threats by severity
        severity_weights = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }
        
        total_score = 0.0
        max_individual_score = 0.0
        
        for threat in threats:
            # Base score from severity
            base_score = severity_weights.get(threat.severity, 0.5)
            
            # Adjust by confidence
            adjusted_score = base_score * threat.confidence
            
            total_score += adjusted_score
            max_individual_score = max(max_individual_score, adjusted_score)
        
        # Combine total and max scores (prevents low-confidence mass detections from dominating)
        risk_score = min(1.0, (total_score * 0.3 + max_individual_score * 0.7))
        
        return risk_score
    
    def _sanitize_content(self, content: str, threats: List[SecurityThreat]) -> str:
        """Sanitize content based on detected threats."""
        
        sanitized = content
        
        # Remove or replace malicious URLs
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        
        def replace_url(match):
            url = match.group(0)
            # Check if this URL was flagged as malicious
            url_threats = [t for t in threats if t.threat_type in ['malicious_url', 'suspicious_url'] and url in t.evidence]
            if url_threats:
                return "[SUSPICIOUS_LINK_REMOVED]"
            return url
        
        sanitized = url_pattern.sub(replace_url, sanitized)
        
        # Mask potential credentials
        cred_pattern = re.compile(r'(?:password|pwd|pin):\s*\w+', re.IGNORECASE)
        sanitized = cred_pattern.sub('[CREDENTIAL_MASKED]', sanitized)
        
        # Mask potential credit card numbers
        cc_pattern = re.compile(r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}')
        sanitized = cc_pattern.sub('[PAYMENT_INFO_MASKED]', sanitized)
        
        return sanitized


class SecurityAuditLogger:
    """Advanced security audit logging system."""
    
    def __init__(self):
        """Initialize the security audit logger."""
        self.logger = get_logger("security_audit")
        
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None
    ):
        """Log security events with comprehensive details."""
        
        event_data = {
            'event_type': event_type,
            'severity': severity,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            'details': details,
        }
        
        if user_id:
            event_data['user_id'] = user_id
        
        if source_ip:
            event_data['source_ip'] = source_ip
        
        # Generate event hash for integrity
        event_hash = self._generate_event_hash(event_data)
        event_data['integrity_hash'] = event_hash
        
        self.logger.warning(
            f"Security Event: {event_type}",
            extra=event_data
        )
        
        # Update security metrics
        _metrics_collector.increment_counter(f"security_event_{event_type}")
        _metrics_collector.increment_counter(f"security_severity_{severity}")
    
    def _generate_event_hash(self, event_data: Dict[str, Any]) -> str:
        """Generate integrity hash for security event."""
        
        # Create canonical string representation
        canonical = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
        
        # Generate HMAC-SHA256 hash
        secret_key = secrets.token_bytes(32)  # In production, use persistent key
        return hmac.new(secret_key, canonical.encode(), hashlib.sha256).hexdigest()


# Global security audit logger instance
security_audit = SecurityAuditLogger()


def perform_security_scan(
    content: str,
    metadata: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> SecurityAnalysisResult:
    """Perform comprehensive security scan of email content."""
    
    scanner = AdvancedSecurityScanner(config)
    result = scanner.scan_content(content, metadata)
    
    # Log security events
    if result.threats:
        security_audit.log_security_event(
            event_type="threats_detected",
            severity="high" if result.risk_score >= 0.6 else "medium",
            details={
                'threat_count': len(result.threats),
                'risk_score': result.risk_score,
                'threat_types': [t.threat_type for t in result.threats],
                'quarantine_recommended': result.quarantine_recommended,
            }
        )
    
    return result
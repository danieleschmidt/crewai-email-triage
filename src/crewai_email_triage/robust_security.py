"""Security validation and sanitization module."""

import re
import html
import logging
from typing import Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class SecurityThreatLevel(Enum):
    """Security threat levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SecurityScanner:
    """Security scanner for email content."""
    
    def __init__(self):
        # Known malicious patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',                # JavaScript protocol
            r'vbscript:',                 # VBScript protocol
            r'on\w+\s*=',                 # Event handlers
            r'<iframe[^>]*>',             # Iframe injection
            r'<object[^>]*>',             # Object embedding
            r'<embed[^>]*>',              # Embed tags
        ]
        
        # Suspicious patterns
        self.suspicious_patterns = [
            r'urgent.*click.*now',
            r'verify.*account.*immediately',
            r'suspend.*account',
            r'winner.*prize.*claim',
            r'bank.*account.*frozen',
            r'tax.*refund.*pending'
        ]
        
        # Phishing indicators
        self.phishing_patterns = [
            r'paypal.*verify',
            r'amazon.*security',
            r'google.*security.*alert',
            r'microsoft.*account.*locked',
            r'apple.*id.*suspended'
        ]
    
    def scan_content(self, content: str) -> Dict[str, any]:
        """Scan content for security threats."""
        threats = []
        threat_level = SecurityThreatLevel.NONE
        
        content_lower = content.lower()
        
        # Check for malicious patterns (highest priority)
        for pattern in self.malicious_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if matches:
                threats.append({
                    "type": "malicious_code",
                    "pattern": pattern,
                    "matches": len(matches),
                    "severity": SecurityThreatLevel.CRITICAL.value
                })
                threat_level = max(threat_level, SecurityThreatLevel.CRITICAL)
        
        # Check for phishing patterns
        for pattern in self.phishing_patterns:
            if re.search(pattern, content_lower):
                threats.append({
                    "type": "phishing",
                    "pattern": pattern,
                    "severity": SecurityThreatLevel.HIGH.value
                })
                threat_level = max(threat_level, SecurityThreatLevel.HIGH)
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, content_lower):
                threats.append({
                    "type": "suspicious",
                    "pattern": pattern,
                    "severity": SecurityThreatLevel.MEDIUM.value
                })
                if threat_level.value < SecurityThreatLevel.MEDIUM.value:
                    threat_level = SecurityThreatLevel.MEDIUM
        
        # Additional security checks
        security_score = self._calculate_security_score(content)
        
        return {
            "threat_level": threat_level.value,
            "threat_count": len(threats),
            "threats": threats,
            "security_score": security_score,
            "is_safe": threat_level.value <= SecurityThreatLevel.LOW.value,
            "quarantine_recommended": threat_level.value >= SecurityThreatLevel.HIGH.value
        }
    
    def _calculate_security_score(self, content: str) -> float:
        """Calculate overall security score (0-1, higher is more suspicious)."""
        score = 0.0
        
        # URL density check
        url_pattern = r'https?://[^\s<>"]+'
        urls = re.findall(url_pattern, content)
        if len(urls) > 5:
            score += 0.3
        elif len(urls) > 2:
            score += 0.1
        
        # Excessive capitalization
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        if caps_ratio > 0.5:
            score += 0.2
        
        # Excessive punctuation
        punct_ratio = sum(1 for c in content if c in '!?') / max(len(content), 1)
        if punct_ratio > 0.05:
            score += 0.1
        
        # Content length anomalies
        if len(content) < 10:
            score += 0.2
        elif len(content) > 50000:
            score += 0.1
        
        return min(score, 1.0)

class ContentSanitizer:
    """Content sanitization utilities."""
    
    @staticmethod
    def sanitize_html(content: str) -> str:
        """Sanitize HTML content."""
        # Remove script tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove dangerous attributes
        content = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)
        content = re.sub(r'javascript:', '', content, flags=re.IGNORECASE)
        content = re.sub(r'vbscript:', '', content, flags=re.IGNORECASE)
        
        # HTML encode remaining content
        content = html.escape(content)
        
        return content
    
    @staticmethod
    def sanitize_email_content(content: str) -> Tuple[str, List[str]]:
        """Sanitize email content and return warnings."""
        warnings = []
        original_length = len(content)
        
        # Remove null bytes
        if '\x00' in content:
            content = content.replace('\x00', '')
            warnings.append("Removed null bytes from content")
        
        # Limit content length
        max_length = 100000
        if len(content) > max_length:
            content = content[:max_length]
            warnings.append(f"Content truncated to {max_length} characters")
        
        # Basic HTML sanitization
        sanitized = ContentSanitizer.sanitize_html(content)
        if len(sanitized) != len(content):
            warnings.append("HTML content was sanitized")
            content = sanitized
        
        return content, warnings

def secure_email_processing(content: str) -> Dict[str, any]:
    """Comprehensive security processing for email content."""
    scanner = SecurityScanner()
    
    # Security scan
    security_result = scanner.scan_content(content)
    
    # Content sanitization
    sanitized_content, sanitization_warnings = ContentSanitizer.sanitize_email_content(content)
    
    # Log security events
    if security_result["threat_level"] > SecurityThreatLevel.LOW.value:
        logger.warning(f"Security threats detected: level {security_result['threat_level']}")
    
    return {
        "original_content": content,
        "sanitized_content": sanitized_content,
        "security_analysis": security_result,
        "sanitization_warnings": sanitization_warnings,
        "processing_safe": security_result["is_safe"] and len(sanitization_warnings) == 0
    }

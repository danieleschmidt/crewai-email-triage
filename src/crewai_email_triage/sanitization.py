"""Email content sanitization and security validation."""

from __future__ import annotations

import html
import re
import urllib.parse
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class SanitizationConfig:
    """Configuration for email content sanitization."""
    
    max_length: int = 50000
    allow_html: bool = False
    strict_url_validation: bool = True
    remove_all_urls: bool = False
    sanitization_level: str = 'standard'  # 'basic', 'standard', 'strict'
    trusted_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=lambda: [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co'  # Common URL shorteners
    ])
    max_url_count: int = 10
    preserve_line_breaks: bool = True


@dataclass
class SanitizationResult:
    """Result of email content sanitization."""
    
    sanitized_content: str
    is_safe: bool
    threats_detected: List[str]
    modifications_made: List[str]
    original_length: int
    sanitized_length: int
    processing_time_ms: float


class EmailSanitizer:
    """Comprehensive email content sanitizer with threat detection."""
    
    def __init__(self, config: Optional[SanitizationConfig] = None):
        self.config = config or SanitizationConfig()
        self.threat_patterns = self._load_threat_patterns()
        self.safe_url_pattern = re.compile(r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*$')
        self.metrics = {
            'total_processed': 0,
            'threats_detected': 0,
            'safe_messages': 0,
            'total_processing_time': 0.0
        }
        
    def _load_threat_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load compiled regex patterns for threat detection."""
        patterns = {
            'script_injection': [
                re.compile(r'<\s*script[^>]*>.*?</\s*script\s*>', re.IGNORECASE | re.DOTALL),
                re.compile(r'javascript\s*:', re.IGNORECASE),
                re.compile(r'vbscript\s*:', re.IGNORECASE),
                re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers
            ],
            'html_injection': [
                re.compile(r'<\s*iframe[^>]*>', re.IGNORECASE),
                re.compile(r'<\s*object[^>]*>', re.IGNORECASE),
                re.compile(r'<\s*embed[^>]*>', re.IGNORECASE),
                re.compile(r'<\s*form[^>]*>', re.IGNORECASE),
                re.compile(r'<\s*meta[^>]*>', re.IGNORECASE),
            ],
            'sql_injection': [
                re.compile(r'\b(union|select|insert|update|delete|drop|create|alter)\b.*?\b(from|into|table|database)\b', re.IGNORECASE),
                re.compile(r'[\'";]\s*;\s*drop\s+table', re.IGNORECASE),
                re.compile(r'\'\s*or\s*\'\s*=\s*\'', re.IGNORECASE),
                re.compile(r'1\s*=\s*1|1\s*or\s*1', re.IGNORECASE),
            ],
            'data_exfiltration': [
                re.compile(r'document\.(cookie|location|referrer)', re.IGNORECASE),
                re.compile(r'window\.(location|open)', re.IGNORECASE),
                re.compile(r'eval\s*\(', re.IGNORECASE),
                re.compile(r'setTimeout\s*\(', re.IGNORECASE),
                re.compile(r'setInterval\s*\(', re.IGNORECASE),
            ],
            'suspicious_urls': [
                re.compile(r'javascript:', re.IGNORECASE),
                re.compile(r'data:', re.IGNORECASE),
                re.compile(r'file://', re.IGNORECASE),
                re.compile(r'ftp://', re.IGNORECASE),
            ],
            'unicode_attacks': [
                re.compile(r'[\u2000-\u200F\u2028-\u202F\u205F-\u206F\uFEFF]'),  # Zero-width/invisible chars
                re.compile(r'[\uFF00-\uFFEF]'),  # Fullwidth characters
            ]
        }
        return patterns
    
    @lru_cache(maxsize=1000)
    def sanitize(self, content: str) -> SanitizationResult:
        """Sanitize email content and detect threats.
        
        Parameters
        ----------
        content : str
            Raw email content to sanitize
            
        Returns
        -------
        SanitizationResult
            Detailed sanitization results
        """
        import time
        start_time = time.perf_counter()
        
        original_length = len(content)
        threats_detected = []
        modifications_made = []
        
        try:
            # Step 1: Unicode normalization
            content = self._normalize_unicode(content)
            
            # Step 2: Decode common encodings
            content, encoding_threats = self._decode_encodings(content)
            threats_detected.extend(encoding_threats)
            if encoding_threats:
                modifications_made.append('encoding_decoded')
            
            # Step 3: Length validation and truncation
            if len(content) > self.config.max_length:
                content = content[:self.config.max_length]
                modifications_made.append('content_truncated')
                logger.warning("Content truncated due to excessive length", 
                             extra={'original_length': original_length, 
                                   'truncated_length': len(content)})
            
            # Step 4: Binary content detection
            if self._detect_binary_content(content):
                threats_detected.append('binary_content')
                content = self._clean_binary_content(content)
                modifications_made.append('content_cleaned')
            
            # Step 5: Threat detection
            detected_threats = self._detect_threats(content)
            threats_detected.extend(detected_threats)
            
            # Step 6: Content sanitization based on detected threats
            content = self._sanitize_content(content, detected_threats, modifications_made)
            
            # Step 7: URL validation and sanitization
            content, url_threats = self._sanitize_urls(content)
            threats_detected.extend(url_threats)
            if url_threats:
                modifications_made.append('url_sanitized')
            
            # Step 8: Email address validation
            content = self._validate_email_addresses(content, modifications_made)
            
            # Step 9: Final cleanup
            content = self._final_cleanup(content, modifications_made)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            is_safe = len(threats_detected) == 0
            
            # Update metrics
            self.metrics['total_processed'] += 1
            self.metrics['total_processing_time'] += processing_time
            if threats_detected:
                self.metrics['threats_detected'] += 1
            else:
                self.metrics['safe_messages'] += 1
            
            if threats_detected:
                logger.info("Threats detected and sanitized", 
                           extra={'threats': threats_detected, 
                                 'modifications': modifications_made,
                                 'processing_time_ms': processing_time})
            
            return SanitizationResult(
                sanitized_content=content,
                is_safe=is_safe,
                threats_detected=threats_detected,
                modifications_made=modifications_made,
                original_length=original_length,
                sanitized_length=len(content),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error("Error during sanitization", extra={'error': str(e)})
            # Return safe fallback
            return SanitizationResult(
                sanitized_content="[Content sanitization failed - content removed for security]",
                is_safe=False,
                threats_detected=['sanitization_error'],
                modifications_made=['content_replaced'],
                original_length=original_length,
                sanitized_length=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def _normalize_unicode(self, content: str) -> str:
        """Normalize Unicode to prevent normalization attacks."""
        # Normalize to NFKC form to handle lookalike characters
        normalized = unicodedata.normalize('NFKC', content)
        return normalized
    
    def _decode_encodings(self, content: str) -> tuple[str, List[str]]:
        """Decode common encoding attacks."""
        threats = []
        original_content = content
        
        # URL decode
        try:
            decoded = urllib.parse.unquote(content)
            if decoded != content:
                content = decoded
                threats.append('encoding_attack')
        except Exception:
            pass
        
        # HTML entity decode
        try:
            decoded = html.unescape(content)
            if decoded != content:
                content = decoded
                if 'encoding_attack' not in threats:
                    threats.append('encoding_attack')
        except Exception:
            pass
        
        # Unicode escape decode
        try:
            if '\\u' in content:
                decoded = content.encode().decode('unicode_escape')
                if decoded != content:
                    content = decoded
                    if 'encoding_attack' not in threats:
                        threats.append('encoding_attack')
        except Exception:
            pass
            
        return content, threats
    
    def _detect_binary_content(self, content: str) -> bool:
        """Detect binary content in text."""
        # Check for null bytes and other binary indicators
        binary_indicators = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05']
        return any(indicator in content for indicator in binary_indicators)
    
    def _clean_binary_content(self, content: str) -> str:
        """Remove binary content from text."""
        # Remove null bytes and other binary characters
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
        return cleaned
    
    def _detect_threats(self, content: str) -> List[str]:
        """Detect various threat patterns in content."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    threats.append(threat_type)
                    break  # Only add each threat type once
        
        return threats
    
    def _sanitize_content(self, content: str, threats: List[str], modifications: List[str]) -> str:
        """Sanitize content based on detected threats."""
        
        # Remove script injections
        if 'script_injection' in threats:
            content = re.sub(r'<\s*script[^>]*>.*?</\s*script\s*>', '', content, flags=re.IGNORECASE | re.DOTALL)
            content = re.sub(r'javascript\s*:[^"\'\s]*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'vbscript\s*:[^"\'\s]*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)
            modifications.append('script_tags_removed')
        
        # Handle HTML based on configuration
        if not self.config.allow_html:
            if 'html_injection' in threats or '<' in content:
                content = re.sub(r'<[^>]*>', '', content)
                modifications.append('html_tags_removed')
        
        # Clean SQL injection patterns
        if 'sql_injection' in threats:
            # Replace dangerous SQL keywords with safe equivalents
            content = re.sub(r'\b(union|select|insert|update|delete|drop|create|alter)\b', '[SQL_KEYWORD]', content, flags=re.IGNORECASE)
            modifications.append('sql_keywords_sanitized')
        
        # Remove data exfiltration patterns
        if 'data_exfiltration' in threats:
            content = re.sub(r'(document|window)\.[a-zA-Z]+', '[BLOCKED]', content, flags=re.IGNORECASE)
            content = re.sub(r'eval\s*\([^)]*\)', '[BLOCKED]', content, flags=re.IGNORECASE)
            modifications.append('data_exfiltration_blocked')
        
        # Handle Unicode attacks
        if 'unicode_attacks' in threats:
            # Remove zero-width and invisible characters
            content = re.sub(r'[\u2000-\u200F\u2028-\u202F\u205F-\u206F\uFEFF]', '', content)
            # Replace fullwidth characters with normal ones
            content = re.sub(r'[\uFF00-\uFFEF]', lambda m: chr(ord(m.group()) - 0xFEE0), content)
            modifications.append('unicode_normalized')
        
        return content
    
    def _sanitize_urls(self, content: str) -> tuple[str, List[str]]:
        """Sanitize URLs in content."""
        threats = []
        
        if self.config.remove_all_urls:
            # Remove all URLs
            content = re.sub(r'https?://[^\s]+', '[URL_REMOVED]', content, flags=re.IGNORECASE)
            threats.append('url_removed')
            return content, threats
        
        # Find all URLs
        url_pattern = re.compile(r'(https?://[^\s]+)', re.IGNORECASE)
        urls = url_pattern.findall(content)
        
        if len(urls) > self.config.max_url_count:
            threats.append('excessive_urls')
            # Keep only first N URLs
            for i, url in enumerate(urls[self.config.max_url_count:], self.config.max_url_count):
                content = content.replace(url, '[URL_REMOVED]', 1)
        
        # Check each URL for threats
        for url in urls[:self.config.max_url_count]:
            # Check for suspicious protocols
            for threat_type, patterns in self.threat_patterns.items():
                if threat_type == 'suspicious_urls':
                    for pattern in patterns:
                        if pattern.search(url):
                            threats.append('suspicious_url')
                            content = content.replace(url, '[SUSPICIOUS_URL_REMOVED]')
                            break
            
            # Check domain against blocklist
            try:
                parsed = urllib.parse.urlparse(url)
                domain = parsed.netloc.lower()
                if any(blocked in domain for blocked in self.config.blocked_domains):
                    threats.append('blocked_domain')
                    content = content.replace(url, '[BLOCKED_DOMAIN]')
                elif self.config.trusted_domains and not any(trusted in domain for trusted in self.config.trusted_domains):
                    if self.config.strict_url_validation:
                        threats.append('untrusted_domain')
                        content = content.replace(url, '[UNTRUSTED_URL]')
            except Exception:
                threats.append('malformed_url')
                content = content.replace(url, '[MALFORMED_URL]')
        
        return content, threats
    
    def _validate_email_addresses(self, content: str, modifications: List[str]) -> str:
        """Validate and sanitize email addresses."""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        emails = email_pattern.findall(content)
        
        for email in emails:
            domain = email.split('@')[1].lower()
            # Check against blocked domains
            if any(blocked in domain for blocked in self.config.blocked_domains):
                content = content.replace(email, '[BLOCKED_EMAIL]')
                modifications.append('email_sanitized')
        
        return content
    
    def _final_cleanup(self, content: str, modifications: List[str]) -> str:
        """Perform final cleanup operations."""
        # Remove excessive whitespace
        content = re.sub(r'\s{3,}', '  ', content)
        
        # Preserve line breaks if configured
        if self.config.preserve_line_breaks:
            # Normalize line breaks
            content = re.sub(r'\r\n|\r', '\n', content)
        else:
            # Replace line breaks with spaces
            content = re.sub(r'[\r\n]+', ' ', content)
        
        # Trim leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def get_metrics(self) -> Dict[str, float]:
        """Get sanitization processing metrics."""
        return self.metrics.copy()


# Global sanitizer instance
_default_sanitizer = None


def get_sanitizer(config: Optional[SanitizationConfig] = None) -> EmailSanitizer:
    """Get the default sanitizer instance or create with custom config."""
    global _default_sanitizer
    if config is not None:
        return EmailSanitizer(config)
    if _default_sanitizer is None:
        _default_sanitizer = EmailSanitizer()
    return _default_sanitizer


def sanitize_email_content(content: str, config: Optional[SanitizationConfig] = None) -> SanitizationResult:
    """Convenience function to sanitize email content.
    
    Parameters
    ----------
    content : str
        Email content to sanitize
    config : SanitizationConfig, optional
        Custom sanitization configuration
        
    Returns
    -------
    SanitizationResult
        Sanitization results
    """
    sanitizer = get_sanitizer(config)
    return sanitizer.sanitize(content)
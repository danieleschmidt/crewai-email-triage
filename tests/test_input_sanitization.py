"""Test input sanitization and validation functionality."""

import pytest
from crewai_email_triage.sanitization import (
    EmailSanitizer,
    sanitize_email_content,
    SanitizationConfig,
    SanitizationResult
)


class TestInputSanitization:
    """Test email content sanitization."""

    def test_basic_text_passthrough(self):
        """Test that normal text passes through unchanged."""
        content = "This is a normal email about our meeting tomorrow."
        result = sanitize_email_content(content)
        
        assert result.sanitized_content == content
        assert result.is_safe is True
        assert result.threats_detected == []
        assert result.modifications_made == []

    def test_html_tag_removal(self):
        """Test that HTML tags are removed from content."""
        content = "Hello <script>alert('xss')</script> world"
        result = sanitize_email_content(content)
        
        assert "<script>" not in result.sanitized_content
        assert "alert" not in result.sanitized_content
        assert "Hello  world" in result.sanitized_content
        assert "html_tags_removed" in result.modifications_made
        assert "script_tag" in result.threats_detected

    def test_javascript_removal(self):
        """Test that JavaScript code is detected and removed."""
        dangerous_content = [
            "javascript:alert('xss')",
            "onclick='malicious()'",
            "eval(user_input)",
            "document.cookie",
            "window.location"
        ]
        
        for content in dangerous_content:
            result = sanitize_email_content(f"Click here: {content}")
            
            assert content not in result.sanitized_content
            assert "javascript_detected" in result.threats_detected
            assert result.is_safe is True

    def test_sql_injection_detection(self):
        """Test that SQL injection patterns are detected."""
        sql_patterns = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "INSERT INTO users VALUES",
            "DELETE FROM accounts WHERE"
        ]
        
        for pattern in sql_patterns:
            result = sanitize_email_content(f"User input: {pattern}")
            
            assert "sql_injection" in result.threats_detected
            # Content should be sanitized but message preserved
            assert len(result.sanitized_content) > 0

    def test_url_validation_and_sanitization(self):
        """Test URL validation and suspicious link detection."""
        test_cases = [
            ("Visit https://example.com", True, "https://example.com"),
            ("Click http://malicious-site.com/phish", False, None),
            ("Go to javascript:alert('xss')", False, None),
            ("Link: data:text/html,<script>alert(1)</script>", False, None),
            ("Check file:///etc/passwd", False, None)
        ]
        
        for content, should_be_safe, expected_url in test_cases:
            result = sanitize_email_content(content)
            
            if not should_be_safe:
                assert "suspicious_url" in result.threats_detected
                assert "url_sanitized" in result.modifications_made

    def test_email_address_validation(self):
        """Test email address validation and obfuscation."""
        content = "Contact me at admin@evil-domain.com or support@legitimate.org"
        result = sanitize_email_content(content)
        
        # Should preserve legitimate email patterns but flag suspicious ones
        assert "support@legitimate.org" in result.sanitized_content
        # Suspicious domains should be flagged
        if "admin@evil-domain.com" not in result.sanitized_content:
            assert "email_sanitized" in result.modifications_made

    def test_excessive_length_handling(self):
        """Test handling of excessively long content."""
        long_content = "A" * 100000  # 100KB of content
        result = sanitize_email_content(long_content)
        
        assert len(result.sanitized_content) <= 50000  # Should be truncated
        assert "content_truncated" in result.modifications_made
        assert result.is_safe is True

    def test_binary_content_detection(self):
        """Test detection and handling of binary content."""
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd".decode('latin-1')
        result = sanitize_email_content(binary_content)
        
        assert "binary_content" in result.threats_detected
        assert "content_cleaned" in result.modifications_made

    def test_unicode_normalization(self):
        """Test Unicode normalization and encoding attacks."""
        # Unicode normalization attack vectors
        unicode_attacks = [
            "café",  # Normal
            "café",  # Different Unicode composition
            "＜script＞alert('xss')＜/script＞",  # Fullwidth characters
            "јаvascript:alert(1)",  # Cyrillic characters that look like Latin
        ]
        
        for content in unicode_attacks:
            result = sanitize_email_content(content)
            
            # Should normalize and detect attacks
            assert result.sanitized_content is not None
            if "script" in content.lower():
                assert "unicode_attack" in result.threats_detected

    def test_encoding_attacks(self):
        """Test various encoding-based attack vectors."""
        encoding_attacks = [
            "%3Cscript%3Ealert(1)%3C/script%3E",  # URL encoded
            "&lt;script&gt;alert(1)&lt;/script&gt;",  # HTML entities
            "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e",  # Unicode escape
        ]
        
        for attack in encoding_attacks:
            result = sanitize_email_content(attack)
            
            # Should decode and then sanitize
            assert "script" not in result.sanitized_content.lower()
            assert "encoding_attack" in result.threats_detected

    def test_sanitization_config_customization(self):
        """Test customizable sanitization configuration."""
        config = SanitizationConfig(
            max_length=1000,
            allow_html=False,
            strict_url_validation=True,
            remove_all_urls=True
        )
        
        content = "Visit https://example.com for more info"
        result = sanitize_email_content(content, config=config)
        
        assert "https://example.com" not in result.sanitized_content
        assert "url_removed" in result.modifications_made

    def test_whitelist_domains(self):
        """Test domain whitelist functionality."""
        config = SanitizationConfig(
            trusted_domains=['example.com', 'trusted.org']
        )
        
        content = "Visit https://example.com and https://malicious.com"
        result = sanitize_email_content(content, config=config)
        
        assert "https://example.com" in result.sanitized_content
        assert "https://malicious.com" not in result.sanitized_content

    def test_sanitization_preserves_legitimate_content(self):
        """Test that legitimate content is preserved during sanitization."""
        legitimate_content = """
        Dear User,
        
        Thank you for your email regarding the quarterly report.
        Please find the attached document with the following details:
        
        - Revenue: $1,234,567
        - Growth: 15.5%
        - Meeting scheduled for 2024-01-15 at 2:30 PM
        
        Contact us at support@company.com for questions.
        
        Best regards,
        Team
        """
        
        result = sanitize_email_content(legitimate_content)
        
        assert result.is_safe is True
        assert result.threats_detected == []
        assert "quarterly report" in result.sanitized_content
        assert "$1,234,567" in result.sanitized_content
        assert "support@company.com" in result.sanitized_content

    def test_progressive_sanitization_levels(self):
        """Test different levels of sanitization strictness."""
        dangerous_content = "Visit <a href='javascript:alert(1)'>this link</a> now!"
        
        # Level 1: Basic sanitization
        result_basic = sanitize_email_content(dangerous_content, 
                                            config=SanitizationConfig(sanitization_level='basic'))
        
        # Level 2: Strict sanitization  
        result_strict = sanitize_email_content(dangerous_content,
                                             config=SanitizationConfig(sanitization_level='strict'))
        
        # Strict should remove more content than basic
        assert len(result_strict.sanitized_content) <= len(result_basic.sanitized_content)
        assert "javascript:" not in result_basic.sanitized_content
        assert "javascript:" not in result_strict.sanitized_content


class TestEmailSanitizer:
    """Test the EmailSanitizer class functionality."""

    def test_sanitizer_initialization(self):
        """Test EmailSanitizer initialization."""
        sanitizer = EmailSanitizer()
        assert sanitizer.config is not None
        assert isinstance(sanitizer.config, SanitizationConfig)

    def test_sanitizer_custom_config(self):
        """Test EmailSanitizer with custom configuration."""
        config = SanitizationConfig(max_length=500, allow_html=True)
        sanitizer = EmailSanitizer(config=config)
        
        assert sanitizer.config.max_length == 500
        assert sanitizer.config.allow_html is True

    def test_sanitizer_threat_patterns_loading(self):
        """Test that threat patterns are loaded correctly."""
        sanitizer = EmailSanitizer()
        
        # Should have threat patterns loaded
        assert len(sanitizer.threat_patterns) > 0
        assert any('script' in pattern for pattern in sanitizer.threat_patterns)
        assert any('javascript' in pattern for pattern in sanitizer.threat_patterns)

    def test_sanitizer_caching(self):
        """Test that sanitization results are cached for performance."""
        sanitizer = EmailSanitizer()
        content = "Test content for caching"
        
        # First call
        result1 = sanitizer.sanitize(content)
        
        # Second call should use cache
        result2 = sanitizer.sanitize(content)
        
        assert result1.sanitized_content == result2.sanitized_content
        assert result1.threats_detected == result2.threats_detected

    def test_sanitizer_metrics_tracking(self):
        """Test that sanitizer tracks processing metrics."""
        sanitizer = EmailSanitizer()
        
        # Process some content
        sanitizer.sanitize("Normal content")
        sanitizer.sanitize("<script>alert('xss')</script>")
        sanitizer.sanitize("Another normal message")
        
        metrics = sanitizer.get_metrics()
        
        assert metrics['total_processed'] >= 3
        assert metrics['threats_detected'] >= 1
        assert metrics['safe_messages'] >= 2


class TestSanitizationIntegration:
    """Test integration of sanitization with the email pipeline."""

    def test_pipeline_integration(self):
        """Test that sanitization integrates with the email triage pipeline."""
        from crewai_email_triage.pipeline import _triage_single
        from crewai_email_triage.classifier import ClassifierAgent
        from crewai_email_triage.priority import PriorityAgent
        from crewai_email_triage.summarizer import SummarizerAgent
        from crewai_email_triage.response import ResponseAgent
        
        dangerous_content = "Meeting tomorrow <script>alert('xss')</script>"
        
        result = _triage_single(
            dangerous_content,
            ClassifierAgent(),
            PriorityAgent(),
            SummarizerAgent(),
            ResponseAgent()
        )
        
        # Should process successfully without the dangerous content
        assert result['category'] is not None
        assert result['priority'] >= 0
        assert '<script>' not in str(result)
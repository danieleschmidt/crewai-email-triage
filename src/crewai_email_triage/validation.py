"""Comprehensive input validation for CrewAI Email Triage."""

from __future__ import annotations

import email
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in input."""

    field: str
    message: str
    severity: ValidationSeverity
    code: str
    value: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
            "code": self.code,
            "value": self.value,
            "suggestions": self.suggestions
        }


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_input: Optional[str] = None
    confidence_score: float = 1.0

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
                  for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "confidence_score": self.confidence_score,
            "issues": [issue.to_dict() for issue in self.issues],
            "sanitized_input": self.sanitized_input
        }


class EmailValidator:
    """Comprehensive email content validator."""

    def __init__(self):
        # Common email patterns
        self.email_patterns = {
            'email_address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            'date': re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b'),
            'money': re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        }

        # Suspicious patterns that might indicate security issues
        self.security_patterns = {
            'script_tag': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            'iframe_tag': re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
            'javascript': re.compile(r'javascript:', re.IGNORECASE),
            'data_uri': re.compile(r'data:[^;]+;base64,', re.IGNORECASE),
            'suspicious_headers': re.compile(r'x-(?:forwarded|real)-ip|x-originating-ip', re.IGNORECASE),
        }

        # Business logic patterns
        self.priority_keywords = {
            'urgent': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'rush'],
            'deadline': ['deadline', 'due date', 'expires', 'cutoff', 'before'],
            'meeting': ['meeting', 'call', 'conference', 'zoom', 'teams', 'appointment'],
            'approval': ['approve', 'review', 'sign off', 'authorize', 'confirm'],
            'financial': ['invoice', 'payment', 'budget', 'cost', 'expense', 'purchase'],
        }

    def validate_content(self, content: str) -> ValidationResult:
        """Validate email content comprehensively."""
        if not isinstance(content, str):
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    field="content",
                    message=f"Content must be a string, got {type(content).__name__}",
                    severity=ValidationSeverity.CRITICAL,
                    code="INVALID_TYPE"
                )]
            )

        issues = []
        sanitized_content = content
        confidence_score = 1.0

        # Basic validation
        issues.extend(self._validate_basic_properties(content))

        # Structure validation
        issues.extend(self._validate_structure(content))

        # Security validation
        security_issues = self._validate_security(content)
        issues.extend(security_issues)

        # Content analysis
        issues.extend(self._analyze_content_quality(content))

        # Business logic validation
        issues.extend(self._validate_business_logic(content))

        # Calculate confidence score based on issues
        error_count = len([i for i in issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)])
        warning_count = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        confidence_score = max(0.0, 1.0 - (error_count * 0.3) - (warning_count * 0.1))

        # Determine if valid (no critical errors)
        is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            sanitized_input=sanitized_content,
            confidence_score=confidence_score
        )

    def _validate_basic_properties(self, content: str) -> List[ValidationIssue]:
        """Validate basic content properties."""
        issues = []

        # Length checks
        if len(content) == 0:
            issues.append(ValidationIssue(
                field="content",
                message="Content is empty",
                severity=ValidationSeverity.ERROR,
                code="EMPTY_CONTENT"
            ))
        elif len(content) < 10:
            issues.append(ValidationIssue(
                field="content",
                message="Content is very short (less than 10 characters)",
                severity=ValidationSeverity.WARNING,
                code="SHORT_CONTENT",
                suggestions=["Consider providing more context or details"]
            ))
        elif len(content) > 50000:
            issues.append(ValidationIssue(
                field="content",
                message="Content is very long (over 50,000 characters)",
                severity=ValidationSeverity.WARNING,
                code="LONG_CONTENT",
                suggestions=["Consider summarizing or breaking into smaller messages"]
            ))

        # Encoding checks
        try:
            content.encode('utf-8')
        except UnicodeEncodeError as e:
            issues.append(ValidationIssue(
                field="content",
                message=f"Content contains invalid UTF-8 characters: {e}",
                severity=ValidationSeverity.ERROR,
                code="INVALID_ENCODING"
            ))

        # Control character checks
        control_chars = [c for c in content if ord(c) < 32 and c not in '\t\n\r']
        if control_chars:
            issues.append(ValidationIssue(
                field="content",
                message=f"Content contains {len(control_chars)} control characters",
                severity=ValidationSeverity.WARNING,
                code="CONTROL_CHARACTERS",
                suggestions=["Remove or replace control characters"]
            ))

        return issues

    def _validate_structure(self, content: str) -> List[ValidationIssue]:
        """Validate email structure and format."""
        issues = []

        # Check if it looks like an email
        has_email_indicators = any([
            'from:' in content.lower(),
            'to:' in content.lower(),
            'subject:' in content.lower(),
            '@' in content,
            'dear' in content.lower(),
            'sincerely' in content.lower(),
            'best regards' in content.lower(),
        ])

        if not has_email_indicators:
            issues.append(ValidationIssue(
                field="content",
                message="Content doesn't appear to be email-like",
                severity=ValidationSeverity.INFO,
                code="NOT_EMAIL_LIKE",
                suggestions=["Ensure content is actual email content"]
            ))

        # Try parsing as email message
        try:
            msg = email.message_from_string(content)
            if msg.get_content_type() == 'text/plain' and not msg.get_payload():
                issues.append(ValidationIssue(
                    field="content",
                    message="Email appears to have no body content",
                    severity=ValidationSeverity.WARNING,
                    code="NO_BODY_CONTENT"
                ))
        except Exception:
            # Content might not be in email format, which is fine
            pass

        # Check for excessive HTML
        html_tags = re.findall(r'<[^>]+>', content)
        if len(html_tags) > 50:
            issues.append(ValidationIssue(
                field="content",
                message=f"Content contains many HTML tags ({len(html_tags)})",
                severity=ValidationSeverity.INFO,
                code="HTML_HEAVY",
                suggestions=["Consider extracting plain text for better processing"]
            ))

        return issues

    def _validate_security(self, content: str) -> List[ValidationIssue]:
        """Validate content for security issues."""
        issues = []

        # Check for suspicious patterns
        for pattern_name, pattern in self.security_patterns.items():
            matches = pattern.findall(content)
            if matches:
                severity = ValidationSeverity.CRITICAL if pattern_name in ['script_tag', 'javascript'] else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    field="content",
                    message=f"Potentially unsafe content detected: {pattern_name}",
                    severity=severity,
                    code=f"SECURITY_{pattern_name.upper()}",
                    value=matches[0][:100] if matches else None,
                    suggestions=["Content should be sanitized before processing"]
                ))

        # Check for suspicious URLs
        urls = self.email_patterns['url'].findall(content)
        suspicious_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
        for url in urls:
            if any(domain in url.lower() for domain in suspicious_domains):
                issues.append(ValidationIssue(
                    field="content",
                    message="Content contains shortened URLs",
                    severity=ValidationSeverity.WARNING,
                    code="SHORTENED_URL",
                    value=url,
                    suggestions=["Be cautious with shortened URLs"]
                ))

        # Check for excessive special characters (possible obfuscation)
        special_char_ratio = len([c for c in content if not c.isalnum() and not c.isspace()]) / len(content)
        if special_char_ratio > 0.3:
            issues.append(ValidationIssue(
                field="content",
                message=f"High ratio of special characters ({special_char_ratio:.1%})",
                severity=ValidationSeverity.WARNING,
                code="HIGH_SPECIAL_CHARS",
                suggestions=["Verify content is not obfuscated"]
            ))

        return issues

    def _analyze_content_quality(self, content: str) -> List[ValidationIssue]:
        """Analyze content quality and structure."""
        issues = []

        # Readability checks
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length > 40:
                issues.append(ValidationIssue(
                    field="content",
                    message=f"Average sentence length is very long ({avg_sentence_length:.1f} words)",
                    severity=ValidationSeverity.INFO,
                    code="LONG_SENTENCES",
                    suggestions=["Consider breaking long sentences for better readability"]
                ))

        # Language detection (basic)
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        if words:
            # Simple English word check
            english_indicators = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            english_word_count = sum(1 for word in words if word in english_indicators)
            english_ratio = english_word_count / len(words)

            if english_ratio < 0.05:
                issues.append(ValidationIssue(
                    field="content",
                    message="Content may not be in English",
                    severity=ValidationSeverity.INFO,
                    code="NON_ENGLISH",
                    suggestions=["Verify language requirements for processing"]
                ))

        # Repetition check
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1

        repeated_words = [(word, count) for word, count in word_freq.items() if count > 10]
        if repeated_words:
            issues.append(ValidationIssue(
                field="content",
                message=f"High word repetition detected: {repeated_words[:3]}",
                severity=ValidationSeverity.INFO,
                code="HIGH_REPETITION",
                suggestions=["Check for spam or automated content"]
            ))

        return issues

    def _validate_business_logic(self, content: str) -> List[ValidationIssue]:
        """Validate business logic and context."""
        issues = []
        content_lower = content.lower()

        # Check for conflicting priorities
        priority_matches = {}
        for priority, keywords in self.priority_keywords.items():
            matches = [kw for kw in keywords if kw in content_lower]
            if matches:
                priority_matches[priority] = matches

        if len(priority_matches) > 3:
            issues.append(ValidationIssue(
                field="content",
                message="Content contains mixed priority indicators",
                severity=ValidationSeverity.INFO,
                code="MIXED_PRIORITIES",
                value=str(list(priority_matches.keys())),
                suggestions=["Clarify the main priority or urgency level"]
            ))

        # Check for incomplete information
        question_words = ['what', 'when', 'where', 'who', 'why', 'how']
        questions = [qw for qw in question_words if f"{qw} " in content_lower or f"{qw}?" in content_lower]
        if len(questions) > 5:
            issues.append(ValidationIssue(
                field="content",
                message="Content contains many questions - may need more context",
                severity=ValidationSeverity.INFO,
                code="MANY_QUESTIONS",
                suggestions=["Provide more background information"]
            ))

        # Check for financial information
        money_matches = self.email_patterns['money'].findall(content)
        if money_matches:
            issues.append(ValidationIssue(
                field="content",
                message="Content contains financial information",
                severity=ValidationSeverity.INFO,
                code="FINANCIAL_CONTENT",
                value=str(money_matches[:3]),
                suggestions=["Ensure proper handling of financial data"]
            ))

        return issues


# Global validator instance
_email_validator: Optional[EmailValidator] = None


def get_email_validator() -> EmailValidator:
    """Get the global email validator instance."""
    global _email_validator
    if _email_validator is None:
        _email_validator = EmailValidator()
    return _email_validator


def validate_email_content(content: str) -> ValidationResult:
    """Validate email content using the global validator."""
    validator = get_email_validator()
    return validator.validate_content(content)


class ConfigValidator:
    """Validator for configuration files and parameters."""

    @staticmethod
    def validate_config_dict(config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary."""
        issues = []

        # Check required sections
        required_sections = ['classifier', 'priority', 'summarizer', 'response']
        for section in required_sections:
            if section not in config:
                issues.append(ValidationIssue(
                    field=f"config.{section}",
                    message=f"Missing required configuration section: {section}",
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_CONFIG_SECTION"
                ))

        # Validate classifier config
        if 'classifier' in config:
            classifier_config = config['classifier']
            if not isinstance(classifier_config, dict):
                issues.append(ValidationIssue(
                    field="config.classifier",
                    message="Classifier configuration must be a dictionary",
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_CONFIG_TYPE"
                ))
            else:
                # Check for empty keyword lists
                for category, keywords in classifier_config.items():
                    if isinstance(keywords, list) and len(keywords) == 0:
                        issues.append(ValidationIssue(
                            field=f"config.classifier.{category}",
                            message=f"Empty keyword list for category: {category}",
                            severity=ValidationSeverity.WARNING,
                            code="EMPTY_KEYWORD_LIST"
                        ))

        # Validate priority config
        if 'priority' in config:
            priority_config = config['priority']
            if isinstance(priority_config, dict):
                for key, value in priority_config.items():
                    if isinstance(value, (int, float)) and not (0 <= value <= 10):
                        issues.append(ValidationIssue(
                            field=f"config.priority.{key}",
                            message=f"Priority score must be between 0-10, got {value}",
                            severity=ValidationSeverity.ERROR,
                            code="INVALID_PRIORITY_SCORE"
                        ))

        is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            confidence_score=1.0 - (len(issues) * 0.1)
        )

"""Data privacy and compliance features."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_utils import get_logger

logger = get_logger(__name__)


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan."""

    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    redacted_content: Optional[str] = None
    original_length: int = 0
    redacted_length: int = 0
    processing_time_ms: float = 0.0


@dataclass
class ComplianceReport:
    """Comprehensive compliance analysis report."""

    framework: ComplianceFramework
    compliant: bool = True
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    data_classification: DataClassification = DataClassification.PUBLIC
    retention_period_days: int = 30
    pii_summary: Optional[PIIDetectionResult] = None
    geographic_restrictions: List[str] = field(default_factory=list)
    processing_lawful_basis: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "framework": self.framework.value,
            "compliant": self.compliant,
            "violations": self.violations,
            "recommendations": self.recommendations,
            "data_classification": self.data_classification.value,
            "retention_period_days": self.retention_period_days,
            "pii_summary": {
                "pii_detected": self.pii_summary.pii_detected if self.pii_summary else False,
                "pii_types": self.pii_summary.pii_types if self.pii_summary else [],
                "redacted_content_available": bool(self.pii_summary.redacted_content if self.pii_summary else False)
            } if self.pii_summary else None,
            "geographic_restrictions": self.geographic_restrictions,
            "processing_lawful_basis": self.processing_lawful_basis
        }


class PIIDetector:
    """Advanced Personally Identifiable Information detector."""

    def __init__(self):
        # Regex patterns for common PII types
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_us": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
            "credit_card": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            "iban": r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b',
            "ip_address": r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        }

    def detect_pii(self, content: str) -> PIIDetectionResult:
        """Detect PII in content and optionally redact it."""
        import re

        start_time = time.perf_counter()
        result = PIIDetectionResult(
            original_length=len(content)
        )

        redacted_content = content

        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)

            if matches:
                result.pii_detected = True
                result.pii_types.append(pii_type)
                result.confidence_scores[pii_type] = min(len(matches) * 0.2 + 0.8, 1.0)

                # Redact the PII
                redacted_content = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]',
                                        redacted_content, flags=re.IGNORECASE)

                logger.info(f"PII detected: {pii_type} ({len(matches)} instances)")

        result.redacted_content = redacted_content if result.pii_detected else None
        result.redacted_length = len(redacted_content)
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000

        return result


class ComplianceChecker:
    """Multi-framework compliance checker."""

    def __init__(self):
        self.pii_detector = PIIDetector()
        self.supported_frameworks = {
            ComplianceFramework.GDPR: self._check_gdpr_compliance,
            ComplianceFramework.CCPA: self._check_ccpa_compliance,
            ComplianceFramework.PDPA: self._check_pdpa_compliance,
            ComplianceFramework.PIPEDA: self._check_pipeda_compliance,
            ComplianceFramework.LGPD: self._check_lgpd_compliance,
        }

    def check_compliance(
        self,
        content: str,
        framework: ComplianceFramework = ComplianceFramework.GDPR,
        user_region: str = "EU",
        processing_purpose: str = "email_triage"
    ) -> ComplianceReport:
        """Perform comprehensive compliance check."""

        logger.info("Starting compliance check",
                   extra={'framework': framework.value, 'region': user_region})

        # Detect PII first
        pii_result = self.pii_detector.detect_pii(content)

        # Run framework-specific checks
        report = ComplianceReport(
            framework=framework,
            pii_summary=pii_result
        )

        if framework in self.supported_frameworks:
            self.supported_frameworks[framework](report, content, user_region, processing_purpose)
        else:
            report.violations.append(f"Unsupported compliance framework: {framework.value}")
            report.compliant = False

        # Final compliance determination
        report.compliant = len(report.violations) == 0

        logger.info("Compliance check completed",
                   extra={
                       'framework': framework.value,
                       'compliant': report.compliant,
                       'violations_count': len(report.violations),
                       'pii_detected': pii_result.pii_detected
                   })

        return report

    def _check_gdpr_compliance(
        self,
        report: ComplianceReport,
        content: str,
        user_region: str,
        processing_purpose: str
    ) -> None:
        """Check GDPR compliance (EU)."""

        # GDPR applies to EU residents or data processed in EU
        if user_region in ["EU", "EEA"] or any(country in user_region.upper() for country in [
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
            "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"
        ]):
            report.geographic_restrictions.append("EU/EEA data subject rights apply")

        # Check for lawful basis
        lawful_bases = {
            "email_triage": "legitimate_interest",
            "customer_support": "contract_performance",
            "marketing": "consent_required",
            "analytics": "legitimate_interest"
        }

        report.processing_lawful_basis = lawful_bases.get(processing_purpose, "consent_required")

        # PII detection requirements
        if report.pii_summary and report.pii_summary.pii_detected:
            sensitive_pii = {"ssn", "credit_card", "health_id"}
            detected_sensitive = set(report.pii_summary.pii_types) & sensitive_pii

            if detected_sensitive:
                report.data_classification = DataClassification.RESTRICTED
                report.violations.append("Sensitive personal data detected - enhanced protections required")
                report.recommendations.append("Implement explicit consent mechanisms for sensitive data")

            report.retention_period_days = 30  # Conservative retention
            report.recommendations.append("Implement data subject access request procedures")
            report.recommendations.append("Provide clear privacy notice and opt-out mechanisms")

        # Technical and organizational measures
        report.recommendations.extend([
            "Implement data encryption at rest and in transit",
            "Maintain data processing records (Article 30)",
            "Conduct Data Protection Impact Assessment if high risk",
            "Ensure data processor agreements are in place"
        ])

    def _check_ccpa_compliance(
        self,
        report: ComplianceReport,
        content: str,
        user_region: str,
        processing_purpose: str
    ) -> None:
        """Check CCPA compliance (California)."""

        if "CA" in user_region.upper() or "CALIFORNIA" in user_region.upper():
            report.geographic_restrictions.append("California consumer rights apply")

            if report.pii_summary and report.pii_summary.pii_detected:
                report.recommendations.extend([
                    "Provide 'Do Not Sell My Personal Information' option",
                    "Implement consumer request procedures (access, delete, opt-out)",
                    "Update privacy policy with CCPA disclosures",
                    "Maintain records of consumer requests and responses"
                ])

                report.retention_period_days = 365  # CCPA allows longer retention

    def _check_pdpa_compliance(
        self,
        report: ComplianceReport,
        content: str,
        user_region: str,
        processing_purpose: str
    ) -> None:
        """Check PDPA compliance (Singapore/Thailand)."""

        if any(country in user_region.upper() for country in ["SG", "SINGAPORE", "TH", "THAILAND"]):
            report.geographic_restrictions.append("PDPA data protection obligations apply")

            if report.pii_summary and report.pii_summary.pii_detected:
                report.recommendations.extend([
                    "Obtain appropriate consent for personal data collection",
                    "Implement data breach notification procedures",
                    "Designate Data Protection Officer if required",
                    "Provide access and correction mechanisms"
                ])

    def _check_pipeda_compliance(
        self,
        report: ComplianceReport,
        content: str,
        user_region: str,
        processing_purpose: str
    ) -> None:
        """Check PIPEDA compliance (Canada)."""

        if "CA" in user_region.upper() and "CALIFORNIA" not in user_region.upper():
            report.geographic_restrictions.append("Canadian privacy obligations apply")
            report.processing_lawful_basis = "reasonable_purposes"

    def _check_lgpd_compliance(
        self,
        report: ComplianceReport,
        content: str,
        user_region: str,
        processing_purpose: str
    ) -> None:
        """Check LGPD compliance (Brazil)."""

        if any(country in user_region.upper() for country in ["BR", "BRAZIL"]):
            report.geographic_restrictions.append("LGPD data protection requirements apply")

            if report.pii_summary and report.pii_summary.pii_detected:
                report.recommendations.extend([
                    "Implement data subject rights procedures (access, rectification, deletion)",
                    "Maintain data processing impact assessments",
                    "Designate Data Protection Officer for public entities",
                    "Implement privacy by design principles"
                ])


def check_compliance_for_content(
    content: str,
    framework: str = "gdpr",
    user_region: str = "EU",
    processing_purpose: str = "email_triage"
) -> ComplianceReport:
    """Convenience function for compliance checking."""

    try:
        framework_enum = ComplianceFramework(framework.lower())
    except ValueError:
        logger.warning(f"Unknown compliance framework '{framework}', defaulting to GDPR")
        framework_enum = ComplianceFramework.GDPR

    checker = ComplianceChecker()
    return checker.check_compliance(content, framework_enum, user_region, processing_purpose)


def redact_pii_from_content(content: str) -> tuple[str, PIIDetectionResult]:
    """Detect and redact PII from content."""
    detector = PIIDetector()
    pii_result = detector.detect_pii(content)

    redacted_content = pii_result.redacted_content if pii_result.redacted_content else content

    return redacted_content, pii_result


def get_supported_compliance_frameworks() -> List[str]:
    """Get list of supported compliance frameworks."""
    return [framework.value for framework in ComplianceFramework]


# Global compliance checker instance
_compliance_checker: Optional[ComplianceChecker] = None


def get_compliance_checker() -> ComplianceChecker:
    """Get the global compliance checker instance."""
    global _compliance_checker
    if _compliance_checker is None:
        _compliance_checker = ComplianceChecker()
    return _compliance_checker

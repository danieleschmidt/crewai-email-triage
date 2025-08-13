"""Global-first features: multi-region, internationalization, and compliance."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging_utils import get_logger
from .metrics_export import get_metrics_collector

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()


class Region(Enum):
    """Supported regions for global deployment."""

    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    BRAZIL = "sa-east-1"
    INDIA = "ap-south-1"


class Language(Enum):
    """Supported languages for internationalization."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    KOREAN = "ko"
    RUSSIAN = "ru"


class ComplianceStandard(Enum):
    """Supported compliance standards."""

    GDPR = "gdpr"          # EU General Data Protection Regulation
    CCPA = "ccpa"          # California Consumer Privacy Act
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore)
    PIPEDA = "pipeda"      # Personal Information Protection and Electronic Documents Act (Canada)
    SOX = "sox"            # Sarbanes-Oxley Act
    HIPAA = "hipaa"        # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci-dss"   # Payment Card Industry Data Security Standard
    ISO_27001 = "iso-27001" # Information Security Management


@dataclass
class ComplianceCheck:
    """Represents a compliance validation result."""

    standard: ComplianceStandard
    compliant: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "standard": self.standard.value,
            "compliant": self.compliant,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
        }


@dataclass
class GlobalContext:
    """Global context for processing emails."""

    region: Region = Region.US_EAST
    language: Language = Language.ENGLISH
    timezone: str = "UTC"
    currency: str = "USD"
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    data_residency_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "region": self.region.value,
            "language": self.language.value,
            "timezone": self.timezone,
            "currency": self.currency,
            "compliance_standards": [s.value for s in self.compliance_standards],
            "data_residency_required": self.data_residency_required,
        }


class InternationalizationManager:
    """Manages internationalization and localization."""

    def __init__(self, default_language: Language = Language.ENGLISH):
        """Initialize internationalization manager."""
        self.default_language = default_language
        self.current_language = default_language
        self.translations = self._load_translations()
        self.date_formats = self._load_date_formats()
        self.currency_formats = self._load_currency_formats()

        logger.info(f"InternationalizationManager initialized with language: {default_language.value}")

    def set_language(self, language: Language):
        """Set the current language."""
        self.current_language = language
        _metrics_collector.increment_counter(f"i18n_language_set_{language.value}")
        logger.debug(f"Language set to: {language.value}")

    def translate(self, key: str, language: Optional[Language] = None, **kwargs) -> str:
        """Translate a key to the specified language."""

        target_language = language or self.current_language

        # Get translation from dictionary
        translation = self.translations.get(target_language, {}).get(key, key)

        # Handle parameter substitution
        try:
            if kwargs:
                translation = translation.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Translation parameter missing for key '{key}': {e}")

        _metrics_collector.increment_counter("i18n_translations")

        return translation

    def format_date(self, timestamp: float, language: Optional[Language] = None) -> str:
        """Format a timestamp according to language conventions."""

        target_language = language or self.current_language
        date_format = self.date_formats.get(target_language, "%Y-%m-%d %H:%M:%S")

        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)

        return dt.strftime(date_format)

    def format_currency(self, amount: float, currency: str = "USD", language: Optional[Language] = None) -> str:
        """Format currency according to language conventions."""

        target_language = language or self.current_language
        format_info = self.currency_formats.get(target_language, {})

        symbol = format_info.get("symbol", "$")
        decimals = format_info.get("decimals", 2)
        separator = format_info.get("separator", ",")
        decimal_point = format_info.get("decimal_point", ".")

        # Format amount with proper decimal places
        formatted_amount = f"{amount:.{decimals}f}"

        # Add thousands separator
        if separator:
            parts = formatted_amount.split(".")
            parts[0] = f"{int(parts[0]):,}".replace(",", separator)
            formatted_amount = decimal_point.join(parts)

        return f"{symbol}{formatted_amount}"

    def detect_language(self, text: str) -> Language:
        """Detect the language of the given text."""

        # Simple language detection based on common words
        language_indicators = {
            Language.ENGLISH: ["the", "and", "you", "that", "was", "for", "are"],
            Language.SPANISH: ["el", "la", "de", "que", "y", "en", "un"],
            Language.FRENCH: ["le", "de", "et", "à", "un", "il", "être"],
            Language.GERMAN: ["der", "die", "und", "in", "den", "von", "zu"],
            Language.PORTUGUESE: ["o", "de", "a", "e", "do", "da", "em"],
            Language.ITALIAN: ["il", "di", "a", "e", "che", "in", "con"],
        }

        text_lower = text.lower()
        language_scores = {}

        for language, indicators in language_indicators.items():
            score = 0
            for indicator in indicators:
                score += text_lower.count(f" {indicator} ")
            language_scores[language] = score

        # Return language with highest score, or default if no clear winner
        best_language = max(language_scores.items(), key=lambda x: x[1])

        if best_language[1] > 0:
            detected = best_language[0]
            _metrics_collector.increment_counter(f"i18n_language_detected_{detected.value}")
            return detected

        return self.default_language

    def _load_translations(self) -> Dict[Language, Dict[str, str]]:
        """Load translation dictionaries for all supported languages."""

        return {
            Language.ENGLISH: {
                "email_processed": "Email processed successfully",
                "high_priority": "High Priority",
                "urgent_message": "Urgent Message",
                "complaint_detected": "Complaint Detected",
                "security_threat": "Security Threat",
                "processing_failed": "Processing Failed",
                "thank_you": "Thank you for your message",
                "we_will_respond": "We will respond shortly",
                "urgent_attention": "This requires urgent attention",
                "escalated": "Escalated to senior team",
            },
            Language.SPANISH: {
                "email_processed": "Correo procesado exitosamente",
                "high_priority": "Alta Prioridad",
                "urgent_message": "Mensaje Urgente",
                "complaint_detected": "Queja Detectada",
                "security_threat": "Amenaza de Seguridad",
                "processing_failed": "Procesamiento Fallido",
                "thank_you": "Gracias por su mensaje",
                "we_will_respond": "Responderemos pronto",
                "urgent_attention": "Esto requiere atención urgente",
                "escalated": "Escalado al equipo senior",
            },
            Language.FRENCH: {
                "email_processed": "E-mail traité avec succès",
                "high_priority": "Haute Priorité",
                "urgent_message": "Message Urgent",
                "complaint_detected": "Plainte Détectée",
                "security_threat": "Menace de Sécurité",
                "processing_failed": "Échec du Traitement",
                "thank_you": "Merci pour votre message",
                "we_will_respond": "Nous répondrons bientôt",
                "urgent_attention": "Ceci nécessite une attention urgente",
                "escalated": "Escaladé à l'équipe senior",
            },
            Language.GERMAN: {
                "email_processed": "E-Mail erfolgreich verarbeitet",
                "high_priority": "Hohe Priorität",
                "urgent_message": "Dringende Nachricht",
                "complaint_detected": "Beschwerde Erkannt",
                "security_threat": "Sicherheitsbedrohung",
                "processing_failed": "Verarbeitung Fehlgeschlagen",
                "thank_you": "Vielen Dank für Ihre Nachricht",
                "we_will_respond": "Wir werden bald antworten",
                "urgent_attention": "Dies erfordert dringende Aufmerksamkeit",
                "escalated": "An das Senior-Team weitergeleitet",
            },
            Language.JAPANESE: {
                "email_processed": "メールの処理が正常に完了しました",
                "high_priority": "高優先度",
                "urgent_message": "緊急メッセージ",
                "complaint_detected": "苦情を検出",
                "security_threat": "セキュリティ脅威",
                "processing_failed": "処理失敗",
                "thank_you": "メッセージをありがとうございます",
                "we_will_respond": "すぐに返信いたします",
                "urgent_attention": "これは緊急の注意が必要です",
                "escalated": "上級チームにエスカレート",
            },
        }

    def _load_date_formats(self) -> Dict[Language, str]:
        """Load date format patterns for different languages."""

        return {
            Language.ENGLISH: "%B %d, %Y at %I:%M %p",
            Language.SPANISH: "%d de %B de %Y a las %H:%M",
            Language.FRENCH: "%d %B %Y à %H:%M",
            Language.GERMAN: "%d. %B %Y um %H:%M",
            Language.JAPANESE: "%Y年%m月%d日 %H:%M",
            Language.CHINESE_SIMPLIFIED: "%Y年%m月%d日 %H:%M",
            Language.PORTUGUESE: "%d de %B de %Y às %H:%M",
            Language.ITALIAN: "%d %B %Y alle %H:%M",
        }

    def _load_currency_formats(self) -> Dict[Language, Dict[str, Any]]:
        """Load currency formatting rules for different languages."""

        return {
            Language.ENGLISH: {"symbol": "$", "decimals": 2, "separator": ",", "decimal_point": "."},
            Language.SPANISH: {"symbol": "€", "decimals": 2, "separator": ".", "decimal_point": ","},
            Language.FRENCH: {"symbol": "€", "decimals": 2, "separator": " ", "decimal_point": ","},
            Language.GERMAN: {"symbol": "€", "decimals": 2, "separator": ".", "decimal_point": ","},
            Language.JAPANESE: {"symbol": "¥", "decimals": 0, "separator": ",", "decimal_point": "."},
            Language.CHINESE_SIMPLIFIED: {"symbol": "¥", "decimals": 2, "separator": ",", "decimal_point": "."},
            Language.PORTUGUESE: {"symbol": "R$", "decimals": 2, "separator": ".", "decimal_point": ","},
            Language.ITALIAN: {"symbol": "€", "decimals": 2, "separator": ".", "decimal_point": ","},
        }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in self.translations.keys()]


class ComplianceManager:
    """Manages compliance with international regulations."""

    def __init__(self):
        """Initialize compliance manager."""
        self.compliance_rules = self._load_compliance_rules()
        self.sensitive_data_patterns = self._load_sensitive_data_patterns()

        logger.info("ComplianceManager initialized")

    def validate_compliance(
        self,
        content: str,
        standards: List[ComplianceStandard],
        context: GlobalContext
    ) -> List[ComplianceCheck]:
        """Validate content against compliance standards."""

        results = []

        for standard in standards:
            check = self._validate_single_standard(content, standard, context)
            results.append(check)

            # Update metrics
            status = "compliant" if check.compliant else "non_compliant"
            _metrics_collector.increment_counter(f"compliance_{standard.value}_{status}")

        return results

    def _validate_single_standard(
        self,
        content: str,
        standard: ComplianceStandard,
        context: GlobalContext
    ) -> ComplianceCheck:
        """Validate content against a single compliance standard."""

        rules = self.compliance_rules.get(standard, {})
        check = ComplianceCheck(standard=standard, compliant=True)

        # Check for sensitive data
        sensitive_matches = self._detect_sensitive_data(content, standard)
        if sensitive_matches:
            check.compliant = False
            check.issues.extend([f"Sensitive data detected: {match}" for match in sensitive_matches])

        # GDPR specific checks
        if standard == ComplianceStandard.GDPR:
            check = self._validate_gdpr(content, context, check)

        # CCPA specific checks
        elif standard == ComplianceStandard.CCPA:
            check = self._validate_ccpa(content, context, check)

        # HIPAA specific checks
        elif standard == ComplianceStandard.HIPAA:
            check = self._validate_hipaa(content, context, check)

        # PCI-DSS specific checks
        elif standard == ComplianceStandard.PCI_DSS:
            check = self._validate_pci_dss(content, context, check)

        # Set confidence based on number of checks performed
        check.confidence = 0.8 if check.issues else 0.9

        logger.debug(f"Compliance check for {standard.value}: {'PASS' if check.compliant else 'FAIL'}")

        return check

    def _detect_sensitive_data(self, content: str, standard: ComplianceStandard) -> List[str]:
        """Detect sensitive data patterns in content."""

        patterns = self.sensitive_data_patterns.get(standard, {})
        matches = []

        for data_type, pattern in patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(data_type)

        return matches

    def _validate_gdpr(self, content: str, context: GlobalContext, check: ComplianceCheck) -> ComplianceCheck:
        """Validate GDPR compliance."""

        # Check for EU region processing
        eu_regions = [Region.EU_WEST, Region.EU_CENTRAL]
        if context.region not in eu_regions and context.data_residency_required:
            check.compliant = False
            check.issues.append("Data processed outside EU without proper safeguards")
            check.recommendations.append("Ensure adequate data protection measures for cross-border transfers")

        # Check for consent language
        consent_keywords = ["consent", "agree", "opt-in", "permission"]
        has_consent_language = any(keyword in content.lower() for keyword in consent_keywords)

        if not has_consent_language and "personal" in content.lower():
            check.recommendations.append("Consider adding explicit consent language for personal data processing")

        # Check for data subject rights mention
        rights_keywords = ["right to access", "right to deletion", "data portability", "withdraw consent"]
        has_rights_mention = any(keyword in content.lower() for keyword in rights_keywords)

        if not has_rights_mention:
            check.recommendations.append("Consider informing data subjects of their rights under GDPR")

        return check

    def _validate_ccpa(self, content: str, context: GlobalContext, check: ComplianceCheck) -> ComplianceCheck:
        """Validate CCPA compliance."""

        # Check for California resident data
        if "california" in content.lower() or "ca resident" in content.lower():

            # Check for opt-out language
            opt_out_keywords = ["do not sell", "opt out", "privacy choices"]
            has_opt_out = any(keyword in content.lower() for keyword in opt_out_keywords)

            if not has_opt_out:
                check.recommendations.append("Consider adding opt-out options for California residents")

            # Check for data categories disclosure
            disclosure_keywords = ["categories of information", "sources of information", "business purposes"]
            has_disclosure = any(keyword in content.lower() for keyword in disclosure_keywords)

            if not has_disclosure:
                check.recommendations.append("Consider disclosing categories of personal information collected")

        return check

    def _validate_hipaa(self, content: str, context: GlobalContext, check: ComplianceCheck) -> ComplianceCheck:
        """Validate HIPAA compliance."""

        # Check for protected health information (PHI)
        phi_indicators = ["medical", "health", "diagnosis", "treatment", "patient", "doctor", "hospital"]
        has_phi = any(indicator in content.lower() for indicator in phi_indicators)

        if has_phi:
            # Check for proper safeguards language
            safeguards_keywords = ["encrypted", "secure", "confidential", "hipaa compliant"]
            has_safeguards = any(keyword in content.lower() for keyword in safeguards_keywords)

            if not has_safeguards:
                check.compliant = False
                check.issues.append("PHI detected without proper safeguards language")
                check.recommendations.append("Ensure PHI is properly protected with encryption and access controls")

        return check

    def _validate_pci_dss(self, content: str, context: GlobalContext, check: ComplianceCheck) -> ComplianceCheck:
        """Validate PCI-DSS compliance."""

        # Check for credit card data
        card_pattern = r'\b(?:\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b'
        if re.search(card_pattern, content):
            check.compliant = False
            check.issues.append("Credit card numbers detected in plaintext")
            check.recommendations.append("Never store or transmit credit card data in plaintext")

        # Check for CVV codes
        cvv_pattern = r'\b(?:cvv|cvc|security code)[\s:]*\d{3,4}\b'
        if re.search(cvv_pattern, content, re.IGNORECASE):
            check.compliant = False
            check.issues.append("CVV/CVC codes detected")
            check.recommendations.append("CVV/CVC codes must never be stored after authorization")

        return check

    def _load_compliance_rules(self) -> Dict[ComplianceStandard, Dict]:
        """Load compliance rules configuration."""

        return {
            ComplianceStandard.GDPR: {
                "requires_consent": True,
                "data_residency": True,
                "right_to_deletion": True,
                "breach_notification": 72,  # hours
            },
            ComplianceStandard.CCPA: {
                "opt_out_required": True,
                "data_disclosure": True,
                "sale_notification": True,
            },
            ComplianceStandard.HIPAA: {
                "phi_encryption": True,
                "access_controls": True,
                "audit_logs": True,
            },
            ComplianceStandard.PCI_DSS: {
                "no_card_storage": True,
                "encryption_required": True,
                "network_security": True,
            },
        }

    def _load_sensitive_data_patterns(self) -> Dict[ComplianceStandard, Dict[str, str]]:
        """Load sensitive data detection patterns."""

        return {
            ComplianceStandard.GDPR: {
                "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                "name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            },
            ComplianceStandard.CCPA: {
                "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
                "driver_license": r'\b[A-Z]\d{7,8}\b',
                "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            },
            ComplianceStandard.HIPAA: {
                "medical_record": r'\bMRN[\s:]?\d+\b',
                "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
                "insurance": r'\b(?:policy|member)\s*(?:id|number)[\s:]\w+\b',
            },
            ComplianceStandard.PCI_DSS: {
                "credit_card": r'\b(?:\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})\b',
                "cvv": r'\b(?:cvv|cvc)[\s:]*\d{3,4}\b',
                "account_number": r'\b(?:account|acct)[\s#:]*\d{8,}\b',
            },
        }

    def get_compliance_report(self, checks: List[ComplianceCheck]) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""

        total_checks = len(checks)
        compliant_checks = len([c for c in checks if c.compliant])

        report = {
            "overall_compliance": compliant_checks == total_checks,
            "compliance_score": compliant_checks / total_checks if total_checks > 0 else 0.0,
            "total_checks": total_checks,
            "compliant_checks": compliant_checks,
            "non_compliant_checks": total_checks - compliant_checks,
            "standards": [check.to_dict() for check in checks],
            "summary": {
                "high_risk_issues": [],
                "recommendations": [],
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }

        # Collect high-risk issues and recommendations
        for check in checks:
            if not check.compliant:
                report["summary"]["high_risk_issues"].extend(check.issues)
            report["summary"]["recommendations"].extend(check.recommendations)

        # Remove duplicates
        report["summary"]["high_risk_issues"] = list(set(report["summary"]["high_risk_issues"]))
        report["summary"]["recommendations"] = list(set(report["summary"]["recommendations"]))

        return report


class GlobalEmailProcessor:
    """Global email processor with i18n and compliance features."""

    def __init__(self, context: GlobalContext):
        """Initialize global email processor."""
        self.context = context
        self.i18n = InternationalizationManager(context.language)
        self.compliance = ComplianceManager()

        logger.info(f"GlobalEmailProcessor initialized for region {context.region.value}")

    def process_email_global(self, content: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Process email with global features."""

        start_time = time.perf_counter()

        # Detect language if not specified
        detected_language = self.i18n.detect_language(content)
        if detected_language != self.context.language:
            self.i18n.set_language(detected_language)
            logger.info(f"Language auto-detected and switched to: {detected_language.value}")

        # Perform compliance validation
        compliance_checks = []
        if self.context.compliance_standards:
            compliance_checks = self.compliance.validate_compliance(
                content, self.context.compliance_standards, self.context
            )

        # Process with standard triage (import here to avoid circular imports)
        import asyncio

        from .ai_enhancements import intelligent_triage_email

        # Run async function in sync context
        triage_result = asyncio.run(intelligent_triage_email(content, headers))

        # Localize the response
        localized_response = self._localize_response(triage_result, detected_language)

        # Generate compliance report
        compliance_report = None
        if compliance_checks:
            compliance_report = self.compliance.get_compliance_report(compliance_checks)

        # Create global result
        global_result = {
            "triage": localized_response.to_dict(),
            "global_context": self.context.to_dict(),
            "detected_language": detected_language.value,
            "compliance": compliance_report,
            "processing_region": self.context.region.value,
            "processing_time_ms": (time.perf_counter() - start_time) * 1000,
            "timestamp": self.i18n.format_date(time.time(), detected_language),
        }

        # Update metrics
        _metrics_collector.increment_counter(f"global_processing_{self.context.region.value}")
        _metrics_collector.increment_counter(f"language_processing_{detected_language.value}")

        if compliance_report and not compliance_report["overall_compliance"]:
            _metrics_collector.increment_counter("compliance_violations")

        return global_result

    def _localize_response(self, triage_result, language: Language):
        """Localize triage response based on detected language."""

        # Localize category
        category_key = f"category_{triage_result.category.lower()}"
        localized_category = self.i18n.translate(category_key, language)
        if localized_category == category_key:  # No translation found
            localized_category = triage_result.category

        # Localize priority description
        if triage_result.priority >= 8:
            priority_desc = self.i18n.translate("high_priority", language)
        elif triage_result.priority >= 5:
            priority_desc = self.i18n.translate("medium_priority", language)
        else:
            priority_desc = self.i18n.translate("low_priority", language)

        # Enhance response with localized elements
        base_response = triage_result.response

        # Add localized greeting/closing based on language and priority
        if triage_result.priority >= 7:
            urgency_note = self.i18n.translate("urgent_attention", language)
            base_response = f"{urgency_note}\n\n{base_response}"

        # Add thank you in appropriate language
        thank_you = self.i18n.translate("thank_you", language)
        response_promise = self.i18n.translate("we_will_respond", language)

        enhanced_response = f"{base_response}\n\n{thank_you}. {response_promise}."

        # Update the result with localized content
        triage_result.category = localized_category
        triage_result.response = enhanced_response

        # Add priority description to summary
        original_summary = triage_result.summary
        triage_result.summary = f"[{priority_desc}] {original_summary}"

        return triage_result

    def get_region_status(self) -> Dict[str, Any]:
        """Get status of current region."""

        return {
            "region": self.context.region.value,
            "language": self.context.language.value,
            "timezone": self.context.timezone,
            "compliance_standards": [s.value for s in self.context.compliance_standards],
            "data_residency_required": self.context.data_residency_required,
            "supported_languages": self.i18n.get_supported_languages(),
            "i18n_active": True,
            "compliance_active": len(self.context.compliance_standards) > 0,
        }


# Global instances for different regions
def create_regional_processor(region: Region, compliance_standards: List[ComplianceStandard] = None) -> GlobalEmailProcessor:
    """Create a regional email processor with appropriate settings."""

    # Regional defaults
    regional_settings = {
        Region.US_EAST: {
            "language": Language.ENGLISH,
            "timezone": "America/New_York",
            "currency": "USD",
            "compliance": [ComplianceStandard.CCPA, ComplianceStandard.SOX],
        },
        Region.EU_WEST: {
            "language": Language.ENGLISH,
            "timezone": "Europe/London",
            "currency": "EUR",
            "compliance": [ComplianceStandard.GDPR, ComplianceStandard.ISO_27001],
        },
        Region.EU_CENTRAL: {
            "language": Language.GERMAN,
            "timezone": "Europe/Berlin",
            "currency": "EUR",
            "compliance": [ComplianceStandard.GDPR, ComplianceStandard.ISO_27001],
        },
        Region.ASIA_PACIFIC: {
            "language": Language.ENGLISH,
            "timezone": "Asia/Singapore",
            "currency": "SGD",
            "compliance": [ComplianceStandard.PDPA],
        },
        Region.ASIA_NORTHEAST: {
            "language": Language.JAPANESE,
            "timezone": "Asia/Tokyo",
            "currency": "JPY",
            "compliance": [],
        },
    }

    settings = regional_settings.get(region, {
        "language": Language.ENGLISH,
        "timezone": "UTC",
        "currency": "USD",
        "compliance": [],
    })

    context = GlobalContext(
        region=region,
        language=settings["language"],
        timezone=settings["timezone"],
        currency=settings["currency"],
        compliance_standards=compliance_standards or settings["compliance"],
        data_residency_required=region in [Region.EU_WEST, Region.EU_CENTRAL],
    )

    return GlobalEmailProcessor(context)


# Convenience functions
def get_eu_processor() -> GlobalEmailProcessor:
    """Get EU-compliant processor with GDPR."""
    return create_regional_processor(Region.EU_WEST, [ComplianceStandard.GDPR])

def get_us_processor() -> GlobalEmailProcessor:
    """Get US processor with CCPA compliance."""
    return create_regional_processor(Region.US_EAST, [ComplianceStandard.CCPA])

def get_asia_processor() -> GlobalEmailProcessor:
    """Get Asia-Pacific processor with PDPA compliance."""
    return create_regional_processor(Region.ASIA_PACIFIC, [ComplianceStandard.PDPA])

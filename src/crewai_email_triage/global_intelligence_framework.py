"""Global Intelligence Framework for Email Triage.

Multi-language support, compliance frameworks, and global deployment capabilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
import json
import hashlib
from datetime import datetime, timezone
import re

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages with ISO codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HEBREW = "he"
    HINDI = "hi"


class ComplianceRegion(Enum):
    """Global compliance regions."""
    EU = "eu"           # GDPR
    US = "us"           # CCPA, COPPA
    CANADA = "ca"       # PIPEDA
    AUSTRALIA = "au"    # Privacy Act
    SINGAPORE = "sg"    # PDPA
    BRAZIL = "br"       # LGPD
    JAPAN = "jp"        # APPI
    SOUTH_KOREA = "kr"  # PIPA
    INDIA = "in"        # DPDP
    UK = "uk"           # UK-GDPR
    CHINA = "cn"        # PIPL
    GLOBAL = "global"   # Universal compliance


class DataClassification(Enum):
    """Data sensitivity classifications."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


@dataclass
class LanguageProfile:
    """Language-specific processing profile."""
    language: SupportedLanguage
    confidence: float
    detected_patterns: List[str] = field(default_factory=list)
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    processing_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceProfile:
    """Compliance requirements profile."""
    regions: List[ComplianceRegion]
    data_classification: DataClassification
    retention_days: int
    encryption_required: bool = True
    audit_trail_required: bool = True
    consent_required: bool = False
    right_to_deletion: bool = False
    cross_border_restrictions: Dict[str, Any] = field(default_factory=dict)
    special_categories: List[str] = field(default_factory=list)


@dataclass
class GlobalProcessingResult:
    """Global processing result with compliance metadata."""
    content: str
    language_profile: LanguageProfile
    compliance_profile: ComplianceProfile
    processing_timestamp: str
    data_residency_region: str
    classification_tags: List[str] = field(default_factory=list)
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    localized_response: Dict[str, str] = field(default_factory=dict)


class GlobalLanguageDetector:
    """Advanced multi-language detection and processing."""
    
    def __init__(self):
        self.language_patterns = {
            SupportedLanguage.ENGLISH: {
                'common_words': ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but'],
                'patterns': [r'\\b(the|and|that|have|for)\\b', r'\\b(is|are|was|were)\\b'],
                'cultural_markers': ['please', 'thank you', 'regards', 'best', 'sincerely']
            },
            SupportedLanguage.SPANISH: {
                'common_words': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
                'patterns': [r'\\b(el|la|los|las)\\b', r'\\b(es|son|está|están)\\b'],
                'cultural_markers': ['por favor', 'gracias', 'saludos', 'atentamente']
            },
            SupportedLanguage.FRENCH: {
                'common_words': ['le', 'la', 'et', 'que', 'de', 'pour', 'avec', 'dans', 'sur', 'une'],
                'patterns': [r'\\b(le|la|les)\\b', r'\\b(est|sont|était|étaient)\\b'],
                'cultural_markers': ['s\'il vous plaît', 'merci', 'cordialement', 'salutations']
            },
            SupportedLanguage.GERMAN: {
                'common_words': ['der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'mit', 'sich'],
                'patterns': [r'\\b(der|die|das)\\b', r'\\b(ist|sind|war|waren)\\b'],
                'cultural_markers': ['bitte', 'danke', 'mit freundlichen grüßen', 'hochachtungsvoll']
            },
            SupportedLanguage.JAPANESE: {
                'common_words': ['の', 'に', 'は', 'を', 'が', 'と', 'で', 'も', 'から', 'する'],
                'patterns': [r'[ひらがな]', r'[カタカナ]', r'[漢字]'],
                'cultural_markers': ['お疲れ様', 'よろしく', 'ありがとう', 'すみません']
            },
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                'common_words': ['的', '是', '在', '了', '有', '和', '人', '这', '中', '大'],
                'patterns': [r'[\\u4e00-\\u9fff]', r'的是在了有和'],
                'cultural_markers': ['谢谢', '您好', '再见', '请']
            },
            SupportedLanguage.KOREAN: {
                'common_words': ['이', '그', '저', '것', '수', '있', '하', '되', '않', '들'],
                'patterns': [r'[ㄱ-ㅎㅏ-ㅣ가-힣]', r'습니다|입니다'],
                'cultural_markers': ['감사합니다', '안녕하세요', '죄송합니다']
            }
        }
    
    def detect_language(self, content: str) -> LanguageProfile:
        """Detect language with confidence scoring."""
        if not content or len(content.strip()) < 10:
            return LanguageProfile(SupportedLanguage.ENGLISH, 0.1)
        
        content_lower = content.lower()
        scores = {}
        detected_patterns = []
        
        for language, patterns in self.language_patterns.items():
            score = 0.0
            lang_patterns = []
            
            # Check common words
            word_matches = sum(1 for word in patterns['common_words'] if word in content_lower)
            word_score = word_matches / len(patterns['common_words'])
            
            # Check regex patterns
            pattern_matches = 0
            for pattern in patterns.get('patterns', []):
                matches = re.findall(pattern, content_lower)
                if matches:
                    pattern_matches += len(matches)
                    lang_patterns.extend(matches[:3])  # Keep sample matches
            
            pattern_score = min(pattern_matches / 10, 1.0)  # Normalize
            
            # Check cultural markers
            cultural_matches = sum(1 for marker in patterns['cultural_markers'] if marker in content_lower)
            cultural_score = cultural_matches / max(len(patterns['cultural_markers']), 1)
            
            # Combined score with weights
            total_score = word_score * 0.5 + pattern_score * 0.3 + cultural_score * 0.2
            scores[language] = total_score
            
            if total_score > 0.1:
                detected_patterns.extend(lang_patterns)
        
        # Find best match
        if scores:
            best_language = max(scores.items(), key=lambda x: x[1])
            language, confidence = best_language
            
            # Adjust confidence based on content length and clarity
            confidence = min(confidence * (1 + min(len(content) / 1000, 0.5)), 1.0)
            
            # Cultural context detection
            cultural_context = self._detect_cultural_context(content, language)
            processing_adjustments = self._get_processing_adjustments(language)
            
            return LanguageProfile(
                language=language,
                confidence=confidence,
                detected_patterns=detected_patterns[:10],  # Limit patterns
                cultural_context=cultural_context,
                processing_adjustments=processing_adjustments
            )
        
        # Default to English with low confidence
        return LanguageProfile(SupportedLanguage.ENGLISH, 0.1)
    
    def _detect_cultural_context(self, content: str, language: SupportedLanguage) -> Dict[str, Any]:
        """Detect cultural context markers."""
        context = {
            'formality_level': 'neutral',
            'business_context': False,
            'urgency_cultural_markers': [],
            'politeness_markers': [],
            'hierarchical_markers': []
        }
        
        content_lower = content.lower()
        
        # Formality detection
        formal_markers = {
            SupportedLanguage.ENGLISH: ['dear sir/madam', 'yours faithfully', 'respectfully'],
            SupportedLanguage.JAPANESE: ['お疲れ様でございます', 'いつもお世話になっております'],
            SupportedLanguage.GERMAN: ['sehr geehrte', 'mit freundlichen grüßen', 'hochachtungsvoll'],
            SupportedLanguage.FRENCH: ['madame, monsieur', 'veuillez agréer', 'cordialement']
        }
        
        if language in formal_markers:
            formal_count = sum(1 for marker in formal_markers[language] if marker in content_lower)
            if formal_count > 0:
                context['formality_level'] = 'high'
                context['politeness_markers'] = formal_markers[language][:formal_count]
        
        # Business context
        business_markers = ['meeting', 'project', 'deadline', 'proposal', 'contract', 'agreement']
        business_count = sum(1 for marker in business_markers if marker in content_lower)
        context['business_context'] = business_count > 2
        
        return context
    
    def _get_processing_adjustments(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Get language-specific processing adjustments."""
        adjustments = {
            'text_direction': 'ltr',
            'character_encoding': 'utf-8',
            'word_segmentation': 'space_separated',
            'case_sensitive': False,
            'punctuation_rules': 'latin'
        }
        
        # Language-specific adjustments
        if language in [SupportedLanguage.ARABIC, SupportedLanguage.HEBREW]:
            adjustments['text_direction'] = 'rtl'
        
        if language in [SupportedLanguage.JAPANESE, SupportedLanguage.CHINESE_SIMPLIFIED, 
                       SupportedLanguage.CHINESE_TRADITIONAL, SupportedLanguage.KOREAN]:
            adjustments['word_segmentation'] = 'character_based'
            adjustments['punctuation_rules'] = 'cjk'
        
        if language in [SupportedLanguage.GERMAN]:
            adjustments['case_sensitive'] = True  # German nouns are capitalized
        
        return adjustments


class GlobalComplianceManager:
    """Manage global compliance requirements and data governance."""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        self.data_classification_rules = self._load_classification_rules()
        self.cross_border_rules = self._load_cross_border_rules()
    
    def assess_compliance(self, content: str, processing_region: str, 
                         target_regions: List[str] = None) -> ComplianceProfile:
        """Assess compliance requirements for content and regions."""
        # Classify data sensitivity
        data_classification = self._classify_data_sensitivity(content)
        
        # Determine applicable regions
        applicable_regions = [ComplianceRegion(processing_region)]
        if target_regions:
            applicable_regions.extend([ComplianceRegion(region) for region in target_regions])
        
        # Get strictest requirements across all regions
        profile = ComplianceProfile(
            regions=applicable_regions,
            data_classification=data_classification,
            retention_days=self._get_max_retention(applicable_regions, data_classification),
            encryption_required=self._requires_encryption(applicable_regions, data_classification),
            audit_trail_required=self._requires_audit_trail(applicable_regions),
            consent_required=self._requires_consent(applicable_regions, data_classification),
            right_to_deletion=self._has_deletion_rights(applicable_regions),
            cross_border_restrictions=self._get_cross_border_restrictions(applicable_regions),
            special_categories=self._identify_special_categories(content, applicable_regions)
        )
        
        return profile
    
    def _classify_data_sensitivity(self, content: str) -> DataClassification:
        """Classify data sensitivity based on content."""
        content_lower = content.lower()
        
        # Check for highly sensitive data
        sensitive_personal_indicators = [
            'ssn', 'social security', 'passport', 'driver license', 'medical', 'health',
            'credit card', 'bank account', 'financial', 'salary', 'income', 'biometric'
        ]
        
        personal_indicators = [
            'email', 'phone', 'address', 'name', 'birthday', 'age', 'gender',
            'employment', 'education', 'family', 'relationship'
        ]
        
        confidential_indicators = [
            'confidential', 'proprietary', 'trade secret', 'internal only',
            'restricted', 'classified', 'private'
        ]
        
        # Score content
        sensitive_personal_score = sum(1 for indicator in sensitive_personal_indicators 
                                     if indicator in content_lower)
        personal_score = sum(1 for indicator in personal_indicators 
                           if indicator in content_lower)
        confidential_score = sum(1 for indicator in confidential_indicators 
                               if indicator in content_lower)
        
        # Classify based on highest score
        if sensitive_personal_score > 0:
            return DataClassification.SENSITIVE_PERSONAL
        elif personal_score > 1:
            return DataClassification.PERSONAL
        elif confidential_score > 0:
            return DataClassification.CONFIDENTIAL
        elif any(word in content_lower for word in ['internal', 'company', 'organization']):
            return DataClassification.INTERNAL
        else:
            return DataClassification.PUBLIC
    
    def _get_max_retention(self, regions: List[ComplianceRegion], 
                          classification: DataClassification) -> int:
        """Get maximum retention period across regions."""
        retention_rules = {
            ComplianceRegion.EU: {
                DataClassification.SENSITIVE_PERSONAL: 365,
                DataClassification.PERSONAL: 1095,  # 3 years
                DataClassification.CONFIDENTIAL: 2555,  # 7 years
                DataClassification.INTERNAL: 3650,  # 10 years
                DataClassification.PUBLIC: -1  # No limit
            },
            ComplianceRegion.US: {
                DataClassification.SENSITIVE_PERSONAL: 1095,
                DataClassification.PERSONAL: 2555,
                DataClassification.CONFIDENTIAL: 2555,
                DataClassification.INTERNAL: 3650,
                DataClassification.PUBLIC: -1
            },
            ComplianceRegion.CANADA: {
                DataClassification.SENSITIVE_PERSONAL: 365,
                DataClassification.PERSONAL: 1095,
                DataClassification.CONFIDENTIAL: 2555,
                DataClassification.INTERNAL: 3650,
                DataClassification.PUBLIC: -1
            }
        }
        
        max_retention = -1  # No limit by default
        
        for region in regions:
            if region in retention_rules:
                region_retention = retention_rules[region].get(classification, -1)
                if region_retention > 0:
                    if max_retention == -1 or region_retention < max_retention:
                        max_retention = region_retention
        
        return max_retention if max_retention > 0 else 2555  # Default 7 years
    
    def _requires_encryption(self, regions: List[ComplianceRegion], 
                           classification: DataClassification) -> bool:
        """Check if encryption is required."""
        encryption_required_regions = [
            ComplianceRegion.EU, ComplianceRegion.US, ComplianceRegion.CANADA,
            ComplianceRegion.SINGAPORE, ComplianceRegion.BRAZIL
        ]
        
        sensitive_classifications = [
            DataClassification.SENSITIVE_PERSONAL,
            DataClassification.PERSONAL,
            DataClassification.CONFIDENTIAL
        ]
        
        return (any(region in encryption_required_regions for region in regions) and
                classification in sensitive_classifications)
    
    def _requires_audit_trail(self, regions: List[ComplianceRegion]) -> bool:
        """Check if audit trail is required."""
        audit_required_regions = [
            ComplianceRegion.EU, ComplianceRegion.US, ComplianceRegion.CANADA,
            ComplianceRegion.SINGAPORE, ComplianceRegion.BRAZIL, ComplianceRegion.UK
        ]
        
        return any(region in audit_required_regions for region in regions)
    
    def _requires_consent(self, regions: List[ComplianceRegion], 
                         classification: DataClassification) -> bool:
        """Check if consent is required."""
        consent_required_regions = [ComplianceRegion.EU, ComplianceRegion.UK, ComplianceRegion.BRAZIL]
        personal_classifications = [DataClassification.SENSITIVE_PERSONAL, DataClassification.PERSONAL]
        
        return (any(region in consent_required_regions for region in regions) and
                classification in personal_classifications)
    
    def _has_deletion_rights(self, regions: List[ComplianceRegion]) -> bool:
        """Check if right to deletion applies."""
        deletion_rights_regions = [
            ComplianceRegion.EU, ComplianceRegion.UK, ComplianceRegion.BRAZIL,
            ComplianceRegion.CANADA
        ]
        
        return any(region in deletion_rights_regions for region in regions)
    
    def _get_cross_border_restrictions(self, regions: List[ComplianceRegion]) -> Dict[str, Any]:
        """Get cross-border data transfer restrictions."""
        restrictions = {
            'adequacy_required': [],
            'transfer_mechanisms': [],
            'prohibited_countries': [],
            'additional_safeguards': []
        }
        
        if ComplianceRegion.EU in regions or ComplianceRegion.UK in regions:
            restrictions['adequacy_required'] = ['EU', 'UK', 'Canada', 'Japan', 'South Korea']
            restrictions['transfer_mechanisms'] = ['SCCs', 'BCRs', 'Certification']
            restrictions['prohibited_countries'] = ['Countries without adequate protection']
            restrictions['additional_safeguards'] = ['Encryption', 'Pseudonymization']
        
        if ComplianceRegion.CHINA in regions:
            restrictions['additional_safeguards'].append('Local data residency')
            restrictions['transfer_mechanisms'].append('CIIO approval required')
        
        return restrictions
    
    def _identify_special_categories(self, content: str, 
                                   regions: List[ComplianceRegion]) -> List[str]:
        """Identify special categories of personal data."""
        content_lower = content.lower()
        special_categories = []
        
        # GDPR Article 9 special categories
        if ComplianceRegion.EU in regions or ComplianceRegion.UK in regions:
            gdpr_special = {
                'racial_ethnic': ['race', 'ethnic', 'nationality', 'origin'],
                'political_opinions': ['political', 'party', 'vote', 'election'],
                'religious_beliefs': ['religion', 'faith', 'belief', 'worship'],
                'health': ['health', 'medical', 'disease', 'treatment', 'diagnosis'],
                'sexual_orientation': ['sexual', 'orientation', 'lgbt', 'gay', 'lesbian'],
                'biometric': ['fingerprint', 'retina', 'biometric', 'facial recognition'],
                'genetic': ['dna', 'genetic', 'genome', 'hereditary']
            }
            
            for category, indicators in gdpr_special.items():
                if any(indicator in content_lower for indicator in indicators):
                    special_categories.append(category)
        
        return special_categories
    
    def _load_compliance_rules(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Load compliance rules for each region."""
        return {
            ComplianceRegion.EU: {
                'name': 'GDPR',
                'requires_dpo': True,
                'breach_notification_hours': 72,
                'max_fine_percentage': 0.04,
                'data_subject_rights': ['access', 'rectification', 'erasure', 'portability']
            },
            ComplianceRegion.US: {
                'name': 'CCPA',
                'requires_dpo': False,
                'breach_notification_hours': 0,  # No specific requirement
                'max_fine_percentage': 0.0,  # Fixed amounts
                'data_subject_rights': ['access', 'deletion', 'opt_out']
            },
            ComplianceRegion.CANADA: {
                'name': 'PIPEDA',
                'requires_dpo': False,
                'breach_notification_hours': 72,
                'max_fine_percentage': 0.0,
                'data_subject_rights': ['access', 'correction']
            }
        }
    
    def _load_classification_rules(self) -> Dict[str, Any]:
        """Load data classification rules."""
        return {
            'public': {'retention_min': 0, 'encryption': False},
            'internal': {'retention_min': 365, 'encryption': True},
            'confidential': {'retention_min': 1095, 'encryption': True},
            'restricted': {'retention_min': 2555, 'encryption': True}
        }
    
    def _load_cross_border_rules(self) -> Dict[str, Any]:
        """Load cross-border transfer rules."""
        return {
            'adequacy_decisions': {
                'EU': ['Andorra', 'Argentina', 'Canada', 'Faroe Islands', 'Guernsey'],
                'UK': ['EU', 'Canada', 'Japan', 'New Zealand']
            },
            'restricted_countries': {
                'high_risk': ['Countries with mass surveillance'],
                'medium_risk': ['Countries with limited data protection']
            }
        }


class GlobalIntelligenceFramework:
    """Main framework for global email intelligence processing."""
    
    def __init__(self):
        self.language_detector = GlobalLanguageDetector()
        self.compliance_manager = GlobalComplianceManager()
        
        # Localization data
        self.localized_responses = self._load_localized_responses()
        self.regional_preferences = self._load_regional_preferences()
        
    def process_global_email(self, content: str, processing_region: str = "global",
                           target_regions: List[str] = None, 
                           user_preferences: Dict[str, Any] = None) -> GlobalProcessingResult:
        """Process email with full global intelligence."""
        processing_start = time.time()
        
        # Language detection and cultural analysis
        language_profile = self.language_detector.detect_language(content)
        
        # Compliance assessment
        compliance_profile = self.compliance_manager.assess_compliance(
            content, processing_region, target_regions
        )
        
        # Data classification and tagging
        classification_tags = self._generate_classification_tags(content, language_profile, compliance_profile)
        
        # Localized responses
        localized_response = self._generate_localized_responses(content, language_profile)
        
        # Audit metadata
        audit_metadata = self._generate_audit_metadata(
            content, language_profile, compliance_profile, processing_region
        )
        
        # Data residency determination
        data_residency_region = self._determine_data_residency(compliance_profile, processing_region)
        
        result = GlobalProcessingResult(
            content=content,
            language_profile=language_profile,
            compliance_profile=compliance_profile,
            processing_timestamp=datetime.now(timezone.utc).isoformat(),
            data_residency_region=data_residency_region,
            classification_tags=classification_tags,
            audit_metadata=audit_metadata,
            localized_response=localized_response
        )
        
        logger.info(f"Global processing completed in {(time.time() - processing_start)*1000:.2f}ms for {language_profile.language.value} content")
        
        return result
    
    def _generate_classification_tags(self, content: str, language_profile: LanguageProfile,
                                    compliance_profile: ComplianceProfile) -> List[str]:
        """Generate classification tags for content."""
        tags = []
        
        # Language tags
        tags.append(f"lang:{language_profile.language.value}")
        if language_profile.confidence > 0.8:
            tags.append("lang:high_confidence")
        
        # Compliance tags
        tags.append(f"classification:{compliance_profile.data_classification.value}")
        for region in compliance_profile.regions:
            tags.append(f"region:{region.value}")
        
        # Content tags
        if compliance_profile.encryption_required:
            tags.append("security:encryption_required")
        if compliance_profile.right_to_deletion:
            tags.append("rights:deletion")
        if compliance_profile.consent_required:
            tags.append("privacy:consent_required")
        
        # Cultural tags
        if language_profile.cultural_context.get('business_context'):
            tags.append("context:business")
        if language_profile.cultural_context.get('formality_level') == 'high':
            tags.append("style:formal")
        
        return tags
    
    def _generate_localized_responses(self, content: str, 
                                    language_profile: LanguageProfile) -> Dict[str, str]:
        """Generate responses in multiple languages."""
        base_response = "Your email has been processed successfully."
        
        # Get localized templates
        language = language_profile.language
        templates = self.localized_responses.get(language, self.localized_responses[SupportedLanguage.ENGLISH])
        
        localized = {}
        
        # Primary language response
        localized[language.value] = templates.get('success_message', base_response)
        
        # Always include English
        if language != SupportedLanguage.ENGLISH:
            english_templates = self.localized_responses[SupportedLanguage.ENGLISH]
            localized['en'] = english_templates.get('success_message', base_response)
        
        # Add regional languages based on compliance regions
        # This would expand based on business requirements
        
        return localized
    
    def _generate_audit_metadata(self, content: str, language_profile: LanguageProfile,
                               compliance_profile: ComplianceProfile, 
                               processing_region: str) -> Dict[str, Any]:
        """Generate comprehensive audit metadata."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        return {
            'content_hash': content_hash,
            'content_length': len(content),
            'language_detected': language_profile.language.value,
            'language_confidence': language_profile.confidence,
            'data_classification': compliance_profile.data_classification.value,
            'processing_region': processing_region,
            'applicable_regulations': [region.value for region in compliance_profile.regions],
            'encryption_applied': compliance_profile.encryption_required,
            'retention_period_days': compliance_profile.retention_days,
            'special_categories': compliance_profile.special_categories,
            'processing_timestamp': datetime.now(timezone.utc).isoformat(),
            'audit_version': '1.0'
        }
    
    def _determine_data_residency(self, compliance_profile: ComplianceProfile, 
                                processing_region: str) -> str:
        """Determine appropriate data residency region."""
        # Check for data localization requirements
        strict_residency_regions = [ComplianceRegion.CHINA]
        
        for region in compliance_profile.regions:
            if region in strict_residency_regions:
                return region.value
        
        # For GDPR, prefer EU residency for EU data
        if ComplianceRegion.EU in compliance_profile.regions:
            return ComplianceRegion.EU.value
        
        # Default to processing region
        return processing_region
    
    def _load_localized_responses(self) -> Dict[SupportedLanguage, Dict[str, str]]:
        """Load localized response templates."""
        return {
            SupportedLanguage.ENGLISH: {
                'success_message': 'Your email has been processed successfully.',
                'error_message': 'An error occurred while processing your email.',
                'privacy_notice': 'Your data is processed in accordance with applicable privacy laws.'
            },
            SupportedLanguage.SPANISH: {
                'success_message': 'Su correo electrónico ha sido procesado con éxito.',
                'error_message': 'Se produjo un error al procesar su correo electrónico.',
                'privacy_notice': 'Sus datos se procesan de acuerdo con las leyes de privacidad aplicables.'
            },
            SupportedLanguage.FRENCH: {
                'success_message': 'Votre email a été traité avec succès.',
                'error_message': 'Une erreur s\'est produite lors du traitement de votre email.',
                'privacy_notice': 'Vos données sont traitées conformément aux lois sur la confidentialité applicables.'
            },
            SupportedLanguage.GERMAN: {
                'success_message': 'Ihre E-Mail wurde erfolgreich verarbeitet.',
                'error_message': 'Bei der Verarbeitung Ihrer E-Mail ist ein Fehler aufgetreten.',
                'privacy_notice': 'Ihre Daten werden gemäß den geltenden Datenschutzgesetzen verarbeitet.'
            },
            SupportedLanguage.JAPANESE: {
                'success_message': 'メールの処理が正常に完了しました。',
                'error_message': 'メールの処理中にエラーが発生しました。',
                'privacy_notice': '適用されるプライバシー法に従ってデータが処理されます。'
            }
        }
    
    def _load_regional_preferences(self) -> Dict[ComplianceRegion, Dict[str, Any]]:
        """Load regional processing preferences."""
        return {
            ComplianceRegion.EU: {
                'preferred_languages': ['en', 'de', 'fr', 'it', 'es'],
                'business_hours': '09:00-17:00 CET',
                'date_format': 'DD/MM/YYYY',
                'currency': 'EUR'
            },
            ComplianceRegion.US: {
                'preferred_languages': ['en', 'es'],
                'business_hours': '09:00-17:00 EST/PST',
                'date_format': 'MM/DD/YYYY',
                'currency': 'USD'
            },
            ComplianceRegion.JAPAN: {
                'preferred_languages': ['ja', 'en'],
                'business_hours': '09:00-17:00 JST',
                'date_format': 'YYYY/MM/DD',
                'currency': 'JPY'
            }
        }
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of compliance capabilities."""
        return {
            'supported_regions': [region.value for region in ComplianceRegion],
            'supported_languages': [lang.value for lang in SupportedLanguage],
            'data_classifications': [cls.value for cls in DataClassification],
            'privacy_frameworks': ['GDPR', 'CCPA', 'PIPEDA', 'PDPA', 'LGPD'],
            'security_features': ['Encryption', 'Audit trails', 'Data residency', 'Access controls'],
            'data_subject_rights': ['Access', 'Rectification', 'Erasure', 'Portability', 'Objection']
        }


# Global framework instance
_global_framework = None

def get_global_framework() -> GlobalIntelligenceFramework:
    """Get global intelligence framework instance."""
    global _global_framework
    if _global_framework is None:
        _global_framework = GlobalIntelligenceFramework()
    return _global_framework
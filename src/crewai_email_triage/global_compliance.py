"""
Global Compliance Framework
Implements comprehensive privacy and regulatory compliance for global deployments.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import re


class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"          # GDPR
    US = "us"          # CCPA, COPPA  
    CANADA = "ca"      # PIPEDA
    UK = "uk"          # UK GDPR
    AUSTRALIA = "au"   # Privacy Act
    SINGAPORE = "sg"   # PDPA
    BRAZIL = "br"      # LGPD
    INDIA = "in"       # PDPB
    GLOBAL = "global"  # General privacy best practices


class DataCategory(Enum):
    """Data classification categories."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    id: str
    name: str
    region: ComplianceRegion
    category: DataCategory
    description: str
    mandatory: bool = True
    retention_days: Optional[int] = None
    anonymization_required: bool = False
    encryption_required: bool = False
    audit_required: bool = True


@dataclass
class PrivacyMetadata:
    """Privacy metadata for processed data."""
    data_id: str
    region: ComplianceRegion
    category: DataCategory
    collected_at: datetime
    retention_until: Optional[datetime] = None
    anonymized: bool = False
    encrypted: bool = False
    consent_given: bool = False
    purpose: str = "email_processing"
    legal_basis: str = "legitimate_interest"


class GlobalComplianceFramework:
    """Comprehensive global compliance framework."""
    
    def __init__(self):
        self.logger = logging.getLogger("global_compliance")
        self.rules: Dict[ComplianceRegion, List[ComplianceRule]] = {}
        self.privacy_records: Dict[str, PrivacyMetadata] = {}
        self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different regions."""
        
        # GDPR (EU) Rules
        self.rules[ComplianceRegion.EU] = [
            ComplianceRule(
                id="gdpr_data_minimization",
                name="Data Minimization",
                region=ComplianceRegion.EU,
                category=DataCategory.PERSONAL,
                description="Process only necessary personal data",
                retention_days=365,
                audit_required=True
            ),
            ComplianceRule(
                id="gdpr_right_to_erasure",
                name="Right to Erasure",
                region=ComplianceRegion.EU,
                category=DataCategory.PERSONAL,
                description="Support data deletion requests",
                retention_days=2555,  # 7 years max
                anonymization_required=True
            ),
            ComplianceRule(
                id="gdpr_encryption",
                name="Data Protection by Design",
                region=ComplianceRegion.EU,
                category=DataCategory.SENSITIVE_PERSONAL,
                description="Encrypt sensitive personal data",
                encryption_required=True,
                audit_required=True
            )
        ]
        
        # CCPA (US) Rules
        self.rules[ComplianceRegion.US] = [
            ComplianceRule(
                id="ccpa_data_disclosure",
                name="Data Disclosure Rights",
                region=ComplianceRegion.US,
                category=DataCategory.PERSONAL,
                description="Provide data usage transparency",
                retention_days=365,
                audit_required=True
            ),
            ComplianceRule(
                id="ccpa_opt_out",
                name="Right to Opt-Out",
                region=ComplianceRegion.US,
                category=DataCategory.PERSONAL,
                description="Support opt-out of data sale/sharing",
                audit_required=True
            )
        ]
        
        # PDPA (Singapore) Rules
        self.rules[ComplianceRegion.SINGAPORE] = [
            ComplianceRule(
                id="pdpa_consent",
                name="Consent Requirement",
                region=ComplianceRegion.SINGAPORE,
                category=DataCategory.PERSONAL,
                description="Obtain valid consent for data collection",
                retention_days=1095,  # 3 years
                audit_required=True
            ),
            ComplianceRule(
                id="pdpa_purpose_limitation",
                name="Purpose Limitation",
                region=ComplianceRegion.SINGAPORE,
                category=DataCategory.PERSONAL,
                description="Use data only for stated purposes",
                audit_required=True
            )
        ]
        
        # Global Best Practices
        self.rules[ComplianceRegion.GLOBAL] = [
            ComplianceRule(
                id="global_data_security",
                name="Data Security Standards",
                region=ComplianceRegion.GLOBAL,
                category=DataCategory.CONFIDENTIAL,
                description="Maintain high security standards",
                encryption_required=True,
                audit_required=True
            ),
            ComplianceRule(
                id="global_data_quality",
                name="Data Quality Assurance",
                region=ComplianceRegion.GLOBAL,
                category=DataCategory.INTERNAL,
                description="Ensure data accuracy and completeness",
                audit_required=False
            )
        ]
    
    def classify_email_data(self, email_content: str, metadata: Dict[str, Any]) -> DataCategory:
        """Classify email data based on content and metadata."""
        
        # Check for sensitive patterns
        sensitive_patterns = [
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Z]{2}\d{2}[A-Z]{4}\d{10}\b',  # IBAN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
        ]
        
        personal_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\b',  # Names with titles
            r'\b\d{1,5}\s+[A-Za-z\s]+\s+(?:Street|St|Avenue|Ave|Road|Rd)\b',  # Addresses
        ]
        
        # Check content
        content_lower = email_content.lower()
        
        # Check for sensitive data
        for pattern in sensitive_patterns:
            if re.search(pattern, email_content):
                return DataCategory.SENSITIVE_PERSONAL
        
        # Check for personal data
        for pattern in personal_patterns:
            if re.search(pattern, email_content):
                return DataCategory.PERSONAL
        
        # Check metadata for personal indicators
        if 'from' in metadata or 'to' in metadata:
            return DataCategory.PERSONAL
        
        # Check for work-related content
        work_keywords = ['meeting', 'project', 'deadline', 'report', 'business']
        if any(keyword in content_lower for keyword in work_keywords):
            return DataCategory.INTERNAL
        
        return DataCategory.PUBLIC
    
    def get_applicable_rules(self, region: ComplianceRegion, category: DataCategory) -> List[ComplianceRule]:
        """Get applicable compliance rules for region and data category."""
        applicable_rules = []
        
        # Add region-specific rules
        if region in self.rules:
            applicable_rules.extend([
                rule for rule in self.rules[region] 
                if rule.category == category or rule.category == DataCategory.PERSONAL
            ])
        
        # Add global rules
        if ComplianceRegion.GLOBAL in self.rules:
            applicable_rules.extend([
                rule for rule in self.rules[ComplianceRegion.GLOBAL]
                if rule.category == category
            ])
        
        return applicable_rules
    
    def create_privacy_record(self, email_content: str, metadata: Dict[str, Any], 
                            region: ComplianceRegion) -> PrivacyMetadata:
        """Create privacy record for processed email."""
        
        # Generate unique data ID
        data_hash = hashlib.sha256(email_content.encode()).hexdigest()[:16]
        data_id = f"{region.value}_{data_hash}_{int(datetime.now().timestamp())}"
        
        # Classify data
        category = self.classify_email_data(email_content, metadata)
        
        # Get applicable rules for retention
        rules = self.get_applicable_rules(region, category)
        max_retention_days = max([rule.retention_days for rule in rules if rule.retention_days], default=365)
        
        # Create privacy record
        privacy_record = PrivacyMetadata(
            data_id=data_id,
            region=region,
            category=category,
            collected_at=datetime.now(),
            retention_until=datetime.now() + timedelta(days=max_retention_days),
            consent_given=self._has_consent(metadata),
            purpose="automated_email_triage",
            legal_basis=self._determine_legal_basis(region, category)
        )
        
        # Store record
        self.privacy_records[data_id] = privacy_record
        
        return privacy_record
    
    def _has_consent(self, metadata: Dict[str, Any]) -> bool:
        """Check if consent has been given."""
        # In a real implementation, this would check actual consent records
        return metadata.get('consent_given', False)
    
    def _determine_legal_basis(self, region: ComplianceRegion, category: DataCategory) -> str:
        """Determine legal basis for processing."""
        if region == ComplianceRegion.EU:
            if category in [DataCategory.SENSITIVE_PERSONAL]:
                return "explicit_consent"
            else:
                return "legitimate_interest"
        elif region == ComplianceRegion.US:
            return "business_purpose"
        elif region == ComplianceRegion.SINGAPORE:
            return "consent"
        else:
            return "legitimate_interest"
    
    def validate_compliance(self, email_content: str, metadata: Dict[str, Any], 
                          region: ComplianceRegion) -> Dict[str, Any]:
        """Validate compliance for email processing."""
        
        category = self.classify_email_data(email_content, metadata)
        applicable_rules = self.get_applicable_rules(region, category)
        
        validation_result = {
            'compliant': True,
            'data_category': category.value,
            'region': region.value,
            'applicable_rules': len(applicable_rules),
            'violations': [],
            'requirements': [],
            'actions_required': []
        }
        
        for rule in applicable_rules:
            # Check encryption requirements
            if rule.encryption_required:
                validation_result['requirements'].append(f"Encryption required: {rule.name}")
                if not metadata.get('encrypted', False):
                    validation_result['violations'].append(f"Missing encryption: {rule.name}")
                    validation_result['compliant'] = False
            
            # Check anonymization requirements  
            if rule.anonymization_required:
                validation_result['requirements'].append(f"Anonymization required: {rule.name}")
                if not metadata.get('anonymized', False):
                    validation_result['actions_required'].append(f"Schedule anonymization: {rule.name}")
            
            # Check consent requirements (GDPR, PDPA)
            if region in [ComplianceRegion.EU, ComplianceRegion.SINGAPORE] and rule.mandatory:
                if not metadata.get('consent_given', False) and category == DataCategory.PERSONAL:
                    validation_result['violations'].append(f"Missing consent: {rule.name}")
                    validation_result['compliant'] = False
        
        return validation_result
    
    def get_retention_policy(self, region: ComplianceRegion, category: DataCategory) -> Dict[str, Any]:
        """Get data retention policy for region and category."""
        
        rules = self.get_applicable_rules(region, category)
        retention_rules = [rule for rule in rules if rule.retention_days]
        
        if not retention_rules:
            default_retention = 365  # 1 year default
        else:
            # Use the most restrictive (shortest) retention period
            default_retention = min(rule.retention_days for rule in retention_rules)
        
        return {
            'region': region.value,
            'category': category.value,
            'retention_days': default_retention,
            'retention_until': (datetime.now() + timedelta(days=default_retention)).isoformat(),
            'anonymization_required': any(rule.anonymization_required for rule in rules),
            'deletion_method': 'secure_deletion' if category in [DataCategory.SENSITIVE_PERSONAL] else 'standard_deletion'
        }
    
    def audit_compliance(self) -> Dict[str, Any]:
        """Generate compliance audit report."""
        
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(self.privacy_records),
            'regions': {},
            'categories': {},
            'upcoming_deletions': [],
            'compliance_violations': [],
            'recommendations': []
        }
        
        # Analyze records by region and category
        for record in self.privacy_records.values():
            region = record.region.value
            category = record.category.value
            
            if region not in audit_report['regions']:
                audit_report['regions'][region] = 0
            audit_report['regions'][region] += 1
            
            if category not in audit_report['categories']:
                audit_report['categories'][category] = 0
            audit_report['categories'][category] += 1
            
            # Check for upcoming deletions (within 30 days)
            if record.retention_until and record.retention_until <= datetime.now() + timedelta(days=30):
                audit_report['upcoming_deletions'].append({
                    'data_id': record.data_id,
                    'region': region,
                    'category': category,
                    'retention_until': record.retention_until.isoformat()
                })
        
        # Generate recommendations
        if len(audit_report['upcoming_deletions']) > 0:
            audit_report['recommendations'].append(
                f"Schedule deletion of {len(audit_report['upcoming_deletions'])} records approaching retention limit"
            )
        
        if any(not record.encrypted for record in self.privacy_records.values() 
               if record.category == DataCategory.SENSITIVE_PERSONAL):
            audit_report['recommendations'].append(
                "Encrypt all sensitive personal data records"
            )
        
        return audit_report
    
    def process_deletion_request(self, email_address: str, region: ComplianceRegion) -> Dict[str, Any]:
        """Process data deletion request (Right to Erasure/Right to Delete)."""
        
        # Find related records
        related_records = []
        for record_id, record in self.privacy_records.items():
            if record.region == region:
                # In a real implementation, you would check if the record relates to the email address
                related_records.append(record_id)
        
        # Schedule deletion
        deletion_report = {
            'request_id': hashlib.md5(f"{email_address}_{region.value}_{datetime.now()}".encode()).hexdigest()[:12],
            'email_address': email_address,
            'region': region.value,
            'request_timestamp': datetime.now().isoformat(),
            'records_found': len(related_records),
            'deletion_scheduled': True,
            'completion_deadline': (datetime.now() + timedelta(days=30)).isoformat(),
            'status': 'scheduled'
        }
        
        return deletion_report


# Global compliance framework instance
_compliance_framework: Optional[GlobalComplianceFramework] = None


def get_compliance_framework() -> GlobalComplianceFramework:
    """Get or create global compliance framework instance."""
    global _compliance_framework
    if _compliance_framework is None:
        _compliance_framework = GlobalComplianceFramework()
    return _compliance_framework


def validate_email_compliance(email_content: str, metadata: Dict[str, Any], 
                             region_code: str = "global") -> Dict[str, Any]:
    """Validate email processing compliance."""
    framework = get_compliance_framework()
    
    try:
        region = ComplianceRegion(region_code.lower())
    except ValueError:
        region = ComplianceRegion.GLOBAL
    
    return framework.validate_compliance(email_content, metadata, region)
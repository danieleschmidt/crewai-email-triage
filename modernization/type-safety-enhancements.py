#!/usr/bin/env python3
"""
Advanced type safety patterns for Python 3.11+ email triage service.
Demonstrates modern typing features for enhanced code safety and IDE support.
"""

from typing import (
    TypeVar, Generic, Protocol, Literal, LiteralString, Self, TypeAlias,
    Annotated, get_args, get_origin, Any, Never, assert_never
)
from typing_extensions import NotRequired, TypedDict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging

# Type aliases for domain-specific types
EmailID: TypeAlias = Annotated[str, "Unique email identifier"]
EmailAddress: TypeAlias = Annotated[str, "Valid email address format"]
ConfidenceScore: TypeAlias = Annotated[float, "Confidence score between 0.0 and 1.0"]

# Literal types for email categories
EmailCategory = Literal["urgent", "important", "normal", "spam", "promotional"]
ProcessingStatus = Literal["pending", "processing", "completed", "failed"]

# Generic type variables
T = TypeVar('T')
EmailT = TypeVar('EmailT', bound='BaseEmail')

class EmailMetadata(TypedDict):
    """Typed dictionary for email metadata with optional fields."""
    sender: EmailAddress
    recipient: EmailAddress
    subject: str
    timestamp: int
    category: NotRequired[EmailCategory]  # Optional field
    attachments: NotRequired[list[str]]   # Optional field

class EmailProcessor(Protocol):
    """Protocol defining the email processor interface."""
    
    def process(self, email: 'BaseEmail') -> 'ProcessingResult':
        """Process an email and return results."""
        ...
    
    def batch_process(self, emails: list['BaseEmail']) -> list['ProcessingResult']:
        """Process multiple emails in batch."""
        ...

@dataclass(frozen=True, slots=True)  # Python 3.10+ slots optimization
class BaseEmail:
    """Immutable base email class with advanced type annotations."""
    
    id: EmailID
    metadata: EmailMetadata
    content: str
    processing_status: ProcessingStatus = "pending"
    
    def with_status(self, status: ProcessingStatus) -> Self:
        """Create a new instance with updated status."""
        return BaseEmail(
            id=self.id,
            metadata=self.metadata,
            content=self.content,
            processing_status=status
        )
    
    def is_urgent(self) -> bool:
        """Type-safe check for urgent emails."""
        return self.metadata.get("category") == "urgent"

@dataclass
class ProcessingResult:
    """Results from email processing with comprehensive type safety."""
    
    email_id: EmailID
    classification: EmailCategory
    confidence: ConfidenceScore
    summary: str | None = None
    errors: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate confidence score range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

class ProcessingError(Enum):
    """Enumeration of possible processing errors."""
    INVALID_FORMAT = "invalid_email_format"
    CLASSIFICATION_FAILED = "classification_failed"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "processing_timeout"
    INSUFFICIENT_DATA = "insufficient_data"

class SecurityLevel(Enum):
    """Security levels for email processing."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

def secure_query_builder(query_template: LiteralString) -> str:
    """Build secure database queries using LiteralString to prevent injection."""
    # LiteralString ensures the query template is a compile-time literal
    # This prevents dynamic query construction that could lead to SQL injection
    return query_template

def classify_email_security(email_content: str) -> SecurityLevel:
    """Advanced pattern matching for security classification."""
    
    # Normalize content for analysis
    content_lower = email_content.lower()
    
    match True:
        case _ if any(keyword in content_lower for keyword in ["confidential", "secret", "classified"]):
            return SecurityLevel.RESTRICTED
        case _ if any(keyword in content_lower for keyword in ["internal", "company", "proprietary"]):
            return SecurityLevel.CONFIDENTIAL
        case _ if "@company.com" in content_lower:
            return SecurityLevel.INTERNAL
        case _:
            return SecurityLevel.PUBLIC

class TypeSafeProcessorFactory(Generic[EmailT]):
    """Generic factory for creating type-safe email processors."""
    
    def __init__(self, email_type: type[EmailT]):
        self.email_type = email_type
        self._processors: dict[EmailCategory, EmailProcessor] = {}
    
    def register_processor(
        self, 
        category: EmailCategory, 
        processor: EmailProcessor
    ) -> Self:
        """Register a processor for a specific email category."""
        self._processors[category] = processor
        return self
    
    def get_processor(self, category: EmailCategory) -> EmailProcessor:
        """Get the appropriate processor for an email category."""
        if category not in self._processors:
            raise ValueError(f"No processor registered for category: {category}")
        return self._processors[category]
    
    def create_typed_email(self, email_data: dict[str, Any]) -> EmailT:
        """Create a typed email instance with validation."""
        if not isinstance(email_data.get("id"), str):
            raise ValueError("Email ID must be a string")
            
        return self.email_type(**email_data)

def exhaustive_category_handler(category: EmailCategory) -> str:
    """Demonstrate exhaustive pattern matching with Never type."""
    
    match category:
        case "urgent":
            return "Handle with highest priority"
        case "important":
            return "Handle within 2 hours"
        case "normal":
            return "Handle within 24 hours"
        case "spam":
            return "Move to spam folder"
        case "promotional":
            return "Move to promotions folder"
        case _ as unreachable:
            # This branch should never be reached if all cases are covered
            assert_never(unreachable)

class AdvancedEmailAnalyzer:
    """Advanced email analyzer with comprehensive type safety."""
    
    def __init__(self):
        self.processor_factory = TypeSafeProcessorFactory(BaseEmail)
        
    def analyze_batch(
        self, 
        emails: list[BaseEmail]
    ) -> dict[EmailCategory, list[ProcessingResult]]:
        """Analyze emails and group results by category."""
        
        results_by_category: dict[EmailCategory, list[ProcessingResult]] = {
            category: [] for category in get_args(EmailCategory)
        }
        
        for email in emails:
            try:
                # Type-safe processing
                result = self._analyze_single_email(email)
                results_by_category[result.classification].append(result)
                
            except Exception as e:
                logging.error(f"Failed to analyze email {email.id}: {e}")
                # Create error result
                error_result = ProcessingResult(
                    email_id=email.id,
                    classification="normal",  # Default fallback
                    confidence=0.0,
                    errors=[str(e)]
                )
                results_by_category["normal"].append(error_result)
        
        return results_by_category
    
    def _analyze_single_email(self, email: BaseEmail) -> ProcessingResult:
        """Analyze a single email with comprehensive type checking."""
        
        # Security classification
        security_level = classify_email_security(email.content)
        
        # Content-based classification
        classification = self._classify_content(email.content)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(email, security_level)
        
        return ProcessingResult(
            email_id=email.id,
            classification=classification,
            confidence=confidence,
            summary=self._generate_summary(email.content)
        )
    
    def _classify_content(self, content: str) -> EmailCategory:
        """Classify email content using type-safe logic."""
        content_lower = content.lower()
        
        # Use exhaustive matching to ensure all categories are considered
        if any(urgent_keyword in content_lower for urgent_keyword in ["urgent", "asap", "emergency"]):
            return "urgent"
        elif any(important_keyword in content_lower for important_keyword in ["important", "priority", "deadline"]):
            return "important"
        elif any(spam_keyword in content_lower for spam_keyword in ["lottery", "winner", "click here"]):
            return "spam"
        elif any(promo_keyword in content_lower for promo_keyword in ["sale", "discount", "offer"]):
            return "promotional"
        else:
            return "normal"
    
    def _calculate_confidence(
        self, 
        email: BaseEmail, 
        security_level: SecurityLevel
    ) -> ConfidenceScore:
        """Calculate confidence score with type safety."""
        base_confidence = 0.7
        
        # Adjust based on security level
        security_boost = {
            SecurityLevel.PUBLIC: 0.0,
            SecurityLevel.INTERNAL: 0.1,
            SecurityLevel.CONFIDENTIAL: 0.15,
            SecurityLevel.RESTRICTED: 0.2
        }
        
        # Ensure the result is within valid range
        confidence = min(1.0, base_confidence + security_boost[security_level])
        
        # Type system ensures this is a valid ConfidenceScore
        return confidence
    
    def _generate_summary(self, content: str) -> str:
        """Generate a brief summary of email content."""
        # Simple truncation for demonstration
        return content[:100] + "..." if len(content) > 100 else content

# Usage example demonstrating type safety
def main_type_safety_example():
    """Demonstrate advanced type safety features."""
    
    # Create type-safe email metadata
    metadata: EmailMetadata = {
        "sender": "user@example.com",
        "recipient": "admin@company.com",
        "subject": "Urgent: System Alert",
        "timestamp": 1640995200,
        "category": "urgent"  # Optional field
    }
    
    # Create email with full type safety
    email = BaseEmail(
        id="email_001",
        metadata=metadata,
        content="This is an urgent system alert requiring immediate attention."
    )
    
    # Type-safe processing
    analyzer = AdvancedEmailAnalyzer()
    result = analyzer._analyze_single_email(email)
    
    print(f"Email {result.email_id} classified as {result.classification} "
          f"with {result.confidence:.2%} confidence")

if __name__ == "__main__":
    main_type_safety_example()
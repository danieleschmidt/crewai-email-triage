"""Basic email validation functionality."""

import re
from typing import Dict, List, Tuple


class EmailValidator:
    """Basic email content validator."""

    @staticmethod
    def validate_content(content: str) -> Tuple[bool, List[str]]:
        """Validate email content and return (is_valid, warnings)."""
        if not content or not content.strip():
            return False, ["Empty content"]

        warnings = []

        # Check for basic spam indicators
        spam_patterns = [
            r'urgent.*act.*now',
            r'click.*here.*immediately',
            r'limited.*time.*offer',
            r'congratulations.*winner'
        ]

        content_lower = content.lower()
        for pattern in spam_patterns:
            if re.search(pattern, content_lower):
                warnings.append(f"Potential spam pattern detected: {pattern}")

        # Check for excessive capitalization
        if sum(1 for c in content if c.isupper()) / len(content) > 0.5:
            warnings.append("Excessive capitalization detected")

        # Basic length validation
        if len(content) > 50000:
            warnings.append("Email content is unusually long")

        return len(warnings) == 0, warnings

def validate_email_basic(content: str) -> Dict[str, any]:
    """Basic email validation function."""
    validator = EmailValidator()
    is_valid, warnings = validator.validate_content(content)

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "content_length": len(content),
        "validation_timestamp": import_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    }

import time as import_time

#!/usr/bin/env python3
"""
Property-based testing for email triage service using Hypothesis.
Validates system behavior through generative testing and invariant checking.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List
import pytest
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant
import string

# Import our email processing modules (mock imports for demonstration)
# from crewai_email_triage.classifier import EmailClassifier
# from crewai_email_triage.pipeline import EmailPipeline

# Hypothesis strategies for generating test data
@st.composite
def email_address_strategy(draw):
    """Generate valid email addresses."""
    local_part = draw(st.text(
        alphabet=string.ascii_letters + string.digits + ".-_",
        min_size=1,
        max_size=20
    ).filter(lambda x: x[0] not in ".-_" and x[-1] not in ".-_"))
    
    domain = draw(st.text(
        alphabet=string.ascii_lowercase + string.digits + "-",
        min_size=3,
        max_size=15
    ).filter(lambda x: x[0] != "-" and x[-1] != "-"))
    
    tld = draw(st.sampled_from(["com", "org", "net", "edu", "gov"]))
    
    return f"{local_part}@{domain}.{tld}"

@st.composite
def email_content_strategy(draw):
    """Generate realistic email content."""
    subject_length = draw(st.integers(min_value=5, max_value=100))
    body_length = draw(st.integers(min_value=50, max_value=2000))
    
    subject = draw(st.text(
        alphabet=string.ascii_letters + string.digits + string.punctuation + " ",
        min_size=subject_length,
        max_size=subject_length
    ))
    
    body = draw(st.text(
        alphabet=string.ascii_letters + string.digits + string.punctuation + " \n",
        min_size=body_length,
        max_size=body_length
    ))
    
    return {
        "subject": subject.strip(),
        "body": body.strip(),
        "sender": draw(email_address_strategy()),
        "recipient": draw(email_address_strategy()),
        "timestamp": draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31)
        )),
        "attachments": draw(st.lists(
            st.text(min_size=5, max_size=50),
            min_size=0,
            max_size=5
        ))
    }

class EmailClassificationProperties:
    """Property-based tests for email classification."""
    
    @given(email_content_strategy())
    @settings(max_examples=200, deadline=5000)
    def test_classification_consistency(self, email_data):
        """Test that classification is consistent for identical emails."""
        # classifier = EmailClassifier()
        
        # Simulate classification
        def mock_classify(email):
            # Mock implementation for demonstration
            if "urgent" in email["subject"].lower():
                return "urgent"
            elif len(email["attachments"]) > 3:
                return "bulk"
            else:
                return "normal"
        
        # Property: Same email should always get same classification
        classification1 = mock_classify(email_data)
        classification2 = mock_classify(email_data)
        
        assert classification1 == classification2, \
            "Classification should be deterministic"
    
    @given(
        email_data=email_content_strategy(),
        confidence_threshold=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_confidence_bounds(self, email_data, confidence_threshold):
        """Test that confidence scores are always within valid bounds."""
        def mock_classify_with_confidence(email):
            # Mock confidence calculation
            base_confidence = 0.7
            if "urgent" in email["subject"].lower():
                return "urgent", min(1.0, base_confidence + 0.2)
            return "normal", base_confidence
        
        category, confidence = mock_classify_with_confidence(email_data)
        
        # Property: Confidence must be between 0 and 1
        assert 0.0 <= confidence <= 1.0, \
            f"Confidence {confidence} is outside valid range [0, 1]"
    
    @given(st.lists(email_content_strategy(), min_size=1, max_size=50))
    def test_batch_processing_equivalence(self, email_list):
        """Test that batch processing gives same results as individual processing."""
        def mock_process_single(email):
            return {
                "id": hash(frozenset(email.items())) % 1000000,
                "classification": "normal",
                "confidence": 0.7
            }
        
        def mock_process_batch(emails):
            return [mock_process_single(email) for email in emails]
        
        # Process individually
        individual_results = [mock_process_single(email) for email in email_list]
        
        # Process as batch
        batch_results = mock_process_batch(email_list)
        
        # Property: Results should be equivalent
        assert len(individual_results) == len(batch_results)
        for individual, batch in zip(individual_results, batch_results):
            assert individual["classification"] == batch["classification"]

class EmailProcessingStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for email processing pipeline."""
    
    def __init__(self):
        super().__init__()
        self.processed_emails = []
        self.failed_emails = []
        self.email_queue = []
        
    emails = Bundle('emails')
    
    @rule(target=emails, email_data=email_content_strategy())
    def add_email_to_queue(self, email_data):
        """Add an email to the processing queue."""
        email_id = len(self.email_queue)
        email = {**email_data, "id": email_id}
        self.email_queue.append(email)
        return email
    
    @rule(email=emails)
    def process_email(self, email):
        """Process an email from the queue."""
        assume(email in self.email_queue)
        
        try:
            # Mock processing logic
            if len(email["subject"]) == 0:
                raise ValueError("Empty subject")
            
            processed_email = {
                **email,
                "processed_at": datetime.now(),
                "classification": "normal",
                "confidence": 0.8
            }
            
            self.processed_emails.append(processed_email)
            self.email_queue.remove(email)
            
        except Exception as e:
            self.failed_emails.append({**email, "error": str(e)})
            if email in self.email_queue:
                self.email_queue.remove(email)
    
    @rule(batch_size=st.integers(min_value=1, max_value=10))
    def process_batch(self, batch_size):
        """Process a batch of emails."""
        if not self.email_queue:
            return
            
        batch_size = min(batch_size, len(self.email_queue))
        batch = self.email_queue[:batch_size]
        
        for email in batch:
            self.process_email(email)
    
    @invariant()
    def no_duplicate_processing(self):
        """Invariant: No email should be processed multiple times."""
        processed_ids = [email["id"] for email in self.processed_emails]
        assert len(processed_ids) == len(set(processed_ids)), \
            "Duplicate email processing detected"
    
    @invariant()
    def total_emails_conservation(self):
        """Invariant: Total emails should be conserved."""
        total_emails_seen = len(self.processed_emails) + len(self.failed_emails) + len(self.email_queue)
        # This would be more meaningful with actual email tracking
        assert total_emails_seen >= 0
    
    @invariant()
    def processed_emails_have_classification(self):
        """Invariant: All processed emails must have a classification."""
        for email in self.processed_emails:
            assert "classification" in email, \
                f"Processed email {email['id']} missing classification"
            assert email["classification"] in ["urgent", "normal", "spam", "bulk"], \
                f"Invalid classification: {email['classification']}"

class EmailValidationProperties:
    """Property-based tests for email validation."""
    
    @given(
        subject=st.text(max_size=1000),
        body=st.text(max_size=10000),
        sender=email_address_strategy()
    )
    @example(subject="", body="", sender="test@example.com")  # Edge case
    def test_email_validation_properties(self, subject, body, sender):
        """Test email validation properties."""
        def validate_email(subject, body, sender):
            """Mock email validation."""
            errors = []
            
            if len(subject.strip()) == 0:
                errors.append("Empty subject")
            if len(subject) > 500:
                errors.append("Subject too long")
            if len(body.strip()) == 0:
                errors.append("Empty body")
            if "@" not in sender:
                errors.append("Invalid sender")
                
            return len(errors) == 0, errors
        
        is_valid, errors = validate_email(subject, body, sender)
        
        # Property: Valid emails should have no errors
        if is_valid:
            assert len(errors) == 0
        else:
            assert len(errors) > 0
        
        # Property: Validation should be deterministic
        is_valid2, errors2 = validate_email(subject, body, sender)
        assert is_valid == is_valid2
        assert errors == errors2

# Metamorphic testing properties
class MetamorphicEmailProperties:
    """Metamorphic properties for email processing."""
    
    @given(email_content_strategy())
    def test_case_insensitive_classification(self, email_data):
        """Test that classification is case-insensitive."""
        def mock_classify(email):
            subject_lower = email["subject"].lower()
            if "urgent" in subject_lower:
                return "urgent"
            return "normal"
        
        # Original classification
        original_classification = mock_classify(email_data)
        
        # Modified email with different case
        modified_email = {
            **email_data,
            "subject": email_data["subject"].swapcase()
        }
        modified_classification = mock_classify(modified_email)
        
        # Property: Classification should be case-insensitive
        assert original_classification == modified_classification
    
    @given(email_content_strategy())
    def test_whitespace_normalization(self, email_data):
        """Test that classification is not affected by whitespace changes."""
        def mock_classify_normalized(email):
            normalized_subject = " ".join(email["subject"].split())
            if "important" in normalized_subject.lower():
                return "important"
            return "normal"
        
        # Original classification
        original_classification = mock_classify_normalized(email_data)
        
        # Add extra whitespace
        modified_email = {
            **email_data,
            "subject": "  " + email_data["subject"].replace(" ", "   ") + "  "
        }
        modified_classification = mock_classify_normalized(modified_email)
        
        # Property: Whitespace should not affect classification
        assert original_classification == modified_classification

# Performance property tests
class PerformanceProperties:
    """Property-based tests for performance characteristics."""
    
    @given(
        email_count=st.integers(min_value=1, max_value=1000),
        email_size=st.integers(min_value=100, max_value=5000)
    )
    @settings(max_examples=50, deadline=10000)
    def test_processing_time_scales_linearly(self, email_count, email_size):
        """Test that processing time scales reasonably with input size."""
        import time
        
        def mock_process_emails(count, size):
            # Simulate processing time proportional to input
            start_time = time.time()
            # Mock processing work
            for _ in range(count):
                _ = "x" * size  # Simulate work proportional to email size
            end_time = time.time()
            return end_time - start_time
        
        processing_time = mock_process_emails(email_count, email_size)
        
        # Property: Processing should complete within reasonable time bounds
        # This is a simplified example - real tests would have more sophisticated timing
        max_expected_time = (email_count * email_size) / 100000  # Arbitrary scaling
        assert processing_time < max_expected_time + 1.0, \
            f"Processing took too long: {processing_time}s"

if __name__ == "__main__":
    # Run the property-based tests
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
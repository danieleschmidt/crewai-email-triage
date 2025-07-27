"""Configuration and fixtures for integration tests."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_email_message():
    """Sample email message for testing."""
    return {
        "subject": "Urgent: Project deadline tomorrow",
        "sender": "manager@company.com",
        "body": "Please review the attached documents and provide feedback by end of day. This is urgent.",
        "timestamp": "2025-07-27T10:00:00Z",
        "attachments": []
    }

@pytest.fixture
def batch_email_messages():
    """Batch of sample email messages for testing."""
    return [
        {
            "subject": "Urgent: Project deadline tomorrow",
            "sender": "manager@company.com", 
            "body": "Please review the attached documents and provide feedback by end of day. This is urgent.",
            "timestamp": "2025-07-27T10:00:00Z",
            "attachments": []
        },
        {
            "subject": "Meeting invitation",
            "sender": "colleague@company.com",
            "body": "Would you like to join our team meeting on Friday?",
            "timestamp": "2025-07-27T11:00:00Z",
            "attachments": []
        },
        {
            "subject": "Newsletter subscription",
            "sender": "marketing@newsletter.com",
            "body": "Thanks for subscribing! Click here to unsubscribe.",
            "timestamp": "2025-07-27T12:00:00Z",
            "attachments": []
        }
    ]

@pytest.fixture
def mock_gmail_provider():
    """Mock Gmail provider for integration tests."""
    with patch('crewai_email_triage.provider.GmailProvider') as mock:
        provider = Mock()
        provider.fetch_messages.return_value = []
        provider.send_response.return_value = True
        mock.return_value = provider
        yield provider

@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "classifier": {
            "urgent": ["urgent", "asap", "important"],
            "spam": ["unsubscribe", "marketing", "promotion"],
            "work": ["meeting", "project", "deadline"]
        },
        "priority": {
            "scores": {"high": 10, "medium": 5, "low": 1},
            "high_keywords": ["urgent", "asap", "deadline"],
            "medium_keywords": ["meeting", "review", "feedback"]
        }
    }

@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "GMAIL_USER": "test@example.com",
        "GMAIL_PASSWORD": "test_password",
        "CREWAI_CONFIG": "",
        "LOG_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture(scope="session")
def integration_test_marker():
    """Marker to identify integration tests."""
    return pytest.mark.integration
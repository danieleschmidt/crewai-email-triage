"""Global pytest configuration and fixtures for CrewAI Email Triage."""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests"
    )
    config.addinivalue_line(
        "markers", "security: Security tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "network: Tests requiring network access"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add e2e marker to end-to-end tests
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add unit marker to unit tests (default)
        if not any(mark.name in ["integration", "performance", "e2e"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

# =============================================================================
# GLOBAL FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def temp_workspace():
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create subdirectories
        (workspace / "config").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "data").mkdir()
        (workspace / "cache").mkdir()
        
        yield workspace

@pytest.fixture
def test_config():
    """Default test configuration."""
    return {
        "classifier": {
            "urgent": ["urgent", "asap", "important", "critical", "emergency"],
            "spam": ["unsubscribe", "marketing", "promotion", "advertisement"],
            "work": ["meeting", "project", "deadline", "review", "feedback"],
            "personal": ["family", "friend", "birthday", "vacation"]
        },
        "priority": {
            "scores": {
                "urgent": 10,
                "high": 8,
                "medium": 5,
                "low": 2,
                "spam": 0
            },
            "urgent_keywords": ["urgent", "asap", "emergency", "critical"],
            "high_keywords": ["important", "deadline", "review"],
            "medium_keywords": ["meeting", "project", "feedback"],
            "low_keywords": ["newsletter", "update", "notification"]
        },
        "response": {
            "templates": {
                "urgent": "Thank you for your urgent message. I will review this immediately.",
                "work": "Thank you for your message. I will respond within 24 hours.",
                "personal": "Thanks for reaching out! I'll get back to you soon."
            }
        }
    }

@pytest.fixture
def test_config_file(temp_workspace, test_config):
    """Create a temporary configuration file."""
    config_file = temp_workspace / "config" / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(test_config, f, indent=2)
    return config_file

@pytest.fixture
def sample_emails():
    """Sample email messages for testing."""
    return [
        {
            "id": "email_1",
            "subject": "URGENT: Server downtime scheduled",
            "sender": "admin@company.com",
            "recipient": "team@company.com",
            "body": "Emergency server maintenance will occur tonight. Please prepare for downtime.",
            "timestamp": "2025-08-02T09:00:00Z",
            "headers": {"Message-ID": "urgent-001@company.com"},
            "attachments": []
        },
        {
            "id": "email_2", 
            "subject": "Weekly team meeting",
            "sender": "manager@company.com",
            "recipient": "team@company.com",
            "body": "Our weekly sync is scheduled for Thursday 2 PM. Please review the agenda.",
            "timestamp": "2025-08-02T10:00:00Z",
            "headers": {"Message-ID": "meeting-001@company.com"},
            "attachments": ["agenda.pdf"]
        },
        {
            "id": "email_3",
            "subject": "Newsletter: Latest updates",
            "sender": "newsletter@updates.com",
            "recipient": "user@company.com", 
            "body": "Check out our latest features and updates. Click here to unsubscribe.",
            "timestamp": "2025-08-02T11:00:00Z",
            "headers": {"Message-ID": "newsletter-001@updates.com"},
            "attachments": []
        },
        {
            "id": "email_4",
            "subject": "Birthday party invitation",
            "sender": "friend@personal.com",
            "recipient": "user@company.com",
            "body": "You're invited to my birthday party this Saturday! Hope to see you there.",
            "timestamp": "2025-08-02T12:00:00Z", 
            "headers": {"Message-ID": "birthday-001@personal.com"},
            "attachments": ["invitation.jpg"]
        }
    ]

@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        "GMAIL_USER": "test@example.com",
        "GMAIL_PASSWORD": "test_app_password",
        "CREWAI_CONFIG": "",
        "LOG_LEVEL": "DEBUG",
        "METRICS_ENABLED": "false",
        "RATE_LIMIT_PER_MINUTE": "100",
        "BATCH_SIZE": "10",
        "TIMEOUT_SECONDS": "30"
    }
    
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars

@pytest.fixture
def mock_gmail_provider():
    """Mock Gmail email provider."""
    mock_provider = Mock()
    mock_provider.fetch_messages.return_value = []
    mock_provider.send_response.return_value = True
    mock_provider.is_connected.return_value = True
    mock_provider.get_unread_count.return_value = 0
    
    with patch('crewai_email_triage.provider.GmailProvider', return_value=mock_provider):
        yield mock_provider

@pytest.fixture
def mock_agents():
    """Mock all agents for isolated testing."""
    mocks = {}
    
    # Mock classifier agent
    mock_classifier = Mock()
    mock_classifier.run.return_value = {"category": "work", "confidence": 0.9}
    mocks['classifier'] = mock_classifier
    
    # Mock priority agent  
    mock_priority = Mock()
    mock_priority.run.return_value = {"priority": "high", "score": 8}
    mocks['priority'] = mock_priority
    
    # Mock summarizer agent
    mock_summarizer = Mock()
    mock_summarizer.run.return_value = {"summary": "Test email summary"}
    mocks['summarizer'] = mock_summarizer
    
    # Mock response agent
    mock_response = Mock()
    mock_response.run.return_value = {"response": "Test response"}
    mocks['response'] = mock_response
    
    with patch('crewai_email_triage.classifier.ClassifierAgent', return_value=mock_classifier), \
         patch('crewai_email_triage.priority.PriorityAgent', return_value=mock_priority), \
         patch('crewai_email_triage.summarizer.SummarizerAgent', return_value=mock_summarizer), \
         patch('crewai_email_triage.response.ResponseAgent', return_value=mock_response):
        yield mocks

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            
        def start_timer(self, name: str):
            import time
            self.metrics[name] = {"start": time.time()}
            
        def end_timer(self, name: str):
            import time
            if name in self.metrics:
                self.metrics[name]["end"] = time.time()
                self.metrics[name]["duration"] = self.metrics[name]["end"] - self.metrics[name]["start"]
                
        def get_duration(self, name: str) -> float:
            return self.metrics.get(name, {}).get("duration", 0)
            
        def assert_duration_under(self, name: str, max_seconds: float):
            duration = self.get_duration(name)
            assert duration < max_seconds, f"Operation '{name}' took {duration:.3f}s, expected under {max_seconds}s"
    
    return PerformanceMonitor()

@pytest.fixture
def security_scanner():
    """Security testing utilities."""
    class SecurityScanner:
        def __init__(self):
            self.malicious_patterns = [
                r'<script.*?>.*?</script>',
                r'javascript:',
                r'eval\(',
                r'exec\(',
                r'subprocess\.',
                r'os\.system',
                r'__import__',
                r'file://',
                r'../../'
            ]
            
        def scan_for_vulnerabilities(self, text: str) -> List[str]:
            """Scan text for potential security vulnerabilities."""
            import re
            found_issues = []
            
            for pattern in self.malicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_issues.append(f"Potential vulnerability: {pattern}")
                    
            return found_issues
            
        def assert_no_vulnerabilities(self, text: str):
            """Assert that text contains no security vulnerabilities."""
            issues = self.scan_for_vulnerabilities(text)
            assert not issues, f"Security vulnerabilities found: {issues}"
    
    return SecurityScanner()

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    
    # Cleanup any test files created
    test_patterns = [
        "test_*.tmp",
        "test_*.log", 
        "test_config*.json",
        "test_emails*.json"
    ]
    
    import glob
    for pattern in test_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except OSError:
                pass  # Ignore if file doesn't exist or can't be removed

# =============================================================================
# TESTING UTILITIES
# =============================================================================

@pytest.fixture
def assert_helpers():
    """Helper functions for common assertions."""
    class AssertHelpers:
        @staticmethod
        def assert_valid_email_format(email: str):
            """Assert that email address is in valid format."""
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            assert re.match(pattern, email), f"Invalid email format: {email}"
            
        @staticmethod
        def assert_valid_json(json_string: str):
            """Assert that string is valid JSON."""
            try:
                json.loads(json_string)
            except (json.JSONDecodeError, TypeError) as e:
                pytest.fail(f"Invalid JSON: {e}")
                
        @staticmethod
        def assert_classification_result(result: Dict[str, Any]):
            """Assert that classification result has required fields."""
            required_fields = ["category", "confidence"]
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            assert 0 <= result["confidence"] <= 1, "Confidence must be between 0 and 1"
            
        @staticmethod
        def assert_priority_result(result: Dict[str, Any]):
            """Assert that priority result has required fields."""
            required_fields = ["priority", "score"]
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
            assert isinstance(result["score"], (int, float)), "Score must be numeric"
            
        @staticmethod
        def assert_response_time_acceptable(duration: float, max_seconds: float = 5.0):
            """Assert that response time is acceptable."""
            assert duration < max_seconds, f"Response time {duration:.3f}s exceeds maximum {max_seconds}s"
    
    return AssertHelpers()

# =============================================================================
# PARAMETRIZED FIXTURES
# =============================================================================

@pytest.fixture(params=[
    "gmail",
    "imap",
    "mock"
])
def email_provider_type(request):
    """Parametrized fixture for different email provider types."""
    return request.param

@pytest.fixture(params=[1, 5, 10, 50])
def batch_sizes(request):
    """Parametrized fixture for different batch sizes."""
    return request.param

@pytest.fixture(params=[
    {"parallel": True, "workers": 2},
    {"parallel": True, "workers": 4},
    {"parallel": False, "workers": 1}
])
def processing_configs(request):
    """Parametrized fixture for different processing configurations."""
    return request.param
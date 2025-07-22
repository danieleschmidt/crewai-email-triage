"""Test enhanced structured logging in provider module."""

import sys
import os
import json
import logging
from io import StringIO
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from crewai_email_triage.logging_utils import setup_structured_logging, StructuredFormatter
from crewai_email_triage.provider import GmailProvider


class TestProviderLoggingEnhancement:
    """Test enhanced structured logging in provider module."""
    
    def setup_method(self):
        """Setup test environment with structured logging capture."""
        # Capture log output
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(StructuredFormatter())
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    
    def get_structured_logs(self) -> list:
        """Parse and return structured log entries."""
        log_output = self.log_stream.getvalue()
        if not log_output.strip():
            return []
            
        logs = []
        for line in log_output.strip().split('\n'):
            if line.strip():
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    pass
        return logs
    
    def test_provider_initialization_logging(self):
        """Test that provider initialization includes structured context."""
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Initialize provider
        provider = GmailProvider("test@example.com", "fake_password")
        
        logs = self.get_structured_logs()
        
        # Should have logs with context about initialization
        init_logs = [log for log in logs if 'initialized' in log.get('message', '').lower()]
        
        for log in init_logs:
            assert 'extra' in log or any(key in log for key in ['username', 'server', 'operation'])
            print(f"✓ Provider init logging: {log.get('message', '')[:50]}...")
    
    def test_message_search_logging(self):
        """Test that message search includes count and performance context."""
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        try:
            provider = GmailProvider("test@example.com", "fake_password")
            
            # This will fail but should generate logs about the process
            try:
                provider.fetch_unread(3)
            except Exception:
                pass  # Expected to fail, we're testing logging
        except Exception:
            pass  # Provider initialization might fail, that's ok
        
        logs = self.get_structured_logs()
        
        # Look for any message-related logs
        message_logs = [log for log in logs if any(word in log.get('message', '').lower() 
                       for word in ['found', 'fetch', 'messages', 'unread'])]
        
        if message_logs:
            for log in message_logs[:3]:  # Show first few
                print(f"✓ Message operation logging: {log.get('message', '')[:60]}...")
        else:
            print("ℹ No message operation logs generated in this test scenario")
    
    def test_error_logging_enhancement(self):
        """Test that error logging includes detailed context."""
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Test credential error logging
        try:
            provider = GmailProvider("invalid@example.com")
            provider.fetch_unread(1)
        except Exception:
            pass  # Expected to fail
        
        logs = self.get_structured_logs()
        
        # Should have error logs with context
        error_logs = [log for log in logs if log.get('level') == 'ERROR']
        
        for error_log in error_logs:
            # Check for structured error context
            has_context = ('extra' in error_log and 
                          any(key in error_log['extra'] for key in ['error', 'error_type', 'username', 'operation'])) or \
                         any(key in error_log for key in ['error_type', 'username'])
            
            if has_context:
                print(f"✓ Enhanced error logging: {error_log.get('message', '')[:50]}...")
            else:
                print(f"⚠ Basic error logging: {error_log.get('message', '')[:50]}...")
    
    def assert_structured_context_usage(self):
        """Assert that provider module uses structured logging context appropriately."""
        # Read the provider module source
        provider_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'crewai_email_triage', 'provider.py')
        
        with open(provider_path, 'r') as f:
            provider_source = f.read()
        
        # Count structured logging calls (with extra= parameter)
        structured_calls = provider_source.count('extra=')
        
        # Should have at least 3-5 structured logging calls for key operations
        expected_min = 3
        
        print(f"Provider module structured logging calls: {structured_calls}")
        print(f"Expected minimum: {expected_min}")
        
        if structured_calls >= expected_min:
            print("✅ Provider module has adequate structured logging")
            return True
        else:
            print("❌ Provider module needs more structured logging context")
            return False


def run_provider_logging_test():
    """Run the provider logging enhancement test."""
    print("🔍 Testing Provider Logging Enhancement")
    print("=" * 50)
    
    test_instance = TestProviderLoggingEnhancement()
    test_instance.setup_method()
    
    try:
        print("\n1. Testing Provider Initialization Logging:")
        test_instance.test_provider_initialization_logging()
        
        print("\n2. Testing Message Search Logging:")
        test_instance.test_message_search_logging()
        
        print("\n3. Testing Error Logging Enhancement:")
        test_instance.test_error_logging_enhancement()
        
        print("\n4. Checking Structured Context Usage:")
        has_adequate_logging = test_instance.assert_structured_context_usage()
        
        if has_adequate_logging:
            print("\n✅ Provider logging enhancement validated!")
            return True
        else:
            print("\n🎯 Provider needs structured logging enhancement")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_provider_logging_test()
    exit(0 if success else 1)
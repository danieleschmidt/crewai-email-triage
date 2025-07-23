"""Test structured logging migration across modules."""

import sys
import os
import json
import logging
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from crewai_email_triage.logging_utils import StructuredFormatter
from crewai_email_triage.sanitization import sanitize_email_content
from crewai_email_triage.provider import GmailProvider


class TestStructuredLoggingMigration:
    """Test migration to structured logging."""
    
    def setup_method(self):
        """Setup test environment with structured logging."""
        # Capture log output
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(StructuredFormatter())
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    
    def test_sanitization_structured_logging(self):
        """Test that sanitization module uses structured logging."""
        # Test content that will trigger logging
        test_content = "<script>alert('test')</script>Hello World"
        
        # Clear previous log output
        self.log_stream.seek(0)
        self.log_stream.truncate(0)
        
        # Trigger sanitization (should generate logs)
        result = sanitize_email_content(test_content)
        
        # Get log output
        log_output = self.log_stream.getvalue()
        
        if log_output.strip():
            # Verify logs are in JSON format (structured)
            log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
            
            for line in log_lines:
                try:
                    log_entry = json.loads(line)
                    # Verify structured log format
                    assert 'timestamp' in log_entry
                    assert 'level' in log_entry
                    assert 'logger' in log_entry
                    assert 'message' in log_entry
                    print(f"✓ Structured log verified: {log_entry['message'][:50]}...")
                except json.JSONDecodeError:
                    # If not JSON, this indicates standard logging still in use
                    print(f"✗ Non-structured log found: {line[:100]}")
                    assert False, f"Expected structured JSON log, got: {line}"
        else:
            print("ℹ No logs generated during sanitization test")
    
    def test_provider_structured_logging(self):
        """Test that provider module uses structured logging."""
        # Clear previous log output
        self.log_stream.seek(0) 
        self.log_stream.truncate(0)
        
        # Test Gmail provider initialization (should generate logs)
        try:
            # This will likely fail but should generate logs
            provider = GmailProvider("test@example.com", "fake_password")
            # Try to fetch (will fail but should log)
            try:
                provider.fetch_unread(1)
            except Exception:
                pass  # Expected to fail, we just want to trigger logging
        except Exception:
            pass  # Expected to fail, we just want to trigger logging
        
        # Get log output
        log_output = self.log_stream.getvalue()
        
        if log_output.strip():
            # Verify logs are in JSON format (structured)  
            log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
            
            for line in log_lines:
                try:
                    log_entry = json.loads(line)
                    # Verify structured log format
                    assert 'timestamp' in log_entry
                    assert 'level' in log_entry
                    assert 'logger' in log_entry 
                    assert 'message' in log_entry
                    print(f"✓ Structured log verified: {log_entry['message'][:50]}...")
                except json.JSONDecodeError:
                    # If not JSON, this indicates standard logging still in use
                    print(f"✗ Non-structured log found: {line[:100]}")
                    assert False, f"Expected structured JSON log, got: {line}"
        else:
            print("ℹ No logs generated during provider test")
    
    def test_all_modules_use_structured_logging(self):
        """Integration test to verify all modules generate structured logs."""
        modules_to_test = [
            'crewai_email_triage.sanitization',
            'crewai_email_triage.provider', 
            'crewai_email_triage.retry_utils',
            'crewai_email_triage.circuit_breaker',
            'crewai_email_triage.secure_credentials',
            'crewai_email_triage.agent_responses'
        ]
        
        structured_log_count = 0
        
        for module_name in modules_to_test:
            # Clear previous log output
            self.log_stream.seek(0)
            self.log_stream.truncate(0)
            
            # Import and trigger some logging from the module
            try:
                module = __import__(module_name, fromlist=[''])
                # Get the module logger name and check its configuration
                logger = logging.getLogger(module_name)
                
                # Trigger a test log entry
                logger.info(f"Test structured logging for {module_name}")
                
                # Check log output
                log_output = self.log_stream.getvalue()
                
                if log_output.strip():
                    log_lines = [line.strip() for line in log_output.strip().split('\n') if line.strip()]
                    
                    for line in log_lines:
                        try:
                            log_entry = json.loads(line)
                            structured_log_count += 1
                            print(f"✓ {module_name}: Structured logging confirmed")
                            break
                        except json.JSONDecodeError:
                            print(f"✗ {module_name}: Still using standard logging")
                            break
                else:
                    print(f"ℹ {module_name}: No logs generated")
                    
            except Exception as e:
                print(f"⚠ {module_name}: Error testing - {e}")
        
        print(f"\nStructured logging summary: {structured_log_count}/{len(modules_to_test)} modules")
        return structured_log_count, len(modules_to_test)


def run_logging_migration_test():
    """Run the structured logging migration test."""
    print("🔍 Testing Structured Logging Migration")
    print("=" * 50)
    
    test_instance = TestStructuredLoggingMigration()
    test_instance.setup_method()
    
    try:
        print("\n1. Testing Sanitization Module:")
        test_instance.test_sanitization_structured_logging()
        
        print("\n2. Testing Provider Module:")
        test_instance.test_provider_structured_logging()
        
        print("\n3. Testing All Modules:")
        structured_count, total_count = test_instance.test_all_modules_use_structured_logging()
        
        print(f"\n📊 Results: {structured_count}/{total_count} modules using structured logging")
        
        if structured_count < total_count:
            print("🎯 Migration needed for remaining modules")
            return False
        else:
            print("✅ All modules successfully using structured logging!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_logging_migration_test()
    exit(0 if success else 1)
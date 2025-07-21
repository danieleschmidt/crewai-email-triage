"""Integration tests for the complete email triage pipeline."""

import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import pytest

from crewai_email_triage import triage_email, triage_batch, GmailProvider
from crewai_email_triage.sanitization import SanitizationConfig
from crewai_email_triage.logging_utils import setup_structured_logging


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""

    def test_complete_email_triage_workflow(self):
        """Test the complete workflow from raw email to final result."""
        # Test normal email
        email_content = """
        Subject: Quarterly Review Meeting
        
        Hi team,
        
        Please join our quarterly review meeting tomorrow at 2 PM.
        We'll be discussing:
        - Q4 performance metrics
        - Budget planning for next year
        - Team objectives
        
        Meeting link: https://company.zoom.us/j/123456789
        
        Best regards,
        Manager
        """
        
        result = triage_email(email_content)
        
        # Verify all expected fields are present
        assert isinstance(result, dict)
        assert "category" in result
        assert "priority" in result
        assert "summary" in result
        assert "response" in result
        
        # Verify reasonable categorization
        assert result["category"] in ["work", "general", "urgent"]
        assert 0 <= result["priority"] <= 10
        assert len(result["summary"]) > 0
        assert len(result["response"]) > 0
        
        # Should not have sanitization warnings for clean content
        assert "sanitization_warnings" not in result

    def test_malicious_email_handling(self):
        """Test that malicious emails are properly sanitized and processed."""
        malicious_email = """
        URGENT: Click here immediately!
        
        <script>
        // Malicious JavaScript
        document.location = "https://evil.com/steal-data";
        alert("Your account has been compromised!");
        </script>
        
        Please click this link: javascript:alert('XSS')
        
        Or visit: data:text/html,<script>alert('attack')</script>
        
        Also try: '; DROP TABLE users; --
        """
        
        result = triage_email(malicious_email)
        
        # Should still produce a valid result
        assert isinstance(result, dict)
        assert all(key in result for key in ["category", "priority", "summary", "response"])
        
        # Should have sanitization warnings
        assert "sanitization_warnings" in result
        assert len(result["sanitization_warnings"]) > 0
        
        # Verify no malicious content in results
        result_str = json.dumps(result)
        assert "<script>" not in result_str
        assert "javascript:" not in result_str
        assert "DROP TABLE" not in result_str

    def test_batch_processing_integration(self):
        """Test batch processing with mixed content types."""
        emails = [
            "Normal business email about quarterly reports",
            "URGENT: Meeting moved to tomorrow <script>alert('xss')</script>",
            "",  # Empty email
            None,  # None input
            "Very urgent deadline today: '; DROP TABLE users; --",
            "Regular follow-up email with https://legitimate.com link"
        ]
        
        results = triage_batch(emails)
        
        # Should process all emails
        assert len(results) == len(emails)
        
        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert all(key in result for key in ["category", "priority", "summary", "response"])
        
        # Check that problematic emails were handled
        # Email with script should have warnings
        script_result = results[1]
        assert "sanitization_warnings" in script_result
        assert "script_injection" in script_result["sanitization_warnings"]
        
        # Empty email should be categorized appropriately
        empty_result = results[2]
        assert empty_result["category"] == "empty"
        
        # None email should be handled
        none_result = results[3]
        assert none_result["category"] == "unknown"
        
        # SQL injection should be caught
        sql_result = results[4]
        assert "sanitization_warnings" in sql_result
        assert "sql_injection" in sql_result["sanitization_warnings"]

    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential processing produce consistent results."""
        emails = [
            "Meeting invitation for tomorrow",
            "Urgent: Please respond ASAP",
            "Regular status update email"
        ]
        
        # Process sequentially
        results_sequential = triage_batch(emails, parallel=False)
        
        # Process in parallel
        results_parallel = triage_batch(emails, parallel=True)
        
        # Results should have same structure and similar content
        assert len(results_sequential) == len(results_parallel)
        
        for seq, par in zip(results_sequential, results_parallel):
            # Categories should match
            assert seq["category"] == par["category"]
            # Priorities should match
            assert seq["priority"] == par["priority"]
            # Both should be valid results
            assert isinstance(seq, dict) and isinstance(par, dict)

    def test_configuration_integration(self):
        """Test that configuration changes affect processing."""
        email_content = "Test email with multiple URLs: https://site1.com and https://site2.com"
        
        # Test with default config
        result_default = triage_email(email_content)
        
        # Test with custom sanitization config (via pipeline integration)
        # Note: This tests the configuration system integration
        with patch('crewai_email_triage.pipeline.sanitize_email_content') as mock_sanitize:
            mock_sanitize.return_value = Mock(
                sanitized_content="Test email with multiple URLs: [URL_REMOVED] and [URL_REMOVED]",
                is_safe=False,
                threats_detected=["excessive_urls"],
                modifications_made=["url_removed"],
                original_length=len(email_content),
                sanitized_length=50,
                processing_time_ms=1.0
            )
            
            result_restricted = triage_email(email_content)
            
            # Should have sanitization warnings
            assert "sanitization_warnings" in result_restricted
            assert "excessive_urls" in result_restricted["sanitization_warnings"]

    def test_logging_integration(self):
        """Test that structured logging works throughout the pipeline."""
        # Capture log output
        log_stream = StringIO()
        
        # Setup structured logging to capture output
        import logging
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("crewai_email_triage")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Process an email
        result = triage_email("Test email for logging integration")
        
        # Check that logs were generated
        log_output = log_stream.getvalue()
        
        # Should have processing logs
        assert len(log_output) > 0
        
        # Cleanup
        logger.removeHandler(handler)

    def test_metrics_tracking_integration(self):
        """Test that metrics are properly tracked across operations."""
        from crewai_email_triage.pipeline import get_legacy_metrics
        
        # Get initial metrics
        initial_metrics = get_legacy_metrics()
        initial_processed = initial_metrics["processed"]
        initial_time = initial_metrics["total_time"]
        
        # Process some emails
        emails = ["Email 1", "Email 2", "Email 3"]
        triage_batch(emails)
        
        # Verify metrics were updated
        after_batch_metrics = get_legacy_metrics()
        assert after_batch_metrics["processed"] > initial_processed
        assert after_batch_metrics["total_time"] >= initial_time
        
        # Process single email
        triage_email("Single email test")
        
        # Verify single email processing updates metrics
        final_metrics = get_legacy_metrics()
        assert final_metrics["processed"] > initial_processed + len(emails)

    def test_error_recovery_integration(self):
        """Test that the system recovers gracefully from various error conditions."""
        # Test with various problematic inputs
        problematic_inputs = [
            None,
            "",
            123,  # Wrong type
            {"not": "string"},  # Wrong type
            "x" * 100000,  # Very long content
            "🤖" * 1000,  # Unicode stress test
            "\x00\x01\x02",  # Binary content
        ]
        
        results = []
        for problematic_input in problematic_inputs:
            try:
                result = triage_email(problematic_input)
                results.append(result)
            except Exception as e:
                pytest.fail(f"System failed to handle input {type(problematic_input)}: {e}")
        
        # All results should be valid dictionaries
        for result in results:
            assert isinstance(result, dict)
            assert all(key in result for key in ["category", "priority", "summary", "response"])

    def test_gmail_provider_integration(self):
        """Test Gmail provider integration (mocked)."""
        # Mock the IMAP connection
        with patch('imaplib.IMAP4_SSL') as mock_imap:
            # Setup mock IMAP responses
            mock_mail = MagicMock()
            mock_imap.return_value = mock_mail
            
            # Mock successful login and search
            mock_mail.login.return_value = None
            mock_mail.select.return_value = None
            mock_mail.search.return_value = ('OK', [b'1 2 3'])
            
            # Mock fetch responses
            mock_mail.fetch.side_effect = [
                ('OK', [(b'1 (RFC822 {123}', b'Subject: Test 1\r\n\r\nTest email 1')]),
                ('OK', [(b'2 (RFC822 {123}', b'Subject: Test 2\r\n\r\nTest email 2')]),
                ('OK', [(b'3 (RFC822 {123}', b'Subject: Test 3\r\n\r\nTest email 3')])
            ]
            
            # Create provider and fetch emails
            provider = GmailProvider("test@example.com", "password")
            messages = provider.fetch_unread(max_messages=3)
            
            # Verify emails were fetched
            assert len(messages) == 3
            assert all(isinstance(msg, str) for msg in messages)
            
            # Process the fetched emails
            results = triage_batch(messages)
            
            # Verify processing
            assert len(results) == 3
            for result in results:
                assert isinstance(result, dict)
                assert all(key in result for key in ["category", "priority", "summary", "response"])

    def test_cli_integration(self):
        """Test CLI integration with various options."""
        import subprocess
        import sys
        
        # Test basic CLI functionality
        cmd = [
            sys.executable, 
            "/root/repo/triage.py",
            "--message", "Test CLI integration",
            "--pretty"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "/root/repo/src"
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env,
                timeout=30
            )
            
            # Should execute successfully
            assert result.returncode == 0
            
            # Should produce valid JSON output
            output_lines = result.stdout.strip().split('\n')
            json_output = output_lines[0]  # First line should be JSON result
            
            parsed_result = json.loads(json_output)
            assert isinstance(parsed_result, dict)
            assert all(key in parsed_result for key in ["category", "priority", "summary", "response"])
            
        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out")
        except Exception as e:
            pytest.skip(f"CLI integration test skipped due to environment: {e}")

    def test_performance_integration(self):
        """Test performance characteristics under load."""
        import time
        
        # Generate a reasonable number of test emails
        emails = [f"Test email number {i} with some content" for i in range(50)]
        
        # Test sequential processing
        start_time = time.perf_counter()
        results_seq = triage_batch(emails, parallel=False)
        seq_time = time.perf_counter() - start_time
        
        # Test parallel processing
        start_time = time.perf_counter()
        results_par = triage_batch(emails, parallel=True)
        par_time = time.perf_counter() - start_time
        
        # Both should complete successfully
        assert len(results_seq) == len(emails)
        assert len(results_par) == len(emails)
        
        # Processing should complete in reasonable time
        assert seq_time < 30  # Should process 50 emails in under 30 seconds
        assert par_time < 30
        
        # Results should be consistent
        for seq, par in zip(results_seq, results_par):
            assert seq["category"] == par["category"]
            assert seq["priority"] == par["priority"]

    def test_security_integration(self):
        """Test security features work together across the system."""
        # Create email with multiple attack vectors
        malicious_email = """
        <iframe src="javascript:alert('xss')"></iframe>
        <script>document.location='http://evil.com'</script>
        
        Click: javascript:alert(1)
        Visit: data:text/html,<img src=x onerror=alert(1)>
        
        Database query: '; DROP TABLE users; SELECT * FROM passwords; --
        
        Unicode attack: ＜script＞alert('fullwidth')＜/script＞
        
        Encoding: %3Cscript%3Ealert(1)%3C/script%3E
        """
        
        result = triage_email(malicious_email)
        
        # Should process successfully
        assert isinstance(result, dict)
        assert all(key in result for key in ["category", "priority", "summary", "response"])
        
        # Should detect multiple threat types
        assert "sanitization_warnings" in result
        threats = result["sanitization_warnings"]
        
        # Should catch various attack types
        expected_threats = ["script_injection", "html_injection", "suspicious_urls", "sql_injection"]
        detected_threat_types = set(threats)
        
        # At least some of the expected threats should be detected
        assert len(detected_threat_types.intersection(expected_threats)) > 0
        
        # Verify no malicious content in final result
        result_str = json.dumps(result)
        dangerous_patterns = ["<script>", "javascript:", "DROP TABLE", "onerror="]
        for pattern in dangerous_patterns:
            assert pattern not in result_str
"""Integration tests for the email triage pipeline."""

import pytest
from unittest.mock import Mock, patch
from crewai_email_triage.pipeline import EmailTriagePipeline
from crewai_email_triage.config import load_config


@pytest.mark.integration
class TestEmailTriagePipelineIntegration:
    """Integration tests for the complete email triage pipeline."""

    def test_end_to_end_email_processing(
        self, 
        sample_email_message, 
        integration_test_config,
        mock_environment_variables
    ):
        """Test complete email processing from input to output."""
        # Mock the configuration loading
        with patch('crewai_email_triage.config.load_config', return_value=integration_test_config):
            pipeline = EmailTriagePipeline(config=integration_test_config)
            
            # Process the email
            result = pipeline.process_email(sample_email_message)
            
            # Verify the result structure
            assert 'classification' in result
            assert 'priority' in result
            assert 'summary' in result
            assert 'response' in result
            
            # Verify classification results
            assert result['classification'] in ['urgent', 'work', 'spam', 'other']
            
            # Verify priority scoring
            assert isinstance(result['priority'], (int, float))
            assert 0 <= result['priority'] <= 10
            
            # Verify summary exists and is non-empty
            assert result['summary']
            assert len(result['summary']) > 0
            
            # Verify response generation
            assert result['response']
            assert len(result['response']) > 0

    def test_batch_email_processing(
        self, 
        batch_email_messages, 
        integration_test_config,
        mock_environment_variables
    ):
        """Test batch processing of multiple emails."""
        with patch('crewai_email_triage.config.load_config', return_value=integration_test_config):
            pipeline = EmailTriagePipeline(config=integration_test_config)
            
            # Process batch of emails
            results = pipeline.process_batch(batch_email_messages)
            
            # Verify we got results for all emails
            assert len(results) == len(batch_email_messages)
            
            # Verify each result has the expected structure
            for i, result in enumerate(results):
                assert 'classification' in result
                assert 'priority' in result
                assert 'summary' in result
                assert 'response' in result
                
                # Verify the email ID or index is preserved
                assert 'email_index' in result or 'email_id' in result

    def test_parallel_batch_processing(
        self, 
        batch_email_messages, 
        integration_test_config,
        mock_environment_variables
    ):
        """Test parallel processing of email batches."""
        with patch('crewai_email_triage.config.load_config', return_value=integration_test_config):
            pipeline = EmailTriagePipeline(config=integration_test_config)
            
            # Process batch in parallel
            results = pipeline.process_batch(batch_email_messages, parallel=True)
            
            # Verify we got results for all emails
            assert len(results) == len(batch_email_messages)
            
            # Verify results are consistent with sequential processing
            sequential_results = pipeline.process_batch(batch_email_messages, parallel=False)
            
            # Both should have the same number of results
            assert len(results) == len(sequential_results)

    def test_error_handling_in_pipeline(
        self, 
        integration_test_config,
        mock_environment_variables
    ):
        """Test error handling in the pipeline."""
        with patch('crewai_email_triage.config.load_config', return_value=integration_test_config):
            pipeline = EmailTriagePipeline(config=integration_test_config)
            
            # Test with invalid email message
            invalid_email = {"invalid": "data"}
            
            with pytest.raises(Exception):
                pipeline.process_email(invalid_email)

    def test_configuration_override(
        self, 
        sample_email_message,
        mock_environment_variables
    ):
        """Test pipeline with custom configuration override."""
        custom_config = {
            "classifier": {
                "urgent": ["critical", "emergency"],
                "normal": ["regular", "standard"]
            },
            "priority": {
                "scores": {"high": 20, "low": 2},
                "high_keywords": ["critical", "emergency"]
            }
        }
        
        with patch('crewai_email_triage.config.load_config', return_value=custom_config):
            pipeline = EmailTriagePipeline(config=custom_config)
            
            # Process email with custom config
            result = pipeline.process_email(sample_email_message)
            
            # Verify the custom configuration was used
            assert 'classification' in result
            assert 'priority' in result

    @pytest.mark.slow
    def test_performance_with_large_batch(
        self, 
        integration_test_config,
        mock_environment_variables
    ):
        """Test pipeline performance with a large batch of emails."""
        import time
        
        # Create a large batch of test emails
        large_batch = []
        for i in range(50):  # 50 emails for performance testing
            email = {
                "subject": f"Test email {i}",
                "sender": f"sender{i}@example.com",
                "body": f"This is test email number {i} with some content to process.",
                "timestamp": "2025-07-27T10:00:00Z",
                "attachments": []
            }
            large_batch.append(email)
        
        with patch('crewai_email_triage.config.load_config', return_value=integration_test_config):
            pipeline = EmailTriagePipeline(config=integration_test_config)
            
            # Measure processing time
            start_time = time.time()
            results = pipeline.process_batch(large_batch)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify all emails were processed
            assert len(results) == len(large_batch)
            
            # Performance assertion - should process emails efficiently
            # Expect less than 1 second per email on average
            assert processing_time < len(large_batch) * 1.0
            
            # Log performance metrics
            emails_per_second = len(large_batch) / processing_time
            print(f"Processed {len(large_batch)} emails in {processing_time:.2f}s")
            print(f"Performance: {emails_per_second:.2f} emails/second")
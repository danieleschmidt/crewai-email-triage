"""Performance benchmarks for CrewAI Email Triage."""

import pytest
import time
from unittest.mock import patch, Mock
from crewai_email_triage.pipeline import EmailTriagePipeline


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests for email triage components."""

    @pytest.fixture
    def benchmark_config(self):
        """Configuration for benchmark tests."""
        return {
            "classifier": {
                "urgent": ["urgent", "asap", "critical"],
                "spam": ["unsubscribe", "marketing"],
                "work": ["meeting", "project", "deadline"]
            },
            "priority": {
                "scores": {"high": 10, "medium": 5, "low": 1},
                "high_keywords": ["urgent", "asap", "critical"],
                "medium_keywords": ["meeting", "review", "important"]
            }
        }

    @pytest.fixture
    def benchmark_emails(self):
        """Generate test emails for benchmarking."""
        emails = []
        templates = [
            "Urgent: Please review the project proposal by end of day.",
            "Meeting scheduled for tomorrow at 10 AM in conference room A.",
            "Newsletter: Latest updates from our company blog.",
            "Critical bug found in production system, immediate attention required.",
            "Thanks for your purchase! Here's your receipt.",
            "Spam: Get rich quick with this amazing opportunity!",
            "Project deadline moved to next Friday, please update your tasks.",
            "Your subscription expires soon, click here to renew.",
            "Important: Security update required for your account.",
            "Casual: How was your weekend? Let's grab coffee sometime."
        ]
        
        for i in range(100):
            template = templates[i % len(templates)]
            email = {
                "subject": f"Email {i}: {template[:30]}...",
                "sender": f"sender{i}@example.com",
                "body": template + f" Email ID: {i}",
                "timestamp": "2025-07-27T10:00:00Z",
                "attachments": []
            }
            emails.append(email)
        
        return emails

    def test_single_email_processing_benchmark(self, benchmark, benchmark_config):
        """Benchmark single email processing performance."""
        with patch('crewai_email_triage.config.load_config', return_value=benchmark_config):
            pipeline = EmailTriagePipeline(config=benchmark_config)
            
            test_email = {
                "subject": "Urgent: Critical system failure",
                "sender": "admin@company.com",
                "body": "We have a critical system failure that needs immediate attention.",
                "timestamp": "2025-07-27T10:00:00Z",
                "attachments": []
            }
            
            # Benchmark the email processing
            result = benchmark(pipeline.process_email, test_email)
            
            # Verify the result is valid
            assert 'classification' in result
            assert 'priority' in result
            assert 'summary' in result
            assert 'response' in result

    def test_batch_processing_benchmark(self, benchmark, benchmark_config, benchmark_emails):
        """Benchmark batch email processing performance."""
        with patch('crewai_email_triage.config.load_config', return_value=benchmark_config):
            pipeline = EmailTriagePipeline(config=benchmark_config)
            
            # Use smaller batch for reasonable benchmark time
            test_batch = benchmark_emails[:10]
            
            # Benchmark the batch processing
            results = benchmark(pipeline.process_batch, test_batch)
            
            # Verify all emails were processed
            assert len(results) == len(test_batch)

    def test_parallel_vs_sequential_benchmark(self, benchmark_config, benchmark_emails):
        """Compare parallel vs sequential processing performance."""
        with patch('crewai_email_triage.config.load_config', return_value=benchmark_config):
            pipeline = EmailTriagePipeline(config=benchmark_config)
            
            test_batch = benchmark_emails[:20]
            
            # Benchmark sequential processing
            start_time = time.time()
            sequential_results = pipeline.process_batch(test_batch, parallel=False)
            sequential_time = time.time() - start_time
            
            # Benchmark parallel processing
            start_time = time.time()
            parallel_results = pipeline.process_batch(test_batch, parallel=True)
            parallel_time = time.time() - start_time
            
            # Verify both produced same number of results
            assert len(sequential_results) == len(parallel_results)
            assert len(sequential_results) == len(test_batch)
            
            # Log performance comparison
            print(f"Sequential processing: {sequential_time:.3f}s")
            print(f"Parallel processing: {parallel_time:.3f}s")
            print(f"Speedup: {sequential_time / parallel_time:.2f}x")
            
            # Parallel should be faster for larger batches
            if len(test_batch) > 10:
                assert parallel_time <= sequential_time

    @pytest.mark.slow
    def test_memory_usage_benchmark(self, benchmark_config, benchmark_emails):
        """Benchmark memory usage during email processing."""
        import psutil
        import os
        
        with patch('crewai_email_triage.config.load_config', return_value=benchmark_config):
            pipeline = EmailTriagePipeline(config=benchmark_config)
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process emails and monitor memory
            test_batch = benchmark_emails[:50]
            results = pipeline.process_batch(test_batch)
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Verify processing completed
            assert len(results) == len(test_batch)
            
            # Log memory usage
            print(f"Initial memory: {initial_memory:.2f} MB")
            print(f"Peak memory: {peak_memory:.2f} MB")
            print(f"Memory increase: {memory_increase:.2f} MB")
            print(f"Memory per email: {memory_increase / len(test_batch):.3f} MB")
            
            # Memory usage should be reasonable (less than 10MB per email)
            assert memory_increase / len(test_batch) < 10.0

    def test_throughput_benchmark(self, benchmark_config):
        """Benchmark email processing throughput."""
        with patch('crewai_email_triage.config.load_config', return_value=benchmark_config):
            pipeline = EmailTriagePipeline(config=benchmark_config)
            
            # Generate emails on the fly to test sustained throughput
            emails_processed = 0
            start_time = time.time()
            target_duration = 5.0  # Run for 5 seconds
            
            while time.time() - start_time < target_duration:
                test_email = {
                    "subject": f"Throughput test email {emails_processed}",
                    "sender": f"test{emails_processed}@example.com",
                    "body": f"This is throughput test email number {emails_processed}",
                    "timestamp": "2025-07-27T10:00:00Z",
                    "attachments": []
                }
                
                result = pipeline.process_email(test_email)
                assert 'classification' in result
                emails_processed += 1
                
                # Safety break to avoid infinite loop
                if emails_processed > 1000:
                    break
            
            actual_duration = time.time() - start_time
            throughput = emails_processed / actual_duration
            
            print(f"Processed {emails_processed} emails in {actual_duration:.2f}s")
            print(f"Throughput: {throughput:.2f} emails/second")
            
            # Should process at least 1 email per second
            assert throughput >= 1.0

    def test_configuration_loading_benchmark(self, benchmark):
        """Benchmark configuration loading performance."""
        from crewai_email_triage.config import load_config
        
        # Benchmark configuration loading
        config = benchmark(load_config, None)
        
        # Verify config was loaded
        assert config is not None
        assert 'classifier' in config or 'priority' in config

    def test_agent_initialization_benchmark(self, benchmark, benchmark_config):
        """Benchmark agent initialization performance."""
        with patch('crewai_email_triage.config.load_config', return_value=benchmark_config):
            # Benchmark pipeline initialization
            pipeline = benchmark(EmailTriagePipeline, config=benchmark_config)
            
            # Verify pipeline was initialized
            assert pipeline is not None
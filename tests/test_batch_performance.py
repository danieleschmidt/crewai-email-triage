"""Test batch processing performance and thread safety."""

from crewai_email_triage.pipeline import triage_batch, get_legacy_metrics


class TestBatchProcessing:
    """Test batch processing functionality and performance."""

    def test_empty_batch(self):
        """Test that empty batch is handled correctly."""
        result = triage_batch([])
        assert result == []

    def test_single_message_batch(self):
        """Test that single message batch works correctly."""
        result = triage_batch(["test message"])
        
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "category" in result[0]
        assert "priority" in result[0]
        assert "summary" in result[0]
        assert "response" in result[0]

    def test_multiple_messages_sequential(self):
        """Test sequential processing of multiple messages."""
        messages = [
            "urgent meeting tomorrow",
            "lunch invitation",
            "project status update"
        ]
        
        result = triage_batch(messages, parallel=False)
        
        assert len(result) == 3
        for item in result:
            assert isinstance(item, dict)
            assert all(key in item for key in ["category", "priority", "summary", "response"])

    def test_multiple_messages_parallel(self):
        """Test parallel processing of multiple messages."""
        messages = [
            "urgent meeting tomorrow",
            "lunch invitation", 
            "project status update",
            "system maintenance notice"
        ]
        
        result = triage_batch(messages, parallel=True)
        
        assert len(result) == 4
        for item in result:
            assert isinstance(item, dict)
            assert all(key in item for key in ["category", "priority", "summary", "response"])

    def test_parallel_vs_sequential_results_consistency(self):
        """Test that parallel and sequential processing produce consistent results."""
        messages = [
            "urgent meeting request",
            "quarterly report deadline"
        ]
        
        seq_result = triage_batch(messages, parallel=False)
        par_result = triage_batch(messages, parallel=True)
        
        assert len(seq_result) == len(par_result)
        
        # Results should be functionally equivalent (categories should match)
        for seq, par in zip(seq_result, par_result):
            assert seq["category"] == par["category"]
            assert seq["priority"] == par["priority"]

    def test_max_workers_parameter(self):
        """Test that max_workers parameter is respected."""
        messages = ["test"] * 5
        
        # Should not raise an exception
        result = triage_batch(messages, parallel=True, max_workers=2)
        assert len(result) == 5

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked during batch processing."""
        initial_metrics = get_legacy_metrics()
        initial_processed = initial_metrics["processed"]
        initial_time = initial_metrics["total_time"]
        
        messages = ["test message 1", "test message 2"]
        triage_batch(messages)
        
        final_metrics = get_legacy_metrics()
        assert final_metrics["processed"] > initial_processed
        assert final_metrics["total_time"] >= initial_time

    def test_iterable_input_handling(self):
        """Test that different iterable types work correctly."""
        # Test with list
        list_result = triage_batch(["message1", "message2"])
        assert len(list_result) == 2
        
        # Test with tuple
        tuple_result = triage_batch(("message1", "message2"))
        assert len(tuple_result) == 2
        
        # Test with generator
        def message_generator():
            yield "message1"
            yield "message2"
            
        gen_result = triage_batch(message_generator())
        assert len(gen_result) == 2

    def test_error_resilience_in_batch(self):
        """Test that errors in one message don't stop batch processing."""
        messages = [
            "normal message",
            None,  # This should be handled gracefully
            "",    # Empty message
            "another normal message"
        ]
        
        # Should not raise an exception
        result = triage_batch(messages)
        assert len(result) == 4
        
        # All results should be dictionaries with required keys
        for item in result:
            assert isinstance(item, dict)
            assert all(key in item for key in ["category", "priority", "summary", "response"])

    def test_performance_logging(self, caplog):
        """Test that performance metrics are logged."""
        messages = ["test message"] * 3
        
        with caplog.at_level("INFO"):
            triage_batch(messages)
        
        # Should log processing start and completion with metrics
        log_messages = [record.message for record in caplog.records]
        
        # Check for processing info
        processing_logs = [msg for msg in log_messages if "Processing" in msg and "messages" in msg]
        assert len(processing_logs) > 0
        
        # Check for completion info with timing
        completion_logs = [msg for msg in log_messages if "Processed" in msg and "msg/s" in msg]
        assert len(completion_logs) > 0

    def test_thread_safety_simulation(self):
        """Test that parallel processing doesn't cause race conditions."""
        messages = ["test message"] * 10
        
        # Run multiple times to catch potential race conditions
        for _ in range(3):
            result = triage_batch(messages, parallel=True, max_workers=4)
            
            assert len(result) == 10
            for item in result:
                assert isinstance(item, dict)
                assert all(key in item for key in ["category", "priority", "summary", "response"])
                # Ensure no None values from race conditions
                assert all(value is not None for value in item.values())
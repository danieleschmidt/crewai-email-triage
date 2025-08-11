"""Comprehensive tests for enhanced AI features."""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from crewai_email_triage.enhanced_pipeline import (
    EnhancedEmailPipeline,
    ProcessingConfig,
    ProcessingMode,
    ProcessingQuality,
    EnhancedTriageResult,
    process_email_enhanced,
    process_batch_enhanced
)
from crewai_email_triage.llm_pipeline import (
    AdvancedLLMPipeline,
    LLMResponse,
    ProcessingContext,
    ModelType
)
from crewai_email_triage.realtime_streaming import (
    RealTimeEventBus,
    StreamingEmailProcessor,
    StreamEvent,
    StreamEventType
)
from crewai_email_triage.intelligent_learning import (
    IntelligentLearningSystem,
    PatternLearner,
    PatternType,
    LearningRecord
)


class TestLLMPipeline:
    """Test the advanced LLM pipeline."""
    
    @pytest.fixture
    def llm_pipeline(self):
        """Create LLM pipeline for testing."""
        return AdvancedLLMPipeline()
    
    @pytest.mark.asyncio
    async def test_process_email_basic(self, llm_pipeline):
        """Test basic email processing."""
        content = "This is an urgent meeting request for tomorrow at 2 PM"
        
        result = await llm_pipeline.process_email(content)
        
        assert isinstance(result, LLMResponse)
        assert result.category in ["urgent", "meeting", "general"]
        assert 1 <= result.priority <= 10
        assert result.summary
        assert result.response_suggestion
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.processing_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_process_email_with_headers(self, llm_pipeline):
        """Test email processing with headers."""
        content = "Please review the attached invoice"
        headers = {
            "from": "billing@company.com",
            "subject": "Invoice #12345",
            "date": "2024-01-01"
        }
        
        result = await llm_pipeline.process_email(content, headers)
        
        assert isinstance(result, LLMResponse)
        assert result.category == "billing"
        assert result.priority >= 6  # Billing should be medium-high priority
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, llm_pipeline):
        """Test streaming email processing."""
        content = "This is a test email for streaming processing"
        
        chunks = []
        async for chunk in llm_pipeline.stream_process_email(content):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert any("Analyzing" in chunk for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, llm_pipeline):
        """Test performance optimization."""
        # Process several emails to build up learning buffer
        test_emails = [
            "Urgent: System down!",
            "Meeting invitation for next week",
            "Invoice payment due",
            "General inquiry about services",
            "Emergency maintenance required"
        ]
        
        for email in test_emails:
            await llm_pipeline.process_email(email)
        
        # Run optimization
        optimization_result = await llm_pipeline.optimize_performance()
        
        assert "status" in optimization_result
        assert optimization_result["status"] in ["optimized", "insufficient_data"]


class TestRealTimeStreaming:
    """Test the real-time streaming system."""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing."""
        return RealTimeEventBus()
    
    @pytest.fixture
    def streaming_processor(self):
        """Create streaming processor for testing."""
        return StreamingEmailProcessor()
    
    @pytest.mark.asyncio
    async def test_event_bus_subscription(self, event_bus):
        """Test event bus subscription and publishing."""
        received_events = []
        
        def event_handler(event: StreamEvent):
            received_events.append(event)
        
        # Subscribe to events
        subscription_id = event_bus.subscribe(
            {StreamEventType.EMAIL_RECEIVED},
            event_handler
        )
        
        # Publish event
        test_event = StreamEvent(
            event_type=StreamEventType.EMAIL_RECEIVED,
            data={"test": "data"}
        )
        
        await event_bus.publish(test_event)
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].event_type == StreamEventType.EMAIL_RECEIVED
        assert received_events[0].data == {"test": "data"}
        
        # Clean up
        event_bus.unsubscribe(subscription_id)
    
    @pytest.mark.asyncio
    async def test_streaming_email_processing(self, streaming_processor):
        """Test streaming email processing."""
        content = "Test email for streaming processing"
        session_id = "test_session_123"
        
        events = []
        async for event in streaming_processor.process_email_stream(content, session_id=session_id):
            events.append(event)
        
        # Check we got expected event types
        event_types = [event.event_type for event in events]
        
        assert StreamEventType.EMAIL_RECEIVED in event_types
        assert StreamEventType.PROCESSING_STARTED in event_types
        assert StreamEventType.PROCESSING_COMPLETE in event_types
        
        # Check session ID is consistent
        for event in events:
            assert event.session_id == session_id
    
    @pytest.mark.asyncio
    async def test_queue_processing(self, streaming_processor):
        """Test queued email processing."""
        await streaming_processor.start_processing()
        
        # Queue an email
        session_id = await streaming_processor.queue_email(
            "Test queued email",
            session_id="queued_session"
        )
        
        assert session_id == "queued_session"
        
        # Check session status
        status = streaming_processor.get_session_status(session_id)
        assert status is not None
        assert status["user_id"] is None


class TestIntelligentLearning:
    """Test the intelligent learning system."""
    
    @pytest.fixture
    def learning_system(self):
        """Create learning system for testing."""
        return IntelligentLearningSystem()
    
    @pytest.fixture
    def pattern_learner(self):
        """Create pattern learner for testing."""
        return PatternLearner(PatternType.CONTENT_CLASSIFICATION)
    
    def test_learning_record_accuracy(self):
        """Test learning record accuracy calculation."""
        # Exact match
        record1 = LearningRecord(
            timestamp=time.time(),
            pattern_type=PatternType.CONTENT_CLASSIFICATION,
            input_features={"content": "test"},
            predicted_output="urgent",
            actual_output="urgent",
            confidence=0.8
        )
        assert record1.calculate_accuracy() == 1.0
        
        # No match
        record2 = LearningRecord(
            timestamp=time.time(),
            pattern_type=PatternType.CONTENT_CLASSIFICATION,
            input_features={"content": "test"},
            predicted_output="urgent",
            actual_output="general",
            confidence=0.8
        )
        assert record2.calculate_accuracy() == 0.0
        
        # Numerical match
        record3 = LearningRecord(
            timestamp=time.time(),
            pattern_type=PatternType.URGENCY_DETECTION,
            input_features={"content": "test"},
            predicted_output=8,
            actual_output=7,
            confidence=0.8
        )
        assert 0.8 <= record3.calculate_accuracy() <= 1.0
    
    def test_pattern_learner_content_classification(self, pattern_learner):
        """Test content classification learning."""
        # Add training records
        records = [
            LearningRecord(
                timestamp=time.time(),
                pattern_type=PatternType.CONTENT_CLASSIFICATION,
                input_features={"content": "urgent meeting tomorrow"},
                predicted_output="meeting",
                actual_output="urgent",
                confidence=0.7
            ),
            LearningRecord(
                timestamp=time.time(),
                pattern_type=PatternType.CONTENT_CLASSIFICATION,
                input_features={"content": "please pay invoice immediately"},
                predicted_output="billing",
                actual_output="billing",
                confidence=0.9
            ),
            LearningRecord(
                timestamp=time.time(),
                pattern_type=PatternType.CONTENT_CLASSIFICATION,
                input_features={"content": "schedule meeting next week"},
                predicted_output="meeting",
                actual_output="meeting",
                confidence=0.8
            )
        ]
        
        for record in records:
            pattern_learner.add_record(record)
        
        # Test prediction
        prediction, confidence = pattern_learner.predict({"content": "urgent billing invoice"})
        
        assert prediction is not None
        assert 0.0 <= confidence <= 1.0
        
        # Check learning stats
        stats = pattern_learner.get_learning_stats()
        assert stats["records_processed"] == 3
        assert stats["patterns_learned"] >= 1
        assert stats["avg_accuracy"] > 0.0
    
    @pytest.mark.asyncio
    async def test_learning_from_email_processing(self, learning_system):
        """Test learning from email processing."""
        content = "Urgent: Payment required immediately"
        headers = {"from": "billing@company.com"}
        
        # Mock LLM response
        llm_response = LLMResponse(
            category="billing",
            priority=8,
            summary="Urgent payment request",
            response_suggestion="I'll process your payment request immediately",
            confidence_score=0.9
        )
        
        # Learn from processing
        await learning_system.learn_from_email_processing(
            content, headers, llm_response
        )
        
        # Get predictions
        predictions = await learning_system.get_enhanced_predictions(content, headers)
        
        assert "individual_predictions" in predictions
        assert "combined_insights" in predictions
        assert "system_confidence" in predictions
        assert "learning_recommendations" in predictions
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization(self, learning_system):
        """Test adaptive optimization."""
        # Add multiple learning records to trigger optimization
        for i in range(101):  # Trigger optimization at 100
            content = f"Test email {i}"
            llm_response = LLMResponse(
                category="general",
                priority=5,
                summary=f"Test email {i}",
                response_suggestion="Test response",
                confidence_score=0.7
            )
            
            await learning_system.learn_from_email_processing(content, None, llm_response)
        
        # Check that optimization was triggered
        stats = learning_system.get_comprehensive_stats()
        assert stats["global_metrics"]["adaptation_cycles"] > 0


class TestEnhancedPipeline:
    """Test the enhanced email processing pipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create enhanced pipeline for testing."""
        config = ProcessingConfig(
            mode=ProcessingMode.INTELLIGENT,
            quality=ProcessingQuality.BALANCED,
            enable_learning=True,
            enable_monitoring=True
        )
        return EnhancedEmailPipeline(config)
    
    @pytest.mark.asyncio
    async def test_standard_processing(self, pipeline):
        """Test standard processing mode."""
        config = ProcessingConfig(mode=ProcessingMode.STANDARD)
        
        result = await pipeline.process_email(
            "This is a test email",
            processing_config=config
        )
        
        assert isinstance(result, EnhancedTriageResult)
        assert result.processing_mode == ProcessingMode.STANDARD
        assert result.category in ["urgent", "meeting", "general", "error"]
        assert 1 <= result.priority <= 10
        assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_ai_enhanced_processing(self, pipeline):
        """Test AI-enhanced processing mode."""
        config = ProcessingConfig(mode=ProcessingMode.AI_ENHANCED)
        
        result = await pipeline.process_email(
            "Urgent meeting request for tomorrow",
            processing_config=config
        )
        
        assert isinstance(result, EnhancedTriageResult)
        assert result.processing_mode == ProcessingMode.AI_ENHANCED
        assert result.llm_response is not None
        assert result.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_intelligent_processing(self, pipeline):
        """Test full intelligent processing mode."""
        config = ProcessingConfig(mode=ProcessingMode.INTELLIGENT)
        
        result = await pipeline.process_email(
            "Please review the attached contract",
            headers={"from": "legal@company.com"},
            processing_config=config
        )
        
        assert isinstance(result, EnhancedTriageResult)
        assert result.processing_mode == ProcessingMode.INTELLIGENT
        assert result.llm_response is not None
        assert result.learning_applied is True
        assert result.intelligent_insights is not None
    
    @pytest.mark.asyncio
    async def test_research_processing(self, pipeline):
        """Test research processing mode."""
        config = ProcessingConfig(mode=ProcessingMode.RESEARCH)
        
        result = await pipeline.process_email(
            "This is a comprehensive test email with lots of content to analyze thoroughly",
            headers={"from": "research@university.edu", "subject": "Research Request"},
            processing_config=config
        )
        
        assert isinstance(result, EnhancedTriageResult)
        assert result.processing_mode == ProcessingMode.RESEARCH
        assert result.intelligent_insights is not None
        
        # Research mode should have detailed analysis
        research_analysis = result.intelligent_insights.get("research_analysis")
        if research_analysis:
            assert "content_analysis" in research_analysis
            assert "linguistic_features" in research_analysis
            assert "temporal_analysis" in research_analysis
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, pipeline):
        """Test batch processing."""
        emails = [
            {"content": "Urgent system alert"},
            {"content": "Meeting invitation"},
            {"content": "Invoice payment due"},
            {"content": "General inquiry"}
        ]
        
        results = await pipeline.process_batch(emails)
        
        assert len(results) == 4
        for result in results:
            assert isinstance(result, EnhancedTriageResult)
            assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_streaming_processing(self, pipeline):
        """Test streaming processing."""
        content = "Test email for streaming"
        
        events_and_results = []
        async for item in pipeline.stream_process_email(content):
            events_and_results.append(item)
        
        # Should have both events and final result
        assert len(events_and_results) > 0
        
        # Last item should be the final result
        final_result = events_and_results[-1]
        assert isinstance(final_result, EnhancedTriageResult)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, pipeline):
        """Test error handling in processing."""
        # Test with invalid input
        result = await pipeline.process_email("")
        
        assert isinstance(result, EnhancedTriageResult)
        assert result.category == "error"
        assert result.error_details is not None
        assert "error_type" in result.error_details
    
    def test_input_validation(self, pipeline):
        """Test input validation."""
        with pytest.raises(ValueError, match="Email content must be a non-empty string"):
            pipeline._validate_inputs(None, None)
        
        with pytest.raises(ValueError, match="Email content cannot be empty"):
            pipeline._validate_inputs("   ", None)
        
        with pytest.raises(ValueError, match="exceeds maximum size limit"):
            pipeline._validate_inputs("x" * 100001, None)
    
    def test_statistics_tracking(self, pipeline):
        """Test statistics tracking."""
        initial_stats = pipeline.stats.copy()
        
        # Stats should be initialized properly
        assert "total_processed" in initial_stats
        assert "successful_processes" in initial_stats
        assert "failed_processes" in initial_stats
        assert "avg_processing_time" in initial_stats
        assert "avg_confidence" in initial_stats
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience functions."""
        # Test single email processing
        result = await process_email_enhanced(
            "Test email for convenience function",
            mode=ProcessingMode.AI_ENHANCED,
            quality=ProcessingQuality.FAST
        )
        
        assert isinstance(result, EnhancedTriageResult)
        assert result.processing_mode == ProcessingMode.AI_ENHANCED
        assert result.processing_quality == ProcessingQuality.FAST
        
        # Test batch processing
        emails = [
            {"content": "Email 1"},
            {"content": "Email 2"}
        ]
        
        batch_results = await process_batch_enhanced(
            emails,
            mode=ProcessingMode.STANDARD,
            max_concurrent=2
        )
        
        assert len(batch_results) == 2
        for result in batch_results:
            assert isinstance(result, EnhancedTriageResult)
            assert result.processing_mode == ProcessingMode.STANDARD


class TestIntegration:
    """Integration tests for all enhanced AI features."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test full pipeline integration with all features."""
        # Create pipeline with full capabilities
        config = ProcessingConfig(
            mode=ProcessingMode.INTELLIGENT,
            quality=ProcessingQuality.QUALITY,
            enable_learning=True,
            enable_streaming=True,
            enable_monitoring=True
        )
        
        pipeline = EnhancedEmailPipeline(config)
        
        # Process email with all features
        content = "Urgent: Critical system failure requires immediate attention!"
        headers = {
            "from": "ops@company.com",
            "subject": "CRITICAL: System Down",
            "priority": "high"
        }
        
        result = await pipeline.process_email(
            content,
            headers,
            session_id="integration_test",
            user_id="test_user"
        )
        
        # Verify all features are working
        assert isinstance(result, EnhancedTriageResult)
        assert result.processing_mode == ProcessingMode.INTELLIGENT
        assert result.processing_quality == ProcessingQuality.QUALITY
        assert result.llm_response is not None
        assert result.intelligent_insights is not None
        assert result.learning_applied is True
        assert result.session_id == "integration_test"
        
        # Check comprehensive stats
        stats = pipeline.get_comprehensive_stats()
        assert "pipeline_stats" in stats
        assert "component_stats" in stats
        assert "system_health" in stats
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test performance under concurrent load."""
        config = ProcessingConfig(
            mode=ProcessingMode.AI_ENHANCED,
            max_concurrent_requests=5
        )
        
        pipeline = EnhancedEmailPipeline(config)
        
        # Create multiple concurrent requests
        emails = [f"Test email {i}" for i in range(20)]
        email_data = [{"content": email} for email in emails]
        
        start_time = time.time()
        results = await pipeline.process_batch(email_data, max_concurrent=5)
        processing_time = time.time() - start_time
        
        # Verify all emails were processed
        assert len(results) == 20
        
        # Verify reasonable processing time (should benefit from concurrency)
        assert processing_time < 60  # Should complete within 1 minute
        
        # Verify all results are valid
        for result in results:
            assert isinstance(result, EnhancedTriageResult)
            assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio 
    async def test_learning_adaptation(self):
        """Test learning system adaptation over time."""
        config = ProcessingConfig(
            mode=ProcessingMode.LEARNING,
            enable_learning=True,
            adaptive_learning=True
        )
        
        pipeline = EnhancedEmailPipeline(config)
        
        # Process emails with feedback to trigger learning
        test_cases = [
            {
                "content": "Urgent payment required now",
                "feedback": {"category": "billing", "priority": 9}
            },
            {
                "content": "Schedule meeting next week",
                "feedback": {"category": "meeting", "priority": 5}
            },
            {
                "content": "System maintenance tonight", 
                "feedback": {"category": "maintenance", "priority": 7}
            }
        ] * 10  # Repeat to build learning data
        
        results = []
        for test_case in test_cases:
            result = await pipeline.process_email(
                test_case["content"],
                feedback=test_case["feedback"]
            )
            results.append(result)
        
        # Verify learning was applied
        learning_applied_count = sum(1 for r in results if r.learning_applied)
        assert learning_applied_count > 0
        
        # Get learning system stats
        learning_system = pipeline.learning_system
        stats = learning_system.get_comprehensive_stats()
        
        assert stats["global_metrics"]["total_learning_events"] > 0
        assert len(stats["learner_statistics"]) > 0


@pytest.mark.asyncio
async def test_error_recovery_and_resilience():
    """Test error recovery and system resilience."""
    config = ProcessingConfig(
        mode=ProcessingMode.AI_ENHANCED,
        max_retries=2
    )
    
    pipeline = EnhancedEmailPipeline(config)
    
    # Test with various error conditions
    error_cases = [
        "",  # Empty content
        None,  # None content (will cause error in processing)
        "x" * 100001,  # Too large content
    ]
    
    for error_content in error_cases:
        try:
            if error_content is None:
                # This will raise an error before reaching process_email
                continue
            result = await pipeline.process_email(error_content)
            
            # Should get error result, not exception
            if error_content == "":
                assert result.category == "error"
                assert result.error_details is not None
        except Exception as e:
            # Some errors might still raise exceptions, which is acceptable
            assert isinstance(e, (ValueError, ValidationError))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
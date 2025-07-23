"""Test structured agent response functionality."""

import pytest
from dataclasses import asdict

from crewai_email_triage.agent_responses import (
    AgentResponse,
    ClassificationResponse,
    PriorityResponse, 
    SummaryResponse,
    ResponseGenerationResponse,
    parse_agent_response,
    AgentResponseError
)
from crewai_email_triage.classifier import ClassifierAgent
from crewai_email_triage.priority import PriorityAgent
from crewai_email_triage.summarizer import SummarizerAgent
from crewai_email_triage.response import ResponseAgent


class TestAgentResponseStructures:
    """Test agent response data structures."""

    def test_base_agent_response(self):
        """Test base AgentResponse functionality."""
        response = AgentResponse(
            agent_type="test",
            success=True,
            raw_output="test output",
            processing_time_ms=1.5
        )
        
        assert response.agent_type == "test"
        assert response.success is True
        assert response.raw_output == "test output"
        assert response.processing_time_ms == 1.5
        assert response.error_message is None

    def test_agent_response_with_error(self):
        """Test AgentResponse with error condition."""
        response = AgentResponse(
            agent_type="test",
            success=False,
            raw_output="",
            processing_time_ms=0.5,
            error_message="Processing failed"
        )
        
        assert response.success is False
        assert response.error_message == "Processing failed"

    def test_classification_response(self):
        """Test ClassificationResponse structure."""
        response = ClassificationResponse(
            agent_type="classifier",
            success=True,
            raw_output="category: urgent",
            processing_time_ms=2.0,
            category="urgent",
            confidence=0.85,
            matched_keywords=["urgent", "asap"]
        )
        
        assert response.category == "urgent"
        assert response.confidence == 0.85
        assert response.matched_keywords == ["urgent", "asap"]
        
        # Test dict conversion
        data = asdict(response)
        assert data["category"] == "urgent"
        assert data["confidence"] == 0.85

    def test_priority_response(self):
        """Test PriorityResponse structure."""
        response = PriorityResponse(
            agent_type="priority",
            success=True,
            raw_output="priority: 8",
            processing_time_ms=1.8,
            priority_score=8,
            reasoning="Contains urgent keywords",
            factors=["urgent_keyword", "exclamation_mark"]
        )
        
        assert response.priority_score == 8
        assert response.reasoning == "Contains urgent keywords"
        assert response.factors == ["urgent_keyword", "exclamation_mark"]

    def test_summary_response(self):
        """Test SummaryResponse structure."""
        response = SummaryResponse(
            agent_type="summarizer",
            success=True,
            raw_output="summary: Meeting tomorrow",
            processing_time_ms=3.2,
            summary="Meeting tomorrow",
            key_points=["meeting", "tomorrow"],
            word_count=15
        )
        
        assert response.summary == "Meeting tomorrow"
        assert response.key_points == ["meeting", "tomorrow"]
        assert response.word_count == 15

    def test_response_generation_response(self):
        """Test ResponseGenerationResponse structure."""
        response = ResponseGenerationResponse(
            agent_type="responder",
            success=True,
            raw_output="response: Thanks for your email",
            processing_time_ms=2.5,
            response_text="Thanks for your email",
            response_type="acknowledgment",
            tone="professional"
        )
        
        assert response.response_text == "Thanks for your email"
        assert response.response_type == "acknowledgment"
        assert response.tone == "professional"

    def test_priority_score_validation(self):
        """Test priority score validation."""
        # Valid score
        response = PriorityResponse(
            agent_type="priority",
            success=True,
            raw_output="priority: 5",
            processing_time_ms=1.0,
            priority_score=5
        )
        assert response.priority_score == 5
        
        # Test with invalid scores in real usage would be handled by validation
        with pytest.raises(ValueError):
            PriorityResponse(
                agent_type="priority",
                success=True,
                raw_output="priority: 15",
                processing_time_ms=1.0,
                priority_score=15  # Out of range
            ).validate_priority_score()


class TestAgentResponseParsing:
    """Test parsing agent output to structured responses."""

    def test_parse_classification_response(self):
        """Test parsing classifier output."""
        raw_output = "category: urgent"
        response = parse_agent_response(raw_output, "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is True
        assert response.category == "urgent"
        assert response.raw_output == raw_output

    def test_parse_priority_response(self):
        """Test parsing priority agent output."""
        raw_output = "priority: 8"
        response = parse_agent_response(raw_output, "priority")
        
        assert isinstance(response, PriorityResponse)
        assert response.success is True
        assert response.priority_score == 8
        assert response.raw_output == raw_output

    def test_parse_summary_response(self):
        """Test parsing summarizer output."""
        raw_output = "summary: Team meeting scheduled for tomorrow at 10 AM"
        response = parse_agent_response(raw_output, "summarizer")
        
        assert isinstance(response, SummaryResponse)
        assert response.success is True
        assert response.summary == "Team meeting scheduled for tomorrow at 10 AM"
        assert response.raw_output == raw_output

    def test_parse_response_generation_response(self):
        """Test parsing response agent output."""
        raw_output = "response: Thank you for your email. I will review and get back to you."
        response = parse_agent_response(raw_output, "responder")
        
        assert isinstance(response, ResponseGenerationResponse)
        assert response.success is True
        assert response.response_text == "Thank you for your email. I will review and get back to you."
        assert response.raw_output == raw_output

    def test_parse_malformed_response(self):
        """Test parsing malformed agent output."""
        raw_output = "invalid output format"
        response = parse_agent_response(raw_output, "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is False
        assert response.category is None
        assert response.error_message is not None
        assert "Failed to parse" in response.error_message

    def test_parse_missing_prefix(self):
        """Test parsing output missing expected prefix."""
        raw_output = "urgent"  # Missing "category: " prefix
        response = parse_agent_response(raw_output, "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is False
        assert response.error_message is not None

    def test_parse_empty_output(self):
        """Test parsing empty agent output."""
        response = parse_agent_response("", "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is False
        assert response.error_message is not None

    def test_parse_none_output(self):
        """Test parsing None agent output."""
        response = parse_agent_response(None, "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is False
        assert response.error_message is not None

    def test_parse_invalid_priority_score(self):
        """Test parsing invalid priority score."""
        raw_output = "priority: invalid"
        response = parse_agent_response(raw_output, "priority")
        
        assert isinstance(response, PriorityResponse)
        assert response.success is False
        assert response.priority_score is None
        assert response.error_message is not None

    def test_parse_out_of_range_priority(self):
        """Test parsing out-of-range priority score."""
        raw_output = "priority: 15"
        response = parse_agent_response(raw_output, "priority")
        
        assert isinstance(response, PriorityResponse)
        assert response.success is False
        assert response.error_message is not None
        assert "out of range" in response.error_message.lower()

    def test_parse_unknown_agent_type(self):
        """Test parsing with unknown agent type."""
        with pytest.raises(AgentResponseError):
            parse_agent_response("test: value", "unknown_agent")

    def test_parse_with_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        raw_output = "  category:   urgent  "
        response = parse_agent_response(raw_output, "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is True
        assert response.category == "urgent"

    def test_parse_with_mixed_case(self):
        """Test parsing with mixed case."""
        raw_output = "Category: URGENT"
        response = parse_agent_response(raw_output, "classifier")
        
        assert isinstance(response, ClassificationResponse)
        assert response.success is True
        assert response.category == "urgent"  # Should be normalized to lowercase


class TestAgentIntegration:
    """Test integration with actual agent classes."""

    def test_classifier_agent_structured_response(self):
        """Test that ClassifierAgent can work with structured responses."""
        agent = ClassifierAgent()
        content = "Urgent meeting tomorrow!"
        
        # Test current behavior
        raw_result = agent.run(content)
        assert isinstance(raw_result, str)
        assert raw_result.startswith("category:")
        
        # Test structured parsing
        structured_result = parse_agent_response(raw_result, "classifier")
        assert isinstance(structured_result, ClassificationResponse)
        assert structured_result.success is True
        assert structured_result.category in ["urgent", "work", "general"]

    def test_priority_agent_structured_response(self):
        """Test that PriorityAgent can work with structured responses."""
        agent = PriorityAgent()
        content = "URGENT: Please respond immediately!"
        
        raw_result = agent.run(content)
        assert isinstance(raw_result, str)
        assert raw_result.startswith("priority:")
        
        structured_result = parse_agent_response(raw_result, "priority")
        assert isinstance(structured_result, PriorityResponse)
        assert structured_result.success is True
        assert 0 <= structured_result.priority_score <= 10

    def test_summarizer_agent_structured_response(self):
        """Test that SummarizerAgent can work with structured responses."""
        agent = SummarizerAgent()
        content = "Please join our quarterly review meeting tomorrow at 2 PM in conference room A."
        
        raw_result = agent.run(content)
        assert isinstance(raw_result, str)
        assert raw_result.startswith("summary:")
        
        structured_result = parse_agent_response(raw_result, "summarizer")
        assert isinstance(structured_result, SummaryResponse)
        assert structured_result.success is True
        assert len(structured_result.summary) > 0

    def test_response_agent_structured_response(self):
        """Test that ResponseAgent can work with structured responses."""
        agent = ResponseAgent()
        content = "Thank you for the meeting invitation."
        
        raw_result = agent.run(content)
        assert isinstance(raw_result, str)
        assert raw_result.startswith("response:")
        
        structured_result = parse_agent_response(raw_result, "responder")
        assert isinstance(structured_result, ResponseGenerationResponse)
        assert structured_result.success is True
        assert len(structured_result.response_text) > 0

    def test_error_handling_with_failing_agent(self):
        """Test error handling when agent fails."""
        # Mock an agent that returns invalid output
        class FailingAgent:
            def run(self, content):
                return "invalid format without prefix"
        
        agent = FailingAgent()
        raw_result = agent.run("test")
        
        structured_result = parse_agent_response(raw_result, "classifier")
        assert isinstance(structured_result, ClassificationResponse)
        assert structured_result.success is False
        assert structured_result.error_message is not None

    def test_backward_compatibility(self):
        """Test that existing string-based parsing still works."""
        # This ensures we can migrate gradually
        raw_outputs = [
            "category: urgent",
            "priority: 8",
            "summary: Meeting tomorrow",
            "response: Thanks for your email"
        ]
        
        for raw_output in raw_outputs:
            # Old string-based parsing
            if raw_output.startswith("category:"):
                old_value = raw_output.replace("category: ", "")
            elif raw_output.startswith("priority:"):
                old_value = raw_output.replace("priority: ", "")
            elif raw_output.startswith("summary:"):
                old_value = raw_output.replace("summary: ", "")
            elif raw_output.startswith("response:"):
                old_value = raw_output.replace("response: ", "")
            
            # New structured parsing
            agent_type = raw_output.split(":")[0]
            if agent_type == "category":
                structured = parse_agent_response(raw_output, "classifier")
                new_value = structured.category
            elif agent_type == "priority":
                structured = parse_agent_response(raw_output, "priority")
                new_value = str(structured.priority_score)
            elif agent_type == "summary":
                structured = parse_agent_response(raw_output, "summarizer")
                new_value = structured.summary
            elif agent_type == "response":
                structured = parse_agent_response(raw_output, "responder")
                new_value = structured.response_text
            
            # Values should match
            assert old_value == new_value, f"Mismatch for {raw_output}: {old_value} != {new_value}"


class TestResponseValidation:
    """Test response validation and constraints."""

    def test_category_validation(self):
        """Test category value validation."""
        valid_categories = ["urgent", "work", "general", "spam", "unknown"]
        
        for category in valid_categories:
            response = ClassificationResponse(
                agent_type="classifier",
                success=True,
                raw_output=f"category: {category}",
                processing_time_ms=1.0,
                category=category
            )
            assert response.category == category

    def test_summary_length_validation(self):
        """Test summary length constraints."""
        long_summary = "A" * 1000
        response = SummaryResponse(
            agent_type="summarizer",
            success=True,
            raw_output=f"summary: {long_summary}",
            processing_time_ms=1.0,
            summary=long_summary
        )
        
        # Should truncate or validate length
        assert len(response.summary) <= 500 or response.error_message is not None

    def test_response_text_validation(self):
        """Test response text validation."""
        response_text = "Thank you for your email. I will get back to you soon."
        response = ResponseGenerationResponse(
            agent_type="responder",
            success=True,
            raw_output=f"response: {response_text}",
            processing_time_ms=1.0,
            response_text=response_text
        )
        
        assert response.response_text == response_text
        assert len(response.response_text) > 0
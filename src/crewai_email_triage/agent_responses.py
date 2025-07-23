"""Structured response types for email triage agents."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

# Constants for consistent behavior across the module
MILLISECONDS_PER_SECOND = 1000  # Conversion factor for timing calculations
MAX_SUMMARY_LENGTH = 500        # Maximum length for summary truncation
MAX_RESPONSE_LENGTH = 1000      # Maximum length for response truncation


class AgentResponseError(Exception):
    """Exception raised for agent response parsing errors."""
    pass


@dataclass
class AgentResponse:
    """Base class for structured agent responses."""
    
    agent_type: str
    success: bool
    raw_output: str
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'agent_type': self.agent_type,
            'success': self.success,
            'raw_output': self.raw_output,
            'processing_time_ms': self.processing_time_ms,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }


@dataclass
class ClassificationResponse(AgentResponse):
    """Structured response from email classification agent."""
    
    category: Optional[str] = None
    confidence: Optional[float] = None
    matched_keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'category': self.category,
            'confidence': self.confidence,
            'matched_keywords': self.matched_keywords
        })
        return base_dict


@dataclass 
class PriorityResponse(AgentResponse):
    """Structured response from priority scoring agent."""
    
    priority_score: Optional[int] = None
    reasoning: Optional[str] = None
    factors: List[str] = field(default_factory=list)
    
    def validate_priority_score(self) -> None:
        """Validate priority score is in valid range."""
        if self.priority_score is not None and not (0 <= self.priority_score <= 10):
            raise ValueError(f"Priority score {self.priority_score} is out of range (0-10)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'priority_score': self.priority_score,
            'reasoning': self.reasoning,
            'factors': self.factors
        })
        return base_dict


@dataclass
class SummaryResponse(AgentResponse):
    """Structured response from email summarization agent."""
    
    summary: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    word_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'summary': self.summary,
            'key_points': self.key_points,
            'word_count': self.word_count
        })
        return base_dict


@dataclass
class ResponseGenerationResponse(AgentResponse):
    """Structured response from email response generation agent."""
    
    response_text: Optional[str] = None
    response_type: Optional[str] = None
    tone: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            'response_text': self.response_text,
            'response_type': self.response_type,
            'tone': self.tone
        })
        return base_dict


def parse_agent_response(raw_output: Optional[str], agent_type: str) -> AgentResponse:
    """Parse raw agent output into structured response.
    
    Parameters
    ----------
    raw_output : str, optional
        Raw output from agent
    agent_type : str
        Type of agent ('classifier', 'priority', 'summarizer', 'responder')
        
    Returns
    -------
    AgentResponse
        Structured response object
        
    Raises
    ------
    AgentResponseError
        If agent_type is unknown
    """
    start_time = time.perf_counter()
    
    # Handle None or empty input
    if not raw_output:
        error_msg = "Empty or None output from agent"
        return _create_error_response(agent_type, "", error_msg, start_time)
    
    # Clean up the output
    cleaned_output = raw_output.strip()
    
    try:
        if agent_type == "classifier":
            return _parse_classification_response(cleaned_output, start_time)
        elif agent_type == "priority":
            return _parse_priority_response(cleaned_output, start_time)
        elif agent_type == "summarizer":
            return _parse_summary_response(cleaned_output, start_time)
        elif agent_type == "responder":
            return _parse_response_generation_response(cleaned_output, start_time)
        else:
            raise AgentResponseError(f"Unknown agent type: {agent_type}")
            
    except Exception as e:
        error_msg = f"Failed to parse {agent_type} response: {str(e)}"
        logger.error(error_msg, extra={'raw_output': raw_output, 'agent_type': agent_type})
        return _create_error_response(agent_type, raw_output, error_msg, start_time)


def _create_error_response(agent_type: str, raw_output: str, error_msg: str, start_time: float) -> AgentResponse:
    """Create an error response for failed parsing."""
    processing_time = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
    
    if agent_type == "classifier":
        return ClassificationResponse(
            agent_type=agent_type,
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message=error_msg
        )
    elif agent_type == "priority":
        return PriorityResponse(
            agent_type=agent_type,
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message=error_msg
        )
    elif agent_type == "summarizer":
        return SummaryResponse(
            agent_type=agent_type,
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message=error_msg
        )
    elif agent_type == "responder":
        return ResponseGenerationResponse(
            agent_type=agent_type,
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message=error_msg
        )
    else:
        return AgentResponse(
            agent_type=agent_type,
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message=error_msg
        )


def _parse_classification_response(raw_output: str, start_time: float) -> ClassificationResponse:
    """Parse classification agent output."""
    processing_time = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
    
    # Handle case-insensitive matching
    match = re.match(r'^\s*category\s*:\s*(.+)\s*$', raw_output, re.IGNORECASE)
    
    if not match:
        return ClassificationResponse(
            agent_type="classifier",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message="Failed to parse category from output"
        )
    
    category = match.group(1).strip().lower()
    
    # Validate category
    valid_categories = {"urgent", "work", "general", "spam", "unknown", "classification_error", "empty"}
    if category not in valid_categories:
        # Allow any category but log warning
        logger.warning("Unknown category detected", extra={'category': category})
    
    return ClassificationResponse(
        agent_type="classifier",
        success=True,
        raw_output=raw_output,
        processing_time_ms=processing_time,
        category=category,
        confidence=0.8  # Default confidence
    )


def _parse_priority_response(raw_output: str, start_time: float) -> PriorityResponse:
    """Parse priority agent output."""
    processing_time = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
    
    # Handle case-insensitive matching
    match = re.match(r'^\s*priority\s*:\s*(.+)\s*$', raw_output, re.IGNORECASE)
    
    if not match:
        return PriorityResponse(
            agent_type="priority",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message="Failed to parse priority from output"
        )
    
    priority_str = match.group(1).strip()
    
    try:
        # Handle float input by converting to int
        priority_score = int(float(priority_str))
        
        # Clamp to valid range
        if priority_score < 0:
            priority_score = 0
            logger.warning("Priority score below 0, clamped to 0")
        elif priority_score > 10:
            priority_score = 10
            logger.warning("Priority score above 10, clamped to 10")
            
        return PriorityResponse(
            agent_type="priority",
            success=True,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            priority_score=priority_score,
            reasoning=f"Parsed score: {priority_str}"
        )
        
    except ValueError:
        return PriorityResponse(
            agent_type="priority",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message=f"Invalid priority score: '{priority_str}'"
        )


def _parse_summary_response(raw_output: str, start_time: float) -> SummaryResponse:
    """Parse summary agent output."""
    processing_time = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
    
    # Handle case-insensitive matching
    match = re.match(r'^\s*summary\s*:\s*(.+)\s*$', raw_output, re.IGNORECASE | re.DOTALL)
    
    if not match:
        return SummaryResponse(
            agent_type="summarizer",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message="Failed to parse summary from output"
        )
    
    summary = match.group(1).strip()
    
    # Handle empty summary
    if not summary:
        return SummaryResponse(
            agent_type="summarizer",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message="Empty summary content"
        )
    
    # Truncate if too long
    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH-3] + "..."
        logger.debug(f"Summary truncated to {MAX_SUMMARY_LENGTH} characters")
    
    # Calculate word count
    word_count = len(summary.split()) if summary else 0
    
    # Extract key points (simple word extraction)
    key_points = [word.lower() for word in summary.split() if len(word) > 3][:5]
    
    return SummaryResponse(
        agent_type="summarizer",
        success=True,
        raw_output=raw_output,
        processing_time_ms=processing_time,
        summary=summary,
        key_points=key_points,
        word_count=word_count
    )


def _parse_response_generation_response(raw_output: str, start_time: float) -> ResponseGenerationResponse:
    """Parse response generation agent output."""
    processing_time = (time.perf_counter() - start_time) * MILLISECONDS_PER_SECOND
    
    # Handle case-insensitive matching
    match = re.match(r'^\s*response\s*:\s*(.+)\s*$', raw_output, re.IGNORECASE | re.DOTALL)
    
    if not match:
        return ResponseGenerationResponse(
            agent_type="responder",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message="Failed to parse response from output"
        )
    
    response_text = match.group(1).strip()
    
    # Handle empty response
    if not response_text:
        return ResponseGenerationResponse(
            agent_type="responder",
            success=False,
            raw_output=raw_output,
            processing_time_ms=processing_time,
            error_message="Empty response content"
        )
    
    # Truncate if too long
    if len(response_text) > MAX_RESPONSE_LENGTH:
        response_text = response_text[:MAX_RESPONSE_LENGTH-3] + "..."
        logger.debug(f"Response truncated to {MAX_RESPONSE_LENGTH} characters")
    
    # Determine response type
    response_type = "acknowledgment"
    if any(word in response_text.lower() for word in ["question", "?"]):
        response_type = "question"
    elif any(word in response_text.lower() for word in ["meeting", "schedule", "calendar"]):
        response_type = "scheduling"
    elif any(word in response_text.lower() for word in ["thanks", "thank you"]):
        response_type = "acknowledgment"
    
    # Determine tone
    tone = "professional"
    if any(word in response_text.lower() for word in ["urgent", "immediately", "asap"]):
        tone = "urgent"
    elif any(word in response_text.lower() for word in ["please", "kindly", "appreciate"]):
        tone = "polite"
    
    return ResponseGenerationResponse(
        agent_type="responder",
        success=True,
        raw_output=raw_output,
        processing_time_ms=processing_time,
        response_text=response_text,
        response_type=response_type,
        tone=tone
    )


def extract_value_from_response(response: AgentResponse) -> Union[str, int, None]:
    """Extract the main value from a structured response.
    
    This provides backward compatibility with the old string replacement approach.
    """
    if not response.success:
        return None
        
    if isinstance(response, ClassificationResponse):
        return response.category
    elif isinstance(response, PriorityResponse):
        return response.priority_score
    elif isinstance(response, SummaryResponse):
        return response.summary
    elif isinstance(response, ResponseGenerationResponse):
        return response.response_text
    else:
        return None


def create_agent_response_wrapper(agent_run_func, agent_type: str):
    """Create a wrapper that converts agent output to structured responses.
    
    This can be used to gradually migrate agents to structured responses.
    """
    def wrapper(content):
        start_time = time.perf_counter()
        try:
            raw_output = agent_run_func(content)
            return parse_agent_response(raw_output, agent_type)
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            return _create_error_response(agent_type, "", error_msg, start_time)
    
    return wrapper
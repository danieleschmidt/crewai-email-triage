"""Enhanced Email Triage Pipeline with Advanced AI Integration.

This module integrates all next-generation AI capabilities into a cohesive,
robust pipeline with comprehensive error handling and monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .health import get_health_checker
from .intelligent_learning import get_learning_system, learn_from_processing
from .llm_pipeline import LLMResponse, get_llm_pipeline
from .logging_utils import LoggingContext
from .performance import get_performance_tracker
from .realtime_streaming import StreamEvent, StreamEventType, get_streaming_processor
from .retry_utils import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)


class ProcessingMode(str, Enum):
    """Processing modes for the enhanced pipeline."""

    STANDARD = "standard"           # Standard pipeline processing
    AI_ENHANCED = "ai_enhanced"     # AI-enhanced with LLM integration
    STREAMING = "streaming"         # Real-time streaming processing
    LEARNING = "learning"          # Learning-enabled processing
    INTELLIGENT = "intelligent"    # Full AI capabilities with learning
    RESEARCH = "research"          # Research mode with experimental features


class ProcessingQuality(str, Enum):
    """Quality levels for processing."""

    FAST = "fast"           # Optimized for speed
    BALANCED = "balanced"   # Balance of speed and quality
    QUALITY = "quality"     # Optimized for highest quality
    RESEARCH = "research"   # Maximum capabilities for research


@dataclass
class ProcessingConfig:
    """Configuration for enhanced processing."""

    mode: ProcessingMode = ProcessingMode.INTELLIGENT
    quality: ProcessingQuality = ProcessingQuality.BALANCED
    enable_learning: bool = True
    enable_streaming: bool = False
    enable_caching: bool = True
    enable_monitoring: bool = True

    # AI-specific settings
    llm_timeout: float = 30.0
    confidence_threshold: float = 0.7
    max_retries: int = 3

    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: float = 60.0

    # Learning settings
    collect_feedback: bool = True
    adaptive_learning: bool = True


class EnhancedTriageResult(BaseModel):
    """Enhanced triage result with comprehensive information."""

    # Core results
    category: str
    priority: int = Field(ge=1, le=10)
    summary: str
    response_suggestion: str
    confidence_score: float = Field(ge=0.0, le=1.0)

    # AI-enhanced fields
    llm_response: Optional[LLMResponse] = None
    intelligent_insights: Optional[Dict[str, Any]] = None
    learning_predictions: Optional[Dict[str, Any]] = None

    # Processing metadata
    processing_mode: ProcessingMode
    processing_quality: ProcessingQuality
    processing_time_ms: float
    model_used: Optional[str] = None

    # Health and performance
    health_status: str = "healthy"
    performance_metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None

    # Streaming information
    session_id: Optional[str] = None
    stream_events_count: Optional[int] = None

    # Learning information
    learning_applied: bool = False
    adaptation_suggestions: Optional[List[str]] = None

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            'category': self.category,
            'priority': self.priority,
            'summary': self.summary,
            'response': self.response_suggestion,
            'confidence': self.confidence_score
        }


class EnhancedEmailPipeline:
    """Enhanced email triage pipeline with all advanced capabilities."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

        # Initialize components
        self.llm_pipeline = get_llm_pipeline()
        self.streaming_processor = get_streaming_processor()
        self.learning_system = get_learning_system()
        self.performance_tracker = get_performance_tracker()
        self.health_checker = get_health_checker()

        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "avg_processing_time": 0.0,
            "avg_confidence": 0.0,
            "mode_usage": {mode.value: 0 for mode in ProcessingMode},
            "quality_usage": {quality.value: 0 for quality in ProcessingQuality}
        }

        # Concurrency control
        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Error tracking
        self.error_history: List[Dict[str, Any]] = []

        logger.info(f"EnhancedEmailPipeline initialized with mode: {self.config.mode.value}")

    @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=1.0))
    async def process_email(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None,
        processing_config: Optional[ProcessingConfig] = None
    ) -> EnhancedTriageResult:
        """Process email with enhanced AI capabilities."""

        # Use provided config or default
        config = processing_config or self.config

        # Acquire processing semaphore
        async with self.processing_semaphore:
            return await self._process_email_internal(
                email_content, email_headers, session_id, user_id, feedback, config
            )

    async def _process_email_internal(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        session_id: Optional[str],
        user_id: Optional[str],
        feedback: Optional[Dict[str, Any]],
        config: ProcessingConfig
    ) -> EnhancedTriageResult:
        """Internal processing logic with comprehensive error handling."""

        start_time = time.time()

        with LoggingContext(
            operation="enhanced_email_processing",
            session_id=session_id,
            user_id=user_id,
            mode=config.mode.value
        ):
            try:
                # Validate inputs
                self._validate_inputs(email_content, email_headers)

                # Check system health
                if config.enable_monitoring:
                    health_result = self.health_checker.check_health()
                    if health_result.status.name != "HEALTHY":
                        logger.warning(f"System health degraded: {health_result.status}")

                # Initialize result
                result = EnhancedTriageResult(
                    category="processing",
                    priority=5,
                    summary="Processing in progress...",
                    response_suggestion="",
                    confidence_score=0.0,
                    processing_mode=config.mode,
                    processing_quality=config.quality,
                    processing_time_ms=0.0,
                    session_id=session_id
                )

                # Process based on mode
                if config.mode == ProcessingMode.STANDARD:
                    result = await self._process_standard(email_content, email_headers, result, config)

                elif config.mode == ProcessingMode.AI_ENHANCED:
                    result = await self._process_ai_enhanced(email_content, email_headers, result, config)

                elif config.mode == ProcessingMode.STREAMING:
                    result = await self._process_streaming(
                        email_content, email_headers, result, config, session_id, user_id
                    )

                elif config.mode == ProcessingMode.LEARNING:
                    result = await self._process_with_learning(
                        email_content, email_headers, result, config, feedback
                    )

                elif config.mode == ProcessingMode.INTELLIGENT:
                    result = await self._process_intelligent(
                        email_content, email_headers, result, config, session_id, user_id, feedback
                    )

                elif config.mode == ProcessingMode.RESEARCH:
                    result = await self._process_research(
                        email_content, email_headers, result, config, session_id, user_id, feedback
                    )

                # Finalize processing
                processing_time = (time.time() - start_time) * 1000
                result.processing_time_ms = processing_time

                # Update statistics
                await self._update_statistics(result, config, True)

                # Track performance
                if config.enable_monitoring:
                    self.performance_tracker.record_operation(
                        f"enhanced_processing_{config.mode.value}",
                        processing_time / 1000,
                        {
                            "confidence": result.confidence_score,
                            "quality": config.quality.value,
                            "success": True
                        }
                    )

                logger.info(
                    f"Enhanced processing completed: {config.mode.value} "
                    f"({processing_time:.2f}ms, confidence: {result.confidence_score:.2f})"
                )

                return result

            except Exception as e:
                # Handle processing errors
                processing_time = (time.time() - start_time) * 1000
                error_result = await self._handle_processing_error(
                    e, email_content, config, processing_time, session_id
                )

                # Update error statistics
                await self._update_statistics(error_result, config, False)

                return error_result

    def _validate_inputs(self, email_content: str, email_headers: Optional[Dict[str, str]]) -> None:
        """Validate input parameters."""
        if not email_content or not isinstance(email_content, str):
            raise ValueError("Email content must be a non-empty string")

        if len(email_content.strip()) == 0:
            raise ValueError("Email content cannot be empty or whitespace only")

        if len(email_content) > 100000:  # 100KB limit
            raise ValueError("Email content exceeds maximum size limit (100KB)")

        if email_headers and not isinstance(email_headers, dict):
            raise ValueError("Email headers must be a dictionary")

    async def _process_standard(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult,
        config: ProcessingConfig
    ) -> EnhancedTriageResult:
        """Standard processing mode (legacy compatibility)."""

        # Use existing core processing logic
        from .core import process_email

        processed_content = process_email(email_content)

        # Parse simple result
        if processed_content.startswith("Processed: "):
            content = processed_content[11:]  # Remove "Processed: " prefix

            # Simple classification
            if any(word in content.lower() for word in ["urgent", "important", "asap"]):
                category = "urgent"
                priority = 8
            elif any(word in content.lower() for word in ["meeting", "schedule"]):
                category = "meeting"
                priority = 6
            else:
                category = "general"
                priority = 5

            result.category = category
            result.priority = priority
            result.summary = f"Email classified as {category}"
            result.response_suggestion = f"Acknowledge receipt of {category} email"
            result.confidence_score = 0.6  # Standard confidence

        return result

    async def _process_ai_enhanced(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult,
        config: ProcessingConfig
    ) -> EnhancedTriageResult:
        """AI-enhanced processing with LLM integration."""

        # Process with LLM pipeline
        llm_result = await self.llm_pipeline.process_email(
            email_content, email_headers, result.session_id
        )

        # Update result with LLM data
        result.category = llm_result.category
        result.priority = llm_result.priority
        result.summary = llm_result.summary
        result.response_suggestion = llm_result.response_suggestion
        result.confidence_score = llm_result.confidence_score
        result.llm_response = llm_result
        result.model_used = llm_result.model_used

        return result

    async def _process_streaming(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult,
        config: ProcessingConfig,
        session_id: Optional[str],
        user_id: Optional[str]
    ) -> EnhancedTriageResult:
        """Streaming processing with real-time updates."""

        events_count = 0

        # Process with streaming
        async for event in self.streaming_processor.process_email_stream(
            email_content, email_headers, session_id, user_id
        ):
            events_count += 1

            # Update result based on final event
            if event.event_type == StreamEventType.PROCESSING_COMPLETE:
                if "result" in event.data:
                    llm_data = event.data["result"]
                    result.category = llm_data.get("category", "general")
                    result.priority = llm_data.get("priority", 5)
                    result.summary = llm_data.get("summary", "Email processed via streaming")
                    result.response_suggestion = llm_data.get("response_suggestion", "")
                    result.confidence_score = llm_data.get("confidence_score", 0.7)
                    result.model_used = llm_data.get("model_used")

        result.stream_events_count = events_count

        return result

    async def _process_with_learning(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult,
        config: ProcessingConfig,
        feedback: Optional[Dict[str, Any]]
    ) -> EnhancedTriageResult:
        """Processing with learning capabilities."""

        # First, get AI-enhanced result
        result = await self._process_ai_enhanced(email_content, email_headers, result, config)

        # Get learning insights
        if result.llm_response:
            learning_insights = await self.learning_system.get_enhanced_predictions(
                email_content, email_headers
            )

            result.learning_predictions = learning_insights

            # Apply learning insights to improve results
            combined_insights = learning_insights.get("combined_insights", {})
            if combined_insights:
                # Update category if learning has high confidence
                if (combined_insights.get("recommended_category") and
                    combined_insights.get("category_confidence", 0) > result.confidence_score):
                    result.category = combined_insights["recommended_category"]

                # Update priority if learning suggests different priority
                if (combined_insights.get("recommended_priority") and
                    combined_insights.get("priority_confidence", 0) > 0.7):
                    result.priority = combined_insights["recommended_priority"]

                # Enhance confidence with learning insights
                learning_confidence = combined_insights.get("overall_confidence", 0)
                result.confidence_score = max(result.confidence_score, learning_confidence)

            # Learn from this processing
            await learn_from_processing(email_content, email_headers, result.llm_response, feedback)
            result.learning_applied = True

        return result

    async def _process_intelligent(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult,
        config: ProcessingConfig,
        session_id: Optional[str],
        user_id: Optional[str],
        feedback: Optional[Dict[str, Any]]
    ) -> EnhancedTriageResult:
        """Full intelligent processing with all capabilities."""

        # Combine all processing modes

        # Start with learning-enhanced processing
        result = await self._process_with_learning(email_content, email_headers, result, config, feedback)

        # Add intelligent insights
        if config.enable_learning:
            intelligent_insights = await self.learning_system.get_enhanced_predictions(
                email_content, email_headers, {"session_id": session_id, "user_id": user_id}
            )
            result.intelligent_insights = intelligent_insights

            # Generate adaptation suggestions
            result.adaptation_suggestions = intelligent_insights.get("learning_recommendations", [])

        # Add performance insights
        if config.enable_monitoring:
            performance_data = self.performance_tracker.get_performance_report()
            result.performance_metrics = {
                "current_load": len(asyncio.all_tasks()),
                "avg_response_time": performance_data.get("avg_response_time", 0),
                "system_health": self.health_checker.get_health_status()
            }

        return result

    async def _process_research(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult,
        config: ProcessingConfig,
        session_id: Optional[str],
        user_id: Optional[str],
        feedback: Optional[Dict[str, Any]]
    ) -> EnhancedTriageResult:
        """Research mode with experimental features and comprehensive analysis."""

        # Start with intelligent processing
        result = await self._process_intelligent(
            email_content, email_headers, result, config, session_id, user_id, feedback
        )

        # Add experimental research features
        research_data = await self._conduct_research_analysis(
            email_content, email_headers, result
        )

        # Enhance result with research insights
        if research_data:
            result.intelligent_insights = {
                **(result.intelligent_insights or {}),
                "research_analysis": research_data
            }

        # Generate comprehensive adaptation suggestions
        research_suggestions = await self._generate_research_recommendations(result)
        if research_suggestions:
            result.adaptation_suggestions = (result.adaptation_suggestions or []) + research_suggestions

        return result

    async def _conduct_research_analysis(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]],
        result: EnhancedTriageResult
    ) -> Dict[str, Any]:
        """Conduct comprehensive research analysis."""

        analysis = {
            "content_analysis": {
                "word_count": len(email_content.split()),
                "character_count": len(email_content),
                "sentence_count": email_content.count('.') + email_content.count('!') + email_content.count('?'),
                "paragraph_count": len([p for p in email_content.split('\n\n') if p.strip()]),
                "avg_word_length": sum(len(word) for word in email_content.split()) / len(email_content.split()) if email_content.split() else 0
            },
            "linguistic_features": {
                "exclamation_marks": email_content.count('!'),
                "question_marks": email_content.count('?'),
                "capital_letters": sum(1 for c in email_content if c.isupper()),
                "capitalization_ratio": sum(1 for c in email_content if c.isupper()) / len(email_content) if email_content else 0
            },
            "temporal_analysis": {
                "processing_timestamp": time.time(),
                "estimated_reading_time_seconds": len(email_content.split()) / 200 * 60,  # Average reading speed
                "complexity_score": min(10, len(email_content.split()) / 50)  # Simple complexity metric
            }
        }

        # Add header analysis if available
        if email_headers:
            analysis["header_analysis"] = {
                "has_sender": "from" in email_headers,
                "has_subject": "subject" in email_headers,
                "has_reply_to": "reply-to" in email_headers,
                "header_count": len(email_headers)
            }

        # Compare with historical patterns
        if result.learning_predictions:
            learning_data = result.learning_predictions
            analysis["pattern_comparison"] = {
                "matches_sender_pattern": bool(learning_data.get("sender_behavior")),
                "matches_content_pattern": bool(learning_data.get("content_classification")),
                "pattern_confidence": learning_data.get("combined_insights", {}).get("overall_confidence", 0)
            }

        return analysis

    async def _generate_research_recommendations(self, result: EnhancedTriageResult) -> List[str]:
        """Generate research-specific recommendations."""
        recommendations = []

        # Analyze processing quality
        if result.confidence_score < 0.8:
            recommendations.append(
                f"Low confidence score ({result.confidence_score:.2f}). Consider additional validation."
            )

        # Analyze learning effectiveness
        if result.learning_predictions:
            learning_confidence = result.learning_predictions.get("combined_insights", {}).get("overall_confidence", 0)
            if learning_confidence < 0.6:
                recommendations.append(
                    "Learning system has low confidence. Consider providing more training examples."
                )

        # Analyze processing time
        if result.processing_time_ms > 5000:  # 5 seconds
            recommendations.append(
                f"High processing time ({result.processing_time_ms:.0f}ms). Consider performance optimization."
            )

        # Analyze model performance
        if result.llm_response and result.llm_response.token_usage:
            token_usage = result.llm_response.token_usage
            if token_usage.get("total_tokens", 0) > 4000:
                recommendations.append(
                    "High token usage detected. Consider input optimization or model selection."
                )

        return recommendations

    async def _handle_processing_error(
        self,
        error: Exception,
        email_content: str,
        config: ProcessingConfig,
        processing_time: float,
        session_id: Optional[str]
    ) -> EnhancedTriageResult:
        """Handle processing errors gracefully."""

        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "processing_mode": config.mode.value,
            "content_length": len(email_content),
            "processing_time": processing_time
        }

        # Log error
        logger.error(f"Processing error: {error_details}", exc_info=True)

        # Store error for analysis
        self.error_history.append({
            "timestamp": time.time(),
            "session_id": session_id,
            **error_details
        })

        # Keep error history manageable
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

        # Create fallback result
        fallback_result = EnhancedTriageResult(
            category="error",
            priority=5,
            summary=f"Processing failed: {type(error).__name__}",
            response_suggestion="I apologize, but I was unable to process this email. Please try again or contact support.",
            confidence_score=0.0,
            processing_mode=config.mode,
            processing_quality=config.quality,
            processing_time_ms=processing_time,
            session_id=session_id,
            health_status="error",
            error_details=error_details
        )

        return fallback_result

    async def _update_statistics(
        self,
        result: EnhancedTriageResult,
        config: ProcessingConfig,
        success: bool
    ) -> None:
        """Update processing statistics."""

        self.stats["total_processed"] += 1

        if success:
            self.stats["successful_processes"] += 1

            # Update averages
            total_success = self.stats["successful_processes"]
            current_avg_time = self.stats["avg_processing_time"]
            current_avg_confidence = self.stats["avg_confidence"]

            self.stats["avg_processing_time"] = (
                (current_avg_time * (total_success - 1) + result.processing_time_ms) / total_success
            )

            self.stats["avg_confidence"] = (
                (current_avg_confidence * (total_success - 1) + result.confidence_score) / total_success
            )
        else:
            self.stats["failed_processes"] += 1

        # Update mode and quality usage
        self.stats["mode_usage"][config.mode.value] += 1
        self.stats["quality_usage"][config.quality.value] += 1

    async def process_batch(
        self,
        emails: List[Dict[str, Any]],
        processing_config: Optional[ProcessingConfig] = None,
        max_concurrent: Optional[int] = None
    ) -> List[EnhancedTriageResult]:
        """Process multiple emails with enhanced capabilities."""

        config = processing_config or self.config
        concurrent_limit = max_concurrent or self.config.max_concurrent_requests

        # Create semaphore for batch processing
        batch_semaphore = asyncio.Semaphore(concurrent_limit)

        async def process_single_email(email_data: Dict[str, Any]) -> EnhancedTriageResult:
            async with batch_semaphore:
                return await self.process_email(
                    email_content=email_data.get("content", ""),
                    email_headers=email_data.get("headers"),
                    session_id=email_data.get("session_id"),
                    user_id=email_data.get("user_id"),
                    feedback=email_data.get("feedback"),
                    processing_config=config
                )

        # Process all emails concurrently
        tasks = [process_single_email(email_data) for email_data in emails]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = EnhancedTriageResult(
                    category="batch_error",
                    priority=5,
                    summary=f"Batch processing error: {str(result)}",
                    response_suggestion="Email could not be processed in batch",
                    confidence_score=0.0,
                    processing_mode=config.mode,
                    processing_quality=config.quality,
                    processing_time_ms=0.0,
                    error_details={"batch_index": i, "error": str(result)}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        logger.info(f"Batch processing completed: {len(processed_results)} emails processed")

        return processed_results

    async def stream_process_email(
        self,
        email_content: str,
        email_headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[Union[StreamEvent, EnhancedTriageResult], None]:
        """Stream process email with enhanced pipeline."""

        # Use streaming mode
        streaming_config = ProcessingConfig(
            mode=ProcessingMode.STREAMING,
            enable_streaming=True,
            enable_monitoring=True
        )

        # Start processing
        processing_task = asyncio.create_task(
            self.process_email(email_content, email_headers, session_id, user_id, None, streaming_config)
        )

        # Stream events
        async for event in self.streaming_processor.process_email_stream(
            email_content, email_headers, session_id, user_id
        ):
            yield event

        # Wait for final result and yield it
        try:
            final_result = await processing_task
            yield final_result
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""

        # Get component stats
        llm_stats = self.llm_pipeline.get_performance_stats()
        streaming_stats = self.streaming_processor.get_processing_stats()
        learning_stats = self.learning_system.get_comprehensive_stats()

        return {
            "pipeline_stats": self.stats,
            "error_history": {
                "total_errors": len(self.error_history),
                "recent_errors": self.error_history[-10:] if self.error_history else [],
                "error_rate": (
                    self.stats["failed_processes"] / max(self.stats["total_processed"], 1)
                )
            },
            "component_stats": {
                "llm_pipeline": llm_stats,
                "streaming_processor": streaming_stats,
                "learning_system": learning_stats
            },
            "system_health": {
                "health_status": self.health_checker.get_health_status(),
                "performance_metrics": self.performance_tracker.get_performance_report(),
                "active_tasks": len(asyncio.all_tasks()),
                "semaphore_available": self.processing_semaphore._value
            },
            "recommendations": self._generate_pipeline_recommendations()
        }

    def _generate_pipeline_recommendations(self) -> List[str]:
        """Generate pipeline optimization recommendations."""
        recommendations = []

        # Check error rate
        error_rate = self.stats["failed_processes"] / max(self.stats["total_processed"], 1)
        if error_rate > 0.1:  # 10% error rate
            recommendations.append(
                f"High error rate ({error_rate:.1%}). Consider reviewing error patterns and improving error handling."
            )

        # Check processing performance
        avg_time = self.stats["avg_processing_time"]
        if avg_time > 5000:  # 5 seconds
            recommendations.append(
                f"High average processing time ({avg_time:.0f}ms). Consider performance optimization."
            )

        # Check confidence levels
        avg_confidence = self.stats["avg_confidence"]
        if avg_confidence < 0.7:
            recommendations.append(
                f"Low average confidence ({avg_confidence:.2f}). Consider model tuning or additional training."
            )

        # Check mode usage patterns
        mode_usage = self.stats["mode_usage"]
        total_usage = sum(mode_usage.values())
        if total_usage > 100:  # Only after significant usage
            intelligent_usage = mode_usage.get(ProcessingMode.INTELLIGENT.value, 0)
            if intelligent_usage / total_usage < 0.5:
                recommendations.append(
                    "Consider using INTELLIGENT mode more frequently for better results."
                )

        return recommendations


# Global instance
_enhanced_pipeline: Optional[EnhancedEmailPipeline] = None


def get_enhanced_pipeline(config: Optional[ProcessingConfig] = None) -> EnhancedEmailPipeline:
    """Get the global enhanced pipeline instance."""
    global _enhanced_pipeline
    if _enhanced_pipeline is None:
        _enhanced_pipeline = EnhancedEmailPipeline(config)
    return _enhanced_pipeline


@asynccontextmanager
async def enhanced_processing_context(config: Optional[ProcessingConfig] = None):
    """Context manager for enhanced processing."""
    pipeline = get_enhanced_pipeline(config)
    try:
        yield pipeline
    finally:
        # Perform any necessary cleanup
        pass


# Convenience functions for easy integration
async def process_email_enhanced(
    email_content: str,
    email_headers: Optional[Dict[str, str]] = None,
    mode: ProcessingMode = ProcessingMode.INTELLIGENT,
    quality: ProcessingQuality = ProcessingQuality.BALANCED,
    enable_learning: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    feedback: Optional[Dict[str, Any]] = None
) -> EnhancedTriageResult:
    """Process email with enhanced capabilities - convenience function."""

    config = ProcessingConfig(
        mode=mode,
        quality=quality,
        enable_learning=enable_learning
    )

    pipeline = get_enhanced_pipeline(config)
    return await pipeline.process_email(
        email_content, email_headers, session_id, user_id, feedback, config
    )


async def process_batch_enhanced(
    emails: List[Dict[str, Any]],
    mode: ProcessingMode = ProcessingMode.INTELLIGENT,
    quality: ProcessingQuality = ProcessingQuality.BALANCED,
    max_concurrent: int = 10
) -> List[EnhancedTriageResult]:
    """Process batch of emails with enhanced capabilities - convenience function."""

    config = ProcessingConfig(
        mode=mode,
        quality=quality,
        max_concurrent_requests=max_concurrent
    )

    pipeline = get_enhanced_pipeline(config)
    return await pipeline.process_batch(emails, config, max_concurrent)

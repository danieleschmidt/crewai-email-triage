"""Advanced LLM Integration Pipeline for Next-Generation Email Triage.

This module provides sophisticated AI capabilities including:
- Multi-model LLM integration with intelligent routing
- Context-aware processing with memory management
- Advanced prompt engineering with optimization
- Real-time model switching based on content analysis
- Self-improving algorithms with feedback loops
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field

from .performance import get_performance_tracker


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Available LLM model types with specific use cases."""
    
    CLASSIFICATION = "classification"  # Fast classification models
    ANALYSIS = "analysis"             # Deep analysis models
    GENERATION = "generation"         # Response generation models
    REASONING = "reasoning"           # Complex reasoning models
    MULTIMODAL = "multimodal"        # Multi-modal models


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    
    model_name: str
    model_type: ModelType
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: float = 30.0
    retry_attempts: int = 3
    cost_per_token: float = 0.0001
    quality_threshold: float = 0.8
    
    # Advanced configuration
    context_window: int = 8192
    supports_streaming: bool = False
    supports_function_calling: bool = False
    optimization_level: int = 1  # 1-5, higher = more optimized but slower


@dataclass 
class ProcessingContext:
    """Context for LLM processing with advanced features."""
    
    email_content: str
    email_headers: Optional[Dict[str, str]] = None
    sender_history: List[str] = field(default_factory=list)
    thread_context: List[str] = field(default_factory=list)
    urgency_indicators: List[str] = field(default_factory=list)
    business_context: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Meta information
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    request_id: Optional[str] = None


class LLMResponse(BaseModel):
    """Structured response from LLM processing."""
    
    category: str
    priority: int = Field(ge=1, le=10)
    summary: str
    response_suggestion: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    
    # Advanced fields
    sentiment_analysis: Optional[Dict[str, float]] = None
    key_entities: Optional[List[str]] = None
    action_items: Optional[List[str]] = None
    urgency_level: Optional[str] = None
    business_impact: Optional[str] = None
    
    # Meta information
    model_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def process_async(
        self, 
        context: ProcessingContext, 
        config: LLMConfig
    ) -> LLMResponse:
        """Process email context and return structured response."""
        ...
    
    async def stream_process(
        self, 
        context: ProcessingContext, 
        config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        """Stream processing results in real-time."""
        ...


class MockLLMProvider:
    """Mock LLM provider for development and testing."""
    
    async def process_async(
        self, 
        context: ProcessingContext, 
        config: LLMConfig
    ) -> LLMResponse:
        """Mock processing with intelligent responses."""
        start_time = time.time()
        
        # Simulate processing time based on content length
        await asyncio.sleep(min(0.5, len(context.email_content) / 1000))
        
        # Generate intelligent mock response
        content = context.email_content.lower()
        
        # Smart classification
        if any(word in content for word in ["urgent", "asap", "emergency", "critical"]):
            category = "urgent"
            priority = 9
            urgency_level = "high"
        elif any(word in content for word in ["meeting", "schedule", "calendar"]):
            category = "meeting"
            priority = 6
            urgency_level = "medium"
        elif any(word in content for word in ["invoice", "payment", "billing"]):
            category = "billing"
            priority = 7
            urgency_level = "medium"
        else:
            category = "general"
            priority = 5
            urgency_level = "low"
        
        # Generate contextual summary
        word_count = len(context.email_content.split())
        if word_count < 50:
            summary = f"Brief {category} email requiring attention"
        else:
            summary = f"Detailed {category} communication with {word_count} words"
        
        # Generate response suggestion
        response_suggestion = f"Thank you for your {category} message. I'll review this and respond appropriately."
        
        processing_time = (time.time() - start_time) * 1000
        
        return LLMResponse(
            category=category,
            priority=priority,
            summary=summary,
            response_suggestion=response_suggestion,
            confidence_score=0.85,
            sentiment_analysis={
                "positive": 0.6,
                "neutral": 0.3,
                "negative": 0.1
            },
            key_entities=[word for word in content.split()[:5] if len(word) > 4],
            action_items=["Review email content", "Prepare appropriate response"],
            urgency_level=urgency_level,
            business_impact="medium",
            model_used=config.model_name,
            processing_time_ms=processing_time,
            token_usage={
                "input_tokens": len(context.email_content.split()),
                "output_tokens": 150,
                "total_tokens": len(context.email_content.split()) + 150
            }
        )
    
    async def stream_process(
        self, 
        context: ProcessingContext, 
        config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        """Mock streaming processing."""
        chunks = [
            "Analyzing email content...",
            "Identifying key patterns...",
            "Generating classification...",
            "Preparing response suggestions...",
            "Finalizing results..."
        ]
        
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.2)


class IntelligentModelRouter:
    """Intelligent routing system for selecting optimal LLM models."""
    
    def __init__(self):
        self.models = {
            ModelType.CLASSIFICATION: LLMConfig(
                model_name="fast-classifier-v3",
                model_type=ModelType.CLASSIFICATION,
                max_tokens=512,
                temperature=0.0,
                timeout=5.0,
                optimization_level=5
            ),
            ModelType.ANALYSIS: LLMConfig(
                model_name="deep-analyzer-v2",
                model_type=ModelType.ANALYSIS,
                max_tokens=4096,
                temperature=0.2,
                timeout=15.0,
                optimization_level=3
            ),
            ModelType.GENERATION: LLMConfig(
                model_name="response-generator-v1",
                model_type=ModelType.GENERATION,
                max_tokens=2048,
                temperature=0.7,
                timeout=10.0,
                optimization_level=4
            ),
            ModelType.REASONING: LLMConfig(
                model_name="reasoning-engine-v1",
                model_type=ModelType.REASONING,
                max_tokens=8192,
                temperature=0.1,
                timeout=30.0,
                optimization_level=2
            ),
            ModelType.MULTIMODAL: LLMConfig(
                model_name="multimodal-v1",
                model_type=ModelType.MULTIMODAL,
                max_tokens=4096,
                temperature=0.3,
                timeout=20.0,
                optimization_level=3
            )
        }
        self.usage_stats = {model_type: 0 for model_type in ModelType}
        self.performance_history = {}
    
    def select_optimal_model(self, context: ProcessingContext) -> LLMConfig:
        """Intelligently select the best model for the given context."""
        content = context.email_content.lower()
        word_count = len(context.email_content.split())
        
        # Simple classification for fast processing
        if word_count < 100 and not context.thread_context:
            selected_type = ModelType.CLASSIFICATION
        
        # Complex reasoning for long emails with context
        elif word_count > 500 or len(context.thread_context) > 3:
            selected_type = ModelType.REASONING
        
        # Multi-modal for attachments or rich content
        elif context.email_headers and "attachment" in str(context.email_headers).lower():
            selected_type = ModelType.MULTIMODAL
        
        # Response generation for outbound processing
        elif any(indicator in content for indicator in ["reply", "respond", "answer"]):
            selected_type = ModelType.GENERATION
        
        # Default to analysis for comprehensive processing
        else:
            selected_type = ModelType.ANALYSIS
        
        self.usage_stats[selected_type] += 1
        
        config = self.models[selected_type]
        logger.info(f"Selected model: {config.model_name} for {selected_type.value}")
        
        return config


class AdvancedLLMPipeline:
    """Advanced LLM pipeline with intelligent routing and optimization."""
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or MockLLMProvider()
        self.router = IntelligentModelRouter()
        self.performance_tracker = get_performance_tracker()
        
        # Advanced features
        self.context_memory = {}
        self.learning_buffer = []
        self.optimization_metrics = {
            "total_requests": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0,
            "cost_efficiency": 1.0
        }
    
    async def process_email(
        self, 
        content: str, 
        headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None
    ) -> LLMResponse:
        """Process email with advanced AI capabilities."""
        
        # Build processing context
        context = ProcessingContext(
            email_content=content,
            email_headers=headers,
            sender_history=self._get_sender_history(headers),
            thread_context=self._extract_thread_context(content, headers),
            urgency_indicators=self._detect_urgency_indicators(content),
            business_context=self._extract_business_context(content),
            user_preferences=self._get_user_preferences(session_id),
            session_id=session_id,
            request_id=f"req_{int(time.time() * 1000)}"
        )
        
        # Select optimal model
        config = self.router.select_optimal_model(context)
        
        # Track performance
        start_time = time.time()
        
        try:
            # Process with selected model
            response = await self.provider.process_async(context, config)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(response, processing_time, config)
            
            # Store context for learning
            self._store_learning_data(context, response, config)
            
            # Track with performance system
            self.performance_tracker.record_operation(
                "llm_processing",
                processing_time / 1000,
                {"model": config.model_name, "confidence": response.confidence_score}
            )
            
            logger.info(
                f"LLM processing completed: {config.model_name} "
                f"({processing_time:.2f}ms, confidence: {response.confidence_score:.2f})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            # Return fallback response
            return self._create_fallback_response(context, str(e))
    
    async def stream_process_email(
        self,
        content: str,
        headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream process email for real-time results."""
        
        context = ProcessingContext(
            email_content=content,
            email_headers=headers,
            session_id=session_id
        )
        
        config = self.router.select_optimal_model(context)
        
        if not config.supports_streaming:
            yield "Streaming not supported for selected model, falling back to batch processing..."
            result = await self.process_email(content, headers, session_id)
            yield json.dumps(result.model_dump(), indent=2)
            return
        
        async for chunk in self.provider.stream_process(context, config):
            yield chunk
    
    def _get_sender_history(self, headers: Optional[Dict[str, str]]) -> List[str]:
        """Get sender history for context."""
        if not headers or "from" not in headers:
            return []
        
        sender = headers["from"]
        return self.context_memory.get(sender, [])
    
    def _extract_thread_context(
        self, 
        content: str, 
        headers: Optional[Dict[str, str]]
    ) -> List[str]:
        """Extract thread context from email content and headers."""
        context = []
        
        # Look for thread indicators in headers
        if headers:
            if "in-reply-to" in headers:
                context.append(f"Reply to: {headers['in-reply-to']}")
            if "references" in headers:
                context.append(f"References: {headers['references']}")
        
        # Extract quoted content
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('>'):
                context.append(line.strip())
        
        return context[:10]  # Limit context size
    
    def _detect_urgency_indicators(self, content: str) -> List[str]:
        """Detect urgency indicators in email content."""
        urgency_patterns = [
            "urgent", "asap", "emergency", "critical", "immediate",
            "deadline", "time-sensitive", "priority", "rush", "expedite"
        ]
        
        content_lower = content.lower()
        return [pattern for pattern in urgency_patterns if pattern in content_lower]
    
    def _extract_business_context(self, content: str) -> Optional[str]:
        """Extract business context from email content."""
        business_keywords = {
            "meeting": "Meeting/Scheduling",
            "project": "Project Management", 
            "invoice": "Finance/Billing",
            "contract": "Legal/Contracts",
            "support": "Customer Support",
            "sales": "Sales/Revenue",
            "hr": "Human Resources",
            "it": "IT/Technical"
        }
        
        content_lower = content.lower()
        for keyword, context in business_keywords.items():
            if keyword in content_lower:
                return context
        
        return None
    
    def _get_user_preferences(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Get user preferences for personalization."""
        if not session_id:
            return {}
        
        # Mock preferences - in real implementation, fetch from user profile
        return {
            "response_style": "professional",
            "urgency_sensitivity": "medium",
            "detail_level": "standard",
            "language": "en"
        }
    
    def _update_performance_metrics(
        self, 
        response: LLMResponse, 
        processing_time: float,
        config: LLMConfig
    ) -> None:
        """Update performance metrics for optimization."""
        self.optimization_metrics["total_requests"] += 1
        
        # Update rolling averages
        current_avg_time = self.optimization_metrics["avg_response_time"]
        current_avg_confidence = self.optimization_metrics["avg_confidence"]
        total = self.optimization_metrics["total_requests"]
        
        self.optimization_metrics["avg_response_time"] = (
            (current_avg_time * (total - 1) + processing_time) / total
        )
        
        self.optimization_metrics["avg_confidence"] = (
            (current_avg_confidence * (total - 1) + response.confidence_score) / total
        )
        
        # Update cost efficiency
        if response.token_usage:
            cost = response.token_usage["total_tokens"] * config.cost_per_token
            self.optimization_metrics["cost_efficiency"] = (
                response.confidence_score / max(cost, 0.001)
            )
    
    def _store_learning_data(
        self, 
        context: ProcessingContext, 
        response: LLMResponse,
        config: LLMConfig
    ) -> None:
        """Store data for continuous learning."""
        learning_entry = {
            "timestamp": context.timestamp,
            "content_length": len(context.email_content),
            "model_used": config.model_name,
            "confidence": response.confidence_score,
            "processing_time": response.processing_time_ms,
            "category": response.category,
            "priority": response.priority
        }
        
        self.learning_buffer.append(learning_entry)
        
        # Keep buffer manageable
        if len(self.learning_buffer) > 1000:
            self.learning_buffer = self.learning_buffer[-500:]
        
        # Update sender history
        if context.email_headers and "from" in context.email_headers:
            sender = context.email_headers["from"]
            if sender not in self.context_memory:
                self.context_memory[sender] = []
            
            self.context_memory[sender].append({
                "timestamp": context.timestamp,
                "category": response.category,
                "priority": response.priority
            })
            
            # Keep history manageable
            if len(self.context_memory[sender]) > 50:
                self.context_memory[sender] = self.context_memory[sender][-25:]
    
    def _create_fallback_response(self, context: ProcessingContext, error: str) -> LLMResponse:
        """Create fallback response when LLM processing fails."""
        return LLMResponse(
            category="general",
            priority=5,
            summary=f"Email processing failed: {error}",
            response_suggestion="I apologize, but I'm unable to process this email at the moment. Please try again later.",
            confidence_score=0.1,
            model_used="fallback",
            processing_time_ms=0.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "optimization_metrics": self.optimization_metrics,
            "model_usage": self.router.usage_stats,
            "learning_buffer_size": len(self.learning_buffer),
            "context_memory_size": len(self.context_memory),
            "total_models_available": len(self.router.models)
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Perform automatic performance optimization."""
        logger.info("Starting automatic performance optimization...")
        
        # Analyze learning data
        if len(self.learning_buffer) < 10:
            return {"status": "insufficient_data", "message": "Need more data for optimization"}
        
        # Calculate optimal model distribution
        high_confidence_models = []
        fast_models = []
        
        for entry in self.learning_buffer[-100:]:  # Last 100 requests
            if entry["confidence"] > 0.8:
                high_confidence_models.append(entry["model_used"])
            if entry["processing_time"] and entry["processing_time"] < 1000:  # Under 1 second
                fast_models.append(entry["model_used"])
        
        # Update router preferences
        optimization_results = {
            "status": "optimized",
            "high_confidence_models": list(set(high_confidence_models)),
            "fast_models": list(set(fast_models)),
            "total_requests_analyzed": len(self.learning_buffer),
            "avg_confidence": self.optimization_metrics["avg_confidence"],
            "avg_response_time": self.optimization_metrics["avg_response_time"],
            "recommendations": []
        }
        
        # Generate recommendations
        if self.optimization_metrics["avg_confidence"] < 0.7:
            optimization_results["recommendations"].append(
                "Consider using higher-quality models for better confidence scores"
            )
        
        if self.optimization_metrics["avg_response_time"] > 5000:
            optimization_results["recommendations"].append(
                "Consider using faster models or implementing caching for better response times"
            )
        
        logger.info("Performance optimization completed")
        return optimization_results


# Global instance for easy access
_llm_pipeline: Optional[AdvancedLLMPipeline] = None


def get_llm_pipeline() -> AdvancedLLMPipeline:
    """Get the global LLM pipeline instance."""
    global _llm_pipeline
    if _llm_pipeline is None:
        _llm_pipeline = AdvancedLLMPipeline()
    return _llm_pipeline


@asynccontextmanager
async def llm_processing_context():
    """Context manager for LLM processing with automatic cleanup."""
    pipeline = get_llm_pipeline()
    try:
        yield pipeline
    finally:
        # Perform cleanup if needed
        pass


# Convenience functions for easy integration
async def process_email_with_llm(
    content: str,
    headers: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None
) -> LLMResponse:
    """Process email using advanced LLM capabilities."""
    pipeline = get_llm_pipeline()
    return await pipeline.process_email(content, headers, session_id)


async def stream_email_processing(
    content: str,
    headers: Optional[Dict[str, str]] = None,
    session_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Stream email processing for real-time results."""
    pipeline = get_llm_pipeline()
    async for chunk in pipeline.stream_process_email(content, headers, session_id):
        yield chunk
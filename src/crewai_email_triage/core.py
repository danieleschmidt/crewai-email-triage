"""Core functionality for CrewAI Email Triage."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# Setup basic logging
logger = logging.getLogger(__name__)


@dataclass
class EmailContext:
    """Enhanced email context with intelligence tracking."""
    content: str
    processing_stage: str = "initial"
    confidence_score: float = 0.0
    language_detected: Optional[str] = None
    sentiment: Optional[str] = None
    urgency_indicators: List[str] = None
    keywords: List[str] = None
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.urgency_indicators is None:
            self.urgency_indicators = []
        if self.keywords is None:
            self.keywords = []


class IntelligentEmailProcessor:
    """Enhanced email processor with multi-dimensional analysis."""
    
    def __init__(self):
        self._urgency_keywords = {
            "critical": ["urgent", "asap", "emergency", "critical", "immediate", "rush"],
            "high": ["important", "priority", "deadline", "soon", "quick"],
            "medium": ["follow up", "reminder", "update", "meeting", "schedule"],
        }
        self._sentiment_indicators = {
            "positive": ["thank", "appreciate", "great", "excellent", "wonderful"],
            "negative": ["problem", "issue", "complaint", "error", "failed", "wrong"],
            "neutral": ["information", "update", "meeting", "schedule", "report"]
        }
    
    def analyze_email_intelligence(self, content: str) -> EmailContext:
        """Perform intelligent multi-dimensional analysis."""
        start_time = time.time()
        
        # Create email context
        context = EmailContext(content=content, processing_stage="analysis")
        
        # Detect language (simple heuristic)
        context.language_detected = self._detect_language(content)
        
        # Analyze urgency
        context.urgency_indicators = self._extract_urgency_indicators(content)
        
        # Extract keywords
        context.keywords = self._extract_keywords(content)
        
        # Sentiment analysis
        context.sentiment = self._analyze_sentiment(content)
        
        # Calculate confidence score
        context.confidence_score = self._calculate_confidence(context)
        
        # Update processing time
        context.processing_time_ms = (time.time() - start_time) * 1000
        context.processing_stage = "completed"
        
        logger.info(f"Email intelligence analysis completed: {context.confidence_score:.2f} confidence")
        return context
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection based on common patterns."""
        content_lower = content.lower()
        
        # Simple heuristics for common languages
        if any(word in content_lower for word in ["the", "and", "that", "have", "for"]):
            return "en"
        elif any(word in content_lower for word in ["de", "und", "das", "haben", "für"]):
            return "de"
        elif any(word in content_lower for word in ["le", "et", "que", "avoir", "pour"]):
            return "fr"
        elif any(word in content_lower for word in ["el", "y", "que", "tener", "para"]):
            return "es"
        else:
            return "unknown"
    
    def _extract_urgency_indicators(self, content: str) -> List[str]:
        """Extract urgency-related keywords."""
        content_lower = content.lower()
        indicators = []
        
        for level, keywords in self._urgency_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    indicators.append(f"{level}:{keyword}")
        
        return indicators
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract significant keywords using simple frequency analysis."""
        # Simple keyword extraction - in production, would use NLP libraries
        words = content.lower().split()
        # Filter common stop words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "a", "an"}
        keywords = [word.strip(".,!?;:") for word in words if len(word) > 3 and word not in stop_words]
        
        # Return most frequent unique keywords (simplified)
        return list(set(keywords))[:10]
    
    def _analyze_sentiment(self, content: str) -> str:
        """Basic sentiment analysis."""
        content_lower = content.lower()
        positive_count = sum(1 for word in self._sentiment_indicators["positive"] if word in content_lower)
        negative_count = sum(1 for word in self._sentiment_indicators["negative"] if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, context: EmailContext) -> float:
        """Calculate confidence score based on analysis results."""
        score = 0.0
        
        # Language detection confidence
        if context.language_detected != "unknown":
            score += 0.3
        
        # Urgency indicators found
        if context.urgency_indicators:
            score += 0.3
        
        # Keywords extracted
        if context.keywords:
            score += 0.2
        
        # Sentiment detected
        if context.sentiment != "neutral":
            score += 0.2
        
        return min(score, 1.0)


# Global processor instance
_processor = IntelligentEmailProcessor()


def process_email(content: str | None) -> str:
    """Process an email and return a simple acknowledgment string.
    
    Enhanced error handling and input validation for production use.

    Parameters
    ----------
    content: str | None
        The email content to process. If ``None``, returns an empty string.

    Returns
    -------
    str
        A simple acknowledgment string.
    """
    # Enhanced error handling
    if content is None:
        return ""

    if not isinstance(content, str):
        raise TypeError(f"Expected str or None, got {type(content)}")

    if len(content.strip()) == 0:
        return "Processed: [Empty message]"
    
    result = f"Processed: {content.strip()}"
    logger.info(f"Processed email content: {len(content)} chars")
    return result


def process_email_intelligent(content: str | None) -> Dict[str, any]:
    """Enhanced email processing with intelligence analysis.
    
    Parameters
    ----------
    content: str | None
        The email content to process.
    
    Returns
    -------
    Dict[str, any]
        Comprehensive analysis results including intelligence insights.
    """
    if content is None:
        return {"error": "No content provided", "processed": False}

    if not isinstance(content, str):
        return {"error": f"Expected str or None, got {type(content)}", "processed": False}

    if len(content.strip()) == 0:
        return {"processed": True, "message": "Empty content", "analysis": None}
    
    # Perform intelligent analysis
    analysis = _processor.analyze_email_intelligence(content.strip())
    
    result = {
        "processed": True,
        "message": f"Processed: {content.strip()[:100]}{'...' if len(content) > 100 else ''}",
        "analysis": {
            "language_detected": analysis.language_detected,
            "sentiment": analysis.sentiment,
            "urgency_indicators": analysis.urgency_indicators,
            "keywords": analysis.keywords[:5],  # Top 5 keywords
            "confidence_score": analysis.confidence_score,
            "processing_time_ms": analysis.processing_time_ms,
        }
    }
    
    logger.info(f"Intelligent email processing completed: {analysis.confidence_score:.2f} confidence, {len(content)} chars")
    return result

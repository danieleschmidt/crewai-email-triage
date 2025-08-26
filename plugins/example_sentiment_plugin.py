"""
Example Sentiment Analysis Plugin
Demonstrates how to create email processor plugins.
"""

import random
from typing import Any, Dict

from crewai_email_triage.plugin_architecture import EmailProcessorPlugin, PluginMetadata, PluginConfig


class SentimentAnalysisPlugin(EmailProcessorPlugin):
    """Example plugin that analyzes email sentiment."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="sentiment_analysis",
            version="1.0.0",
            description="Analyzes email sentiment and emotional tone",
            author="CrewAI Team",
            dependencies=[],
            min_api_version="1.0.0",
            max_api_version="2.0.0"
        )
    
    def initialize(self) -> bool:
        """Initialize sentiment analysis resources."""
        self.logger.info("Initializing sentiment analysis plugin")
        # In a real plugin, you might load ML models here
        self.sentiment_keywords = {
            'positive': ['great', 'excellent', 'fantastic', 'wonderful', 'amazing', 'love', 'perfect'],
            'negative': ['terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointed', 'angry'],
            'urgent': ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'deadline']
        }
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up sentiment analysis plugin")
    
    def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email sentiment."""
        try:
            content_lower = email_content.lower()
            
            # Simple keyword-based sentiment analysis
            positive_score = sum(1 for keyword in self.sentiment_keywords['positive'] 
                               if keyword in content_lower)
            negative_score = sum(1 for keyword in self.sentiment_keywords['negative'] 
                               if keyword in content_lower)
            urgency_score = sum(1 for keyword in self.sentiment_keywords['urgent'] 
                              if keyword in content_lower)
            
            # Calculate overall sentiment
            total_words = len(email_content.split())
            sentiment_score = (positive_score - negative_score) / max(total_words / 10, 1)
            
            # Determine sentiment category
            if sentiment_score > 0.1:
                sentiment = "positive"
            elif sentiment_score < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Determine emotional intensity
            intensity = min(abs(sentiment_score) * 10, 1.0)
            
            return {
                "sentiment": sentiment,
                "sentiment_score": round(sentiment_score, 3),
                "emotional_intensity": round(intensity, 3),
                "urgency_indicators": urgency_score,
                "positive_keywords": positive_score,
                "negative_keywords": negative_score,
                "analysis_confidence": min(0.95, 0.5 + (positive_score + negative_score) * 0.1)
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                "sentiment": "unknown",
                "sentiment_score": 0.0,
                "emotional_intensity": 0.0,
                "error": str(e)
            }


class EmailComplexityAnalysisPlugin(EmailProcessorPlugin):
    """Example plugin that analyzes email complexity and readability."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="complexity_analysis",
            version="1.0.0", 
            description="Analyzes email complexity, readability, and structure",
            author="CrewAI Team",
            dependencies=[],
            min_api_version="1.0.0",
            max_api_version="2.0.0"
        )
    
    def initialize(self) -> bool:
        """Initialize complexity analysis."""
        self.logger.info("Initializing complexity analysis plugin")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up complexity analysis plugin")
    
    def get_processing_priority(self) -> int:
        """Lower priority than sentiment analysis."""
        return 200
    
    def process_email(self, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze email complexity."""
        try:
            # Basic text metrics
            words = email_content.split()
            sentences = email_content.split('.')
            paragraphs = [p.strip() for p in email_content.split('\n\n') if p.strip()]
            
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len(paragraphs)
            
            # Average word length
            avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / max(word_count, 1)
            
            # Average sentence length
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Complexity indicators
            complex_words = len([word for word in words if len(word.strip('.,!?;:')) > 6])
            complex_word_ratio = complex_words / max(word_count, 1)
            
            # Simple readability score (Flesch-like)
            readability_score = max(0, min(100, 
                206.835 - (1.015 * avg_sentence_length) - (84.6 * complex_word_ratio)
            ))
            
            # Determine complexity level
            if readability_score > 80:
                complexity_level = "simple"
            elif readability_score > 60:
                complexity_level = "moderate"
            elif readability_score > 40:
                complexity_level = "complex"
            else:
                complexity_level = "very_complex"
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "complex_words": complex_words,
                "complex_word_ratio": round(complex_word_ratio, 3),
                "readability_score": round(readability_score, 1),
                "complexity_level": complexity_level,
                "estimated_read_time_minutes": max(1, word_count // 200)
            }
            
        except Exception as e:
            self.logger.error(f"Error in complexity analysis: {e}")
            return {
                "word_count": 0,
                "complexity_level": "unknown",
                "error": str(e)
            }
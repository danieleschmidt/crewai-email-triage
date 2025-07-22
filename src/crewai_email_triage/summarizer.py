"""Enhanced summarizer agent with intelligent text summarization."""

from __future__ import annotations
import re
import threading
from typing import Dict, Any, List

from .agent import Agent


class SummarizerAgent(Agent):
    """Agent that returns a short summary of the content."""
    
    def __init__(self, config_dict: Dict[str, Any] | None = None):
        """Initialize summarizer with optional configuration injection.
        
        Args:
            config_dict: Configuration dictionary with summarizer settings.
                        If None, falls back to default behavior.
        """
        super().__init__()
        self._config = config_dict
        self._config_lock = threading.RLock()
    
    def _get_summarizer_config(self) -> Dict[str, Any]:
        """Get summarizer configuration, with fallback to defaults."""
        with self._config_lock:
            if self._config is not None:
                return self._config.get("summarizer", {})
            return {}

    def run(self, content: str | None) -> str:
        """Return an intelligent summary of the content."""
        if not content or not content.strip():
            return "summary:"
        
        # Get configuration
        summarizer_config = self._get_summarizer_config()
        max_length = summarizer_config.get("max_length", 100)
        strategy = summarizer_config.get("strategy", "auto")
        
        # Clean and prepare content
        cleaned_content = self._clean_content(content.strip())
        
        # Handle very short content
        if len(cleaned_content) <= 20:
            return f"summary: {cleaned_content}"
        
        # Apply summarization strategy
        if strategy == "truncation":
            summary = self._truncation_summary(cleaned_content, max_length)
        elif strategy == "extractive":
            summary = self._extractive_summary(cleaned_content, max_length)
        else:  # auto strategy
            summary = self._auto_summary(cleaned_content, max_length)
        
        return f"summary: {summary}"
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing email fluff and normalizing whitespace."""
        # Remove common email greetings and closings
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip common email fluff
            lower_line = line.lower()
            if any(fluff in lower_line for fluff in [
                'hi team', 'hello', 'dear', 'greetings',
                'best regards', 'thanks', 'sincerely', 'kind regards',
                'let me know if you have', 'please reach out',
                'if you have any questions', 'hope this email finds you'
            ]):
                continue
            
            # Skip signature-like lines (short lines with names/titles)
            if len(line) < 30 and any(word in lower_line for word in [
                'operations', 'manager', 'team', 'department'
            ]):
                continue
                
            cleaned_lines.append(line)
        
        # Rejoin and normalize whitespace
        cleaned = ' '.join(cleaned_lines)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip() if cleaned.strip() else content
    
    def _truncation_summary(self, content: str, max_length: int) -> str:
        """Simple truncation-based summarization."""
        if len(content) <= max_length:
            return content
        
        # Try to truncate at sentence boundary
        sentences = self._split_sentences(content)
        result = ""
        
        for sentence in sentences:
            if len(result + sentence) > max_length - 3:
                break
            result += sentence + " "
        
        if result.strip():
            return result.strip()
        else:
            # Fallback to character truncation
            return content[:max_length-3] + "..."
    
    def _extractive_summary(self, content: str, max_length: int) -> str:
        """Extract key sentences based on importance scoring."""
        sentences = self._split_sentences(content)
        
        if len(sentences) <= 1:
            return self._truncation_summary(content, max_length)
        
        # Score sentences based on key indicators
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence)
            scored_sentences.append((score, sentence))
        
        # Sort by score (highest first) and select best sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        result = ""
        for score, sentence in scored_sentences:
            if len(result + sentence) > max_length - 3:
                break
            result += sentence + " "
        
        return result.strip() if result.strip() else self._truncation_summary(content, max_length)
    
    def _auto_summary(self, content: str, max_length: int) -> str:
        """Automatic strategy selection based on content analysis."""
        sentences = self._split_sentences(content)
        
        # For short content, use truncation
        if len(sentences) <= 2 or len(content) <= max_length * 1.5:
            return self._truncation_summary(content, max_length)
        else:
            # For longer content, use extractive summarization
            return self._extractive_summary(content, max_length)
    
    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences using simple heuristics."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+\s+', content)
        
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Filter very short fragments
                # Add back the period if it doesn't end with punctuation
                if not sentence[-1] in '.!?':
                    sentence += '.'
                result.append(sentence)
        
        return result
    
    def _score_sentence(self, sentence: str) -> float:
        """Score sentence importance based on various indicators."""
        score = 0.0
        lower_sentence = sentence.lower()
        
        # Boost for urgency indicators
        urgency_words = ['urgent', 'immediate', 'asap', 'deadline', 'critical', 'important']
        score += sum(2.0 for word in urgency_words if word in lower_sentence)
        
        # Boost for action words
        action_words = ['need', 'must', 'should', 'required', 'please', 'will', 'schedule']
        score += sum(1.0 for word in action_words if word in lower_sentence)
        
        # Boost for time references
        time_words = ['today', 'tomorrow', 'friday', 'monday', 'week', 'month', 'pm', 'am']
        score += sum(1.5 for word in time_words if word in lower_sentence)
        
        # Boost for numbers and specifics
        if re.search(r'\d+', sentence):
            score += 1.0
        
        # Penalize very short or very long sentences
        if len(sentence) < 10:
            score -= 1.0
        elif len(sentence) > 200:
            score -= 0.5
        
        # Boost for sentences in middle position (often contain key info)
        score += 0.5
        
        return max(score, 0.0)

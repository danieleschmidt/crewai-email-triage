"""Simple priority agent for assessing email urgency."""

from __future__ import annotations
from typing import Dict, Any

from .agent import Agent
from . import config


class PriorityAgent(Agent):
    """Agent that scores email urgency on a 1–10 scale."""
    
    def __init__(self, config_dict: Dict[str, Any] | None = None):
        """Initialize priority agent with optional configuration injection.
        
        Args:
            config_dict: Configuration dictionary with priority settings.
                        If None, falls back to global configuration.
        """
        super().__init__()
        self._config = config_dict
    
    def _get_priority_config(self) -> Dict[str, Any]:
        """Get priority configuration, with fallback to global config."""
        if self._config is not None:
            return self._config.get("priority", {})
        return config.CONFIG.get("priority", {})

    def run(self, content: str | None) -> str:
        """Return a priority score for ``content`` using simple heuristics."""
        if not content:
            return "priority: 0"

        # Optimize: Cache normalized content and perform string checks once
        normalized_content = content.lower()
        priority_config = self._get_priority_config()
        
        # Handle empty or invalid configuration with fallbacks
        high_keywords = priority_config.get("high_keywords", [])
        medium_keywords = priority_config.get("medium_keywords", [])
        scores = priority_config.get("scores", {"high": 10, "medium": 5, "low": 1})
        
        # Ensure scores have required keys
        high_score = scores.get("high", 10)
        medium_score = scores.get("medium", 5)
        low_score = scores.get("low", 1)
        
        # Optimize: Check uppercase and exclamation once, store results
        is_uppercase = content.isupper()
        has_exclamation = "!" in content
        
        # Optimize: Early return for high priority conditions
        if is_uppercase or (isinstance(high_keywords, list) and any(keyword in normalized_content for keyword in high_keywords)):
            return f"priority: {high_score}"
        
        # Optimize: Early return for medium priority conditions
        if has_exclamation or (isinstance(medium_keywords, list) and any(keyword in normalized_content for keyword in medium_keywords)):
            return f"priority: {medium_score}"
        
        return f"priority: {low_score}"

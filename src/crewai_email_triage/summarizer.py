"""Simple summarizer agent."""

from __future__ import annotations
from typing import Dict, Any

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
    
    def _get_summarizer_config(self) -> Dict[str, Any]:
        """Get summarizer configuration, with fallback to defaults."""
        if self._config is not None:
            return self._config.get("summarizer", {})
        return {}

    def run(self, content: str | None) -> str:
        """Return a summary string for ``content``."""
        if not content:
            return "summary:"
        
        # Get configuration for potential future enhancements
        summarizer_config = self._get_summarizer_config()
        max_length = summarizer_config.get("max_length", 100)
        
        # Current simple implementation: return first sentence
        first_sentence = content.split(".", 1)[0].strip()
        if content.strip().endswith("."):
            first_sentence += "."
            
        # Apply max_length if configured
        if len(first_sentence) > max_length:
            first_sentence = first_sentence[:max_length-3] + "..."
            
        return f"summary: {first_sentence}"

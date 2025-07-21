"""Simple priority agent for assessing email urgency."""

from __future__ import annotations

from .agent import Agent
from . import config


class PriorityAgent(Agent):
    """Agent that scores email urgency on a 1–10 scale."""

    def run(self, content: str | None) -> str:
        """Return a priority score for ``content`` using simple heuristics."""
        if not content:
            return "priority: 0"

        # Optimize: Cache normalized content and perform string checks once
        normalized_content = content.lower()
        priority_config = config.CONFIG["priority"]
        high_keywords = priority_config["high_keywords"]
        medium_keywords = priority_config["medium_keywords"]
        scores = priority_config["scores"]
        
        # Optimize: Check uppercase and exclamation once, store results
        is_uppercase = content.isupper()
        has_exclamation = "!" in content
        
        # Optimize: Early return for high priority conditions
        if is_uppercase or any(keyword in normalized_content for keyword in high_keywords):
            return f"priority: {scores['high']}"
        
        # Optimize: Early return for medium priority conditions
        if has_exclamation or any(keyword in normalized_content for keyword in medium_keywords):
            return f"priority: {scores['medium']}"
        
        return f"priority: {scores['low']}"

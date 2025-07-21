"""Simple email classification agent."""

from __future__ import annotations

from .agent import Agent
from . import config


class ClassifierAgent(Agent):
    """Agent that categorizes email content using keywords."""

    def run(self, content: str | None) -> str:
        """Return a category string for ``content``."""
        if not content:
            return "category: unknown"
        
        # Optimize: Cache normalized content for efficient repeated access
        normalized_content = content.lower()
        classifier_config = config.CONFIG["classifier"]
        
        # Optimize: Single iteration through categories with early return
        for category, keywords in classifier_config.items():
            if any(keyword in normalized_content for keyword in keywords):
                return f"category: {category}"
        return "category: general"

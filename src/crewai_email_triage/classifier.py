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
        text = content.lower()
        for category, keywords in config.CONFIG["classifier"].items():
            if any(word in text for word in keywords):
                return f"category: {category}"
        return "category: general"

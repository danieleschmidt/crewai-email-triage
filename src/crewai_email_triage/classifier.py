"""Simple email classification agent."""

from __future__ import annotations

from .agent import Agent


class ClassifierAgent(Agent):
    """Agent that categorizes email content using keywords."""

    def run(self, content: str | None) -> str:
        """Return a category string for ``content``."""
        if not content:
            return "category: unknown"
        text = content.lower()
        if "urgent" in text:
            return "category: urgent"
        if "unsubscribe" in text or "spam" in text:
            return "category: spam"
        if "meeting" in text or "schedule" in text:
            return "category: work"
        return "category: general"

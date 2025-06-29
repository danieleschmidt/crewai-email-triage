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

        normalized = content.lower()
        high_urgency = set(config.CONFIG["priority"]["high_keywords"])
        medium_urgency = set(config.CONFIG["priority"]["medium_keywords"])

        if any(word in normalized for word in high_urgency) or content.isupper():
            score = config.CONFIG["priority"]["scores"]["high"]
        elif any(word in normalized for word in medium_urgency) or "!" in content:
            score = config.CONFIG["priority"]["scores"]["medium"]
        else:
            score = config.CONFIG["priority"]["scores"]["low"]

        return f"priority: {score}"

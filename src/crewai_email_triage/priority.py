"""Simple priority agent for assessing email urgency."""

from __future__ import annotations

from .agent import Agent


HIGH_URGENCY = {
    "urgent",
    "asap",
    "immediately",
    "right away",
    "high priority",
}

MEDIUM_URGENCY = {
    "soon",
    "important",
    "deadline",
    "tomorrow",
    "today",
    "eod",
    "end of day",
}


class PriorityAgent(Agent):
    """Agent that scores email urgency on a 1–10 scale."""

    def run(self, content: str | None) -> str:
        """Return a priority score for ``content`` using simple heuristics."""
        if not content:
            return "priority: 0"

        normalized = content.lower()

        if any(word in normalized for word in HIGH_URGENCY) or content.isupper():
            score = 10
        elif any(word in normalized for word in MEDIUM_URGENCY) or "!" in content:
            score = 8
        else:
            score = 5

        return f"priority: {score}"

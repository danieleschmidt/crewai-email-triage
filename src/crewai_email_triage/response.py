"""Simple response agent."""

from __future__ import annotations

from .agent import Agent


class ResponseAgent(Agent):
    """Agent that drafts a basic reply."""

    def run(self, content: str | None) -> str:
        """Return a reply string for ``content``."""
        if not content:
            return "response:"
        return "response: Thanks for your email"

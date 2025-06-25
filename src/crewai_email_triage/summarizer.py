"""Simple summarizer agent."""

from __future__ import annotations

from .agent import Agent


class SummarizerAgent(Agent):
    """Agent that returns a short summary of the content."""

    def run(self, content: str | None) -> str:
        """Return a summary string for ``content``."""
        if not content:
            return "summary:"
        first_sentence = content.split(".", 1)[0].strip()
        if content.strip().endswith("."):
            first_sentence += "."
        return f"summary: {first_sentence}"

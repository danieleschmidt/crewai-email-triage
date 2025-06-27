"""Agent for assigning urgency priority to email content."""

from __future__ import annotations

from .agent import Agent


class PriorityAgent(Agent):
    """Agent that assigns a priority level based on keywords."""

    #: Default keyword groups used to determine priority levels.
    DEFAULT_HIGH = {"urgent", "asap", "immediately"}
    DEFAULT_MEDIUM = {"soon", "tomorrow"}

    def __init__(
        self,
        high_keywords: set[str] | None = None,
        medium_keywords: set[str] | None = None,
    ) -> None:
        """Create a new ``PriorityAgent`` instance.

        Parameters
        ----------
        high_keywords : set[str] | None, optional
            Custom keywords that trigger a high priority result. If ``None``,
            :pydata:`DEFAULT_HIGH` is used. Keywords are matched case-
            insensitively.
        medium_keywords : set[str] | None, optional
            Custom keywords that trigger a medium priority result. If ``None``,
            :pydata:`DEFAULT_MEDIUM` is used. Keywords are matched case-
            insensitively.
        """

        self.high_keywords = {
            kw.lower() for kw in (high_keywords or self.DEFAULT_HIGH)
        }
        self.medium_keywords = {
            kw.lower() for kw in (medium_keywords or self.DEFAULT_MEDIUM)
        }

    def run(self, content: str | None) -> str:
        """Return a priority string for ``content``."""
        if not content:
            return "priority: low"
        text = content.lower()
        if any(word in text for word in self.high_keywords):
            return "priority: high"
        if any(word in text for word in self.medium_keywords):
            return "priority: medium"
        return "priority: low"

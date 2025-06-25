"""Base Agent implementation for CrewAI Email Triage."""

from __future__ import annotations

from .core import process_email


class Agent:
    """Base agent with a simple :py:meth:`run` method."""

    def run(self, content: str | None) -> str:
        """Process ``content`` and return a response.

        Parameters
        ----------
        content : str | None
            The input text to process. If ``None`` or empty, returns an empty
            string.

        Returns
        -------
        str
            The processed result string.
        """
        return process_email(content)

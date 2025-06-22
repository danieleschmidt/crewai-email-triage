"""Core functionality for CrewAI Email Triage."""

from __future__ import annotations


def process_email(content: str | None) -> str:
    """Process an email and return a simple acknowledgment string.

    Parameters
    ----------
    content: str | None
        The email content to process. If ``None``, returns an empty string.

    Returns
    -------
    str
        A simple acknowledgment string.
    """
    if content is None:
        return ""
    return f"Processed: {content.strip()}"

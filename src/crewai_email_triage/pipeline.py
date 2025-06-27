"""Pipeline for running all agents in sequence."""

from __future__ import annotations

from .classifier import ClassifierAgent
from .priority import PriorityAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent


def triage_email(
    content: str | None,
    *,
    high_keywords: set[str] | None = None,
    medium_keywords: set[str] | None = None,
) -> dict[str, str]:
    """Process ``content`` through the agent pipeline.

    Parameters
    ----------
    content : str | None
        The email text to process. ``None`` or empty strings return default
        low-priority, unknown-category responses.
    high_keywords : set[str] | None, optional
        Custom high-priority keywords passed to :class:`PriorityAgent`.
    medium_keywords : set[str] | None, optional
        Custom medium-priority keywords passed to :class:`PriorityAgent`.

    Returns
    -------
    dict[str, str]
        A mapping of stage names to their results.
    """

    classifier = ClassifierAgent()
    priority = PriorityAgent(
        high_keywords=high_keywords, medium_keywords=medium_keywords
    )
    summarizer = SummarizerAgent()
    response = ResponseAgent()

    return {
        "category": classifier.run(content),
        "priority": priority.run(content),
        "summary": summarizer.run(content),
        "response": response.run(content),
    }


def triage_emails(
    emails: list[str | None],
    *,
    high_keywords: set[str] | None = None,
    medium_keywords: set[str] | None = None,
) -> list[dict[str, str]]:
    """Run :func:`triage_email` on a list of emails."""

    return [
        triage_email(
            email,
            high_keywords=high_keywords,
            medium_keywords=medium_keywords,
        )
        for email in emails
    ]

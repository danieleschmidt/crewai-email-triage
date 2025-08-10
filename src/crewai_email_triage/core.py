"""Core functionality for CrewAI Email Triage."""

from __future__ import annotations
import logging

# Setup basic logging
logger = logging.getLogger(__name__)


def process_email(content: str | None) -> str:
    """Process an email and return a simple acknowledgment string.
    
    Enhanced error handling and input validation for production use.

    Parameters
    ----------
    content: str | None
        The email content to process. If ``None``, returns an empty string.

    Returns
    -------
    str
        A simple acknowledgment string.
    """
    # Enhanced error handling
    if content is None:
        return ""
    
    if not isinstance(content, str):
        raise TypeError(f"Expected str or None, got {type(content)}")
    
    if len(content.strip()) == 0:
        return "Processed: [Empty message]"
    result = f"Processed: {content.strip()}"
    logger.info(f"Processed email content: {len(content)} chars")
    return result

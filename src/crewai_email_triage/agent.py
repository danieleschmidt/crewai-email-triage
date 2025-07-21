"""Abstract Base Agent implementation for CrewAI Email Triage."""

from __future__ import annotations
from abc import ABC, abstractmethod

from .core import process_email


class Agent(ABC):
    """Abstract base agent that defines the common interface for all email processing agents.
    
    This class enforces a contract for all concrete agent implementations,
    ensuring consistency across the codebase and making extension easier.
    """

    @abstractmethod
    def run(self, content: str | None) -> str:
        """Process email content and return a formatted response.

        Parameters
        ----------
        content : str | None
            The input email text to process. If ``None`` or empty, 
            implementations should handle gracefully.

        Returns
        -------
        str
            The processed result string in the format expected by the pipeline.
            Format should be "field: value" (e.g., "category: urgent", "priority: 8").
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented by a concrete subclass.
        """
        raise NotImplementedError("Subclasses must implement the run method")


class LegacyAgent(Agent):
    """Legacy agent implementation for backward compatibility.
    
    This class provides the original behavior for any existing code
    that directly instantiates the base Agent class.
    """

    def run(self, content: str | None) -> str:
        """Process ``content`` using the legacy core processing function.

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

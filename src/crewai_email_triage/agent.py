"""Abstract Base Agent implementation for CrewAI Email Triage."""

from __future__ import annotations
from abc import ABC, abstractmethod

from .core import process_email


class Agent(ABC):
    """Abstract base agent that defines the common interface for all email processing agents.
    
    This class enforces a contract for all concrete agent implementations,
    ensuring consistency across the codebase and making extension easier.
    
    Agent Contract Overview:
    -----------------------
    All agents must implement the `run(content: str | None) -> str` method and follow
    these conventions:
    
    1. **Input Handling**: Accept None, empty strings, and malformed content gracefully
    2. **Output Format**: Return structured "field: value" format (e.g., "category: urgent")
    3. **Configuration**: Support optional config_dict injection in constructor
    4. **Thread Safety**: Use threading.RLock for configuration access if needed
    5. **Error Handling**: Handle errors gracefully without raising exceptions to pipeline
    
    Configuration Injection:
    ------------------------
    Agents should accept optional config_dict in constructor to enable dependency injection:
    
        def __init__(self, config_dict: Dict[str, Any] | None = None):
            super().__init__()
            self._config = config_dict
            self._config_lock = threading.RLock()  # If thread safety needed
    
    Expected Agent Types and Outputs:
    ---------------------------------
    - **ClassifierAgent**: Returns "category: {category_name}"
      - Categories determined by keyword matching from config["classifier"]
      - Default category: "general" for unmatched content
      
    - **PriorityAgent**: Returns "priority: {numeric_score}"
      - Scores based on keyword matching from config["priority"]
      - Default score: 0 for empty content, 1-10 scale typically used
      
    - **SummarizerAgent**: Returns "summary: {summary_text}"
      - Configurable max_length from config["summarizer"]["max_length"]
      - Default: truncated first sentence
      
    - **ResponseAgent**: Returns "response: {response_text}"
      - Configurable template and signature from config["response"]
      - Default: "Thanks for your email"
    
    Configuration Structure:
    ------------------------
    Expected configuration format (all sections optional):
    
        {
          "classifier": {
            "category_name": ["keyword1", "keyword2", ...]
          },
          "priority": {
            "scores": {"high": 10, "medium": 5, "low": 1},
            "high_keywords": ["urgent", "asap", ...],
            "medium_keywords": ["important", "soon", ...]
          },
          "summarizer": {
            "max_length": 100
          },
          "response": {
            "template": "Thanks for your email",
            "signature": "Best regards, ..."
          }
        }
    
    Thread Safety:
    --------------
    Agents used in parallel processing should protect configuration access with locks.
    Use threading.RLock to allow reentrant access within the same thread.
    
    Error Handling Best Practices:
    ------------------------------
    - Return default/fallback values instead of raising exceptions
    - Log errors for debugging but don't break the pipeline
    - Handle None, empty string, and malformed inputs gracefully
    - Use specific exception types if exceptions are unavoidable
    
    Extension Examples:
    -------------------
    
        class CustomAgent(Agent):
            def __init__(self, config_dict: Dict[str, Any] | None = None):
                super().__init__()
                self._config = config_dict
                self._config_lock = threading.RLock()
            
            def run(self, content: str | None) -> str:
                if not content:
                    return "custom: default_value"
                
                with self._config_lock:
                    config = self._config.get("custom", {}) if self._config else {}
                    
                # Process content and return structured result
                return f"custom: {processed_result}"
    """

    @abstractmethod
    def run(self, content: str | None) -> str:
        """Process email content and return a formatted response.

        This method must be implemented by all concrete agent subclasses and should
        follow the agent contract defined in the class docstring.

        Parameters
        ----------
        content : str | None
            The input email text to process. May be None, empty string, or contain
            malformed content. Implementations should handle all cases gracefully
            without raising exceptions to avoid breaking the pipeline.

        Returns
        -------
        str
            The processed result string in the required "field: value" format.
            
            Examples by agent type:
            - ClassifierAgent: "category: urgent" or "category: general"
            - PriorityAgent: "priority: 8" or "priority: 0"
            - SummarizerAgent: "summary: This is a summary..." or "summary:"
            - ResponseAgent: "response: Thanks for your email" or "response:"
            
            The field name should match the agent's purpose, and the value should
            be meaningful for downstream processing or human consumption.
        
        Raises
        ------
        NotImplementedError
            If the method is not implemented by a concrete subclass.
            
        Notes
        -----
        Implementations should:
        1. Return default values for None/empty inputs rather than raising exceptions
        2. Use configuration from self._config if available, with fallback to defaults
        3. Protect configuration access with self._config_lock if thread safety needed
        4. Log errors for debugging but continue processing with fallback values
        5. Ensure output format is always "field: value" for pipeline compatibility
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

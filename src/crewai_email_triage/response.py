"""Simple response agent."""

from __future__ import annotations
from typing import Dict, Any

from .agent import Agent


class ResponseAgent(Agent):
    """Agent that drafts a basic reply."""
    
    def __init__(self, config_dict: Dict[str, Any] | None = None):
        """Initialize response agent with optional configuration injection.
        
        Args:
            config_dict: Configuration dictionary with response settings.
                        If None, falls back to default behavior.
        """
        super().__init__()
        self._config = config_dict
    
    def _get_response_config(self) -> Dict[str, Any]:
        """Get response configuration, with fallback to defaults."""
        if self._config is not None:
            return self._config.get("response", {})
        return {}

    def run(self, content: str | None) -> str:
        """Return a reply string for ``content``."""
        if not content:
            return "response:"
        
        # Get configuration for customizable responses
        response_config = self._get_response_config()
        template = response_config.get("template", "Thanks for your email")
        signature = response_config.get("signature", "")
        
        # Build response
        response_text = template
        if signature:
            response_text += f"\n\n{signature}"
            
        return f"response: {response_text}"

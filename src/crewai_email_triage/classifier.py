"""Simple email classification agent."""

from __future__ import annotations

import threading
from typing import Any, Dict

from . import config
from .agent import Agent


class ClassifierAgent(Agent):
    """Agent that categorizes email content using keywords."""

    def __init__(self, config_dict: Dict[str, Any] | None = None):
        """Initialize classifier with optional configuration injection.
        
        Args:
            config_dict: Configuration dictionary with classifier settings.
                        If None, falls back to global configuration.
        """
        super().__init__()
        self._config = config_dict
        self._config_lock = threading.RLock()

    def _get_classifier_config(self) -> Dict[str, Any]:
        """Get classifier configuration, with fallback to global config."""
        with self._config_lock:
            if self._config is not None:
                return self._config.get("classifier", {})
            return config.CONFIG.get("classifier", {})

    def run(self, content: str | None) -> str:
        """Return a category string for ``content``."""
        if not content:
            return "category: unknown"

        # Optimize: Cache normalized content for efficient repeated access
        normalized_content = content.lower()
        classifier_config = self._get_classifier_config()

        # Handle empty or invalid configuration
        if not classifier_config:
            return "category: general"

        # Optimize: Single iteration through categories with early return
        for category, keywords in classifier_config.items():
            if isinstance(keywords, list) and any(keyword in normalized_content for keyword in keywords):
                return f"category: {category}"
        return "category: general"

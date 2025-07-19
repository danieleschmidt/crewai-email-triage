from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEF_PATH = Path(__file__).with_name("default_config.json")

# Default fallback configuration
_FALLBACK_CONFIG = {
    "classifier": {
        "urgent": ["urgent", "asap", "immediately"],
        "work": ["meeting", "schedule", "project"],
        "spam": ["unsubscribe", "click here", "offer"]
    },
    "priority": {
        "scores": {"high": 10, "medium": 5, "low": 1},
        "high_keywords": ["urgent", "asap", "immediately", "critical"],
        "medium_keywords": ["important", "soon", "deadline", "today"]
    }
}


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Return configuration from ``path`` or the default file.
    
    Falls back to a basic configuration if file loading fails.
    """
    env_path = os.environ.get("CREWAI_CONFIG")
    cfg_path = Path(path or env_path) if (path or env_path) else _DEF_PATH
    
    try:
        if not cfg_path.exists():
            logger.warning("Config file %s not found, using fallback", cfg_path)
            return _FALLBACK_CONFIG.copy()
            
        with cfg_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
            
        # Validate required sections
        if not isinstance(config, dict):
            logger.error("Config must be a JSON object, using fallback")
            return _FALLBACK_CONFIG.copy()
            
        if "classifier" not in config or "priority" not in config:
            logger.warning("Config missing required sections, merging with fallback")
            merged = _FALLBACK_CONFIG.copy()
            merged.update(config)
            return merged
            
        return config
        
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.error("Failed to load config from %s: %s. Using fallback.", cfg_path, str(e))
        return _FALLBACK_CONFIG.copy()
    except Exception as e:
        logger.error("Unexpected error loading config: %s. Using fallback.", str(e))
        return _FALLBACK_CONFIG.copy()


def set_config(path: str) -> None:
    """Load configuration from ``path`` and store globally."""
    global CONFIG
    CONFIG = load_config(path)


CONFIG = load_config()

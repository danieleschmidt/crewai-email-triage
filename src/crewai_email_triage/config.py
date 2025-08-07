from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .logging_utils import get_logger

logger = get_logger(__name__)

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


def _calculate_config_hash(config: Dict[str, Any]) -> str:
    """Calculate a hash of the configuration for change detection."""
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:12]


def _get_config_stats(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration statistics for logging."""
    stats = {
        'sections': list(config.keys()),
        'section_count': len(config)
    }

    if 'classifier' in config:
        classifier_config = config['classifier']
        if isinstance(classifier_config, dict):
            stats['classifier_categories'] = len(classifier_config)
            stats['classifier_total_keywords'] = sum(
                len(keywords) if isinstance(keywords, list) else 0
                for keywords in classifier_config.values()
            )

    if 'priority' in config:
        priority_config = config['priority']
        if isinstance(priority_config, dict):
            high_kw = priority_config.get('high_keywords', [])
            medium_kw = priority_config.get('medium_keywords', [])
            stats['priority_high_keywords'] = len(high_kw) if isinstance(high_kw, list) else 0
            stats['priority_medium_keywords'] = len(medium_kw) if isinstance(medium_kw, list) else 0

    return stats


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Return configuration from ``path`` or the default file.
    
    Falls back to a basic configuration if file loading fails.
    """
    from .env_config import get_app_config
    app_config = get_app_config()
    env_path = app_config.config_path
    cfg_path = Path(path or env_path) if (path or env_path) else _DEF_PATH

    # Log configuration loading attempt
    logger.info("Loading configuration", extra={
        'operation': 'config_load',
        'config_path': str(cfg_path),
        'is_default_path': cfg_path == _DEF_PATH,
        'source': 'parameter' if path else ('environment' if env_path else 'default')
    })

    try:
        if not cfg_path.exists():
            logger.warning("Config file not found, using fallback configuration", extra={
                'operation': 'config_load',
                'config_path': str(cfg_path),
                'fallback_reason': 'file_not_found',
                'config_hash': _calculate_config_hash(_FALLBACK_CONFIG),
                **_get_config_stats(_FALLBACK_CONFIG)
            })
            return _FALLBACK_CONFIG.copy()

        with cfg_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        file_size = cfg_path.stat().st_size

        # Validate required sections
        if not isinstance(config, dict):
            logger.error("Config validation failed: must be JSON object, using fallback", extra={
                'operation': 'config_load',
                'config_path': str(cfg_path),
                'validation_error': 'invalid_type',
                'actual_type': type(config).__name__,
                'fallback_reason': 'validation_failed'
            })
            return _FALLBACK_CONFIG.copy()

        if "classifier" not in config or "priority" not in config:
            logger.warning("Config missing required sections, merging with fallback", extra={
                'operation': 'config_load',
                'config_path': str(cfg_path),
                'missing_sections': [s for s in ['classifier', 'priority'] if s not in config],
                'provided_sections': list(config.keys()),
                'action': 'merge_with_fallback'
            })
            merged = _FALLBACK_CONFIG.copy()
            merged.update(config)

            logger.info("Configuration successfully loaded with fallback merge", extra={
                'operation': 'config_load',
                'config_path': str(cfg_path),
                'file_size_bytes': file_size,
                'config_hash': _calculate_config_hash(merged),
                'load_result': 'merged_with_fallback',
                **_get_config_stats(merged)
            })
            return merged

        # Successful load
        logger.info("Configuration successfully loaded", extra={
            'operation': 'config_load',
            'config_path': str(cfg_path),
            'file_size_bytes': file_size,
            'config_hash': _calculate_config_hash(config),
            'load_result': 'success',
            **_get_config_stats(config)
        })
        return config

    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.error("Failed to load configuration, using fallback", extra={
            'operation': 'config_load',
            'config_path': str(cfg_path),
            'error_type': type(e).__name__,
            'error_message': str(e),
            'fallback_reason': 'load_error',
            'config_hash': _calculate_config_hash(_FALLBACK_CONFIG)
        })
        return _FALLBACK_CONFIG.copy()
    except Exception as e:
        logger.error("Unexpected error loading configuration, using fallback", extra={
            'operation': 'config_load',
            'config_path': str(cfg_path),
            'error_type': type(e).__name__,
            'error_message': str(e),
            'fallback_reason': 'unexpected_error'
        })
        return _FALLBACK_CONFIG.copy()


def set_config(path: str) -> None:
    """Load configuration from ``path`` and store globally."""
    global CONFIG

    # Store previous config for change detection
    previous_config = CONFIG
    previous_hash = _calculate_config_hash(previous_config)

    logger.info("Setting new global configuration", extra={
        'operation': 'config_change',
        'previous_config_hash': previous_hash,
        'new_config_path': path,
        'action': 'global_config_update'
    })

    new_config = load_config(path)
    new_hash = _calculate_config_hash(new_config)

    # Log configuration change details
    if previous_hash != new_hash:
        logger.info("Global configuration changed", extra={
            'operation': 'config_change',
            'previous_config_hash': previous_hash,
            'new_config_hash': new_hash,
            'config_changed': True,
            'change_trigger': 'set_config',
            **_get_config_stats(new_config)
        })
    else:
        logger.info("Global configuration unchanged", extra={
            'operation': 'config_change',
            'config_hash': new_hash,
            'config_changed': False,
            'change_trigger': 'set_config'
        })

    CONFIG = new_config


CONFIG = load_config()

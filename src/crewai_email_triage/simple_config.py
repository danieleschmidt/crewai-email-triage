"""Simple configuration system for enhanced functionality."""

import json
import os
from pathlib import Path
from typing import Dict, Any

class SimpleConfig:
    """Simple configuration manager."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('EMAIL_TRIAGE_CONFIG')
        self._config = self._load_default_config()
        
        if self.config_path and Path(self.config_path).exists():
            self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "processing": {
                "max_content_length": 50000,
                "enable_validation": True,
                "enable_logging": True
            },
            "validation": {
                "check_spam_patterns": True,
                "max_caps_ratio": 0.5,
                "spam_patterns": [
                    "urgent.*act.*now",
                    "click.*here.*immediately",
                    "limited.*time.*offer"
                ]
            },
            "output": {
                "include_warnings": True,
                "include_metadata": False,
                "default_format": "text"
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file."""
        try:
            with open(self.config_path) as f:
                file_config = json.load(f)
            
            # Merge with defaults
            self._merge_config(self._config, file_config)
        except Exception as e:
            print(f"Warning: Failed to load config file {self.config_path}: {e}")
    
    def _merge_config(self, default: Dict, override: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()

# Global configuration instance
_config = SimpleConfig()

def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value."""
    if key is None:
        return _config.get_all()
    return _config.get(key, default)

def set_config_file(config_path: str):
    """Set configuration file path."""
    global _config
    _config = SimpleConfig(config_path)

"""Centralized environment configuration system for CrewAI Email Triage.

This module provides a unified approach to environment variable management,
following Twelve-Factor App principles and improving maintainability.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass
from typing import Any, Dict, Optional, Union, get_type_hints


def _parse_bool(value: str) -> bool:
    """Parse a string value into a boolean.
    
    Supports common boolean representations:
    True: 'true', 'True', 'TRUE', '1', 'yes', 'on'
    False: 'false', 'False', 'FALSE', '0', 'no', 'off', ''
    """
    if not isinstance(value, str):
        return bool(value)

    true_values = {'true', '1', 'yes', 'on'}
    false_values = {'false', '0', 'no', 'off', ''}

    value_lower = value.lower()
    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        # For unexpected values, follow common convention: non-empty = True
        return len(value.strip()) > 0


def _parse_env_value(value: str, target_type: type) -> Any:
    """Parse environment variable value to target type with proper error handling."""
    if value == "" and target_type is not str:
        return None

    try:
        if target_type is bool:
            return _parse_bool(value)
        elif target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is str:
            return value
        else:
            return value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert environment value '{value}' to {target_type.__name__}: {e}")


@dataclass
class EnvironmentConfig:
    """Base class for environment-based configuration.
    
    Provides common functionality for loading configuration from environment variables
    with type conversion, validation, and documentation support.
    """

    @classmethod
    def from_environment(cls) -> EnvironmentConfig:
        """Create configuration instance from environment variables."""
        env_mapping = cls._get_env_mapping()
        annotations = get_type_hints(cls)
        defaults = {}

        # Get default values from dataclass field defaults
        if hasattr(cls, '__dataclass_fields__'):
            for field_name, field_info in cls.__dataclass_fields__.items():
                if field_info.default is not MISSING:
                    defaults[field_name] = field_info.default
                elif field_info.default_factory is not MISSING:
                    defaults[field_name] = field_info.default_factory()

        kwargs = {}
        for field_name, env_var_name in env_mapping.items():
            env_value = os.environ.get(env_var_name)

            if env_value is not None:
                # Get the target type from annotations
                target_type = annotations.get(field_name, str)
                # Handle Optional[Type] annotations
                if hasattr(target_type, '__origin__') and target_type.__origin__ is Union:
                    # For Optional[T], get T
                    args = target_type.__args__
                    target_type = next((arg for arg in args if arg is not type(None)), str)

                kwargs[field_name] = _parse_env_value(env_value, target_type)
            elif field_name in defaults:
                kwargs[field_name] = defaults[field_name]

        return cls(**kwargs)

    @classmethod
    def _get_env_mapping(cls) -> Dict[str, str]:
        """Return mapping of field names to environment variable names.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_env_mapping")

    @classmethod
    def get_environment_docs(cls) -> Dict[str, str]:
        """Get documentation for environment variables used by this config."""
        env_mapping = cls._get_env_mapping()
        annotations = getattr(cls, '__annotations__', {})
        docs = {}

        for field_name, env_var_name in env_mapping.items():
            field_type = annotations.get(field_name, 'str')
            # Handle Optional[Type] display
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                args = field_type.__args__
                non_none_types = [arg for arg in args if arg is not type(None)]
                if non_none_types:
                    type_name = non_none_types[0].__name__
                else:
                    type_name = 'Any'
                type_display = f"Optional[{type_name}]"
            else:
                type_display = getattr(field_type, '__name__', str(field_type))

            docs[env_var_name] = f"{field_name} ({type_display})"

        return docs

    def validate(self) -> None:
        """Validate configuration values. Override in subclasses for specific validation."""
        pass


@dataclass
class RetryEnvironmentConfig(EnvironmentConfig):
    """Retry configuration loaded from environment variables."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_factor: float = 2.0
    jitter: bool = True

    @classmethod
    def _get_env_mapping(cls) -> Dict[str, str]:
        return {
            'max_attempts': 'RETRY_MAX_ATTEMPTS',
            'base_delay': 'RETRY_BASE_DELAY',
            'max_delay': 'RETRY_MAX_DELAY',
            'exponential_factor': 'RETRY_EXPONENTIAL_FACTOR',
            'jitter': 'RETRY_JITTER'
        }

    def validate(self) -> None:
        """Validate retry configuration values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_factor <= 1:
            raise ValueError("exponential_factor must be > 1")


@dataclass
class MetricsEnvironmentConfig(EnvironmentConfig):
    """Metrics configuration loaded from environment variables."""

    enabled: bool = True
    export_port: int = 8080
    export_path: str = "/metrics"
    namespace: str = "crewai_email_triage"
    histogram_max_size: int = 1000

    @classmethod
    def _get_env_mapping(cls) -> Dict[str, str]:
        return {
            'enabled': 'METRICS_ENABLED',
            'export_port': 'METRICS_EXPORT_PORT',
            'export_path': 'METRICS_EXPORT_PATH',
            'namespace': 'METRICS_NAMESPACE',
            'histogram_max_size': 'METRICS_HISTOGRAM_MAX_SIZE'
        }

    def validate(self) -> None:
        """Validate metrics configuration values."""
        if self.export_port < 1 or self.export_port > 65535:
            raise ValueError("export_port must be between 1 and 65535")
        if self.histogram_max_size < 1:
            raise ValueError("histogram_max_size must be at least 1")
        if not self.export_path.startswith('/'):
            raise ValueError("export_path must start with '/'")


@dataclass
class ProviderEnvironmentConfig(EnvironmentConfig):
    """Email provider configuration loaded from environment variables."""

    gmail_user: Optional[str] = None
    gmail_password: Optional[str] = None

    @classmethod
    def _get_env_mapping(cls) -> Dict[str, str]:
        return {
            'gmail_user': 'GMAIL_USER',
            'gmail_password': 'GMAIL_PASSWORD'
        }


@dataclass
class RateLimitEnvironmentConfig(EnvironmentConfig):
    """Rate limiting configuration loaded from environment variables."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    enabled: bool = True
    backpressure_threshold: float = 0.8
    backpressure_delay: float = 0.1

    @classmethod
    def _get_env_mapping(cls) -> Dict[str, str]:
        return {
            'requests_per_second': 'RATE_LIMIT_REQUESTS_PER_SECOND',
            'burst_size': 'RATE_LIMIT_BURST_SIZE',
            'enabled': 'RATE_LIMIT_ENABLED',
            'backpressure_threshold': 'RATE_LIMIT_BACKPRESSURE_THRESHOLD',
            'backpressure_delay': 'RATE_LIMIT_BACKPRESSURE_DELAY'
        }

    def validate(self):
        """Validate rate limiting configuration values."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size < 1:
            raise ValueError("burst_size must be at least 1")
        if not 0 < self.backpressure_threshold <= 1:
            raise ValueError("backpressure_threshold must be between 0 and 1")
        if self.backpressure_delay < 0:
            raise ValueError("backpressure_delay must be non-negative")


@dataclass
class AppEnvironmentConfig(EnvironmentConfig):
    """Main application configuration loaded from environment variables."""

    config_path: Optional[str] = None

    @classmethod
    def _get_env_mapping(cls) -> Dict[str, str]:
        return {
            'config_path': 'CREWAI_CONFIG'
        }


# Global configuration instances (following existing patterns)
_retry_config: Optional[RetryEnvironmentConfig] = None
_metrics_config: Optional[MetricsEnvironmentConfig] = None
_provider_config: Optional[ProviderEnvironmentConfig] = None
_app_config: Optional[AppEnvironmentConfig] = None
_rate_limit_config: Optional[RateLimitEnvironmentConfig] = None


def get_retry_config() -> RetryEnvironmentConfig:
    """Get the global retry configuration instance."""
    global _retry_config
    if _retry_config is None:
        _retry_config = RetryEnvironmentConfig.from_environment()
        _retry_config.validate()
    return _retry_config


def get_metrics_config() -> MetricsEnvironmentConfig:
    """Get the global metrics configuration instance."""
    global _metrics_config
    if _metrics_config is None:
        _metrics_config = MetricsEnvironmentConfig.from_environment()
        _metrics_config.validate()
    return _metrics_config


def get_provider_config() -> ProviderEnvironmentConfig:
    """Get the global provider configuration instance."""
    global _provider_config
    if _provider_config is None:
        _provider_config = ProviderEnvironmentConfig.from_environment()
    return _provider_config


def get_app_config() -> AppEnvironmentConfig:
    """Get the global application configuration instance."""
    global _app_config
    if _app_config is None:
        _app_config = AppEnvironmentConfig.from_environment()
    return _app_config


def get_rate_limit_config() -> RateLimitEnvironmentConfig:
    """Get the global rate limiting configuration instance."""
    global _rate_limit_config
    if _rate_limit_config is None:
        _rate_limit_config = RateLimitEnvironmentConfig.from_environment()
        _rate_limit_config.validate()
    return _rate_limit_config


def reset_config_cache() -> None:
    """Reset cached configuration instances. Useful for testing."""
    global _retry_config, _metrics_config, _provider_config, _app_config, _rate_limit_config
    _retry_config = None
    _metrics_config = None
    _provider_config = None
    _app_config = None
    _rate_limit_config = None


def get_all_environment_docs() -> Dict[str, Dict[str, str]]:
    """Get documentation for all environment variables used by the system."""
    return {
        "Retry Configuration": RetryEnvironmentConfig.get_environment_docs(),
        "Metrics Configuration": MetricsEnvironmentConfig.get_environment_docs(),
        "Provider Configuration": ProviderEnvironmentConfig.get_environment_docs(),
        "Application Configuration": AppEnvironmentConfig.get_environment_docs(),
    }

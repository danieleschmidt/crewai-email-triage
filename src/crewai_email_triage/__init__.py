"""CrewAI Email Triage package."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import tomllib

from .core import process_email
from .agent import Agent, LegacyAgent
from .classifier import ClassifierAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent
from .priority import PriorityAgent
from .pipeline import triage_email, triage_email_enhanced, triage_batch, TriageResult
from .provider import GmailProvider
from .sanitization import sanitize_email_content, EmailSanitizer, SanitizationConfig
from .agent_responses import (
    AgentResponse, ClassificationResponse, PriorityResponse, 
    SummaryResponse, ResponseGenerationResponse, parse_agent_response
)
from .rate_limiter import RateLimiter, RateLimitConfig, get_rate_limiter
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .retry_utils import retry_with_backoff, RetryConfig
from .logging_utils import get_logger, setup_structured_logging
from .metrics_export import get_metrics_collector, MetricsCollector
from .health import get_health_checker, HealthChecker, HealthMonitor, HealthStatus
from .validation import get_email_validator, validate_email_content, EmailValidator, ValidationResult, ConfigValidator
from .cache import get_smart_cache, get_persistent_cache, SmartCache, LRUCache, cached_agent_operation
from .performance import get_performance_tracker, get_resource_monitor, Timer, timed, enable_performance_monitoring



def _read_version_from_pyproject() -> str:
    """Return the version declared in ``pyproject.toml``."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)
    return project["project"]["version"]


try:  # Grab the installed package version if available
    __version__ = _pkg_version("crewai_email_triage")
except PackageNotFoundError:  # Local source without installation
    try:
        __version__ = _read_version_from_pyproject()
    except Exception:
        __version__ = "0.0.0"

__all__ = [
    "process_email",
    "Agent",
    "LegacyAgent",
    "ClassifierAgent",
    "SummarizerAgent",
    "ResponseAgent",
    "PriorityAgent",
    "triage_email",
    "triage_email_enhanced", 
    "triage_batch",
    "TriageResult",
    "GmailProvider",
    "sanitize_email_content",
    "EmailSanitizer",
    "SanitizationConfig",
    "AgentResponse",
    "ClassificationResponse",
    "PriorityResponse",
    "SummaryResponse",
    "ResponseGenerationResponse",
    "parse_agent_response",
    "RateLimiter",
    "RateLimitConfig", 
    "get_rate_limiter",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "retry_with_backoff",
    "RetryConfig",
    "get_logger",
    "setup_structured_logging",
    "get_metrics_collector",
    "MetricsCollector",
    "get_health_checker",
    "HealthChecker", 
    "HealthMonitor",
    "HealthStatus",
    "get_email_validator",
    "validate_email_content",
    "EmailValidator",
    "ValidationResult",
    "ConfigValidator",
    "get_smart_cache",
    "get_persistent_cache", 
    "SmartCache",
    "LRUCache",
    "cached_agent_operation",
    "get_performance_tracker",
    "get_resource_monitor",
    "Timer",
    "timed",
    "enable_performance_monitoring",
    "__version__",
]

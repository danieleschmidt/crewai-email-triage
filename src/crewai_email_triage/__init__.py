"""CrewAI Email Triage package."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import tomllib

from .advanced_scaling import (
    AdaptiveLoadBalancer,
    HighPerformanceProcessor,
    ScalingMetrics,
    get_performance_insights,
    high_performance_processor,
    optimize_system_performance,
    process_batch_high_performance,
)
from .advanced_security import (
    AdvancedSecurityScanner,
    SecurityAnalysisResult,
    SecurityAuditLogger,
    SecurityThreat,
    perform_security_scan,
)
from .agent import Agent, LegacyAgent
from .agent_responses import (
    AgentResponse,
    ClassificationResponse,
    PriorityResponse,
    ResponseGenerationResponse,
    SummaryResponse,
    parse_agent_response,
)
from .ai_enhancements import (
    AdvancedEmailAnalyzer,
    EmailContext,
    IntelligentTriageResult,
    SmartResponseGenerator,
    intelligent_triage_email,
)
from .cache import (
    LRUCache,
    SmartCache,
    cached_agent_operation,
    get_persistent_cache,
    get_smart_cache,
)
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .classifier import ClassifierAgent
from .cli_enhancements import (
    AdvancedCLIProcessor,
    run_async_cli_function,
)
from .core import process_email
from .health import HealthChecker, HealthMonitor, HealthStatus, get_health_checker
from .logging_utils import get_logger, setup_structured_logging
from .metrics_export import MetricsCollector, get_metrics_collector
from .performance import (
    Timer,
    enable_performance_monitoring,
    get_performance_tracker,
    get_resource_monitor,
    timed,
)
from .pipeline import (
    EmailTriagePipeline,
    TriageResult,
    triage_batch,
    triage_email,
    triage_email_enhanced,
)
from .priority import PriorityAgent
from .provider import GmailProvider
from .rate_limiter import RateLimitConfig, RateLimiter, get_rate_limiter
from .resilience import (
    AdaptiveRetry,
    BulkheadIsolation,
    GracefulDegradation,
    HealthCheck,
    ResilienceOrchestrator,
    resilience,
)
from .response import ResponseAgent
from .retry_utils import RetryConfig, retry_with_backoff
from .sanitization import EmailSanitizer, SanitizationConfig, sanitize_email_content
from .scalability import (
    AdaptiveScalingProcessor,
    BatchOptimizationConfig,
    ProcessingStats,
    benchmark_performance,
    get_adaptive_processor,
    process_batch_with_scaling,
)
from .summarizer import SummarizerAgent
from .validation import (
    ConfigValidator,
    EmailValidator,
    ValidationResult,
    get_email_validator,
    validate_email_content,
)


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
    "EmailTriagePipeline",
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
    "AdaptiveScalingProcessor",
    "BatchOptimizationConfig",
    "ProcessingStats",
    "benchmark_performance",
    "get_adaptive_processor",
    "process_batch_with_scaling",
    "EmailContext",
    "IntelligentTriageResult",
    "AdvancedEmailAnalyzer",
    "SmartResponseGenerator",
    "intelligent_triage_email",
    "AdvancedCLIProcessor",
    "run_async_cli_function",
    "SecurityThreat",
    "SecurityAnalysisResult",
    "AdvancedSecurityScanner",
    "SecurityAuditLogger",
    "perform_security_scan",
    "BulkheadIsolation",
    "GracefulDegradation",
    "AdaptiveRetry",
    "HealthCheck",
    "ResilienceOrchestrator",
    "resilience",
    "ScalingMetrics",
    "AdaptiveLoadBalancer",
    "HighPerformanceProcessor",
    "process_batch_high_performance",
    "get_performance_insights",
    "optimize_system_performance",
    "high_performance_processor",
    "__version__",
]

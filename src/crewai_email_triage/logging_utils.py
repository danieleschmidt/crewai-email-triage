"""Structured logging utilities with request correlation."""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable to store request ID across async/threaded operations
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs with request correlation."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': time.time(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_entry['request_id'] = request_id

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage'):
                extra_fields[key] = value

        if extra_fields:
            log_entry['extra'] = extra_fields

        return json.dumps(log_entry)


def generate_request_id() -> str:
    """Generate a new unique request ID."""
    return str(uuid.uuid4())[:8]  # Short UUID for readability


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the request ID for the current context.
    
    Parameters
    ----------
    request_id : str, optional
        Request ID to set. If None, generates a new one.
        
    Returns
    -------
    str
        The request ID that was set.
    """
    if request_id is None:
        request_id = generate_request_id()
    _request_id.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return _request_id.get()


def clear_request_id() -> None:
    """Clear the request ID from context."""
    _request_id.set(None)


class RequestContextLogger:
    """Logger wrapper that automatically includes request context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with automatic request context injection."""
        extra = kwargs.get('extra', {})
        request_id = get_request_id()
        if request_id:
            extra['request_id'] = request_id
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)


def setup_structured_logging(level: int = logging.INFO,
                           structured: bool = True) -> None:
    """Setup structured logging for the application.
    
    Parameters
    ----------
    level : int
        Logging level (default: INFO)
    structured : bool
        Whether to use structured JSON output (default: True)
    """
    formatter = StructuredFormatter() if structured else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    root_logger.addHandler(handler)


def get_logger(name: str) -> RequestContextLogger:
    """Get a logger with automatic request context support.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
        
    Returns
    -------
    RequestContextLogger
        Logger with request context support
    """
    return RequestContextLogger(logging.getLogger(name))


class LoggingContext:
    """Context manager for setting request ID and timing operations."""

    def __init__(self, request_id: Optional[str] = None,
                 operation: Optional[str] = None):
        self.request_id = request_id or generate_request_id()
        self.operation = operation
        self.start_time = None
        self.logger = get_logger(__name__)

    def __enter__(self):
        self.start_time = time.perf_counter()
        set_request_id(self.request_id)

        if self.operation:
            self.logger.info("Starting operation: %s", self.operation,
                           extra={'operation': self.operation, 'phase': 'start'})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0

        if exc_type:
            self.logger.error("Operation failed: %s (%.3fs)",
                            self.operation or 'unknown', elapsed,
                            extra={'operation': self.operation, 'phase': 'error',
                                  'duration': elapsed, 'error_type': exc_type.__name__})
        elif self.operation:
            self.logger.info("Completed operation: %s (%.3fs)",
                           self.operation, elapsed,
                           extra={'operation': self.operation, 'phase': 'complete',
                                 'duration': elapsed})

        clear_request_id()

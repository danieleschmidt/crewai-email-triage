"""Test structured logging functionality."""

import json
import logging
from io import StringIO

import pytest

from crewai_email_triage.logging_utils import (
    StructuredFormatter,
    generate_request_id,
    set_request_id,
    get_request_id,
    clear_request_id,
    RequestContextLogger,
    setup_structured_logging,
    get_logger,
    LoggingContext
)
from crewai_email_triage.pipeline import triage_email


class TestStructuredLogging:
    """Test structured logging components."""

    def test_generate_request_id(self):
        """Test request ID generation."""
        request_id = generate_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) == 8  # Short UUID
        
        # Generate multiple IDs to ensure uniqueness
        ids = [generate_request_id() for _ in range(10)]
        assert len(set(ids)) == 10  # All unique

    def test_request_id_context(self):
        """Test request ID context management."""
        # Initially no request ID
        assert get_request_id() is None
        
        # Set a request ID
        test_id = "test123"
        set_request_id(test_id)
        assert get_request_id() == test_id
        
        # Clear request ID
        clear_request_id()
        assert get_request_id() is None

    def test_set_request_id_auto_generation(self):
        """Test that set_request_id generates ID if none provided."""
        request_id = set_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) == 8
        assert get_request_id() == request_id

    def test_structured_formatter(self):
        """Test structured JSON formatter."""
        formatter = StructuredFormatter()
        
        # Create a log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse as JSON
        log_data = json.loads(formatted)
        
        # Verify required fields
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert log_data["message"] == "Test message"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

    def test_structured_formatter_with_request_id(self):
        """Test formatter includes request ID when available."""
        formatter = StructuredFormatter()
        
        # Set request ID
        test_id = "test123"
        set_request_id(test_id)
        
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["request_id"] == test_id
        
        clear_request_id()

    def test_request_context_logger(self):
        """Test RequestContextLogger functionality."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        base_logger = logging.getLogger("test_context")
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.DEBUG)
        
        context_logger = RequestContextLogger(base_logger)
        
        # Set request ID
        test_id = "context123"
        set_request_id(test_id)
        
        # Log a message
        context_logger.info("Test context message")
        
        # Check output
        output = stream.getvalue()
        log_data = json.loads(output.strip())
        
        assert log_data["message"] == "Test context message"
        assert log_data["request_id"] == test_id
        
        clear_request_id()

    def test_logging_context_manager(self):
        """Test LoggingContext manager."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        logger = logging.getLogger("crewai_email_triage.logging_utils")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Use context manager
        with LoggingContext(operation="test_operation") as ctx:
            assert get_request_id() is not None
            assert ctx.operation == "test_operation"
        
        # Request ID should be cleared after context
        assert get_request_id() is None
        
        # Check log output
        output = stream.getvalue()
        lines = output.strip().split('\n')
        
        start_log = json.loads(lines[0])
        end_log = json.loads(lines[1])
        
        assert "Starting operation: test_operation" in start_log["message"]
        assert "Completed operation: test_operation" in end_log["message"]
        assert start_log["extra"]["phase"] == "start"
        assert end_log["extra"]["phase"] == "complete"

    def test_logging_context_exception_handling(self):
        """Test LoggingContext handles exceptions properly."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        logger = logging.getLogger("crewai_email_triage.logging_utils")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Test exception in context
        with pytest.raises(ValueError):
            with LoggingContext(operation="failing_operation"):
                raise ValueError("Test error")
        
        # Check error was logged
        output = stream.getvalue()
        lines = output.strip().split('\n')
        
        error_log = json.loads(lines[1])  # Second line should be error
        assert "Operation failed: failing_operation" in error_log["message"]
        assert error_log["extra"]["phase"] == "error"
        assert error_log["extra"]["error_type"] == "ValueError"

    def test_setup_structured_logging(self):
        """Test logging setup function."""
        # Test structured logging setup
        setup_structured_logging(level=logging.INFO, structured=True)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0
        
        # Check formatter type
        formatter = root_logger.handlers[0].formatter
        assert isinstance(formatter, StructuredFormatter)

    def test_get_logger_returns_context_logger(self):
        """Test get_logger returns RequestContextLogger."""
        logger = get_logger("test")
        assert isinstance(logger, RequestContextLogger)

    def test_integration_with_triage_email(self):
        """Test that triage_email uses structured logging."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        # Set up logger to capture pipeline logs
        pipeline_logger = logging.getLogger("crewai_email_triage.pipeline")
        pipeline_logger.addHandler(handler)
        pipeline_logger.setLevel(logging.INFO)
        
        # Run triage
        result = triage_email("test message")
        
        # Check logs were generated
        output = stream.getvalue()
        assert output  # Should have log output
        
        # Parse logs
        lines = [line for line in output.strip().split('\n') if line]
        logs = [json.loads(line) for line in lines]
        
        # Should have start and completion logs
        start_logs = [log for log in logs if "Starting" in log["message"]]
        complete_logs = [log for log in logs if "completed" in log["message"]]
        
        assert len(start_logs) > 0
        assert len(complete_logs) > 0
        
        # All logs should have request IDs
        for log in logs:
            if "request_id" in log:
                assert isinstance(log["request_id"], str)

    def test_batch_logging_has_separate_request_ids(self):
        """Test that batch processing has proper request ID separation."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        # Set up logger
        pipeline_logger = logging.getLogger("crewai_email_triage.pipeline")
        pipeline_logger.addHandler(handler)
        pipeline_logger.setLevel(logging.INFO)
        
        from crewai_email_triage.pipeline import triage_batch
        
        # Run batch
        messages = ["message 1", "message 2"]
        results = triage_batch(messages)
        
        # Check that we have logs
        output = stream.getvalue()
        assert output
        
        # Should have multiple request IDs in parallel mode
        lines = [line for line in output.strip().split('\n') if line]
        logs = [json.loads(line) for line in lines if json.loads(line).get("request_id")]
        
        request_ids = set(log["request_id"] for log in logs)
        assert len(request_ids) >= 1  # At least one request ID
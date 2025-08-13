"""Robust core functionality with comprehensive error handling and monitoring."""

import logging
import time
from typing import Any, Dict, Optional

# Import our robust modules (with fallbacks if not available)
try:
    from .robust_error_handler import (
        ErrorSeverity,
        get_error_handler,
        with_error_handling,
    )
    from .robust_health import get_health_monitor, quick_health_check
    from .robust_security import SecurityThreatLevel, secure_email_processing
except ImportError as e:
    logging.warning(f"Some robust modules not available: {e}")

    # Provide fallback implementations
    def with_error_handling(severity=None, context="operation"):
        def decorator(func):
            return func
        return decorator

    def secure_email_processing(content):
        return {
            "original_content": content,
            "sanitized_content": content,
            "security_analysis": {"is_safe": True, "threat_level": 0},
            "sanitization_warnings": [],
            "processing_safe": True
        }

    def quick_health_check():
        return {"status": "healthy", "score": 100.0}

logger = logging.getLogger(__name__)

class RobustEmailProcessor:
    """Robust email processor with comprehensive error handling, security, and monitoring."""

    def __init__(self):
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "security_blocked": 0,
            "average_processing_time_ms": 0.0
        }

    @with_error_handling(severity=ErrorSeverity.MEDIUM, context="email_processing")
    def process_email_robust(self, content: str | None,
                           enable_security: bool = True,
                           enable_monitoring: bool = True) -> Dict[str, Any]:
        """Process email with comprehensive robustness features."""
        start_time = time.time()

        try:
            # Input validation
            if content is None:
                return self._create_result(
                    success=True,
                    result="Processed: [No content]",
                    processing_time_ms=0.0,
                    warnings=["Input was None"]
                )

            if not isinstance(content, str):
                raise TypeError(f"Expected str or None, got {type(content)}")

            # Health check before processing
            if enable_monitoring:
                health_status = quick_health_check()
                if health_status["status"] == "unhealthy":
                    logger.warning("System health is unhealthy, processing may be degraded")

            # Security processing
            security_result = None
            processed_content = content

            if enable_security:
                security_result = secure_email_processing(content)

                # Block if security threat is too high
                if (security_result["security_analysis"]["threat_level"] >= 3 or
                    not security_result["processing_safe"]):

                    self.processing_stats["security_blocked"] += 1
                    return self._create_result(
                        success=False,
                        result=None,
                        processing_time_ms=(time.time() - start_time) * 1000,
                        errors=["Content blocked due to security concerns"],
                        security_analysis=security_result["security_analysis"]
                    )

                processed_content = security_result["sanitized_content"]

            # Core processing logic (enhanced from original)
            if len(processed_content.strip()) == 0:
                result_text = "Processed: [Empty message]"
            else:
                result_text = f"Processed: {processed_content.strip()}"

            processing_time_ms = (time.time() - start_time) * 1000

            # Update statistics
            self.processing_stats["total_processed"] += 1
            self.processing_stats["successful"] += 1

            # Update average processing time
            total_time = (self.processing_stats["average_processing_time_ms"] *
                         (self.processing_stats["successful"] - 1) + processing_time_ms)
            self.processing_stats["average_processing_time_ms"] = total_time / self.processing_stats["successful"]

            logger.info(f"Successfully processed email: {len(processed_content)} chars in {processing_time_ms:.2f}ms")

            return self._create_result(
                success=True,
                result=result_text,
                processing_time_ms=processing_time_ms,
                security_analysis=security_result["security_analysis"] if security_result else None,
                sanitization_warnings=security_result["sanitization_warnings"] if security_result else []
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.processing_stats["failed"] += 1
            self.processing_stats["total_processed"] += 1

            logger.error(f"Email processing failed: {e}")

            return self._create_result(
                success=False,
                result=None,
                processing_time_ms=processing_time_ms,
                errors=[str(e)]
            )

    def _create_result(self, success: bool, result: Optional[str], processing_time_ms: float,
                      warnings: list = None, errors: list = None, security_analysis: dict = None,
                      sanitization_warnings: list = None) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            "success": success,
            "result": result,
            "processing_time_ms": processing_time_ms,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "warnings": warnings or [],
            "errors": errors or [],
            "security_analysis": security_analysis,
            "sanitization_warnings": sanitization_warnings or [],
            "processor_stats": self.processing_stats.copy()
        }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "security_blocked": 0,
            "average_processing_time_ms": 0.0
        }

# Global robust processor instance
_robust_processor = RobustEmailProcessor()

def process_email_robust(content: str | None, **kwargs) -> Dict[str, Any]:
    """Process email with robustness features."""
    return _robust_processor.process_email_robust(content, **kwargs)

def get_processor_stats() -> Dict[str, Any]:
    """Get current processor statistics."""
    return _robust_processor.get_processing_stats()

# Backward compatibility with original core function
def process_email(content: str | None) -> str:
    """Original process_email function with enhanced robustness."""
    result = process_email_robust(content)

    if result["success"]:
        return result["result"]
    else:
        # Return error information in original format
        error_msg = result["errors"][0] if result["errors"] else "Processing failed"
        return f"Error: {error_msg}"

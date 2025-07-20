# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-01-01
- Initial release with classifier, priority, summarizer and response agents.

## [0.2.0] - 2024-02-01
- Configuration file for keywords and priority scores
- Added logging and metrics with --verbose flag

## [0.3.0] - 2024-03-01
- Batch processing with shared agents
- Gmail integration via IMAP
- Documentation cleanup

## [0.4.0] - 2024-07-20
- Comprehensive metrics export system with Prometheus format
- Thread-safe metrics collection (counters, gauges, histograms)
- HTTP endpoint for metrics serving (/metrics)
- CLI options for metrics export (--export-metrics, --metrics-port)
- Agent-level performance tracking and sanitization metrics
- Backward compatibility with legacy METRICS dict

## [0.4.1] - 2024-07-20 (Security Fix)
- **SECURITY**: Fixed critical vulnerability in email sanitization caching
- Removed @lru_cache decorator from sanitize method to prevent PII exposure
- Added security documentation and tests for sanitization
- Maintained excellent performance (28k+ emails/sec) without caching
- Updated test suite to verify no sensitive data caching

## [0.4.2] - 2024-07-20 (HTTP Security Hardening)
- **SECURITY**: Comprehensive HTTP endpoint security improvements
- Added proper HTTP method validation (405 Method Not Allowed responses)
- Implemented security headers (X-Content-Type-Options, X-Frame-Options, etc.)
- Added /health endpoint for monitoring and health checks
- Proper HEAD request support and standardized error responses
- Hidden server version information for security

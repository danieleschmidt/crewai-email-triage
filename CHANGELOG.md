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

## [0.4.3] - 2024-07-20 (Memory Management Fix)
- **STABILITY**: Fixed memory leak vulnerability in histogram collection
- Implemented bounded histogram storage using deque with configurable limits
- Added METRICS_HISTOGRAM_MAX_SIZE environment variable (default: 1000)
- Enhanced Prometheus export with efficient histogram statistics
- Comprehensive thread safety testing for concurrent metrics collection
- Maintains performance while preventing unbounded memory growth

## [0.4.4] - 2024-07-20 (Exception Handling Improvements)
- **DEBUGGING**: Enhanced exception handling with specific exception types
- Replaced generic Exception catches with targeted error handling
- Added detailed error categorization for sanitization (Unicode, Memory, Regex errors)
- Improved agent operation error handling (Timeout, JSON, Connection errors)
- Enhanced pipeline critical error logging with context and stack traces
- Added specific metrics for different exception types to improve monitoring

## [0.4.6] - 2024-07-20 (Secure Credential Storage)
- **SECURITY**: Implemented secure credential storage system with encryption
- Added memory-safe password handling to prevent exposure in memory dumps
- Created SecureCredentialManager with AES-256 encryption and PBKDF2 key derivation
- Automatic migration from environment variables to secure storage
- Thread-safe credential operations with file locking and secure permissions (600)
- Enhanced GmailProvider to use secure credential storage instead of plaintext memory storage
- Comprehensive test suite for credential security and memory safety
- Backward compatibility maintained for existing environment variable usage

## [0.4.7] - 2024-07-21 (Abstract Agent Base Class)
- **MAINTAINABILITY**: Implemented abstract base class pattern for all email processing agents
- Added ABC (Abstract Base Class) enforcement for Agent interface with @abstractmethod decorator
- Created LegacyAgent class for backward compatibility with existing direct Agent instantiation
- Enhanced interface documentation with clear input/output format specifications
- Added comprehensive test suite for abstract behavior validation and contract enforcement
- Improved code maintainability and extensibility for future agent implementations
- All concrete agents (Classifier, Priority, Summarizer, Response) now enforce the run() method contract

## [0.4.5] - 2024-07-20 (Network Retry Logic)
- **RELIABILITY**: Implemented comprehensive retry logic with exponential backoff
- Added configurable retry utilities with exponential backoff and jitter
- Applied retry logic to all IMAP operations (connect, authenticate, search, fetch)
- Added retry logic to all agent operations (classifier, priority, summarizer, responder)
- Environment-configurable retry parameters (attempts, delays, backoff factors)
- Comprehensive retry logging and failure metrics tracking
- Prevents temporary network failures from causing data loss or processing failures

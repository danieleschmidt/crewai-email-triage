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

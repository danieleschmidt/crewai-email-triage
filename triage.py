"""Command line interface for the email triage pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from crewai_email_triage.pipeline import get_legacy_metrics, triage_batch
from crewai_email_triage.scalability import benchmark_performance, process_batch_with_scaling

from crewai_email_triage import __version__, triage_email, GmailProvider
from crewai_email_triage.pipeline import triage_email_enhanced
from crewai_email_triage.health import get_health_checker, HealthMonitor
from crewai_email_triage.performance import get_performance_tracker, get_resource_monitor, PerformanceReport
from crewai_email_triage.cache import get_smart_cache
from crewai_email_triage.config import set_config
from crewai_email_triage.logging_utils import setup_structured_logging, LoggingContext
from crewai_email_triage.metrics_export import (
    get_metrics_collector, PrometheusExporter, MetricsEndpoint, MetricsConfig
)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run email triage")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--message", help="Email content to triage")
    group.add_argument(
        "--stdin", action="store_true", help="Read message content from standard input"
    )
    group.add_argument(
        "--file", type=argparse.FileType("r"), help="Read message content from a file"
    )
    group.add_argument(
        "--batch-file",
        type=argparse.FileType("r"),
        help="Read multiple messages from a file, one per line",
    )
    group.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    group.add_argument(
        "--gmail", action="store_true", help="Process unread Gmail messages"
    )
    group.add_argument(
        "--health", action="store_true", help="Run system health checks"
    )
    group.add_argument(
        "--performance", action="store_true", help="Show performance statistics"
    )
    group.add_argument(
        "--cache-stats", action="store_true", help="Show cache statistics"
    )
    group.add_argument(
        "--benchmark", action="store_true", help="Run comprehensive performance benchmark"
    )
    parser.add_argument(
        "--output", type=argparse.FileType("w"), help="Write JSON result to the given file"
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--structured-logs", action="store_true", help="Output structured JSON logs")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--max-messages", type=int, default=10, help="Maximum Gmail messages to process")
    parser.add_argument("--disable-sanitization", action="store_true", help="Disable content sanitization (not recommended)")
    parser.add_argument("--sanitization-level", choices=['basic', 'standard', 'strict'], default='standard', 
                       help="Content sanitization level (default: standard)")
    parser.add_argument("--export-metrics", action="store_true", help="Start HTTP server to export Prometheus metrics")
    parser.add_argument("--metrics-port", type=int, default=8080, help="Port for metrics HTTP server (default: 8080)")
    parser.add_argument("--metrics-path", default="/metrics", help="Path for metrics endpoint (default: /metrics)")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced mode with detailed metadata")
    parser.add_argument("--output-format", choices=['json', 'yaml', 'table', 'summary'], default='json',
                       help="Output format (default: json)")
    parser.add_argument("--show-timing", action="store_true", help="Show detailed timing information")
    parser.add_argument("--show-metadata", action="store_true", help="Show processing metadata")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing for batch operations")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads (default: 4)")
    parser.add_argument("--health-checks", nargs='+', metavar='CHECK', 
                       help="Specific health checks to run (memory, cpu, disk, agents, metrics, rate_limiter)")
    parser.add_argument("--start-monitor", action="store_true", help="Start continuous health monitoring")
    
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive scaling processor for optimal performance")
    parser.add_argument("--monitor-interval", type=float, default=30.0, help="Health monitoring interval in seconds (default: 30)")
    parser.add_argument("--enable-perf-monitoring", action="store_true", help="Enable performance monitoring")
    parser.add_argument("--enable-caching", action="store_true", help="Enable intelligent caching")
    parser.add_argument("--cache-clear", action="store_true", help="Clear all caches")
    return parser


def _dump(data: object, pretty: bool) -> str:
    return json.dumps(data, indent=2 if pretty else None)


def _format_output(result, output_format: str, show_timing: bool = False, show_metadata: bool = False) -> str:
    """Format triage result according to specified format."""
    if output_format == 'json':
        if hasattr(result, 'to_dict'):
            data = result.to_dict() if show_metadata else {
                'category': result.category,
                'priority': result.priority,
                'summary': result.summary,
                'response': result.response
            }
            if show_timing and hasattr(result, 'processing_time_ms'):
                data['processing_time_ms'] = result.processing_time_ms
        else:
            data = result
        return json.dumps(data, indent=2)
    
    elif output_format == 'yaml':
        try:
            import yaml
            if hasattr(result, 'to_dict'):
                data = result.to_dict() if show_metadata else {
                    'category': result.category,
                    'priority': result.priority,
                    'summary': result.summary,
                    'response': result.response
                }
            else:
                data = result
            return yaml.dump(data, default_flow_style=False)
        except ImportError:
            return "Error: PyYAML not installed. Install with: pip install PyYAML"
    
    elif output_format == 'table':
        category = result.category if hasattr(result, 'category') else result.get('category', 'unknown')
        priority = result.priority if hasattr(result, 'priority') else result.get('priority', 0)
        summary = result.summary if hasattr(result, 'summary') else result.get('summary', 'No summary')
        
        lines = [
            "=" * 60,
            f"Category: {category}",
            f"Priority: {priority}",
            f"Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}",
            "=" * 60
        ]
        
        if show_timing and hasattr(result, 'processing_time_ms'):
            lines.insert(-1, f"Processing Time: {result.processing_time_ms:.2f}ms")
        
        return "\n".join(lines)
    
    elif output_format == 'summary':
        category = result.category if hasattr(result, 'category') else result.get('category', 'unknown')
        priority = result.priority if hasattr(result, 'priority') else result.get('priority', 0)
        
        output = f"[{category.upper()}] Priority {priority}"
        if show_timing and hasattr(result, 'processing_time_ms'):
            output += f" ({result.processing_time_ms:.2f}ms)"
        return output
    
    return str(result)


def _run_interactive(pretty: bool, config_dict: dict = None, enhanced: bool = False, 
                     output_format: str = 'json', show_timing: bool = False, show_metadata: bool = False) -> None:
    while True:
        try:
            sys.stderr.write("message> ")
            sys.stderr.flush()
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            sys.stderr.write("\n")
            break
        if not line:
            break
        
        if enhanced:
            result = triage_email_enhanced(line, config_dict=config_dict)
        else:
            result = triage_email(line, config_dict=config_dict)
        
        if output_format == 'json' and not enhanced:
            print(_dump(result, pretty))
        else:
            print(_format_output(result, output_format, show_timing, show_metadata))


def _read_single_message(args: argparse.Namespace) -> str:
    if args.stdin:
        return sys.stdin.read()
    if args.file:
        with args.file as fh:
            return fh.read()
    return args.message


def _read_gmail(max_messages: int) -> list[str]:
    client = GmailProvider.from_env()
    return client.fetch_unread(max_messages)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_structured_logging(level=log_level, structured=args.structured_logs)

    # Load configuration
    config_dict = None
    if args.config:
        from crewai_email_triage.config import load_config
        config_dict = load_config(args.config)
        set_config(args.config)  # Also set global for backward compatibility
    
    # Enable performance monitoring if requested
    if args.enable_perf_monitoring:
        from crewai_email_triage.performance import enable_performance_monitoring
        enable_performance_monitoring()
        logging.info("Performance monitoring enabled")

    # Setup metrics export if requested
    metrics_endpoint = None
    if args.export_metrics:
        config = MetricsConfig(
            enabled=True,
            export_port=args.metrics_port,
            export_path=args.metrics_path
        )
        collector = get_metrics_collector()
        exporter = PrometheusExporter(collector)
        metrics_endpoint = MetricsEndpoint(exporter, config)
        
        try:
            metrics_endpoint.start()
            logging.info("Metrics endpoint started on http://localhost:%d%s", 
                        args.metrics_port, args.metrics_path)
        except Exception as e:
            logging.error("Failed to start metrics endpoint: %s", e)
            metrics_endpoint = None

    if args.health:
        health_checker = get_health_checker()
        health_result = health_checker.check_health(args.health_checks)
        
        if args.output_format == 'json':
            output = json.dumps(health_result.to_dict(), indent=2 if args.pretty else None)
        elif args.output_format == 'table':
            lines = [
                "=" * 80,
                f"System Health: {health_result.status.value.upper()}",
                f"Response Time: {health_result.response_time_ms:.2f}ms",
                "=" * 80
            ]
            for check in health_result.checks:
                status_icon = "✓" if check.status.name == "HEALTHY" else "⚠" if check.status.name == "DEGRADED" else "✗"
                lines.append(f"{status_icon} {check.name}: {check.message} ({check.response_time_ms:.2f}ms)")
            lines.append("=" * 80)
            output = "\n".join(lines)
        else:
            output = f"System Health: {health_result.status.value} ({health_result.response_time_ms:.2f}ms)"
            
        print(output)
        
        if args.start_monitor:
            monitor = HealthMonitor(args.monitor_interval)
            monitor.start()
            logging.info("Health monitor started. Press Ctrl+C to stop.")
            try:
                import signal
                signal.pause()
            except KeyboardInterrupt:
                monitor.stop()
                logging.info("Health monitor stopped")
        return

    if args.performance:
        performance_tracker = get_performance_tracker()
        report = PerformanceReport(performance_tracker)
        perf_data = report.generate_report()
        
        if args.output_format == 'json':
            output = json.dumps(perf_data, indent=2 if args.pretty else None)
        elif args.output_format == 'table':
            lines = [
                "=" * 80,
                "Performance Report",
                "=" * 80,
                f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(perf_data['timestamp']))}",
                f"Total Metrics: {perf_data['summary']['total_metrics']}",
                f"Avg Response Time: {perf_data['summary']['avg_response_time']:.2f}ms",
                f"Memory Usage: {perf_data['summary']['memory_usage_mb']:.1f}MB",
                "=" * 80,
                "Recommendations:",
            ]
            for rec in perf_data['recommendations']:
                lines.append(f"- {rec}")
            lines.append("=" * 80)
            output = "\n".join(lines)
        else:
            output = f"Performance Report - {perf_data['summary']['total_metrics']} metrics tracked"
            
        print(output)
        return

    if args.cache_stats:
        cache = get_smart_cache()
        cache_stats = cache.get_stats()
        
        if args.output_format == 'json':
            output = json.dumps(cache_stats, indent=2 if args.pretty else None)
        elif args.output_format == 'table':
            lines = [
                "=" * 80,
                "Cache Statistics",
                "=" * 80
            ]
            for cache_name, stats in cache_stats.items():
                lines.extend([
                    f"{cache_name.upper()} Cache:",
                    f"  Hit Rate: {stats.get('hit_rate', 0):.1%}",
                    f"  Entries: {stats.get('current_size', 0)}/{stats.get('max_size', 0)}",
                    f"  Memory: {stats.get('size_bytes', 0) / (1024*1024):.1f}MB",
                    f"  Hits: {stats.get('hits', 0)}, Misses: {stats.get('misses', 0)}",
                    ""
                ])
            lines.append("=" * 80)
            output = "\n".join(lines)
        else:
            total_hits = sum(stats.get('hits', 0) for stats in cache_stats.values())
            total_misses = sum(stats.get('misses', 0) for stats in cache_stats.values())
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            output = f"Cache Stats - Hit Rate: {hit_rate:.1%} ({total_hits} hits, {total_misses} misses)"
            
        print(output)
        return

    if args.cache_clear:
        cache = get_smart_cache()
        cache.clear_all()
        print("All caches cleared")
        return
        
    if args.benchmark:
        print("Running comprehensive performance benchmark...")
        benchmark_results = benchmark_performance()
        
        if args.output_format == 'json':
            output = json.dumps(benchmark_results, indent=2 if args.pretty else None)
        else:
            lines = ["=" * 60, "Performance Benchmark Results", "=" * 60]
            for config_name, results in benchmark_results.items():
                if config_name != 'summary':
                    throughput = results['throughput_items_per_second']
                    avg_time = results['avg_time_seconds']
                    lines.extend([
                        f"{config_name}:",
                        f"  Throughput: {throughput:.1f} items/sec",
                        f"  Avg Time: {avg_time:.3f}s",
                        ""
                    ])
            
            summary = benchmark_results['summary']
            lines.extend([
                f"🏆 Winner: {summary['best_configuration']}",
                f"📈 Best Throughput: {summary['best_throughput']:.1f} items/sec",
                f"📊 Test Messages: {summary['test_message_count']}",
                "=" * 60
            ])
            output = "\n".join(lines)
            
        print(output)
        return

    if args.interactive:
        _run_interactive(args.pretty, config_dict, args.enhanced, args.output_format, 
                        args.show_timing, args.show_metadata)
        metrics = get_legacy_metrics()
        logging.info("Processed %d message(s)", metrics["processed"])
        return

    with LoggingContext(operation="cli_operation"):
        if args.gmail:
            messages = _read_gmail(args.max_messages)
            if args.enhanced:
                results = [triage_email_enhanced(msg, config_dict=config_dict) for msg in messages]
                if args.output_format == 'json':
                    output = json.dumps([r.to_dict() if args.show_metadata else {
                        'category': r.category, 'priority': r.priority, 
                        'summary': r.summary, 'response': r.response
                    } for r in results], indent=2)
                else:
                    output = '\n\n'.join(_format_output(r, args.output_format, args.show_timing, args.show_metadata) for r in results)
            else:
                batch_result = triage_batch(messages, config_dict=config_dict, parallel=args.parallel, max_workers=args.max_workers)
                output = _dump(batch_result, args.pretty)
        elif args.batch_file:
            with args.batch_file as fh:
                messages = [line.strip() for line in fh if line.strip()]
            if args.enhanced:
                results = [triage_email_enhanced(msg, config_dict=config_dict) for msg in messages]
                if args.output_format == 'json':
                    output = json.dumps([r.to_dict() if args.show_metadata else {
                        'category': r.category, 'priority': r.priority, 
                        'summary': r.summary, 'response': r.response
                    } for r in results], indent=2)
                else:
                    output = '\n\n'.join(_format_output(r, args.output_format, args.show_timing, args.show_metadata) for r in results)
            else:
                batch_result = triage_batch(messages, config_dict=config_dict, parallel=args.parallel, max_workers=args.max_workers)
                output = _dump(batch_result, args.pretty)
        else:
            message = _read_single_message(args)
            if args.enhanced:
                result = triage_email_enhanced(message, config_dict=config_dict)
                output = _format_output(result, args.output_format, args.show_timing, args.show_metadata)
            else:
                result = triage_email(message, config_dict=config_dict)
                output = _dump(result, args.pretty)

    if args.output:
        with args.output as fh:
            fh.write(output + "\n")
    else:
        print(output)

    metrics = get_legacy_metrics()
    logging.info("Processed %d message(s) in %.3fs", metrics["processed"], metrics["total_time"])
    
    # Cleanup metrics endpoint
    if metrics_endpoint:
        metrics_endpoint.stop()
        logging.info("Metrics endpoint stopped")


if __name__ == "__main__":
    main()

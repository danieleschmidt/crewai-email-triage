"""Command line interface for the email triage pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from crewai_email_triage.pipeline import get_legacy_metrics, triage_batch

from crewai_email_triage import __version__, triage_email, GmailProvider
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
    return parser


def _dump(data: object, pretty: bool) -> str:
    return json.dumps(data, indent=2 if pretty else None)


def _run_interactive(pretty: bool, config_dict: dict = None) -> None:
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
        print(_dump(triage_email(line, config_dict=config_dict), pretty))


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

    if args.interactive:
        _run_interactive(args.pretty, config_dict)
        metrics = get_legacy_metrics()
        logging.info("Processed %d message(s)", metrics["processed"])
        return

    with LoggingContext(operation="cli_operation"):
        if args.gmail:
            messages = _read_gmail(args.max_messages)
            output = _dump(triage_batch(messages, config_dict=config_dict), args.pretty)
        elif args.batch_file:
            with args.batch_file as fh:
                messages = [line.strip() for line in fh if line.strip()]
            output = _dump(triage_batch(messages, config_dict=config_dict), args.pretty)
        else:
            message = _read_single_message(args)
            output = _dump(triage_email(message, config_dict=config_dict), args.pretty)

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

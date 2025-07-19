"""Command line interface for the email triage pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from crewai_email_triage.pipeline import METRICS, triage_batch

from crewai_email_triage import __version__, triage_email, GmailProvider
from crewai_email_triage.config import set_config
from crewai_email_triage.logging_utils import setup_structured_logging, LoggingContext


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
    return parser


def _dump(data: object, pretty: bool) -> str:
    return json.dumps(data, indent=2 if pretty else None)


def _run_interactive(pretty: bool) -> None:
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
        print(_dump(triage_email(line), pretty))


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

    if args.config:
        set_config(args.config)

    if args.interactive:
        _run_interactive(args.pretty)
        logging.info("Processed %d message(s)", METRICS["processed"])
        return

    with LoggingContext(operation="cli_operation"):
        if args.gmail:
            messages = _read_gmail(args.max_messages)
            output = _dump(triage_batch(messages), args.pretty)
        elif args.batch_file:
            with args.batch_file as fh:
                messages = [line.strip() for line in fh if line.strip()]
            output = _dump(triage_batch(messages), args.pretty)
        else:
            message = _read_single_message(args)
            output = _dump(triage_email(message), args.pretty)

    if args.output:
        with args.output as fh:
            fh.write(output + "\n")
    else:
        print(output)

    logging.info("Processed %d message(s) in %.3fs", METRICS["processed"], METRICS["total_time"])


if __name__ == "__main__":
    main()

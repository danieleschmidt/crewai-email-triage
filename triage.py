#!/usr/bin/env python
"""Command-line interface for the email triage pipeline."""

from __future__ import annotations

import argparse
import sys

from crewai_email_triage import triage_email


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the triage pipeline on an email string"
    )
    parser.add_argument(
        "text", nargs="?", help="Email text to process. Reads from stdin if omitted."
    )
    parser.add_argument(
        "-f", "--file", type=argparse.FileType("r"), help="Path to a file containing email text"
    )
    parser.add_argument(
        "--high-keywords",
        help="Comma-separated keywords that indicate high priority",
    )
    parser.add_argument(
        "--medium-keywords",
        help="Comma-separated keywords that indicate medium priority",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    args = parser.parse_args()

    if args.file:
        content = args.file.read()
    elif args.text:
        content = args.text
    else:
        content = sys.stdin.read()

    high = set(args.high_keywords.split(",")) if args.high_keywords else None
    medium = set(args.medium_keywords.split(",")) if args.medium_keywords else None

    result = triage_email(
        content.strip(), high_keywords=high, medium_keywords=medium
    )
    if args.json:
        import json

        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()

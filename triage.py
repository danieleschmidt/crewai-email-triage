import argparse
import json
import sys

from crewai_email_triage import __version__, triage_email


def main() -> None:
    parser = argparse.ArgumentParser(description="Run email triage")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--message",
        help="Email content to triage",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="Read message content from standard input",
    )
    group.add_argument(
        "--file",
        type=argparse.FileType("r"),
        help="Read message content from a file",
    )
    group.add_argument(
        "--batch-file",
        type=argparse.FileType("r"),
        help="Read multiple messages from a file, one per line",
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        help="Write JSON result to the given file",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    if args.interactive:
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
            print(
                json.dumps(
                    triage_email(line),
                    indent=2 if args.pretty else None,
                )
            )
        return

    message = args.message

    output_data = None
    if args.batch_file:
        messages = [line.strip() for line in args.batch_file if line.strip()]
        args.batch_file.close()
        results = [triage_email(msg) for msg in messages]
        output_data = json.dumps(results, indent=2 if args.pretty else None)
    else:
        if args.stdin:
            message = sys.stdin.read()
        elif args.file:
            message = args.file.read()
            args.file.close()

        result = triage_email(message)
        output_data = json.dumps(result, indent=2 if args.pretty else None)

    if args.output:
        args.output.write(output_data + "\n")
        args.output.close()
    else:
        print(output_data)


if __name__ == "__main__":
    main()

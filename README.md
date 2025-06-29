# CrewAI Email Triage

Smart assistant that classifies, prioritizes, summarizes and drafts replies for email messages.

## Features

- **Multi-Agent Pipeline**: classifier, priority scorer, summarizer and response generator
- **Configurable Keywords**: edit `default_config.json`, set `CREWAI_CONFIG`, or use `--config`
- **Batch Processing**: reuse agents to handle multiple messages
- **Gmail Integration**: fetch unread messages via IMAP
- **Verbose Metrics**: `--verbose` flag shows processing statistics

## Quick Start

```bash
# Install in editable mode with test extras
pip install -e .[test]

# Triage a single message
python triage.py --message "Urgent meeting tomorrow!" --pretty

# Process multiple messages from a file
python triage.py --batch-file messages.txt

# Process unread Gmail messages (requires $GMAIL_USER and $GMAIL_PASSWORD)
python triage.py --gmail --max-messages 5
```

## Configuration

The package ships with a `default_config.json` containing classifier keywords and priority scores.
You can supply a custom JSON file via `--config /path/to/file` or set the environment variable `CREWAI_CONFIG`.

```bash
# Example custom config
echo '{"classifier": {"urgent": ["urgent", "asap"]}}' > mycfg.json
python triage.py --message "ASAP reply needed" --config mycfg.json
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and commit guidelines.

## License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.

# CrewAI Email Triage

Smart assistant that classifies, prioritizes, summarizes and drafts replies for email messages.

## Features

- **Multi-Agent Pipeline**: classifier, priority scorer, summarizer and response generator
- **Configurable Keywords**: edit `default_config.json`, set `CREWAI_CONFIG`, or use `--config`
- **Batch Processing**: reuse agents to handle multiple messages
- **Parallel Batch**: ``triage_batch(messages, parallel=True)`` for concurrency
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

### Test Setup

Install the package with test dependencies:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with test dependencies
pip install -e ".[test]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests in parallel for speed
pytest -n auto -q

# Run with coverage
pytest --cov=src/crewai_email_triage

# Run specific test file
pytest tests/test_pipeline.py
```

### Pre-commit Hooks

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and commit guidelines.
Install pre-commit hooks to catch lint and secret issues before committing:

```bash
pre-commit install
```

## License

Distributed under the MIT license. See [LICENSE](LICENSE) for details.

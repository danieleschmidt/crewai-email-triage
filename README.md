# crewai-email-triage

A multi-agent email triage system built with pure Python (stdlib only). Classifies, summarizes, and prioritizes emails using keyword matching and sender reputation rules — no LLM required.

## Architecture

Three specialized agents orchestrated by a pipeline:

```
EmailTriagePipeline
├── TriageAgent      → classify into URGENT / ACTION_REQUIRED / INFO / NEWSLETTER / SPAM
├── SummaryAgent     → generate 1-2 sentence body summary
└── PriorityRanker   → score 0.0–10.0 (category + sender trust + recency + keyword intensity)
```

### Categories

| Category | Description |
|---|---|
| `URGENT` | Outages, security alerts, critical deadlines |
| `ACTION_REQUIRED` | Needs review, approval, or response |
| `INFO` | Informational, no action needed |
| `NEWSLETTER` | Mailing lists, digests, marketing |
| `SPAM` | Junk, scams, phishing |

## Quick Start

```bash
# Run the demo (10 sample emails)
python demo.py

# Run tests
python -m pytest tests/ -v
```

## Usage

```python
from datetime import datetime, timezone
from src.email_triage import Email, EmailTriagePipeline

emails = [
    Email(
        id="001",
        subject="URGENT: Production outage",
        sender="pagerduty@pagerduty.com",
        body="The payments service is down. Incident #INC-4821. Join war room immediately.",
        timestamp=datetime.now(tz=timezone.utc),
    ),
    # ... more emails
]

pipeline = EmailTriagePipeline()
report = pipeline.run(emails)
pipeline.print_report(report)
```

### Sample Output

```
════════════════════════════════════════════════════════════════
  EMAIL TRIAGE REPORT  —  2026-03-19 12:13:33
════════════════════════════════════════════════════════════════
  10 emails  |  ACTION_REQUIRED: 4  INFO: 2  NEWSLETTER: 1  SPAM: 1  URGENT: 2

SCORE  CATEGORY          SUBJECT                              SUMMARY
───────────────────────────────────────────────────────────────
 10.0  [URGENT        ]  URGENT: Production outage…           CRITICAL ALERT: The payments-service…
 10.0  [URGENT        ]  Security alert: new login…           We detected a sign-in to your GitHub…
  6.7  [ACTION_REQUIRED]  Your AWS bill is due tomorrow        Your AWS account has an outstanding…
  0.0  [SPAM          ]  Congratulations! You've won…         You have been selected as our lucky…
```

## Project Structure

```
src/email_triage/
├── __init__.py      # Public API
├── models.py        # Email, TriageResult, TriageReport dataclasses
├── agents.py        # TriageAgent, SummaryAgent, PriorityRanker
└── pipeline.py      # EmailTriagePipeline orchestrator

tests/
└── test_triage.py   # 30 tests covering all components

demo.py              # 10-email demo with mixed types
```

## No External Dependencies

Pure stdlib — no pip installs, no LLM API keys, no cloud services. Runs anywhere Python 3.10+ is available.

## License

MIT

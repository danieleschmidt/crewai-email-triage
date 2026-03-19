#!/usr/bin/env python3
"""Demo: triage 10 sample emails and print a prioritized report."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datetime import datetime, timedelta, timezone
from email_triage import Email, EmailTriagePipeline


def main() -> None:
    now = datetime.now(tz=timezone.utc)

    emails = [
        Email(
            id="001",
            subject="URGENT: Production outage — payments service down",
            sender="pagerduty@pagerduty.com",
            body=(
                "CRITICAL ALERT: The payments-service has been down for 15 minutes. "
                "Error rate is at 100%. Customers cannot complete purchases. "
                "Incident #INC-4821 has been opened. Please acknowledge immediately and join "
                "the war room: https://meet.example.com/incident-room"
            ),
            timestamp=now - timedelta(minutes=5),
        ),
        Email(
            id="002",
            subject="Q3 budget approval needed by EOD",
            sender="cfo@acmecorp.com",
            body=(
                "Hi team, I need your approval on the Q3 budget proposal before 5pm today. "
                "Please review the attached spreadsheet and sign off so we can proceed with "
                "vendor contracts. Let me know if you have questions."
            ),
            timestamp=now - timedelta(hours=2),
        ),
        Email(
            id="003",
            subject="Your AWS bill is due tomorrow",
            sender="billing@aws.amazon.com",
            body=(
                "Your AWS account #123456789 has an outstanding balance of $4,218.44 due on "
                "2026-03-20. To avoid service interruption, please ensure your payment method "
                "is up to date. Log in to the billing console to review or pay your invoice."
            ),
            timestamp=now - timedelta(hours=6),
        ),
        Email(
            id="004",
            subject="Weekly engineering newsletter — issue #47",
            sender="digest@engineering-weekly.substack.com",
            body=(
                "This week in engineering: Rust 2.0 release candidate is out, "
                "OpenAI announces new reasoning model, and our picks for top GitHub repos "
                "of the week. Read the full digest online. Unsubscribe at any time."
            ),
            timestamp=now - timedelta(hours=10),
        ),
        Email(
            id="005",
            subject="Meeting notes from yesterday's standup",
            sender="alice@acmecorp.com",
            body=(
                "Hey all, here are the notes from yesterday's standup. "
                "Main items: deployment is on track for Friday, Bob is blocked on the "
                "authentication ticket, and we'll skip the retro next week. "
                "Let me know if I missed anything."
            ),
            timestamp=now - timedelta(hours=18),
        ),
        Email(
            id="006",
            subject="Security alert: new login from unrecognized device",
            sender="security@github.com",
            body=(
                "We detected a sign-in to your GitHub account from a new device in Lagos, Nigeria. "
                "If this was you, no action is needed. If you don't recognize this activity, "
                "please secure your account immediately by changing your password and enabling 2FA. "
                "This is an automated security notification."
            ),
            timestamp=now - timedelta(minutes=30),
        ),
        Email(
            id="007",
            subject="Congratulations! You've won a $1000 Amazon gift card",
            sender="winner@prize-claim-center.net",
            body=(
                "You have been selected as our lucky winner! Claim your $1000 Amazon gift card now. "
                "Act now — this offer expires in 24 hours! Click here to claim your prize: "
                "http://totallylegit.prize-claim-center.net/claim?id=abc123"
            ),
            timestamp=now - timedelta(days=1),
        ),
        Email(
            id="008",
            subject="Can you review my PR before the release?",
            sender="bob@acmecorp.com",
            body=(
                "Hey, could you please review PR #284 when you get a chance? "
                "We're hoping to merge it before the Friday release. It's a small change — "
                "just the rate-limiting fix we discussed. Should be a quick review. Thanks!"
            ),
            timestamp=now - timedelta(hours=3),
        ),
        Email(
            id="009",
            subject="New blog post: 10 tips for writing cleaner Python",
            sender="hello@realpython.com",
            body=(
                "We just published a new tutorial on writing cleaner, more Pythonic code. "
                "Topics include: type hints, dataclasses, pathlib, and walrus operators. "
                "Check it out at realpython.com. You're receiving this because you subscribed "
                "to Real Python updates. Unsubscribe anytime."
            ),
            timestamp=now - timedelta(hours=14),
        ),
        Email(
            id="010",
            subject="FYI: office closed next Monday for the holiday",
            sender="hr@acmecorp.com",
            body=(
                "Just a reminder that the office will be closed next Monday, March 23rd, "
                "in observance of the company holiday. Normal operations resume Tuesday. "
                "Have a great long weekend!"
            ),
            timestamp=now - timedelta(days=2),
        ),
    ]

    pipeline = EmailTriagePipeline()
    report = pipeline.run(emails)
    pipeline.print_report(report)


if __name__ == "__main__":
    main()

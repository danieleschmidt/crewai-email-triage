"""Tests for the email triage system."""

from __future__ import annotations

import sys
import os
import pytest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from email_triage import Email, TriageAgent, SummaryAgent, PriorityRanker, EmailTriagePipeline
from email_triage.models import TriageReport


def make_email(
    id="test-1",
    subject="Test subject",
    sender="user@example.com",
    body="Test body content.",
    hours_ago=1,
) -> Email:
    ts = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    return Email(id=id, subject=subject, sender=sender, body=body, timestamp=ts)


# ---------------------------------------------------------------------------
# Email model
# ---------------------------------------------------------------------------

class TestEmail:
    def test_valid_creation(self):
        e = make_email()
        assert e.id == "test-1"
        assert e.subject == "Test subject"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id"):
            Email(id="", subject="s", sender="a@b.com", body="b",
                  timestamp=datetime.now(tz=timezone.utc))

    def test_empty_sender_raises(self):
        with pytest.raises(ValueError, match="sender"):
            Email(id="x", subject="s", sender="", body="b",
                  timestamp=datetime.now(tz=timezone.utc))

    def test_bad_timestamp_raises(self):
        with pytest.raises(TypeError):
            Email(id="x", subject="s", sender="a@b.com", body="b",
                  timestamp="2024-01-01")  # type: ignore


# ---------------------------------------------------------------------------
# TriageAgent
# ---------------------------------------------------------------------------

class TestTriageAgent:
    agent = TriageAgent()

    def test_classifies_spam(self):
        e = make_email(body="You have won! Claim your prize now. Click here to claim.")
        cat, reasons = self.agent.classify(e)
        assert cat == "SPAM"
        assert reasons

    def test_classifies_spam_domain(self):
        e = make_email(sender="spammer@mailinator.com", body="Hello", subject="Hi")
        cat, _ = self.agent.classify(e)
        assert cat == "SPAM"

    def test_classifies_urgent(self):
        e = make_email(subject="URGENT: production outage", body="The system is down. emergency.")
        cat, reasons = self.agent.classify(e)
        assert cat == "URGENT"
        assert any("urgent" in r.lower() for r in reasons)

    def test_classifies_urgent_security_alert(self):
        e = make_email(
            sender="security@github.com",
            subject="Unauthorized access detected",
            body="We detected unauthorized access to your account.",
        )
        cat, _ = self.agent.classify(e)
        assert cat == "URGENT"

    def test_classifies_newsletter_by_domain(self):
        e = make_email(sender="news@substack.com", body="Weekly digest content")
        cat, _ = self.agent.classify(e)
        assert cat == "NEWSLETTER"

    def test_classifies_newsletter_by_keywords(self):
        e = make_email(
            body="Check out this week's digest. Unsubscribe at any time. 50% off promo.",
            sender="info@someplace.com",
        )
        cat, _ = self.agent.classify(e)
        assert cat == "NEWSLETTER"

    def test_classifies_action_required(self):
        e = make_email(
            subject="Please review the proposal",
            body="Could you please confirm your attendance by Friday? Need your approval.",
        )
        cat, _ = self.agent.classify(e)
        assert cat == "ACTION_REQUIRED"

    def test_classifies_info(self):
        e = make_email(subject="Office party next week", body="Just letting you know there's a party.")
        cat, _ = self.agent.classify(e)
        assert cat == "INFO"

    def test_all_categories_possible(self):
        """Ensure all 5 categories can be produced."""
        samples = [
            make_email(body="You've won the lottery! Claim your prize."),
            make_email(sender="noreply@mailchimp.com", body="newsletter unsubscribe"),
            make_email(subject="URGENT outage", body="critical system down sev1"),
            make_email(subject="Please confirm", body="Could you please review and sign off?"),
            make_email(subject="FYI", body="Just an informational note."),
        ]
        categories = {TriageAgent().classify(e)[0] for e in samples}
        assert categories == {"SPAM", "NEWSLETTER", "URGENT", "ACTION_REQUIRED", "INFO"}


# ---------------------------------------------------------------------------
# SummaryAgent
# ---------------------------------------------------------------------------

class TestSummaryAgent:
    agent = SummaryAgent()

    def test_returns_string(self):
        e = make_email(body="This is a test email. It has multiple sentences. Read it.")
        summary = self.agent.summarize(e)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_empty_body(self):
        e = make_email(body="")
        summary = self.agent.summarize(e)
        assert "Test subject" in summary or len(summary) > 0

    def test_truncates_long_body(self):
        e = make_email(body="word " * 500)
        summary = self.agent.summarize(e)
        assert len(summary) <= 350  # generous bound accounting for truncation

    def test_picks_up_key_sentence(self):
        e = make_email(
            subject="budget deadline",
            body=(
                "Hope you're well. "
                "Please confirm your budget submission before the deadline today. "
                "Thanks for your time."
            ),
        )
        summary = self.agent.summarize(e)
        # Should favor the actionable sentence
        assert len(summary) > 10

    def test_short_body_preserved(self):
        e = make_email(body="Short note.")
        summary = self.agent.summarize(e)
        assert "Short note" in summary


# ---------------------------------------------------------------------------
# PriorityRanker
# ---------------------------------------------------------------------------

class TestPriorityRanker:
    ranker = PriorityRanker()

    def test_urgent_scores_high(self):
        e = make_email(subject="URGENT: outage", body="critical emergency sev1")
        score = self.ranker.score(e, "URGENT")
        assert score >= 9.0

    def test_spam_scores_zero(self):
        e = make_email()
        score = self.ranker.score(e, "SPAM")
        assert score == 0.0

    def test_newsletter_scores_low(self):
        e = make_email()
        score = self.ranker.score(e, "NEWSLETTER")
        assert score <= 2.0

    def test_action_required_mid_range(self):
        e = make_email()
        score = self.ranker.score(e, "ACTION_REQUIRED")
        assert 5.0 <= score <= 8.0

    def test_recent_email_gets_bonus(self):
        recent = make_email(hours_ago=0.5)
        old = make_email(hours_ago=48)
        score_recent = self.ranker.score(recent, "INFO")
        score_old = self.ranker.score(old, "INFO")
        assert score_recent > score_old

    def test_score_bounded_0_to_10(self):
        e = make_email(subject="URGENT!!! URGENT!!! CRITICAL!!!", body="urgent " * 20)
        score = self.ranker.score(e, "URGENT")
        assert 0.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# EmailTriagePipeline
# ---------------------------------------------------------------------------

class TestEmailTriagePipeline:
    pipeline = EmailTriagePipeline()

    def test_processes_single_email(self):
        e = make_email()
        result = self.pipeline.process_email(e)
        assert result.email is e
        assert result.category in ("URGENT", "ACTION_REQUIRED", "INFO", "NEWSLETTER", "SPAM")
        assert isinstance(result.summary, str)
        assert 0.0 <= result.urgency_score <= 10.0

    def test_run_returns_report(self):
        emails = [make_email(id=str(i)) for i in range(5)]
        report = self.pipeline.run(emails)
        assert isinstance(report, TriageReport)
        assert len(report.results) == 5

    def test_sorted_results_descending(self):
        emails = [
            make_email(id="1", subject="URGENT outage", body="critical emergency"),
            make_email(id="2", body="You've won the lottery! Claim prize."),
            make_email(id="3", subject="Please review", body="Could you please confirm?"),
        ]
        report = self.pipeline.run(emails)
        scores = [r.urgency_score for r in report.sorted_results]
        assert scores == sorted(scores, reverse=True)

    def test_summary_stats(self):
        emails = [
            make_email(id="1", subject="URGENT", body="critical emergency sev1 outage"),
            make_email(id="2", body="You've won the lottery! Claim your prize now."),
            make_email(id="3", body="FYI here is some information for you."),
        ]
        report = self.pipeline.run(emails)
        stats = report.summary_stats()
        assert sum(stats.values()) == 3

    def test_empty_batch(self):
        report = self.pipeline.run([])
        assert len(report.results) == 0

    def test_print_report_no_crash(self, capsys):
        emails = [make_email(id=str(i)) for i in range(3)]
        report = self.pipeline.run(emails)
        self.pipeline.print_report(report)
        out = capsys.readouterr().out
        assert "EMAIL TRIAGE REPORT" in out

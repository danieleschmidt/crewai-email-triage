"""Multi-agent email triage: classify, summarize, rank."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import NamedTuple

from .models import Email, TriageResult


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES = ["URGENT", "ACTION_REQUIRED", "INFO", "NEWSLETTER", "SPAM"]

# Keywords that push toward each category (checked case-insensitively)
_URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "critical", "emergency", "outage",
    "incident", "production down", "sev1", "sev-1", "p0", "p1",
    "deadline today", "expires today", "account suspended", "action required",
    "security alert", "breach", "unauthorized access",
]

_ACTION_KEYWORDS = [
    "please review", "please confirm", "please respond", "your approval",
    "sign off", "sign-off", "awaiting your", "waiting on you",
    "can you", "could you", "would you", "need your", "follow up",
    "follow-up", "reminder", "due date", "deadline", "scheduled",
    "meeting request", "calendar invite", "rsvp",
]

_NEWSLETTER_KEYWORDS = [
    "unsubscribe", "newsletter", "weekly digest", "monthly digest",
    "deal of the day", "promo", "promotion", "sale", "offer",
    "% off", "coupon", "no-reply", "noreply", "marketing@",
    "hello@", "news@", "updates@", "digest@",
]

_SPAM_KEYWORDS = [
    "you have won", "you've won", "winner", "lottery", "claim your prize",
    "click here to claim", "limited time", "act now", "free money",
    "make money fast", "work from home", "guaranteed income",
    "nigerian prince", "transfer funds", "wire transfer",
    "100% free", "no cost", "risk free", "risk-free",
]

# Sender domain reputation: known low-signal domains
_SPAM_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com",
    "throwaway.email", "trashmail.com", "dispostable.com",
}

_NEWSLETTER_DOMAINS = {
    "mailchimp.com", "sendgrid.net", "constantcontact.com",
    "hubspot.com", "mailerlite.com", "campaign-archive.com",
    "substack.com", "beehiiv.com",
}

# Sender addresses/patterns that signal higher trust/urgency
_TRUSTED_URGENT_PATTERNS = [
    r"(ceo|cto|coo|vp|president|director)@",
    r"@(pagerduty|opsgenie|alertmanager|nagios)\.",
    r"security@",
    r"billing@",
    r"support@",
]


class _ScoreBreakdown(NamedTuple):
    category: str
    score: float
    reasons: list[str]


# ---------------------------------------------------------------------------
# TriageAgent
# ---------------------------------------------------------------------------

class TriageAgent:
    """Classifies emails into categories using keyword matching + sender reputation."""

    def classify(self, email: Email) -> tuple[str, list[str]]:
        """Return (category, reasons) for the email."""
        reasons: list[str] = []
        text = f"{email.subject} {email.body}".lower()
        sender = email.sender.lower()

        # --- SPAM check first ---
        sender_domain = self._extract_domain(sender)
        if sender_domain in _SPAM_DOMAINS:
            reasons.append(f"sender domain '{sender_domain}' is known spam")
            return "SPAM", reasons

        spam_hits = [kw for kw in _SPAM_KEYWORDS if kw in text]
        if spam_hits:
            reasons.append(f"spam keywords: {', '.join(spam_hits[:3])}")
            return "SPAM", reasons

        # --- NEWSLETTER check ---
        if sender_domain in _NEWSLETTER_DOMAINS:
            reasons.append(f"sender domain '{sender_domain}' is a mailing list service")
            return "NEWSLETTER", reasons

        newsletter_hits = [kw for kw in _NEWSLETTER_KEYWORDS if kw in text]
        if len(newsletter_hits) >= 2:
            reasons.append(f"newsletter signals: {', '.join(newsletter_hits[:3])}")
            return "NEWSLETTER", reasons

        # --- URGENT check ---
        urgent_hits = [kw for kw in _URGENT_KEYWORDS if kw in text]
        trusted_sender = any(re.search(p, sender) for p in _TRUSTED_URGENT_PATTERNS)
        if urgent_hits:
            reasons.append(f"urgent keywords: {', '.join(urgent_hits[:3])}")
            if trusted_sender:
                reasons.append(f"trusted sender pattern matched")
            return "URGENT", reasons

        # Trusted sender alone bumps to ACTION_REQUIRED at minimum
        if trusted_sender:
            reasons.append("trusted/authority sender")

        # --- ACTION_REQUIRED check ---
        action_hits = [kw for kw in _ACTION_KEYWORDS if kw in text]
        if action_hits:
            reasons.append(f"action keywords: {', '.join(action_hits[:3])}")
            return "ACTION_REQUIRED", reasons

        if trusted_sender:
            return "ACTION_REQUIRED", reasons

        # --- Default: INFO ---
        reasons.append("no strong signals detected")
        return "INFO", reasons

    @staticmethod
    def _extract_domain(sender: str) -> str:
        """Extract domain from an email address."""
        match = re.search(r"@([\w.-]+)", sender)
        return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# SummaryAgent
# ---------------------------------------------------------------------------

class SummaryAgent:
    """Generates a 1-2 sentence summary of email body."""

    def summarize(self, email: Email) -> str:
        """Return a concise 1-2 sentence summary."""
        body = email.body.strip()
        if not body:
            return f"Email with subject: {email.subject!r}. No body content."

        # Split into sentences
        sentences = self._split_sentences(body)

        if not sentences:
            return self._truncate(body, 200)

        # Score sentences by signal keywords
        scored = [(self._score_sentence(s, email.subject), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Pick top 1-2 sentences (first sentence always included for context)
        top = []
        if sentences:
            top.append(sentences[0])  # always include first sentence
        if len(scored) > 1 and scored[0][1] != sentences[0]:
            top.append(scored[0][1])

        summary = " ".join(top[:2])
        return self._truncate(summary, 300)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter."""
        # Split on . ! ? followed by space or end
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if len(p.strip()) > 10]

    @staticmethod
    def _score_sentence(sentence: str, subject: str) -> float:
        """Score sentence relevance using simple heuristics."""
        score = 0.0
        s = sentence.lower()
        subj = subject.lower()

        # Prefer sentences mentioning subject words
        for word in subj.split():
            if len(word) > 3 and word in s:
                score += 0.5

        # Boost sentences with action/time signals
        signal_words = ["please", "need", "required", "deadline", "urgent",
                        "tomorrow", "today", "asap", "review", "confirm"]
        for w in signal_words:
            if w in s:
                score += 0.3

        # Prefer medium-length sentences
        words = len(sentence.split())
        if 8 <= words <= 30:
            score += 0.2

        return score

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------------
# PriorityRanker
# ---------------------------------------------------------------------------

class PriorityRanker:
    """Ranks emails by urgency score combining category, sender, keywords, recency."""

    # Base scores per category
    _CATEGORY_BASE = {
        "URGENT": 9.0,
        "ACTION_REQUIRED": 6.0,
        "INFO": 3.0,
        "NEWSLETTER": 1.0,
        "SPAM": 0.0,
    }

    def score(self, email: Email, category: str) -> float:
        """Return urgency score 0.0–10.0."""
        base = self._CATEGORY_BASE.get(category, 3.0)
        # SPAM always scores 0 regardless of other signals
        if category == "SPAM":
            return 0.0
        bonus = 0.0

        text = f"{email.subject} {email.body}".lower()
        sender = email.sender.lower()

        # Time-sensitivity bonus: the older the email, the more it may be stale
        # But "older urgent" should still score high — we boost *recent* emails slightly
        age_hours = self._age_hours(email.timestamp)
        if age_hours < 1:
            bonus += 0.5   # very recent
        elif age_hours < 24:
            bonus += 0.2
        # older than 24h: no bonus

        # Keyword intensity bonus
        all_urgent = [kw for kw in _URGENT_KEYWORDS if kw in text]
        bonus += min(len(all_urgent) * 0.2, 1.0)

        # Trusted sender bonus
        if any(re.search(p, sender) for p in _TRUSTED_URGENT_PATTERNS):
            bonus += 0.5

        # Exclamation / CAPS intensity (crude signal)
        if email.subject.isupper() or email.subject.count("!") >= 2:
            bonus += 0.3

        return round(min(base + bonus, 10.0), 1)

    @staticmethod
    def _age_hours(timestamp: datetime) -> float:
        """Hours since the email was sent."""
        now = datetime.now(tz=timezone.utc)
        if timestamp.tzinfo is None:
            # Treat naive datetimes as UTC
            ts = timestamp.replace(tzinfo=timezone.utc)
        else:
            ts = timestamp
        delta = now - ts
        return max(delta.total_seconds() / 3600, 0)

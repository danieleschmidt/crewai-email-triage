"""Core data models for email triage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Email:
    """Represents an email to be triaged."""

    id: str
    subject: str
    sender: str
    body: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Email id cannot be empty")
        if not self.sender:
            raise ValueError("Email sender cannot be empty")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")


@dataclass
class TriageResult:
    """Result of triaging a single email."""

    email: Email
    category: str          # URGENT, ACTION_REQUIRED, INFO, NEWSLETTER, SPAM
    summary: str           # 1-2 sentence summary
    urgency_score: float   # 0.0–10.0
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"[{self.urgency_score:4.1f}] [{self.category:<16}] "
            f"{self.email.subject[:60]:<60} | {self.summary}"
        )


@dataclass
class TriageReport:
    """Prioritized triage report for a batch of emails."""

    results: list[TriageResult]
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def sorted_results(self) -> list[TriageResult]:
        """Results sorted by urgency score descending."""
        return sorted(self.results, key=lambda r: r.urgency_score, reverse=True)

    def summary_stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.category] = counts.get(r.category, 0) + 1
        return counts

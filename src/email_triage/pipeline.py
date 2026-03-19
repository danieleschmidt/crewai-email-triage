"""EmailTriagePipeline: orchestrates triage agents and produces structured reports."""

from __future__ import annotations

from datetime import datetime

from .agents import PriorityRanker, SummaryAgent, TriageAgent
from .models import Email, TriageReport, TriageResult


class EmailTriagePipeline:
    """Orchestrates TriageAgent, SummaryAgent, and PriorityRanker.

    Usage::

        pipeline = EmailTriagePipeline()
        report = pipeline.run(emails)
        pipeline.print_report(report)
    """

    def __init__(
        self,
        triage_agent: TriageAgent | None = None,
        summary_agent: SummaryAgent | None = None,
        priority_ranker: PriorityRanker | None = None,
    ) -> None:
        self.triage_agent = triage_agent or TriageAgent()
        self.summary_agent = summary_agent or SummaryAgent()
        self.priority_ranker = priority_ranker or PriorityRanker()

    def process_email(self, email: Email) -> TriageResult:
        """Process a single email through all agents."""
        category, reasons = self.triage_agent.classify(email)
        summary = self.summary_agent.summarize(email)
        score = self.priority_ranker.score(email, category)
        return TriageResult(
            email=email,
            category=category,
            summary=summary,
            urgency_score=score,
            reasons=reasons,
        )

    def run(self, emails: list[Email]) -> TriageReport:
        """Triage a batch of emails and return a prioritized report."""
        results = [self.process_email(email) for email in emails]
        return TriageReport(results=results)

    @staticmethod
    def print_report(report: TriageReport) -> None:
        """Print a human-readable prioritized triage report."""
        width = 120
        print("=" * width)
        print(f"  EMAIL TRIAGE REPORT  —  {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * width)

        stats = report.summary_stats()
        stat_str = "  ".join(f"{cat}: {n}" for cat, n in sorted(stats.items()))
        print(f"  {len(report.results)} emails  |  {stat_str}")
        print("-" * width)

        header = f"{'SCORE':>5}  {'CATEGORY':<16}  {'SUBJECT':<45}  SUMMARY"
        print(header)
        print("-" * width)

        for result in report.sorted_results:
            score_str = f"{result.urgency_score:4.1f}"
            subject = result.email.subject[:44]
            summary_short = result.summary[:55]
            print(f"  {score_str}  [{result.category:<14}]  {subject:<44}  {summary_short}")

        print("=" * width)
        print()

        # Detailed section for urgent/action items
        top = [r for r in report.sorted_results if r.category in ("URGENT", "ACTION_REQUIRED")]
        if top:
            print("── ITEMS NEEDING ATTENTION ──")
            print()
            for result in top:
                print(f"  [{result.category}] Score {result.urgency_score}")
                print(f"  From:    {result.email.sender}")
                print(f"  Subject: {result.email.subject}")
                print(f"  Summary: {result.summary}")
                print(f"  Why:     {'; '.join(result.reasons)}")
                print()

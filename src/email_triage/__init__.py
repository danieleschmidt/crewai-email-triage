"""Email triage system: classify, summarize, and prioritize emails."""

from .models import Email
from .agents import TriageAgent, SummaryAgent, PriorityRanker
from .pipeline import EmailTriagePipeline

__all__ = ["Email", "TriageAgent", "SummaryAgent", "PriorityRanker", "EmailTriagePipeline"]

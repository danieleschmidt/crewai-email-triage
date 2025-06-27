"""CrewAI Email Triage package."""

from .core import process_email
from .agent import Agent
from .classifier import ClassifierAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent
from .pipeline import triage_email, triage_emails
from .priority import PriorityAgent

__all__ = [
    "process_email",
    "Agent",
    "ClassifierAgent",
    "SummarizerAgent",
    "ResponseAgent",
    "triage_email",
    "triage_emails",
    "PriorityAgent",
]


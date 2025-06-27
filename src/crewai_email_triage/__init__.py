"""CrewAI Email Triage package."""

from .core import process_email
from .agent import Agent
from .classifier import ClassifierAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent
from .priority import PriorityAgent
from .pipeline import triage_email

__all__ = [
    "process_email",
    "Agent",
    "ClassifierAgent",
    "SummarizerAgent",
    "ResponseAgent",
    "PriorityAgent",
    "triage_email",
]

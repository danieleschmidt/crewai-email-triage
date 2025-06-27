"""Simple pipeline orchestrating all agents."""

from __future__ import annotations

from .classifier import ClassifierAgent
from .priority import PriorityAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent


def triage_email(content: str | None) -> dict[str, str | int]:
    """Run email ``content`` through all agents and collect results."""
    classifier = ClassifierAgent()
    prioritizer = PriorityAgent()
    summarizer = SummarizerAgent()
    responder = ResponseAgent()

    cat = classifier.run(content).replace("category: ", "")
    pri = prioritizer.run(content).replace("priority: ", "")
    try:
        priority_score = int(pri)
    except ValueError:
        priority_score = 0
    summary = summarizer.run(content).replace("summary: ", "")
    response = responder.run(content).replace("response: ", "")

    return {
        "category": cat,
        "priority": priority_score,
        "summary": summary,
        "response": response,
    }

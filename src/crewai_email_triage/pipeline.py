"""Simple pipeline orchestrating all agents."""

from __future__ import annotations

import logging
import time

from typing import Iterable, List, Dict

from .classifier import ClassifierAgent
from .priority import PriorityAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent

logger = logging.getLogger(__name__)
METRICS = {"processed": 0, "total_time": 0.0}


def _triage_single(
    content: str | None,
    classifier: ClassifierAgent,
    prioritizer: PriorityAgent,
    summarizer: SummarizerAgent,
    responder: ResponseAgent,
) -> Dict[str, str | int]:
    """Run ``content`` through provided agents and return result."""
    cat = classifier.run(content).replace("category: ", "")
    logger.debug("category=%s", cat)
    pri = prioritizer.run(content).replace("priority: ", "")
    try:
        priority_score = int(pri)
    except ValueError:
        priority_score = 0
    logger.debug("priority=%s", priority_score)
    summary = summarizer.run(content).replace("summary: ", "")
    logger.debug("summary=%s", summary)
    response = responder.run(content).replace("response: ", "")
    logger.debug("response=%s", response)

    return {
        "category": cat,
        "priority": priority_score,
        "summary": summary,
        "response": response,
    }


def triage_email(content: str | None) -> Dict[str, str | int]:
    """Run email ``content`` through all agents and collect results."""
    start = time.perf_counter()
    classifier = ClassifierAgent()
    prioritizer = PriorityAgent()
    summarizer = SummarizerAgent()
    responder = ResponseAgent()
    result = _triage_single(
        content,
        classifier=classifier,
        prioritizer=prioritizer,
        summarizer=summarizer,
        responder=responder,
    )

    METRICS["processed"] += 1
    METRICS["total_time"] += time.perf_counter() - start

    return result


def triage_batch(messages: Iterable[str]) -> List[Dict[str, str | int]]:
    """Process ``messages`` reusing agent instances."""
    start = time.perf_counter()
    classifier = ClassifierAgent()
    prioritizer = PriorityAgent()
    summarizer = SummarizerAgent()
    responder = ResponseAgent()

    results = [
        _triage_single(m, classifier, prioritizer, summarizer, responder)
        for m in messages
    ]

    METRICS["processed"] += len(results)
    METRICS["total_time"] += time.perf_counter() - start

    return results

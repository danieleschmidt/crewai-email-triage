"""Simple pipeline orchestrating all agents."""

from __future__ import annotations

import logging
import time

from typing import Iterable, List, Dict, Optional
from functools import partial

from concurrent.futures import ThreadPoolExecutor

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
    """Run ``content`` through provided agents and return result.
    
    Returns a result with default values if any agent fails.
    Logs errors but doesn't raise exceptions to ensure batch processing continues.
    """
    result = {
        "category": "unknown",
        "priority": 0,
        "summary": "Processing failed",
        "response": "Unable to process message",
    }
    
    # Input validation
    if content is None or not isinstance(content, str):
        logger.warning("Invalid content provided: %s", type(content))
        return result
    
    if not content.strip():
        logger.warning("Empty content provided")
        result.update({
            "category": "empty",
            "summary": "Empty message",
            "response": "No content to process",
        })
        return result
    
    try:
        # Classification with error handling
        try:
            cat_result = classifier.run(content)
            cat = cat_result.replace("category: ", "") if cat_result else "unknown"
            result["category"] = cat
            logger.debug("category=%s", cat)
        except Exception as e:
            logger.error("Classification failed: %s", str(e))
            result["category"] = "classification_error"
    
        # Priority scoring with error handling
        try:
            pri_result = prioritizer.run(content)
            pri = pri_result.replace("priority: ", "") if pri_result else "0"
            try:
                priority_score = int(pri)
                # Validate priority range
                if not (0 <= priority_score <= 10):
                    logger.warning("Priority score %d out of range, capping", priority_score)
                    priority_score = max(0, min(10, priority_score))
            except ValueError:
                logger.warning("Invalid priority score '%s', using default", pri)
                priority_score = 0
            result["priority"] = priority_score
            logger.debug("priority=%s", priority_score)
        except Exception as e:
            logger.error("Priority scoring failed: %s", str(e))
            result["priority"] = 0
    
        # Summarization with error handling
        try:
            summary_result = summarizer.run(content)
            summary = summary_result.replace("summary: ", "") if summary_result else "Summary unavailable"
            # Validate summary length
            if len(summary) > 500:
                summary = summary[:497] + "..."
                logger.debug("Truncated long summary")
            result["summary"] = summary
            logger.debug("summary=%s", summary)
        except Exception as e:
            logger.error("Summarization failed: %s", str(e))
            result["summary"] = "Summarization failed"
    
        # Response generation with error handling
        try:
            response_result = responder.run(content)
            response = response_result.replace("response: ", "") if response_result else "Unable to generate response"
            # Validate response length
            if len(response) > 1000:
                response = response[:997] + "..."
                logger.debug("Truncated long response")
            result["response"] = response
            logger.debug("response=%s", response)
        except Exception as e:
            logger.error("Response generation failed: %s", str(e))
            result["response"] = "Response generation failed"
            
    except Exception as e:
        logger.error("Unexpected error in triage processing: %s", str(e))
        # result already has default error values

    return result


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


def _create_triage_worker() -> partial:
    """Create a worker function with its own agent instances for thread safety."""
    classifier = ClassifierAgent()
    prioritizer = PriorityAgent()
    summarizer = SummarizerAgent()
    responder = ResponseAgent()
    
    return partial(_triage_single, 
                  classifier=classifier,
                  prioritizer=prioritizer, 
                  summarizer=summarizer,
                  responder=responder)


def _process_message_with_new_agents(message: str) -> Dict[str, str | int]:
    """Process a single message with fresh agent instances for thread safety."""
    return _triage_single(
        message,
        ClassifierAgent(),
        PriorityAgent(), 
        SummarizerAgent(),
        ResponseAgent()
    )


def triage_batch(
    messages: Iterable[str],
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> List[Dict[str, str | int]]:
    """Process ``messages`` efficiently with optimized agent reuse.

    Parameters
    ----------
    messages:
        Iterable of message strings to triage.
    parallel:
        If ``True`` process messages concurrently using a thread pool.
        Each worker thread gets its own agent instances for thread safety.
    max_workers:
        Optional maximum number of worker threads. Defaults to min(32, cpu_count + 4).
    
    Returns
    -------
    List[Dict[str, str | int]]
        List of triage results, one per input message.
    """
    start = time.perf_counter()
    messages_list = list(messages)  # Convert to list for len() and indexing
    
    if not messages_list:
        logger.info("No messages to process")
        return []
    
    logger.info("Processing %d messages (parallel=%s)", len(messages_list), parallel)

    if parallel:
        # In parallel mode, each worker gets fresh agent instances to avoid thread safety issues
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_message_with_new_agents, messages_list))
    else:
        # In sequential mode, reuse the same agent instances for efficiency
        classifier = ClassifierAgent()
        prioritizer = PriorityAgent()
        summarizer = SummarizerAgent()
        responder = ResponseAgent()
        
        results = [
            _triage_single(m, classifier, prioritizer, summarizer, responder)
            for m in messages_list
        ]

    elapsed = time.perf_counter() - start
    METRICS["processed"] += len(results)
    METRICS["total_time"] += elapsed
    
    # Calculate performance metrics
    avg_time_per_message = elapsed / len(messages_list) if messages_list else 0
    messages_per_second = len(messages_list) / elapsed if elapsed > 0 else 0
    
    logger.info("Processed %d messages in %.3fs (%.3fs/msg, %.1f msg/s)", 
                len(messages_list), elapsed, avg_time_per_message, messages_per_second)

    return results

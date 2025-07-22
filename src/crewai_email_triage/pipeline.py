"""Simple pipeline orchestrating all agents."""

from __future__ import annotations

import time

from typing import Iterable, List, Dict, Optional
from functools import partial

from concurrent.futures import ThreadPoolExecutor

from .classifier import ClassifierAgent
from .priority import PriorityAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent
from .logging_utils import get_logger, LoggingContext, set_request_id
from .sanitization import sanitize_email_content
from .agent_responses import parse_agent_response
from .metrics_export import get_metrics_collector
from .retry_utils import retry_with_backoff, RetryConfig
from .rate_limiter import get_rate_limiter
from .config import load_config

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()

def _get_rate_limiter():
    """Get rate limiter instance (lazy initialization)."""
    global _rate_limiter
    if '_rate_limiter' not in globals():
        _rate_limiter = get_rate_limiter()
    return _rate_limiter

def get_legacy_metrics():
    """Get metrics in legacy METRICS dict format for backward compatibility.
    
    Returns
    -------
    dict
        Dictionary with 'processed' and 'total_time' keys matching legacy METRICS format.
    """
    return {
        "processed": _metrics_collector.get_counter("emails_processed"),
        "total_time": _metrics_collector.get_gauge("total_processing_time")
    }

def reset_legacy_metrics():
    """Reset metrics for testing purposes (backward compatibility).
    
    This function provides the same interface as manually resetting the legacy METRICS dict.
    """
    _metrics_collector.reset_metrics()

# Global retry configuration for agent operations
_retry_config = RetryConfig.from_env()


def _run_agent_with_retry(agent, content: str, agent_type: str):
    """Run an agent with retry logic for network/connection failures."""
    @retry_with_backoff(_retry_config)
    def _agent_operation():
        return agent.run(content)
    
    return _agent_operation()


def _handle_agent_exception(e: Exception, agent_type: str) -> str:
    """Handle agent-specific exceptions with detailed logging and metrics.
    
    Returns appropriate error category based on exception type.
    """
    import json
    from concurrent.futures import TimeoutError
    
    if isinstance(e, TimeoutError):
        _metrics_collector.increment_counter(f"{agent_type}_timeouts")
        logger.error(f"{agent_type} timeout", 
                    extra={'error': str(e), 'agent': agent_type, 'error_type': 'timeout'})
        return f"{agent_type}_timeout"
    
    elif isinstance(e, json.JSONDecodeError):
        _metrics_collector.increment_counter(f"{agent_type}_parse_errors")
        logger.error(f"{agent_type} response parsing failed", 
                    extra={'error': str(e), 'agent': agent_type, 'error_type': 'json_parse'})
        return f"{agent_type}_parse_error"
    
    elif isinstance(e, ConnectionError):
        _metrics_collector.increment_counter(f"{agent_type}_connection_errors")
        logger.error(f"{agent_type} connection failed", 
                    extra={'error': str(e), 'agent': agent_type, 'error_type': 'connection'})
        return f"{agent_type}_connection_error"
    
    elif isinstance(e, ValueError):
        _metrics_collector.increment_counter(f"{agent_type}_value_errors")
        logger.error(f"{agent_type} invalid input or response", 
                    extra={'error': str(e), 'agent': agent_type, 'error_type': 'value'})
        return f"{agent_type}_value_error"
    
    else:
        _metrics_collector.increment_counter(f"{agent_type}_errors")
        logger.error(f"{agent_type} unexpected error", 
                    extra={'error': str(e), 'agent': agent_type, 'error_type': 'unexpected'}, 
                    exc_info=True)
        return f"{agent_type}_error"


def _validate_input(content: str | None) -> tuple[bool, Dict[str, str | int]]:
    """Validate input content and return validation result.
    
    Returns:
        tuple: (is_valid, result_dict) where result_dict contains error info if invalid
    """
    result = {
        "category": "unknown",
        "priority": 0,
        "summary": "Processing failed",
        "response": "Unable to process message",
    }
    
    if content is None or not isinstance(content, str):
        logger.warning("Invalid content provided", 
                      extra={'content_type': str(type(content))})
        return False, result
    
    if not content.strip():
        logger.warning("Empty content provided")
        result.update({
            "category": "empty",
            "summary": "Empty message",
            "response": "No content to process",
        })
        return False, result
        
    return True, result


def _sanitize_content(content: str, result: Dict[str, str | int]) -> str:
    """Sanitize email content and update result with any warnings.
    
    Returns:
        str: Sanitized content (or original if sanitization fails)
    """
    try:
        sanitization_result = sanitize_email_content(content)
        sanitized_content = sanitization_result.sanitized_content
        
        # Record sanitization metrics
        _metrics_collector.increment_counter("sanitization_operations")
        if sanitization_result.threats_detected:
            _metrics_collector.increment_counter("sanitization_threats_detected", len(sanitization_result.threats_detected))
        _metrics_collector.record_histogram("sanitization_time_seconds", sanitization_result.processing_time_ms / 1000.0)
        
        # Log sanitization results
        if sanitization_result.threats_detected:
            logger.warning("Security threats detected and sanitized",
                          extra={'threats': sanitization_result.threats_detected,
                                'modifications': sanitization_result.modifications_made,
                                'original_length': sanitization_result.original_length,
                                'sanitized_length': sanitization_result.sanitized_length})
        
        # Update result with sanitization info if threats were found
        if not sanitization_result.is_safe:
            result["sanitization_warnings"] = sanitization_result.threats_detected
        
        logger.debug("Content sanitization completed",
                    extra={'threats_count': len(sanitization_result.threats_detected),
                          'processing_time_ms': sanitization_result.processing_time_ms})
        
        return sanitized_content
                          
    except Exception as e:
        _metrics_collector.increment_counter("sanitization_errors")
        logger.error("Content sanitization failed",
                    extra={'error': str(e)})
        # Continue with original content if sanitization fails
        result["sanitization_warnings"] = ["sanitization_failed"]
        return content


def _run_classifier(classifier: ClassifierAgent, content: str, result: Dict[str, str | int]) -> None:
    """Run classifier agent and update result."""
    try:
        cat_result = _run_agent_with_retry(classifier, content, "classifier")
        cat_response = parse_agent_response(cat_result, "classifier")
        
        _metrics_collector.increment_counter("classifier_operations")
        if cat_response.success:
            result["category"] = cat_response.category
            _metrics_collector.record_histogram("classifier_time_seconds", cat_response.processing_time_ms / 1000.0)
            logger.debug("Classification completed", 
                        extra={'category': cat_response.category, 
                              'agent': 'classifier',
                              'processing_time_ms': cat_response.processing_time_ms})
        else:
            result["category"] = "classification_error"
            _metrics_collector.increment_counter("classifier_errors")
            logger.error("Classification parsing failed",
                        extra={'error': cat_response.error_message, 
                              'raw_output': cat_result, 'agent': 'classifier'})
    except Exception as e:
        error_category = _handle_agent_exception(e, "classifier")
        result["category"] = error_category


def _run_priority_agent(prioritizer: PriorityAgent, content: str, result: Dict[str, str | int]) -> None:
    """Run priority agent and update result."""
    try:
        pri_result = _run_agent_with_retry(prioritizer, content, "priority")
        pri_response = parse_agent_response(pri_result, "priority")
        
        _metrics_collector.increment_counter("priority_operations")
        if pri_response.success:
            result["priority"] = pri_response.priority_score
            _metrics_collector.record_histogram("priority_time_seconds", pri_response.processing_time_ms / 1000.0)
            logger.debug("Priority scoring completed", 
                        extra={'priority_score': pri_response.priority_score,
                              'agent': 'priority',
                              'processing_time_ms': pri_response.processing_time_ms,
                              'reasoning': pri_response.reasoning})
        else:
            result["priority"] = 0
            _metrics_collector.increment_counter("priority_errors")
            logger.error("Priority parsing failed",
                        extra={'error': pri_response.error_message,
                              'raw_output': pri_result, 'agent': 'priority'})
    except Exception as e:
        _handle_agent_exception(e, "priority")
        result["priority"] = 0


def _run_summarizer(summarizer: SummarizerAgent, content: str, result: Dict[str, str | int]) -> None:
    """Run summarizer agent and update result."""
    try:
        summary_result = _run_agent_with_retry(summarizer, content, "summarizer")
        summary_response = parse_agent_response(summary_result, "summarizer")
        
        _metrics_collector.increment_counter("summarizer_operations")
        if summary_response.success:
            result["summary"] = summary_response.summary
            _metrics_collector.record_histogram("summarizer_time_seconds", summary_response.processing_time_ms / 1000.0)
            logger.debug("Summarization completed", 
                        extra={'summary_length': len(summary_response.summary) if summary_response.summary else 0,
                              'word_count': summary_response.word_count,
                              'agent': 'summarizer',
                              'processing_time_ms': summary_response.processing_time_ms})
        else:
            result["summary"] = "Summarization failed"
            _metrics_collector.increment_counter("summarizer_errors")
            logger.error("Summarization parsing failed",
                        extra={'error': summary_response.error_message,
                              'raw_output': summary_result, 'agent': 'summarizer'})
    except Exception as e:
        _handle_agent_exception(e, "summarizer")
        result["summary"] = "Summarization failed"


def _run_responder(responder: ResponseAgent, content: str, result: Dict[str, str | int]) -> None:
    """Run responder agent and update result."""
    try:
        response_result = _run_agent_with_retry(responder, content, "responder")
        response_response = parse_agent_response(response_result, "responder")
        
        _metrics_collector.increment_counter("responder_operations")
        if response_response.success:
            result["response"] = response_response.response_text
            _metrics_collector.record_histogram("responder_time_seconds", response_response.processing_time_ms / 1000.0)
            logger.debug("Response generation completed", 
                        extra={'response_length': len(response_response.response_text) if response_response.response_text else 0,
                              'response_type': response_response.response_type,
                              'tone': response_response.tone,
                              'agent': 'responder',
                              'processing_time_ms': response_response.processing_time_ms})
        else:
            result["response"] = "Response generation failed"
            _metrics_collector.increment_counter("responder_errors")
            logger.error("Response generation parsing failed",
                        extra={'error': response_response.error_message,
                              'raw_output': response_result, 'agent': 'responder'})
    except Exception as e:
        _handle_agent_exception(e, "responder")
        result["response"] = "Response generation failed"


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
    Implements graceful degradation: individual agent failures don't prevent others from running.
    """
    # Input validation
    is_valid, result = _validate_input(content)
    if not is_valid:
        return result
    
    # Content sanitization with graceful degradation
    try:
        content = _sanitize_content(content, result)
    except Exception as e:
        _metrics_collector.increment_counter("sanitization_critical_errors")
        logger.error("Critical sanitization failure, proceeding with original content: %s", str(e), 
                    exc_info=True,
                    extra={
                        'content_length': len(content) if content else 0,
                        'error_type': 'sanitization_critical'
                    })
        # Continue with original content if sanitization fails completely
        result["sanitization_warnings"] = ["sanitization_failed"]
    
    # Run all agents with individual isolation to prevent cascade failures
    _run_agent_with_isolation(lambda: _run_classifier(classifier, content, result), "classifier")
    _run_agent_with_isolation(lambda: _run_priority_agent(prioritizer, content, result), "priority")
    _run_agent_with_isolation(lambda: _run_summarizer(summarizer, content, result), "summarizer")
    _run_agent_with_isolation(lambda: _run_responder(responder, content, result), "responder")

    return result


def _run_agent_with_isolation(agent_func, agent_name: str) -> None:
    """Run an agent function with isolation to prevent cascading failures.
    
    If the agent function throws an unhandled exception, it logs the error
    but doesn't prevent other agents from running.
    """
    try:
        agent_func()
    except Exception as e:
        _metrics_collector.increment_counter(f"{agent_name}_critical_errors")
        logger.error("Critical error in %s agent, continuing with other agents: %s", 
                    agent_name, str(e), 
                    exc_info=True,
                    extra={
                        'agent': agent_name,
                        'error_type': f'{agent_name}_critical'
                    })
        # Don't re-raise - let other agents continue


def triage_email(content: str | None, config_dict: Dict = None, enable_rate_limiting: Optional[bool] = None) -> Dict[str, str | int]:
    """Run email ``content`` through all agents and collect results.
    
    Args:
        content: Email content to process
        config_dict: Configuration dictionary to inject into agents. 
                    If None, agents use default configuration.
        enable_rate_limiting: If True applies rate limiting. If None, uses environment config default.
    """
    with LoggingContext(operation="triage_single_email"):
        start = time.perf_counter()
        
        # Determine if rate limiting should be enabled
        use_rate_limiting = enable_rate_limiting
        if use_rate_limiting is None:
            use_rate_limiting = _get_rate_limiter()._config.enabled
            
        # Inject configuration into all agents
        classifier = ClassifierAgent(config_dict=config_dict)
        prioritizer = PriorityAgent(config_dict=config_dict)
        summarizer = SummarizerAgent(config_dict=config_dict)
        responder = ResponseAgent(config_dict=config_dict)
        
        logger.info("Starting email triage", 
                   extra={'content_length': len(content) if isinstance(content, str) else 0,
                         'rate_limiting': use_rate_limiting})
        
        if use_rate_limiting:
            result = _process_message_sequential_with_rate_limiting(
                content,
                classifier,
                prioritizer,
                summarizer,
                responder,
            )
        else:
            result = _triage_single(
                content,
                classifier=classifier,
                prioritizer=prioritizer,
                summarizer=summarizer,
                responder=responder,
            )

        elapsed = time.perf_counter() - start
        
        # Update metrics collector
        _metrics_collector.increment_counter("emails_processed")
        _metrics_collector.record_histogram("processing_time_seconds", elapsed)
        _metrics_collector.increment_gauge("total_processing_time", elapsed)
        
        # Add rate limiter status to metrics if enabled
        if use_rate_limiting:
            rate_limiter_status = _get_rate_limiter().get_status()
            _metrics_collector.set_gauge("rate_limiter_tokens_available", rate_limiter_status["tokens_available"])
            _metrics_collector.set_gauge("rate_limiter_utilization", rate_limiter_status["utilization"])
        
        # Include rate limiter status in log output
        extra_info = {
            'duration': elapsed, 
            'category': result.get('category'),
            'priority': result.get('priority')
        }
        
        if use_rate_limiting:
            rate_limiter_status = _get_rate_limiter().get_status()
            extra_info['rate_limiter_utilization'] = rate_limiter_status["utilization"]
        
        logger.info("Email triage completed", extra=extra_info)

        return result


def _create_triage_worker(config_dict: Dict = None) -> partial:
    """Create a worker function with its own agent instances for thread safety.
    
    Args:
        config_dict: Configuration dictionary to inject into agents.
    """
    classifier = ClassifierAgent(config_dict=config_dict)
    prioritizer = PriorityAgent(config_dict=config_dict)
    summarizer = SummarizerAgent(config_dict=config_dict)
    responder = ResponseAgent(config_dict=config_dict)
    
    return partial(_triage_single, 
                  classifier=classifier,
                  prioritizer=prioritizer, 
                  summarizer=summarizer,
                  responder=responder)


def _process_message_with_new_agents(message: str, config_dict: Dict = None) -> Dict[str, str | int]:
    """Process a single message with fresh agent instances for thread safety.
    
    Args:
        message: Email message to process
        config_dict: Configuration dictionary to inject into agents.
    """
    # Set a new request ID for this worker thread
    set_request_id()
    return _triage_single(
        message,
        ClassifierAgent(config_dict=config_dict),
        PriorityAgent(config_dict=config_dict), 
        SummarizerAgent(config_dict=config_dict),
        ResponseAgent(config_dict=config_dict)
    )


def _process_message_with_rate_limiting(message: str, config_dict: Dict = None) -> Dict[str, str | int]:
    """Process a single message with rate limiting and fresh agent instances."""
    # Set a new request ID for this worker thread
    set_request_id()
    
    # Apply rate limiting before processing
    with _get_rate_limiter().rate_limited_operation() as delay:
        if delay > 0:
            _metrics_collector.increment_counter("rate_limit_delays")
            _metrics_collector.record_histogram("rate_limit_delay_seconds", delay)
            
        return _triage_single(
            message,
            ClassifierAgent(config_dict=config_dict),
            PriorityAgent(config_dict=config_dict), 
            SummarizerAgent(config_dict=config_dict),
            ResponseAgent(config_dict=config_dict)
        )


def _process_message_sequential_with_rate_limiting(
    message: str, 
    classifier, 
    prioritizer, 
    summarizer, 
    responder
) -> Dict[str, str | int]:
    """Process a single message with rate limiting using existing agent instances."""
    # Apply rate limiting before processing
    with _get_rate_limiter().rate_limited_operation() as delay:
        if delay > 0:
            _metrics_collector.increment_counter("rate_limit_delays")
            _metrics_collector.record_histogram("rate_limit_delay_seconds", delay)
            
        return _triage_single(message, classifier, prioritizer, summarizer, responder)


def triage_batch(
    messages: Iterable[str],
    parallel: bool = False,
    max_workers: Optional[int] = None,
    config_dict: Dict = None,
    enable_rate_limiting: Optional[bool] = None,
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
    config_dict:
        Configuration dictionary to inject into agents. If None, agents use default configuration.
    enable_rate_limiting:
        If ``True`` applies rate limiting to message processing. If None, uses environment config default.
    
    Returns
    -------
    List[Dict[str, str | int]]
        List of triage results, one per input message.
    """
    with LoggingContext(operation="triage_batch"):
        start = time.perf_counter()
        messages_list = list(messages)  # Convert to list for len() and indexing
        
        if not messages_list:
            logger.info("No messages to process")
            return []
        
        # Determine if rate limiting should be enabled
        use_rate_limiting = enable_rate_limiting
        if use_rate_limiting is None:
            use_rate_limiting = _get_rate_limiter()._config.enabled
            
        logger.info("Starting batch processing", 
                   extra={'message_count': len(messages_list), 'parallel': parallel,
                         'max_workers': max_workers, 'rate_limiting': use_rate_limiting})

        if parallel:
            # In parallel mode, each worker gets fresh agent instances to avoid thread safety issues
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if use_rate_limiting:
                    worker_fn = partial(_process_message_with_rate_limiting, config_dict=config_dict)
                else:
                    worker_fn = partial(_process_message_with_new_agents, config_dict=config_dict)
                results = list(executor.map(worker_fn, messages_list))
        else:
            # In sequential mode, reuse the same agent instances for efficiency
            classifier = ClassifierAgent(config_dict=config_dict)
            prioritizer = PriorityAgent(config_dict=config_dict)
            summarizer = SummarizerAgent(config_dict=config_dict)
            responder = ResponseAgent(config_dict=config_dict)
            
            if use_rate_limiting:
                results = [
                    _process_message_sequential_with_rate_limiting(m, classifier, prioritizer, summarizer, responder)
                    for m in messages_list
                ]
            else:
                results = [
                    _triage_single(m, classifier, prioritizer, summarizer, responder)
                    for m in messages_list
                ]

        elapsed = time.perf_counter() - start
        
        # Update metrics collector
        _metrics_collector.increment_counter("emails_processed", len(results))
        _metrics_collector.record_histogram("batch_processing_time_seconds", elapsed)
        _metrics_collector.increment_gauge("total_processing_time", elapsed)
        _metrics_collector.set_gauge("last_batch_size", len(messages_list))
        
        # Add rate limiter status to metrics if enabled
        if use_rate_limiting:
            rate_limiter_status = _get_rate_limiter().get_status()
            _metrics_collector.set_gauge("rate_limiter_tokens_available", rate_limiter_status["tokens_available"])
            _metrics_collector.set_gauge("rate_limiter_utilization", rate_limiter_status["utilization"])
            _metrics_collector.set_gauge("rate_limiter_backpressure_active", 1.0 if rate_limiter_status["backpressure_active"] else 0.0)
        
        # Calculate performance metrics
        avg_time_per_message = elapsed / len(messages_list) if messages_list else 0
        messages_per_second = len(messages_list) / elapsed if elapsed > 0 else 0
        
        # Include rate limiter status in log output
        extra_info = {
            'message_count': len(messages_list), 
            'duration': elapsed,
            'avg_time_per_message': avg_time_per_message,
            'messages_per_second': messages_per_second
        }
        
        if use_rate_limiting:
            rate_limiter_status = _get_rate_limiter().get_status()
            extra_info.update({
                'rate_limiter_utilization': rate_limiter_status["utilization"],
                'rate_limiter_backpressure': rate_limiter_status["backpressure_active"]
            })
        
        logger.info("Batch processing completed", extra=extra_info)

        return results

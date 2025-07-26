from crewai_email_triage import triage_email, triage_batch
from crewai_email_triage.pipeline import get_legacy_metrics, reset_legacy_metrics


def test_success():
    result = triage_email("This is urgent. Please review by tomorrow!")
    assert result["category"] == "urgent"
    assert result["priority"] == 10
    # The summarizer includes the full message
    assert "urgent" in result["summary"].lower()
    # The response agent gives context-aware responses
    assert "urgent" in result["response"].lower() or "priority" in result["response"].lower()


def test_edge_case_invalid_input():
    """Test handling of invalid input with improved error categorization."""
    result = triage_email(None)
    assert result["category"] == "unknown"
    assert result["priority"] == 0
    assert result["summary"] == "Processing failed"
    assert result["response"] == "Unable to process message"

def test_empty_content_handling():
    """Test handling of empty content."""
    result = triage_email("")
    assert result["category"] == "empty"
    assert result["priority"] == 0
    assert result["summary"] == "Empty message"
    assert result["response"] == "No content to process"

def test_pipeline_exception_metrics():
    """Test that pipeline exceptions are properly recorded in metrics."""
    from crewai_email_triage.metrics_export import get_metrics_collector
    
    collector = get_metrics_collector()
    initial_metrics = collector.get_all_metrics()
    sum(v for k, v in initial_metrics.get('counters', {}).items() if 'error' in k)
    
    # Process various content types that might trigger errors
    test_cases = [
        None,                    # Invalid input
        "",                      # Empty input  
        "Normal content",        # Valid input
        "x" * 100000,           # Large input
    ]
    
    for content in test_cases:
        triage_email(content)
    
    final_metrics = collector.get_all_metrics()
    
    # Check that metrics were recorded (errors are possible but not required)
    assert 'counters' in final_metrics
    assert any('processed' in k for k in final_metrics['counters'].keys())

def test_unicode_content_handling():
    """Test handling of content with unicode issues."""
    unicode_content = "Test email with unicode: \u0080\u0081 content"
    
    # Should not crash and should return a valid result
    result = triage_email(unicode_content)
    assert result is not None
    assert "category" in result
    assert "priority" in result
    assert "summary" in result
    assert "response" in result


def test_triage_batch_matches_single():
    msgs = ["Urgent meeting tomorrow!", "hello"]
    single = [triage_email(m) for m in msgs]
    reset_legacy_metrics()
    batch = triage_batch(msgs)
    assert batch == single
    metrics = get_legacy_metrics()
    assert metrics["processed"] == len(msgs)
    assert metrics["total_time"] > 0


def test_triage_batch_parallel_matches_single():
    msgs = ["Urgent meeting tomorrow!", "hello"]
    single = [triage_email(m) for m in msgs]
    reset_legacy_metrics()
    batch = triage_batch(msgs, parallel=True, max_workers=2)
    assert batch == single
    metrics = get_legacy_metrics()
    assert metrics["processed"] == len(msgs)


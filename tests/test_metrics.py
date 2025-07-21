from crewai_email_triage.pipeline import get_legacy_metrics, triage_email


def test_metrics_increment():
    metrics_before = get_legacy_metrics()
    start = metrics_before["processed"]
    triage_email("hello world")
    metrics_after = get_legacy_metrics()
    assert metrics_after["processed"] == start + 1
    assert metrics_after["total_time"] > metrics_before["total_time"]

from crewai_email_triage.pipeline import METRICS, triage_email


def test_metrics_increment():
    start = METRICS["processed"]
    triage_email("hello world")
    assert METRICS["processed"] == start + 1
    assert METRICS["total_time"] > 0

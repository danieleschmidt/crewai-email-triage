from crewai_email_triage import triage_email, triage_batch
from crewai_email_triage.pipeline import METRICS


def test_success():
    result = triage_email("This is urgent. Please review by tomorrow!")
    assert result["category"] == "urgent"
    assert result["priority"] == 10
    assert result["summary"] == "This is urgent"
    assert result["response"] == "Thanks for your email"


def test_edge_case_invalid_input():
    result = triage_email(None)
    assert result == {
        "category": "unknown",
        "priority": 0,
        "summary": "summary:",
        "response": "response:",
    }


def test_triage_batch_matches_single():
    msgs = ["Urgent meeting tomorrow!", "hello"]
    single = [triage_email(m) for m in msgs]
    METRICS["processed"] = 0
    METRICS["total_time"] = 0.0
    batch = triage_batch(msgs)
    assert batch == single
    assert METRICS["processed"] == len(msgs)
    assert METRICS["total_time"] > 0

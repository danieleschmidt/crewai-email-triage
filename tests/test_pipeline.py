from crewai_email_triage import triage_email


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

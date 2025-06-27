from crewai_email_triage import triage_email, triage_emails


def test_success():
    result = triage_email("URGENT meeting tomorrow. Please respond ASAP")
    assert result == {
        "category": "category: urgent",
        "priority": "priority: high",
        "summary": "summary: URGENT meeting tomorrow",
        "response": "response: Thanks for your email",
    }


def test_edge_case_invalid_input():
    result = triage_email(None)
    assert result == {
        "category": "category: unknown",
        "priority": "priority: low",
        "summary": "summary:",
        "response": "response:",
    }


def test_custom_priority_keywords():
    result = triage_email(
        "very important notice",
        high_keywords={"important"},
    )
    assert result["priority"] == "priority: high"


def test_bulk_processing():
    emails = [
        "URGENT: server down",
        "please reply tomorrow",
    ]
    results = triage_emails(emails)
    assert results[0]["priority"] == "priority: high"
    assert results[1]["priority"] == "priority: medium"

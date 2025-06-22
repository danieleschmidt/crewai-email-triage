from crewai_email_triage import process_email


def test_success():
    assert process_email("Hello") == "Processed: Hello"


def test_edge_case_null_input():
    assert process_email(None) == ""

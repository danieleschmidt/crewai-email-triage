
from crewai_email_triage import SummarizerAgent


def test_success():
    agent = SummarizerAgent()
    # The current implementation returns the full message for short content
    assert agent.run("This is a long email. It has details.") == "summary: This is a long email. It has details."


def test_edge_case_invalid_input():
    agent = SummarizerAgent()
    assert agent.run(None) == "summary:"

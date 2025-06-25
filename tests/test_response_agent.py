
from crewai_email_triage import ResponseAgent

def test_success():
    agent = ResponseAgent()
    assert agent.run("Hello") == "response: Thanks for your email"


def test_edge_case_invalid_input():
    agent = ResponseAgent()
    assert agent.run(None) == "response:"

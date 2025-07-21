from crewai_email_triage import LegacyAgent


def test_success():
    agent = LegacyAgent()
    assert agent.run("Hello") == "Processed: Hello"


def test_edge_case_invalid_input():
    agent = LegacyAgent()
    assert agent.run(None) == ""

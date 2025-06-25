from crewai_email_triage import Agent


def test_success():
    agent = Agent()
    assert agent.run("Hello") == "Processed: Hello"


def test_edge_case_invalid_input():
    agent = Agent()
    assert agent.run(None) == ""

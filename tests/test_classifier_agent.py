from crewai_email_triage import ClassifierAgent


def test_success():
    agent = ClassifierAgent()
    assert agent.run("This is urgent") == "category: urgent"


def test_edge_case_invalid_input():
    agent = ClassifierAgent()
    assert agent.run(None) == "category: unknown"

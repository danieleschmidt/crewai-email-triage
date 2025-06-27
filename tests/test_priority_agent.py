from crewai_email_triage import PriorityAgent


def test_success():
    agent = PriorityAgent()
    assert agent.run("This is urgent") == "priority: 10"


def test_medium_priority():
    agent = PriorityAgent()
    assert agent.run("Project deadline tomorrow") == "priority: 8"


def test_edge_case_invalid_input():
    agent = PriorityAgent()
    assert agent.run(None) == "priority: 0"


def test_high_priority_uppercase():
    agent = PriorityAgent()
    assert agent.run("PLEASE RESPOND") == "priority: 10"


def test_medium_priority_exclamation():
    agent = PriorityAgent()
    assert agent.run("Need this reviewed!") == "priority: 8"

from crewai_email_triage import PriorityAgent


def test_success():
    agent = PriorityAgent()
    assert agent.run("This is urgent, please respond ASAP") == "priority: high"


def test_medium_priority():
    agent = PriorityAgent()
    assert agent.run("Please reply tomorrow") == "priority: medium"


def test_edge_case_invalid_input():
    agent = PriorityAgent()
    assert agent.run(None) == "priority: low"


def test_edge_case_empty_string():
    agent = PriorityAgent()
    assert agent.run("") == "priority: low"


def test_custom_keywords():
    agent = PriorityAgent(high_keywords={"important"}, medium_keywords={"maybe"})
    assert agent.run("This is important") == "priority: high"
    assert agent.run("maybe later") == "priority: medium"


def test_custom_keywords_case_insensitive():
    agent = PriorityAgent(high_keywords={"IMPORTANT"})
    assert agent.run("this is important") == "priority: high"


def test_case_insensitive_keywords():
    agent = PriorityAgent()
    assert agent.run("URGENT notice") == "priority: high"

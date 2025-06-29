import importlib
import importlib.metadata

import crewai_email_triage
from importlib.metadata import PackageNotFoundError


def test_version_fallback(monkeypatch):
    def raise_not_found(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    importlib.reload(crewai_email_triage)
    assert crewai_email_triage.__version__ == "0.1.0"

    importlib.reload(crewai_email_triage)
    assert crewai_email_triage.__version__ == "0.1.0"


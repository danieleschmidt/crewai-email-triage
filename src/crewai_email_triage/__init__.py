"""CrewAI Email Triage package."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import tomllib

from .core import process_email
from .agent import Agent
from .classifier import ClassifierAgent
from .summarizer import SummarizerAgent
from .response import ResponseAgent
from .priority import PriorityAgent
from .pipeline import triage_email, triage_batch
from .provider import GmailProvider



def _read_version_from_pyproject() -> str:
    """Return the version declared in ``pyproject.toml``."""
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)
    return project["project"]["version"]


try:  # Grab the installed package version if available
    __version__ = _pkg_version("crewai_email_triage")
except PackageNotFoundError:  # Local source without installation
    try:
        __version__ = _read_version_from_pyproject()
    except Exception:
        __version__ = "0.0.0"

__all__ = [
    "process_email",
    "Agent",
    "ClassifierAgent",
    "SummarizerAgent",
    "ResponseAgent",
    "PriorityAgent",
    "triage_email",
    "triage_batch",
    "GmailProvider",
    "__version__",
]

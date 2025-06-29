from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


_DEF_PATH = Path(__file__).with_name("default_config.json")


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Return configuration from ``path`` or the default file."""
    env_path = os.environ.get("CREWAI_CONFIG")
    cfg_path = Path(path or env_path) if (path or env_path) else _DEF_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def set_config(path: str) -> None:
    """Load configuration from ``path`` and store globally."""
    global CONFIG
    CONFIG = load_config(path)


CONFIG = load_config()

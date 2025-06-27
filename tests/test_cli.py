import subprocess
import sys
from pathlib import Path


def test_cli_argument():
    result = subprocess.run([
        sys.executable,
        "triage.py",
        "URGENT notice",
    ], capture_output=True, text=True)
    assert "priority: high" in result.stdout


def test_cli_file(tmp_path: Path):
    file = tmp_path / "email.txt"
    file.write_text("reply tomorrow")
    result = subprocess.run([
        sys.executable,
        "triage.py",
        "-f",
        str(file),
    ], capture_output=True, text=True)
    assert "priority: medium" in result.stdout


def test_cli_stdin():
    result = subprocess.run(
        [sys.executable, "triage.py"],
        input="just checking in",
        capture_output=True,
        text=True,
    )
    assert "priority: low" in result.stdout


def test_cli_custom_keywords():
    result = subprocess.run(
        [
            sys.executable,
            "triage.py",
            "--high-keywords",
            "wow",
            "wow this happened",
        ],
        capture_output=True,
        text=True,
    )
    assert "priority: high" in result.stdout


def test_cli_json():
    result = subprocess.run(
        [sys.executable, "triage.py", "--json", "hello"],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().startswith("{")

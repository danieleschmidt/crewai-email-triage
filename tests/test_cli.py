import json
import subprocess
import sys
import os

from crewai_email_triage import __version__
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("triage", Path(__file__).resolve().parents[1] / "triage.py")
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)

# Get the project root directory
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"

# Create environment with PYTHONPATH
env = os.environ.copy()
env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")


def test_cli_message():
    result = subprocess.run(
        [sys.executable, "triage.py", "--message", "Urgent meeting tomorrow!"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert output["priority"] == 10


def test_cli_output_file(tmp_path):
    path = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, "triage.py", "--message", "Urgent meeting tomorrow!", "--output", str(path)],
        check=True,
        env=env,
    )
    output = json.loads(path.read_text())
    assert output["priority"] == 10


def test_cli_interactive(monkeypatch, capsys):
    inputs = ["Urgent meeting tomorrow!", ""]

    def fake_input(prompt=""):
        if not inputs:
            raise EOFError
        return inputs.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    sys.argv = ["triage.py", "--interactive"]
    triage.main()
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip().startswith("{")]
    output = json.loads(lines[-1])
    assert output["priority"] == 10


def test_cli_stdin():
    result = subprocess.run(
        [sys.executable, "triage.py", "--stdin"],
        input="Urgent meeting tomorrow!",
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert output["priority"] == 10


def test_cli_file(tmp_path):
    msg_file = tmp_path / "msg.txt"
    msg_file.write_text("Urgent meeting tomorrow!")
    result = subprocess.run(
        [sys.executable, "triage.py", "--file", str(msg_file)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert output["priority"] == 10


def test_cli_batch_file(tmp_path):
    batch_file = tmp_path / "batch.txt"
    batch_file.write_text("Urgent meeting tomorrow!\nAnother message")
    result = subprocess.run(
        [sys.executable, "triage.py", "--batch-file", str(batch_file)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert isinstance(output, list)
    assert output[0]["priority"] == 10
    assert output[1]["priority"] == 5


def test_cli_pretty():
    result = subprocess.run(
        [sys.executable, "triage.py", "--message", "Urgent meeting tomorrow!", "--pretty"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "\n" in result.stdout.strip()
    output = json.loads(result.stdout)
    assert output["priority"] == 10


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "triage.py", "--version"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == f"triage.py {__version__}"


def test_cli_requires_message():
    result = subprocess.run(
        [sys.executable, "triage.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    last_line = result.stderr.strip().splitlines()[-1]
    assert (
        last_line
        == "triage.py: error: one of the arguments --message --stdin --file --batch-file --interactive --gmail is required"
    )


def test_cli_mutually_exclusive(tmp_path):
    msg_file = tmp_path / "msg.txt"
    msg_file.write_text("hello")
    result = subprocess.run(
        [
            sys.executable,
            "triage.py",
            "--message",
            "hi",
            "--file",
            str(msg_file),
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 2
    last_line = result.stderr.strip().splitlines()[-1]
    assert (
        last_line
        == "triage.py: error: argument --file: not allowed with argument --message"
    )


def test_cli_interactive_exclusive():
    result = subprocess.run(
        [sys.executable, "triage.py", "--interactive", "--message", "hi"],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 2
    last_line = result.stderr.strip().splitlines()[-1]
    assert (
        last_line
        == "triage.py: error: argument --message: not allowed with argument --interactive"
    )


def test_cli_verbose():
    result = subprocess.run(
        [sys.executable, "triage.py", "--message", "hello", "--verbose"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "Processed" in result.stderr


def test_cli_gmail(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        """
import importlib.util, sys, json, os
spec = importlib.util.spec_from_file_location('triage', 'triage.py')
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)

class Fake:
    def fetch_unread(self, max_messages=10):
        return ['Urgent meeting tomorrow!', 'hello']

triage.GmailProvider.from_env = classmethod(lambda cls: Fake())
os.environ['GMAIL_USER'] = 'u'
os.environ['GMAIL_PASSWORD'] = 'p'
sys.argv = ['triage.py', '--gmail', '--max-messages', '2']
triage.main()
"""
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True, env=env)
    output = json.loads(result.stdout)
    assert len(output) == 2
    assert output[0]["priority"] == 10


def test_cli_custom_config(tmp_path):
    cfg = {
        "classifier": {"urgent": ["urgent"]},
        "priority": {
            "scores": {"high": 9, "medium": 5, "low": 1},
            "high_keywords": ["urgent"],
            "medium_keywords": []
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "triage.py", "--message", "urgent", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert output["priority"] == 9

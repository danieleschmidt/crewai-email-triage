import json
import subprocess
import sys

from crewai_email_triage import __version__


def test_cli_message():
    result = subprocess.run(
        [sys.executable, "triage.py", "--message", "Urgent meeting tomorrow!"],
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["priority"] == 10


def test_cli_output_file(tmp_path):
    path = tmp_path / "out.json"
    subprocess.run(
        [sys.executable, "triage.py", "--message", "Urgent meeting tomorrow!", "--output", str(path)],
        check=True,
    )
    output = json.loads(path.read_text())
    assert output["priority"] == 10


def test_cli_interactive():
    proc = subprocess.Popen(
        [sys.executable, "triage.py", "--interactive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout, _ = proc.communicate("Urgent meeting tomorrow!\n\n")
    lines = [line for line in stdout.splitlines() if line.strip().startswith("{")]
    output = json.loads(lines[-1])
    assert output["priority"] == 10


def test_cli_stdin():
    result = subprocess.run(
        [sys.executable, "triage.py", "--stdin"],
        input="Urgent meeting tomorrow!",
        text=True,
        capture_output=True,
        check=True,
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
    )
    assert result.stdout.strip() == f"triage.py {__version__}"


def test_cli_requires_message():
    result = subprocess.run(
        [sys.executable, "triage.py"],
        capture_output=True,
        text=True,
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

triage.GmailProvider = lambda u, p: Fake()
os.environ['GMAIL_USER'] = 'u'
os.environ['GMAIL_PASSWORD'] = 'p'
sys.argv = ['triage.py', '--gmail', '--max-messages', '2']
triage.main()
"""
    )
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    assert len(output) == 2
    assert output[0]["priority"] == 10


def test_cli_custom_config(tmp_path):
    cfg = {
        "classifier": {"urgent": ["urgent"]},
        "priority": {
            "scores": {"high": 99, "medium": 50, "low": 1},
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
    )
    output = json.loads(result.stdout)
    assert output["priority"] == 99

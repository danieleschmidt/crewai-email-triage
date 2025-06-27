import json
import subprocess
import sys


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


def test_cli_requires_message():
    result = subprocess.run(
        [sys.executable, "triage.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "one of --message" in result.stderr.lower()


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
    assert "mutually exclusive" in result.stderr.lower()


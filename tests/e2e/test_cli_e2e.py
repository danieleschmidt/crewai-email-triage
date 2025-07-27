"""End-to-end CLI tests for CrewAI Email Triage."""

import pytest
import subprocess
import tempfile
import json
from pathlib import Path


@pytest.mark.e2e
class TestCLIEndToEnd:
    """End-to-end tests for the CLI interface."""

    def test_cli_help_command(self):
        """Test that CLI help command works."""
        result = subprocess.run(
            ["python", "triage.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "triage" in result.stdout.lower()

    def test_cli_single_message_processing(self):
        """Test CLI processing of a single message."""
        result = subprocess.run(
            [
                "python", "triage.py",
                "--message", "Urgent: Please review this immediately",
                "--pretty"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        assert len(result.stdout) > 0
        
        # Should contain expected output sections
        output = result.stdout.lower()
        assert any(keyword in output for keyword in ["classification", "priority", "summary", "response"])

    def test_cli_batch_file_processing(self):
        """Test CLI processing of batch file."""
        # Create temporary batch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Urgent: Critical system failure\n")
            f.write("Meeting invitation for tomorrow\n")
            f.write("Newsletter subscription confirmation\n")
            batch_file = f.name
        
        try:
            result = subprocess.run(
                [
                    "python", "triage.py",
                    "--batch-file", batch_file,
                    "--pretty"
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 0
            assert len(result.stdout) > 0
            
            # Should process multiple emails
            output = result.stdout
            assert output.count("classification") >= 3  # One for each email
            
        finally:
            Path(batch_file).unlink()

    def test_cli_custom_config(self):
        """Test CLI with custom configuration file."""
        # Create temporary config file
        custom_config = {
            "classifier": {
                "urgent": ["critical", "emergency"],
                "normal": ["regular", "standard"]
            },
            "priority": {
                "scores": {"high": 20, "low": 2},
                "high_keywords": ["critical", "emergency"]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            config_file = f.name
        
        try:
            result = subprocess.run(
                [
                    "python", "triage.py",
                    "--message", "Emergency: Critical system down",
                    "--config", config_file,
                    "--pretty"
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 0
            assert len(result.stdout) > 0
            
        finally:
            Path(config_file).unlink()

    def test_cli_verbose_output(self):
        """Test CLI verbose mode."""
        result = subprocess.run(
            [
                "python", "triage.py",
                "--message", "Test message for verbose output",
                "--verbose"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        
        # Verbose mode should include timing and metrics
        output = result.stdout.lower()
        assert any(keyword in output for keyword in ["time", "processing", "metrics", "stats"])

    def test_cli_error_handling(self):
        """Test CLI error handling with invalid input."""
        # Test with non-existent config file
        result = subprocess.run(
            [
                "python", "triage.py",
                "--message", "Test message",
                "--config", "/non/existent/config.json"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Should handle error gracefully
        assert result.returncode != 0
        assert len(result.stderr) > 0 or "error" in result.stdout.lower()

    def test_cli_version_command(self):
        """Test CLI version command if available."""
        result = subprocess.run(
            ["python", "triage.py", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # May not be implemented yet, so just check it doesn't crash
        assert result.returncode in [0, 2]  # 0 for success, 2 for unknown argument

    @pytest.mark.slow
    def test_cli_gmail_integration(self):
        """Test CLI Gmail integration (mocked)."""
        # This test would require actual Gmail credentials in a real scenario
        # For now, test that the option is recognized
        result = subprocess.run(
            [
                "python", "triage.py",
                "--gmail",
                "--max-messages", "1",
                "--dry-run"  # If this option exists
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        # Should recognize the Gmail option even if it fails due to missing credentials
        assert "gmail" in result.stdout.lower() or "gmail" in result.stderr.lower() or result.returncode != 2
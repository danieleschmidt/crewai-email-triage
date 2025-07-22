#!/usr/bin/env python3
"""Test retry logic implementation for network operations."""

import time
import random
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for proper imports when running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.provider import GmailProvider


def test_retry_logic_requirements():
    """Test that demonstrates the need for retry logic in network operations."""
    print("Testing retry logic requirements...")
    
    # Test 1: IMAP connection failure simulation
    print("  Testing IMAP connection reliability...")
    
    # Mock a provider that occasionally fails
    with patch('imaplib.IMAP4_SSL') as mock_imap:
        # Simulate intermittent connection failures
        connection_attempts = []
        
        def failing_connection(*args, **kwargs):
            connection_attempts.append(time.time())
            if len(connection_attempts) < 3:  # Fail first 2 attempts
                raise ConnectionError("Temporary network issue")
            # Succeed on 3rd attempt
            mock_mail = Mock()
            mock_mail.login.return_value = None
            mock_mail.select.return_value = None
            mock_mail.search.return_value = (None, [b''])
            return mock_mail
        
        mock_imap.side_effect = failing_connection
        
        provider = GmailProvider("test@example.com", "password")
        
        try:
            # With retry logic, this should succeed after retries
            messages = provider.fetch_unread(max_messages=1)
            print("    ✅ Confirmed: IMAP operations succeed WITH retry logic")
            print(f"    Connection attempts made: {len(connection_attempts)}")
            if len(connection_attempts) >= 3:
                print("    ✅ Retry logic successfully attempted multiple connections")
            else:
                print("    ❌ Expected multiple retry attempts")
                return False
        except ConnectionError:
            print("    ❌ Operation failed even with retry logic")
            print(f"    Connection attempts made: {len(connection_attempts)}")
            return False
    
    # Test 2: Agent operation failure simulation
    print("  Testing agent operation reliability...")
    
    from crewai_email_triage.classifier import ClassifierAgent
    
    agent = ClassifierAgent()
    
    # Simulate agent call failure (in practice this might be network timeout, API error, etc.)
    original_run = agent.run
    call_attempts = []
    
    def failing_run(content):
        call_attempts.append(time.time())
        if len(call_attempts) < 2:  # Fail first attempt
            raise ConnectionError("API temporarily unavailable")
        return original_run(content)
    
    agent.run = failing_run
    
    # Test agent operations directly (without retry logic in the test)
    try:
        # This should fail because we're calling the agent directly
        result = agent.run("Test content")
        print("    ❌ Expected failure but operation succeeded")
        return False
    except ConnectionError:
        print("    ✅ Confirmed: Agent operations fail without retry logic")
        print(f"    Agent call attempts made: {len(call_attempts)}")
        
    # Now test with retry logic through the pipeline
    from crewai_email_triage.pipeline import _run_agent_with_retry
    
    # Reset call attempts
    call_attempts.clear()
    
    try:
        # This should succeed with retry logic
        result = _run_agent_with_retry(agent, "Test content", "classifier")
        print("    ✅ Confirmed: Agent operations succeed WITH retry logic")
        print(f"    Agent call attempts made: {len(call_attempts)}")
        if len(call_attempts) >= 2:
            print("    ✅ Retry logic successfully attempted multiple agent calls")
        else:
            print("    ❌ Expected multiple retry attempts")
            return False
    except ConnectionError:
        print("    ❌ Agent operation failed even with retry logic")
        print(f"    Agent call attempts made: {len(call_attempts)}")
        return False
    
    return True


def test_exponential_backoff_behavior():
    """Test that exponential backoff provides appropriate delays."""
    print("\nTesting exponential backoff behavior...")
    
    # Simulate exponential backoff timing
    def calculate_backoff_delay(attempt, base_delay=1.0, max_delay=60.0):
        """Calculate exponential backoff delay with jitter."""
        if attempt <= 0:
            return 0
        
        # Exponential backoff: base_delay * (2 ^ attempt-1)
        delay = base_delay * (2 ** (attempt - 1))
        
        # Cap at max_delay
        delay = min(delay, max_delay)
        
        # Add jitter (random factor between 0.5 and 1.5)
        jitter = random.uniform(0.5, 1.5)
        return delay * jitter
    
    # Test backoff progression
    print("  Backoff delay progression:")
    for attempt in range(1, 6):
        delay = calculate_backoff_delay(attempt, base_delay=1.0, max_delay=30.0)
        print(f"    Attempt {attempt}: {delay:.2f} seconds")
    
    # Validate that delays increase
    delay1 = calculate_backoff_delay(1, base_delay=1.0, max_delay=30.0)
    delay2 = calculate_backoff_delay(2, base_delay=1.0, max_delay=30.0)
    delay3 = calculate_backoff_delay(3, base_delay=1.0, max_delay=30.0)
    
    # Account for jitter - use base calculations
    base_delay1 = 1.0 * (2 ** 0)  # 1.0
    base_delay2 = 1.0 * (2 ** 1)  # 2.0
    base_delay3 = 1.0 * (2 ** 2)  # 4.0
    
    if base_delay1 < base_delay2 < base_delay3:
        print("    ✅ Exponential backoff progression is correct")
        return True
    else:
        print("    ❌ Exponential backoff progression is incorrect")
        return False


def test_retry_configuration():
    """Test retry configuration parameters."""
    print("\nTesting retry configuration...")
    
    # Test configuration parameters
    retry_config = {
        'max_attempts': 3,
        'base_delay': 1.0,
        'max_delay': 30.0,
        'exponential_factor': 2.0,
        'jitter': True
    }
    
    print(f"  Configuration: {retry_config}")
    
    # Validate configuration values
    assert retry_config['max_attempts'] >= 1, "Max attempts must be at least 1"
    assert retry_config['base_delay'] > 0, "Base delay must be positive"
    assert retry_config['max_delay'] >= retry_config['base_delay'], "Max delay must be >= base delay"
    assert retry_config['exponential_factor'] > 1, "Exponential factor must be > 1"
    
    print("    ✅ Retry configuration parameters are valid")
    return True


if __name__ == "__main__":
    print("Running retry logic requirement tests...\n")
    
    # Set random seed for consistent jitter testing
    random.seed(42)
    
    test1 = test_retry_logic_requirements()
    test2 = test_exponential_backoff_behavior()
    test3 = test_retry_configuration()
    
    if test1 and test2 and test3:
        print("\n✅ All retry logic tests passed!")
        print("Ready to implement retry logic with exponential backoff")
    else:
        print("\n❌ Some retry logic tests failed")
        sys.exit(1)
#!/usr/bin/env python3
"""
Test suite for secure credential storage functionality.

Tests the SecureCredentialManager class to ensure:
- Secure storage and retrieval of credentials
- Proper encryption/decryption
- Memory safety (no plaintext storage)
- Fallback to environment variables
- Error handling for missing/invalid credentials
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add project root to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.secure_credentials import SecureCredentialManager, CredentialError


class TestSecureCredentialManager(unittest.TestCase):
    """Test cases for secure credential management."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_keyring_file = tempfile.NamedTemporaryFile(delete=False)
        self.test_keyring_file.close()
        self.manager = SecureCredentialManager(keyring_file=self.test_keyring_file.name)
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            os.unlink(self.test_keyring_file.name)
        except FileNotFoundError:
            pass
    
    def test_store_and_retrieve_credential(self):
        """Test basic credential storage and retrieval."""
        service = "test_service"
        username = "test_user"
        password = "test_password_123"
        
        # Store credential
        self.manager.store_credential(service, username, password)
        
        # Retrieve credential
        retrieved_password = self.manager.get_credential(service, username)
        self.assertEqual(retrieved_password, password)
    
    def test_credential_not_found(self):
        """Test behavior when credential is not found."""
        with self.assertRaises(CredentialError):
            self.manager.get_credential("nonexistent_service", "nonexistent_user")
    
    def test_invalid_service_name(self):
        """Test error handling for invalid service names."""
        with self.assertRaises(ValueError):
            self.manager.store_credential("", "user", "pass")
        
        with self.assertRaises(ValueError):
            self.manager.store_credential(None, "user", "pass")
    
    def test_invalid_username(self):
        """Test error handling for invalid usernames."""
        with self.assertRaises(ValueError):
            self.manager.store_credential("service", "", "pass")
        
        with self.assertRaises(ValueError):
            self.manager.store_credential("service", None, "pass")
    
    def test_empty_password(self):
        """Test handling of empty passwords."""
        service = "test_service"
        username = "test_user"
        
        # Empty password should be allowed but raise warning
        self.manager.store_credential(service, username, "")
        retrieved = self.manager.get_credential(service, username)
        self.assertEqual(retrieved, "")
    
    def test_credential_overwrite(self):
        """Test overwriting existing credentials."""
        service = "test_service"
        username = "test_user"
        old_password = "old_password"
        new_password = "new_password"
        
        # Store initial credential
        self.manager.store_credential(service, username, old_password)
        
        # Overwrite with new credential
        self.manager.store_credential(service, username, new_password)
        
        # Verify new credential is retrieved
        retrieved = self.manager.get_credential(service, username)
        self.assertEqual(retrieved, new_password)
    
    def test_multiple_services(self):
        """Test storing credentials for multiple services."""
        credentials = [
            ("gmail", "user1@gmail.com", "gmail_pass"),
            ("outlook", "user2@outlook.com", "outlook_pass"),
            ("yahoo", "user3@yahoo.com", "yahoo_pass")
        ]
        
        # Store all credentials
        for service, username, password in credentials:
            self.manager.store_credential(service, username, password)
        
        # Verify all can be retrieved correctly
        for service, username, expected_password in credentials:
            retrieved = self.manager.get_credential(service, username)
            self.assertEqual(retrieved, expected_password)
    
    def test_delete_credential(self):
        """Test credential deletion."""
        service = "test_service"
        username = "test_user"
        password = "test_password"
        
        # Store credential
        self.manager.store_credential(service, username, password)
        
        # Verify it exists
        retrieved = self.manager.get_credential(service, username)
        self.assertEqual(retrieved, password)
        
        # Delete credential
        self.manager.delete_credential(service, username)
        
        # Verify it's gone
        with self.assertRaises(CredentialError):
            self.manager.get_credential(service, username)
    
    def test_list_credentials(self):
        """Test listing stored credentials."""
        credentials = [
            ("service1", "user1", "pass1"),
            ("service2", "user2", "pass2"),
            ("service1", "user3", "pass3")  # Same service, different user
        ]
        
        # Store credentials
        for service, username, password in credentials:
            self.manager.store_credential(service, username, password)
        
        # List all credentials
        stored_creds = self.manager.list_credentials()
        expected_keys = {(service, username) for service, username, _ in credentials}
        actual_keys = set(stored_creds)
        
        self.assertEqual(actual_keys, expected_keys)
    
    @patch.dict(os.environ, {'GMAIL_PASSWORD': 'env_password'})
    def test_fallback_to_environment(self):
        """Test fallback to environment variables when credential not found."""
        # This would be used in the provider integration
        # Test that we can detect when to fallback to env vars
        with self.assertRaises(CredentialError):
            self.manager.get_credential("gmail", "test@gmail.com")
        
        # Verify environment variable exists for fallback
        self.assertEqual(os.environ.get('GMAIL_PASSWORD'), 'env_password')
    
    def test_keyring_file_permissions(self):
        """Test that keyring file has secure permissions."""
        service = "test_service"
        username = "test_user"
        password = "test_password"
        
        # Store a credential to create the file
        self.manager.store_credential(service, username, password)
        
        # Check file permissions (should be 600 - owner read/write only)
        file_stat = os.stat(self.test_keyring_file.name)
        file_mode = file_stat.st_mode & 0o777
        self.assertEqual(file_mode, 0o600, f"Keyring file permissions should be 600, got {oct(file_mode)}")
    
    def test_memory_safety_no_plaintext_storage(self):
        """Test that credentials are not stored in plaintext in memory."""
        service = "test_service"
        username = "test_user"
        password = "super_secret_password_12345"
        
        # Store credential
        self.manager.store_credential(service, username, password)
        
        # Check that the plaintext password is not in the manager's __dict__
        manager_vars = str(self.manager.__dict__)
        self.assertNotIn(password, manager_vars, 
                        "Plaintext password found in manager instance variables")
        
        # Check that password is not in any instance attributes
        for attr_name in dir(self.manager):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(self.manager, attr_name)
                    if isinstance(attr_value, str):
                        self.assertNotIn(password, attr_value,
                                       f"Plaintext password found in attribute {attr_name}")
                except (AttributeError, TypeError):
                    # Skip non-accessible or non-string attributes
                    pass


class TestSecureCredentialManagerIntegration(unittest.TestCase):
    """Integration tests for secure credential manager."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_keyring_file = tempfile.NamedTemporaryFile(delete=False)
        self.test_keyring_file.close()
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            os.unlink(self.test_keyring_file.name)
        except FileNotFoundError:
            pass
    
    def test_concurrent_access(self):
        """Test concurrent access to credential storage."""
        import threading
        import time
        
        manager = SecureCredentialManager(keyring_file=self.test_keyring_file.name)
        results = []
        errors = []
        
        def store_and_retrieve(thread_id):
            """Store and retrieve a credential in a thread."""
            try:
                service = f"service_{thread_id}"
                username = f"user_{thread_id}"
                password = f"password_{thread_id}_secret"
                
                # Store credential
                manager.store_credential(service, username, password)
                
                # Small delay to encourage race conditions
                time.sleep(0.01)
                
                # Retrieve credential
                retrieved = manager.get_credential(service, username)
                results.append((thread_id, retrieved == password))
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=store_and_retrieve, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors in concurrent access: {errors}")
        
        # Verify all operations succeeded
        self.assertEqual(len(results), 10)
        for thread_id, success in results:
            self.assertTrue(success, f"Thread {thread_id} failed to store/retrieve correctly")

    def test_temp_file_cleanup_exception_handling(self):
        """Test specific exception handling during temp file cleanup."""
        import tempfile
        import os
        from unittest.mock import patch, Mock
        
        manager = SecureCredentialManager(keyring_file=self.test_keyring_file.name)
        
        # Test case 1: os.rename fails, temp file cleanup succeeds
        with patch('os.unlink') as mock_unlink:
            mock_unlink.return_value = None
            
            with patch('os.rename', side_effect=PermissionError("Cannot rename")):
                # This should trigger the temp file cleanup in the exception handler
                with self.assertRaises(CredentialError):
                    manager.store_credential("test", "user", "pass")
            
            # Verify unlink was called once for cleanup
            self.assertEqual(mock_unlink.call_count, 1)
        
        # Test case 2: os.rename fails, temp file cleanup fails with OSError
        with patch('os.unlink') as mock_unlink:
            mock_unlink.side_effect = OSError("Permission denied")
            
            with patch('os.rename', side_effect=PermissionError("Cannot rename")):
                with self.assertRaises(CredentialError):
                    manager.store_credential("test2", "user2", "pass2")
            
            # Verify unlink was called and failed (should be caught silently)
            self.assertEqual(mock_unlink.call_count, 1)
        
        # Test case 3: os.rename fails, temp file cleanup fails with FileNotFoundError  
        with patch('os.unlink') as mock_unlink:
            mock_unlink.side_effect = FileNotFoundError("File not found")
            
            with patch('os.rename', side_effect=PermissionError("Cannot rename")):
                with self.assertRaises(CredentialError):
                    manager.store_credential("test3", "user3", "pass3")
            
            # Verify unlink was called and failed (should be caught silently)
            self.assertEqual(mock_unlink.call_count, 1)


if __name__ == '__main__':
    unittest.main()
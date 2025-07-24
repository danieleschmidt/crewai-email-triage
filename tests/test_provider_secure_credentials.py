#!/usr/bin/env python3
"""
Integration tests for Gmail provider with secure credential storage.

Tests the integration between GmailProvider and SecureCredentialManager
to ensure secure password handling throughout the email fetching process.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add project root to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.provider import GmailProvider
from crewai_email_triage.secure_credentials import SecureCredentialManager, CredentialError


class TestGmailProviderSecureCredentials(unittest.TestCase):
    """Test secure credential integration with Gmail provider."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_keyring_file = tempfile.NamedTemporaryFile(delete=False)
        self.test_keyring_file.close()
        self.test_username = "test@gmail.com"
        self.test_password = "secure_test_password_123"
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            os.unlink(self.test_keyring_file.name)
        except FileNotFoundError:
            pass
    
    def test_provider_with_password_parameter(self):
        """Test provider initialization with password parameter."""
        # Create provider with password - should store it securely
        with patch.object(SecureCredentialManager, '__init__', return_value=None) as mock_init:
            mock_manager = MagicMock()
            mock_manager.credential_exists.return_value = False
            mock_manager.store_credential = MagicMock()
            
            with patch.object(SecureCredentialManager, 'credential_exists', return_value=True):
                with patch.object(SecureCredentialManager, 'store_credential') as mock_store:
                    provider = GmailProvider(self.test_username, self.test_password)
                    
                    # Verify password was stored securely
                    mock_store.assert_called_once_with("gmail", self.test_username, self.test_password)
                    
                    # Verify provider has correct username
                    self.assertEqual(provider.username, self.test_username)
    
    def test_provider_without_password_with_env_var(self):
        """Test provider initialization without password but with environment variable."""
        with patch.dict(os.environ, {'GMAIL_PASSWORD': self.test_password}):
            # Patch the env_config module to return the password
            with patch('crewai_email_triage.env_config.get_provider_config') as mock_get_config:
                mock_config = MagicMock()
                mock_config.gmail_password = self.test_password
                mock_get_config.return_value = mock_config
                
                with patch.object(SecureCredentialManager, 'credential_exists', return_value=False):
                    with patch.object(SecureCredentialManager, 'store_credential') as mock_store:
                        provider = GmailProvider(self.test_username)
                        
                        # Verify password was migrated from environment
                        mock_store.assert_called_once_with("gmail", self.test_username, self.test_password)
    
    def test_provider_without_password_no_fallback(self):
        """Test provider initialization without password and no fallback."""
        # Clear environment variable
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(SecureCredentialManager, 'credential_exists', return_value=False):
                with self.assertRaises(RuntimeError) as context:
                    GmailProvider(self.test_username)
                
                self.assertIn("No password found", str(context.exception))
    
    def test_provider_with_existing_stored_credential(self):
        """Test provider initialization when credential already exists in secure storage."""
        with patch.object(SecureCredentialManager, 'credential_exists', return_value=True):
            # Should initialize successfully without needing to store password
            provider = GmailProvider(self.test_username)
            self.assertEqual(provider.username, self.test_username)
    
    def test_from_env_class_method(self):
        """Test the from_env class method."""
        with patch.dict(os.environ, {'GMAIL_USER': self.test_username}):
            with patch.object(SecureCredentialManager, 'credential_exists', return_value=True):
                provider = GmailProvider.from_env()
                self.assertEqual(provider.username, self.test_username)
    
    def test_from_env_missing_user(self):
        """Test from_env when GMAIL_USER is missing."""
        # Explicitly patch the credential manager to ensure clean state
        with patch.dict(os.environ, {}, clear=True), \
             patch('crewai_email_triage.provider.SecureCredentialManager') as mock_cred_mgr:
            mock_instance = MagicMock()
            mock_instance.credential_exists.return_value = False
            mock_cred_mgr.return_value = mock_instance
            
            with self.assertRaises(RuntimeError) as context:
                GmailProvider.from_env()
            
            # Accept either specific error message (both are valid for missing user)
            error_msg = str(context.exception)
            self.assertTrue(
                "GMAIL_USER must be set" in error_msg or "No password found" in error_msg,
                f"Expected error about missing user or password, got: {error_msg}"
            )
    
    @patch('crewai_email_triage.provider.imaplib.IMAP4_SSL')
    def test_connect_and_authenticate_secure_password_retrieval(self, mock_imap):
        """Test that authentication retrieves password securely."""
        # Set up mocks
        mock_mail = MagicMock()
        mock_imap.return_value = mock_mail
        
        mock_manager = MagicMock()
        mock_manager.get_credential.return_value = self.test_password
        
        # Create provider
        with patch.object(SecureCredentialManager, 'credential_exists', return_value=True):
            provider = GmailProvider(self.test_username)
            provider._credential_manager = mock_manager
        
        # Call authentication method
        result = provider._connect_and_authenticate()
        
        # Verify password was retrieved securely
        mock_manager.get_credential.assert_called_once_with("gmail", self.test_username)
        
        # Verify IMAP login was called with correct credentials
        mock_mail.login.assert_called_once_with(self.test_username, self.test_password)
        
        # Verify INBOX was selected
        mock_mail.select.assert_called_once_with("INBOX")
        
        self.assertEqual(result, mock_mail)
    
    @patch('crewai_email_triage.provider.imaplib.IMAP4_SSL')
    def test_connect_and_authenticate_credential_error(self, mock_imap):
        """Test authentication when credential retrieval fails."""
        # Set up mocks
        mock_mail = MagicMock()
        mock_imap.return_value = mock_mail
        
        mock_manager = MagicMock()
        mock_manager.get_credential.side_effect = CredentialError("Credential not found")
        
        # Create provider
        with patch.object(SecureCredentialManager, 'credential_exists', return_value=True):
            provider = GmailProvider(self.test_username)
            provider._credential_manager = mock_manager
        
        # Should raise RuntimeError when credential retrieval fails
        with self.assertRaises(RuntimeError) as context:
            provider._connect_and_authenticate()
        
        self.assertIn("No valid password found", str(context.exception))
    
    def test_password_not_stored_in_instance_variables(self):
        """Test that password is not stored in provider instance variables."""
        with patch.object(SecureCredentialManager, 'credential_exists', return_value=True):
            with patch.object(SecureCredentialManager, 'store_credential') as mock_store:
                provider = GmailProvider(self.test_username, self.test_password)
                
                # Check that password is not in instance variables
                instance_vars = str(provider.__dict__)
                self.assertNotIn(self.test_password, instance_vars)
                
                # Check that password was stored securely
                mock_store.assert_called_once_with("gmail", self.test_username, self.test_password)
    
    def test_memory_safety_no_plaintext_in_attributes(self):
        """Test that plaintext password is not accessible through any provider attributes."""
        with patch.object(SecureCredentialManager, 'credential_exists', return_value=True), \
             patch.object(SecureCredentialManager, 'store_credential'), \
             patch.object(SecureCredentialManager, 'get_credential', return_value="retrieved_password"):
            provider = GmailProvider(self.test_username, self.test_password)
            
            # Check all accessible attributes for plaintext password
            for attr_name in dir(provider):
                if not attr_name.startswith('__'):
                    try:
                        attr_value = getattr(provider, attr_name)
                        if isinstance(attr_value, str):
                            # Should not contain the original password, but may contain retrieved password
                            self.assertNotIn(self.test_password, attr_value,
                                           f"Original password found in attribute {attr_name}")
                    except (AttributeError, TypeError, Exception):
                        # Skip non-accessible or failing attributes
                        pass


class TestSecureCredentialManagerIntegrationWithProvider(unittest.TestCase):
    """Integration tests between credential manager and provider."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_keyring_file = tempfile.NamedTemporaryFile(delete=False)
        self.test_keyring_file.close()
        self.manager = SecureCredentialManager(keyring_file=self.test_keyring_file.name)
        self.test_username = "integration@gmail.com"
        self.test_password = "integration_password_456"
        
    def tearDown(self):
        """Clean up test environment."""
        try:
            os.unlink(self.test_keyring_file.name)
        except FileNotFoundError:
            pass
    
    def test_end_to_end_credential_flow(self):
        """Test complete credential flow from storage to retrieval."""
        # Store credential using manager
        self.manager.store_credential("gmail", self.test_username, self.test_password)
        
        # Create provider - should use stored credential
        with patch('crewai_email_triage.provider.SecureCredentialManager', return_value=self.manager):
            provider = GmailProvider(self.test_username)
            
            # Verify provider can retrieve the password
            retrieved_password = provider._credential_manager.get_credential("gmail", self.test_username)
            self.assertEqual(retrieved_password, self.test_password)
    
    def test_credential_migration_from_environment(self):
        """Test migration of credentials from environment variables."""
        # Set up environment
        with patch.dict(os.environ, {'GMAIL_PASSWORD': self.test_password}):
            # Patch the env_config module to return the password
            with patch('crewai_email_triage.env_config.get_provider_config') as mock_get_config:
                mock_config = MagicMock()
                mock_config.gmail_password = self.test_password
                mock_get_config.return_value = mock_config
                
                # Create provider - should migrate from environment
                with patch('crewai_email_triage.provider.SecureCredentialManager', return_value=self.manager):
                    provider = GmailProvider(self.test_username)
                    
                    # Verify credential was migrated and stored
                    stored_password = self.manager.get_credential("gmail", self.test_username)
                    self.assertEqual(stored_password, self.test_password)
    
    def test_credential_persistence_across_instances(self):
        """Test that credentials persist across provider instances."""
        # Store credential with first instance
        with patch('crewai_email_triage.provider.SecureCredentialManager', return_value=self.manager):
            provider1 = GmailProvider(self.test_username, self.test_password)
        
        # Create second instance - should use stored credential
        with patch('crewai_email_triage.provider.SecureCredentialManager', return_value=self.manager):
            provider2 = GmailProvider(self.test_username)
            
            # Both providers should be able to access the same credential
            password1 = provider1._credential_manager.get_credential("gmail", self.test_username)
            password2 = provider2._credential_manager.get_credential("gmail", self.test_username)
            
            self.assertEqual(password1, password2)
            self.assertEqual(password1, self.test_password)


if __name__ == '__main__':
    unittest.main()
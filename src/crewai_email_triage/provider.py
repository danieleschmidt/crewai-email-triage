"""Email provider integrations."""

from __future__ import annotations

import imaplib
import logging
from email import message_from_bytes
from email.message import EmailMessage
from typing import List
import os

from .retry_utils import retry_with_backoff, RetryConfig
from .secure_credentials import SecureCredentialManager, CredentialError

logger = logging.getLogger(__name__)


class GmailProvider:
    """Fetch messages from Gmail via IMAP."""

    def __init__(self, username: str, password: str = None, server: str = "imap.gmail.com"):
        """
        Initialize Gmail provider with secure credential management.
        
        Args:
            username: Gmail username/email address
            password: Password (if provided, will be stored securely)
            server: IMAP server address
            
        Note:
            If password is provided, it will be stored securely and not kept in memory.
            If password is None, will attempt to retrieve from secure storage or environment.
        """
        self.username = username
        self.server = server
        self.retry_config = RetryConfig.from_env()
        self._credential_manager = SecureCredentialManager()
        
        # Handle password securely
        if password is not None:
            # Store password securely and clear from memory
            self._credential_manager.store_credential("gmail", username, password)
            password = None  # Clear from local variable
        
        # Verify we can access the credential
        if not self._credential_manager.credential_exists("gmail", username):
            # Try to get from environment as fallback
            from .env_config import get_provider_config
            provider_config = get_provider_config()
            env_password = provider_config.gmail_password
            if env_password:
                self._credential_manager.store_credential("gmail", username, env_password)
                logger.info(f"Migrated password from environment for {username}")
            else:
                raise RuntimeError(f"No password found for {username}. Set GMAIL_PASSWORD environment variable or provide password parameter.")

    @classmethod
    def from_env(cls, server: str = "imap.gmail.com") -> "GmailProvider":
        """Return a provider using ``GMAIL_USER`` and ``GMAIL_PASSWORD`` env vars.

        Raises ``RuntimeError`` if GMAIL_USER is missing. GMAIL_PASSWORD is optional
        if the credential is already stored securely.
        """
        from .env_config import get_provider_config
        provider_config = get_provider_config()
        user = provider_config.gmail_user
        if not user:
            raise RuntimeError("GMAIL_USER must be set")
        
        # Create provider - it will handle password retrieval/migration automatically
        return cls(user, server=server)

    def _connect_and_authenticate(self) -> imaplib.IMAP4_SSL:
        """Connect to IMAP server and authenticate. This method has retry logic."""
        mail = imaplib.IMAP4_SSL(self.server)
        
        # Retrieve password securely
        try:
            password = self._credential_manager.get_credential("gmail", self.username)
        except CredentialError as e:
            logger.error(f"Failed to retrieve password for {self.username}: {e}")
            raise RuntimeError(f"No valid password found for {self.username}")
        
        mail.login(self.username, password)
        
        # Clear password from local variable immediately
        password = None
        
        mail.select("INBOX")
        return mail

    def _search_unread_messages(self, mail: imaplib.IMAP4_SSL, max_messages: int) -> List[bytes]:
        """Search for unread messages. This method has retry logic."""
        _typ, data = mail.search(None, "UNSEEN")
        if not data or not data[0]:
            logger.info("No unread messages found")
            return []
        
        message_nums = data[0].split()[:max_messages]
        logger.info("Found %d unread messages to fetch", len(message_nums))
        return message_nums

    def _fetch_message_content(self, mail: imaplib.IMAP4_SSL, message_num: bytes) -> str:
        """Fetch and parse content for a single message. This method has retry logic."""
        _typ, msg_data = mail.fetch(message_num, "(RFC822)")
        if not msg_data or not msg_data[0]:
            logger.warning("Empty message data for message %s", message_num)
            return ""
            
        # Safely parse the email message
        raw_email = msg_data[0][1]
        if not raw_email:
            logger.warning("No raw email content for message %s", message_num)
            return ""
            
        email_msg = message_from_bytes(raw_email)
        if not isinstance(email_msg, EmailMessage):
            logger.warning("Failed to parse email message %s", message_num)
            return ""
        
        # Extract payload with proper error handling
        payload = email_msg.get_payload(decode=True)
        if payload is None:
            # Try getting the payload without decoding
            payload = email_msg.get_payload()
            if isinstance(payload, str):
                content = payload
            else:
                logger.warning("Unable to extract payload from message %s", message_num)
                return ""
        else:
            # Decode bytes to string with robust error handling
            if isinstance(payload, bytes):
                try:
                    content = payload.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        content = payload.decode("latin-1")
                    except UnicodeDecodeError:
                        content = payload.decode("utf-8", errors="replace")
                        logger.warning("Used 'replace' error handling for message %s encoding", message_num)
            else:
                content = str(payload)
        
        if content and content.strip():
            return content.strip()
        else:
            logger.warning("Empty content for message %s", message_num)
            return ""

    def fetch_unread(self, max_messages: int = 10) -> List[str]:
        """Return up to ``max_messages`` unread messages as raw strings.
        
        Returns empty list if any errors occur during fetching.
        Logs errors for debugging but doesn't raise exceptions.
        """
        messages: List[str] = []
        mail = None
        
        try:
            # Connect and authenticate with retry logic
            connect_with_retry = retry_with_backoff(self.retry_config)(self._connect_and_authenticate)
            mail = connect_with_retry()
            
            # Search for unread messages with retry logic
            search_with_retry = retry_with_backoff(self.retry_config)(self._search_unread_messages)
            message_nums = search_with_retry(mail, max_messages)
            
            if not message_nums:
                return messages
            
            # Fetch each message with individual retry logic
            fetch_with_retry = retry_with_backoff(self.retry_config)(self._fetch_message_content)
            
            for num in message_nums:
                try:
                    content = fetch_with_retry(mail, num)
                    if content:
                        messages.append(content)
                except Exception as e:
                    logger.error("Error processing message %s after retries: %s", num, str(e))
                    continue
                    
        except imaplib.IMAP4.error as e:
            logger.error("IMAP error: %s", str(e))
        except Exception as e:
            logger.error("Unexpected error fetching emails: %s", str(e))
        finally:
            if mail:
                try:
                    mail.logout()
                except Exception as e:
                    logger.warning("Error during logout: %s", str(e))

        logger.info("Successfully fetched %d messages", len(messages))
        return messages

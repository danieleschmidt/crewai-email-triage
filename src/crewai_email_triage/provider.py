"""Email provider integrations."""

from __future__ import annotations

import imaplib
import logging
from email import message_from_bytes
from email.message import EmailMessage
from typing import List
import os

logger = logging.getLogger(__name__)


class GmailProvider:
    """Fetch messages from Gmail via IMAP."""

    def __init__(self, username: str, password: str, server: str = "imap.gmail.com"):
        self.username = username
        self.password = password
        self.server = server

    @classmethod
    def from_env(cls, server: str = "imap.gmail.com") -> "GmailProvider":
        """Return a provider using ``GMAIL_USER`` and ``GMAIL_PASSWORD`` env vars.

        Raises ``RuntimeError`` if either variable is missing.
        """
        user = os.environ.get("GMAIL_USER")
        password = os.environ.get("GMAIL_PASSWORD")
        if not user or not password:
            raise RuntimeError("GMAIL_USER and GMAIL_PASSWORD must be set")
        return cls(user, password, server)

    def fetch_unread(self, max_messages: int = 10) -> List[str]:
        """Return up to ``max_messages`` unread messages as raw strings.
        
        Returns empty list if any errors occur during fetching.
        Logs errors for debugging but doesn't raise exceptions.
        """
        messages: List[str] = []
        mail = None
        
        try:
            mail = imaplib.IMAP4_SSL(self.server)
            mail.login(self.username, self.password)
            mail.select("INBOX")
            
            _typ, data = mail.search(None, "UNSEEN")
            if not data or not data[0]:
                logger.info("No unread messages found")
                return messages
                
            message_nums = data[0].split()[:max_messages]
            logger.info("Fetching %d unread messages", len(message_nums))
            
            for num in message_nums:
                try:
                    _typ, msg_data = mail.fetch(num, "(RFC822)")
                    if not msg_data or not msg_data[0]:
                        logger.warning("Empty message data for message %s", num)
                        continue
                        
                    # Safely parse the email message
                    raw_email = msg_data[0][1]
                    if not raw_email:
                        logger.warning("No raw email content for message %s", num)
                        continue
                        
                    email_msg = message_from_bytes(raw_email)
                    if not isinstance(email_msg, EmailMessage):
                        logger.warning("Failed to parse email message %s", num)
                        continue
                    
                    # Extract payload with proper error handling
                    payload = email_msg.get_payload(decode=True)
                    if payload is None:
                        # Try getting the payload without decoding
                        payload = email_msg.get_payload()
                        if isinstance(payload, str):
                            content = payload
                        else:
                            logger.warning("Unable to extract payload from message %s", num)
                            continue
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
                                    logger.warning("Used 'replace' error handling for message %s encoding", num)
                        else:
                            content = str(payload)
                    
                    if content and content.strip():
                        messages.append(content.strip())
                    else:
                        logger.warning("Empty content for message %s", num)
                        
                except Exception as e:
                    logger.error("Error processing message %s: %s", num, str(e))
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

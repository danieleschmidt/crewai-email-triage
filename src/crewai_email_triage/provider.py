"""Email provider integrations."""

from __future__ import annotations

import imaplib
from email import message_from_bytes
from typing import List
import os


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
        """Return up to ``max_messages`` unread messages as raw strings."""
        messages: List[str] = []
        mail = imaplib.IMAP4_SSL(self.server)
        try:
            mail.login(self.username, self.password)
            mail.select("INBOX")
            _typ, data = mail.search(None, "UNSEEN")
            for num in data[0].split()[:max_messages]:
                _typ, msg_data = mail.fetch(num, "(RFC822)")
                if msg_data and msg_data[0]:
                    messages.append(
                        message_from_bytes(msg_data[0][1])
                        .get_payload(decode=True)
                        .decode("utf-8", errors="ignore")
                    )
        finally:
            mail.logout()

        return messages

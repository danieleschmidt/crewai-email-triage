from crewai_email_triage.provider import GmailProvider
import imaplib
import os

class FakeIMAP:
    def __init__(self):
        self.box = [b"Subject: hi\n\nUrgent"]
    def login(self, user, password):
        return 'OK'
    def select(self, mailbox):
        return 'OK', []
    def search(self, charset, criteria):
        return 'OK', [b'1']
    def fetch(self, num, spec):
        return 'OK', [(b'1', self.box[int(num.decode())-1])]
    def logout(self):
        return 'OK'

def test_fetch_unread(monkeypatch):
    monkeypatch.setattr(imaplib, 'IMAP4_SSL', lambda server: FakeIMAP())
    client = GmailProvider('u', 'p')
    msgs = client.fetch_unread(max_messages=1)
    assert msgs == ['Urgent']


def test_from_env(monkeypatch):
    monkeypatch.setattr(imaplib, 'IMAP4_SSL', lambda server: FakeIMAP())
    monkeypatch.setitem(os.environ, 'GMAIL_USER', 'u')
    monkeypatch.setitem(os.environ, 'GMAIL_PASSWORD', 'p')
    client = GmailProvider.from_env()
    assert client.username == 'u'
    assert client.password == 'p'
    assert client.fetch_unread(1) == ['Urgent']

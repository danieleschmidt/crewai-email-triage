from crewai_email_triage.provider import GmailProvider
import imaplib

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

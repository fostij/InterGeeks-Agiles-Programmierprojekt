"""IMAP-Client zum Abrufen ungelesener E-Mails.

Stellt eine Verbindung zum IMAP-Server her, ruft ungesehene Nachrichten
ab und markiert sie nach dem Abruf als gelesen, damit sie beim nächsten
Durchlauf nicht erneut verarbeitet werden.
"""

import email
import imaplib
import logging
from collections.abc import Generator
from email.message import Message

logger = logging.getLogger(__name__)


class ImapClient:
    """Verbindet sich per IMAP4_SSL und liefert ungesehene Nachrichten."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        folder: str = "INBOX",
    ) -> None:
        """Initialisiert den IMAP-Client mit den Verbindungsdaten.

        Args:
            host: IMAP-Serveradresse (z. B. "imap.gmail.com").
            username: E-Mail-Adresse des Postfachs.
            password: Passwort oder App-Passwort.
            folder: Zu überwachender Ordner (Standard: "INBOX").
        """
        self.host = host
        self.username = username
        self.password = password
        self.folder = folder

    def get_unseen_messages(self) -> Generator[tuple[bytes, Message], None, None]:
        """Verbindet sich, liefert ungesehene Nachrichten und markiert sie als gelesen.

        Yields:
            Tupel (uid, message) für jede ungesehene Nachricht.

        Die Verbindung wird in jedem Aufruf neu aufgebaut und im
        finally-Block garantiert geschlossen — auch bei Ausnahmen.
        """
        conn = imaplib.IMAP4_SSL(self.host)
        try:
            conn.login(self.username, self.password)
            conn.select(self.folder)

            _, data = conn.search(None, "UNSEEN")
            uids = data[0].split()
            logger.info("IMAP: %d ungesehene Nachrichten gefunden.", len(uids))

            for uid in uids:
                try:
                    _, msg_data = conn.fetch(uid, "(RFC822)")
                    raw_message = msg_data[0][1]
                    message = email.message_from_bytes(raw_message)

                    # Als gelesen markieren — verhindert doppelte Verarbeitung
                    conn.store(uid, "+FLAGS", "\\Seen")

                    yield uid, message

                except Exception as exc:
                    logger.error("Fehler beim Abrufen der Nachricht UID=%s: %s", uid, exc)
                    continue

        finally:
            try:
                conn.logout()
            except Exception:
                pass
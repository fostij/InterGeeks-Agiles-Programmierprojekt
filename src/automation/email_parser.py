"""Parser für eingehende E-Mail-Nachrichten.

Extrahiert aus einer MIME-Nachricht den Textkörper, ein optionales
Foto-Attachment (als Bytes) sowie Metadaten wie Absender, Betreff und
Message-ID. Die Bytes werden direkt zurückgegeben und nicht auf dem
Dateisystem gespeichert — die temporäre Datei-Verwaltung übernimmt
Pipeline.run() (siehe src/services/pipeline.py).
"""

import logging
from dataclasses import dataclass
from email.message import Message
from src.exceptions import EmailParsingError

logger = logging.getLogger(__name__)


@dataclass
class ParsedEmail:
    """Extrahierte Daten aus einer eingehenden E-Mail.

    Attributes:
        sender: Absenderadresse (für die Antwort-Mail).
        subject: Betreff der Nachricht.
        text: Klartext-Körper der Nachricht (Eingabe für die Pipeline).
        photo_bytes: Rohe Bytes des ersten Bild-Attachments, falls vorhanden.
            Wird direkt an Pipeline.run(photo_bytes=...) übergeben.
        message_id: Message-ID-Header, nützlich für Deduplizierung oder
            Antwort-Referenzierung in zukünftigen Erweiterungen.
    """

    sender: str
    subject: str
    text: str
    photo_bytes: bytes | None
    message_id: str


class EmailParser:
    """Parst eine email.message.Message in ein ParsedEmail-Objekt."""

    def parse(self, message: Message) -> ParsedEmail:
        """Extrahiert Text, Foto-Bytes und Metadaten aus einer Nachricht.

        Args:
            message: Geparstes E-Mail-Objekt (aus email.message_from_bytes).

        Returns:
            ParsedEmail mit Textkörper, optionalen Foto-Bytes und Metadaten.
        """
        sender = message.get("From", "")
        subject = message.get("Subject", "")
        message_id = message.get("Message-ID", "")

        text = ""
        photo_bytes = None

        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))

                if (
                    content_type == "text/plain"
                    and "attachment" not in disposition
                    and not text
                ):
                    payload = part.get_payload(decode=True)
                    if payload:
                        text = payload.decode(errors="ignore")

                elif (
                    content_type.startswith("image/")
                    and photo_bytes is None
                ):
                    photo_bytes = part.get_payload(decode=True)
                    logger.debug("Foto-Anhang gefunden: Content-Type=%s", content_type)

        else:
            payload = message.get_payload(decode=True)
            if payload:
                text = payload.decode(errors="ignore")

        logger.info(
            "E-Mail geparst: von=%s, hat_foto=%s, Textlänge=%d Zeichen.",
            sender, photo_bytes is not None, len(text),
        )

        if not sender:
            raise EmailParsingError("Absenderadresse fehlt — Nachricht kann nicht beantwortet werden.")
 
        if not text.strip():
            raise EmailParsingError(f"Nachricht von {sender} hat keinen lesbaren Textkörper.")


        return ParsedEmail(
            sender=sender,
            subject=subject,
            text=text,
            photo_bytes=photo_bytes,
            message_id=message_id,
        )
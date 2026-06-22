"""SMTP-Client zum Versenden von Antwort-E-Mails.

Unterstützt zwei Verbindungsmodi:
- SMTP_SSL (Port 465): direkte TLS-Verbindung (Standard bei Gmail, usw.)
- STARTTLS (Port 587): TLS-Upgrade nach dem Verbindungsaufbau (Standard
  bei Office 365, vielen Unternehmensservern)

Den passenden Modus über use_starttls=True/False wählen.
"""

import logging
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class SmtpClient:
    """Versendet E-Mails über SMTP_SSL oder STARTTLS."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        use_starttls: bool = False,
    ) -> None:
        """Initialisiert den SMTP-Client.

        Args:
            host: SMTP-Serveradresse (z. B. "smtp.gmail.com").
            port: Port — typisch 465 für SSL, 587 für STARTTLS.
            username: Absender-E-Mail-Adresse.
            password: Passwort oder App-Passwort.
            use_starttls: True für STARTTLS (Port 587),
                False für direktes SSL (Port 465, Standard).
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_starttls = use_starttls

    def send(
        self,
        recipient: str,
        subject: str,
        body: str,
    ) -> None:
        """Versendet eine Textnachricht an einen Empfänger.

        Args:
            recipient: Empfänger-E-Mail-Adresse.
            subject: Betreff der Nachricht.
            body: Klartext-Inhalt der Nachricht.

        Raises:
            smtplib.SMTPException: Wenn der Versand fehlschlägt.
        """
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = recipient
        msg.set_content(body)

        try:
            if self.use_starttls:
                with smtplib.SMTP(self.host, self.port) as smtp:
                    smtp.starttls()
                    smtp.login(self.username, self.password)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP_SSL(self.host, self.port) as smtp:
                    smtp.login(self.username, self.password)
                    smtp.send_message(msg)

            logger.info("Antwort-Mail gesendet an: %s", recipient)

        except smtplib.SMTPException as exc:
            logger.error("E-Mail-Versand an %s fehlgeschlagen: %s", recipient, exc)
            raise
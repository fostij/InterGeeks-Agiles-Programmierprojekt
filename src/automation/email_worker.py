"""E-Mail-Worker: überwacht ein Postfach und verarbeitet eingehende Schadensmeldungen.

Läuft als dauerhafter Prozess in einer Schleife. Für jede ungesehene
Nachricht wird die Pipeline aufgerufen und das Ergebnis als Antwort-Mail
an den Absender zurückgeschickt. Fehler bei einzelnen Nachrichten
unterbrechen die Schleife nicht — sie werden geloggt und die nächste
Nachricht wird verarbeitet.

Starten:
    python -m src.automation.email_worker
"""

import logging
import time

from src.automation.email_parser import EmailParser
from src.automation.imap_client import ImapClient
from src.automation.smtp_client import SmtpClient
from src.services.pipeline import Pipeline, PredictionResult

logger = logging.getLogger(__name__)


class EmailWorker:
    """Orchestriert den E-Mail-Verarbeitungskreislauf."""

    def __init__(
        self,
        pipeline: Pipeline,
        imap_client: ImapClient,
        parser: EmailParser,
        smtp_client: SmtpClient,
        interval: int = 60,
    ) -> None:
        """Initialisiert den Worker mit allen benötigten Komponenten.

        Args:
            pipeline: Initialisierte Pipeline-Instanz (wird einmalig
                übergeben und für alle Nachrichten wiederverwendet —
                kein Neuladen der Modelle pro Nachricht).
            imap_client: Client zum Abrufen ungesehener Nachrichten.
            parser: Parser zum Extrahieren von Text und Foto aus der Mail.
            smtp_client: Client zum Versenden der Antwort-Mail.
            interval: Wartezeit in Sekunden zwischen zwei Postfach-Prüfungen
                (Standard: 60 Sekunden).
        """
        self.pipeline = pipeline
        self.imap_client = imap_client
        self.parser = parser
        self.smtp_client = smtp_client
        self.interval = interval

    def run(self) -> None:
        """Startet den dauerhaften Verarbeitungskreislauf.

        Läuft bis zum Prozessabbruch (z. B. Ctrl+C oder SIGTERM).
        Fehler bei einzelnen Nachrichten werden geloggt und übersprungen,
        der Worker läuft weiter.
        """
        logger.info("E-Mail-Worker gestartet. Prüfintervall: %d Sekunden.", self.interval)

        while True:
            try:
                for uid, message in self.imap_client.get_unseen_messages():
                    self._process_message(uid, message)
            except Exception as exc:
                logger.error("Fehler beim Abrufen der Nachrichten: %s", exc)

            time.sleep(self.interval)

    def _process_message(self, uid: bytes, message) -> None:
        """Verarbeitet eine einzelne Nachricht: parsen → Pipeline → Antwort senden.

        Fehler werden geloggt und nicht weitergereicht, damit die
        übergeordnete Schleife weiterläuft.
        """
        try:
            email_data = self.parser.parse(message)

            logger.info("Verarbeite Nachricht von: %s", email_data.sender)

            result = self.pipeline.run(
                text=email_data.text,
                photo_bytes=email_data.photo_bytes,
                source="email",
            )

            body = self._build_response(result)

            self.smtp_client.send(
                recipient=email_data.sender,
                subject="Re: " + (email_data.subject or "Ihre Schadensmeldung"),
                body=body,
            )

            logger.info("Nachricht von %s erfolgreich verarbeitet.", email_data.sender)

        except Exception as exc:
            logger.error(
                "Verarbeitung der Nachricht UID=%s fehlgeschlagen: %s",
                uid, exc,
            )

    def _build_response(self, result: PredictionResult) -> str:
        """Formatiert das PredictionResult als lesbaren Antworttext.

        Args:
            result: Ergebnis eines Pipeline-Durchlaufs.

        Returns:
            Formatierter Antworttext für die Antwort-Mail.
        """
        amount = (
            f"{result.predicted_amount:,.2f} EUR"
            if result.predicted_amount is not None
            else "Nicht verfügbar"
        )

        severity = result.final_severity or "Nicht bestimmt"

        confidence = (
            f"{result.text_confidence:.1%}"
            if result.text_confidence is not None
            else "–"
        )

        source_label = (
            "Textanalyse" if result.severity_source == "text" else "Bildanalyse"
        )

        missing = (
            ", ".join(result.missing_fields)
            if result.missing_fields
            else "keine"
        )

        return (
            "Sehr geehrte Damen und Herren,\n\n"
            "vielen Dank für Ihre Schadensmeldung. "
            "Hier ist das Ergebnis unserer automatisierten Analyse:\n\n"
            f"  Geschätzte Schadenhöhe:  {amount}\n"
            f"  Schweregrad:             {severity}\n"
            f"  Konfidenz:               {confidence} ({source_label})\n"
            f"  Fehlende Angaben:        {missing}\n\n"
            "Bitte beachten Sie, dass es sich um eine vorläufige Schätzung handelt.\n"
            "Ein Sachbearbeiter wird sich in Kürze bei Ihnen melden.\n\n"
            "Mit freundlichen Grüßen\n"
            "Ihr KFZ-Schadenprognose-System"
        )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.logging_config import setup_logging
    from src.db.pipeline_repository import PipelineResultRepository

    load_dotenv()
    setup_logging()

    repo = PipelineResultRepository()
    pipeline = Pipeline(repository=repo)

    imap = ImapClient(
        host=os.getenv("EMAIL_IMAP_HOST", ""),
        username=os.getenv("EMAIL_USERNAME", ""),
        password=os.getenv("EMAIL_PASSWORD", ""),
        folder=os.getenv("EMAIL_IMAP_FOLDER", "INBOX"),
    )
    smtp = SmtpClient(
        host=os.getenv("EMAIL_SMTP_HOST", ""),
        port=int(os.getenv("EMAIL_SMTP_PORT", "465")),
        username=os.getenv("EMAIL_USERNAME", ""),
        password=os.getenv("EMAIL_PASSWORD", ""),
        use_starttls=os.getenv("EMAIL_USE_STARTTLS", "false").lower() == "true",
    )
    parser = EmailParser()

    worker = EmailWorker(
        pipeline=pipeline,
        imap_client=imap,
        parser=parser,
        smtp_client=smtp,
        interval=int(os.getenv("EMAIL_POLL_INTERVAL", "60")),
    )
    worker.run()
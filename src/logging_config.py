"""Zentrale Logging-Konfiguration für das gesamte Projekt.

Dieses Modul konfiguriert den Root-Logger genau einmal pro Prozess und
stellt sicher, dass alle Module über `logging.getLogger(__name__)` ein
einheitlich formatiertes Log erhalten — egal ob beim Training, bei der
Inferenz oder in den Hilfsskripten.

Verwendung in einem beliebigen Modul:

    import logging
    from src.logging_config import setup_logging

    logger = logging.getLogger(__name__)

    def main() -> None:
        setup_logging()
        logger.info("Starte Verarbeitung")

`setup_logging()` sollte nur einmal aufgerufen werden, idealerweise im
Einstiegspunkt eines Skripts (z. B. unter `if __name__ == "__main__"`).
Module, die als Bibliothek importiert werden, rufen `setup_logging()`
NICHT selbst auf — sie erstellen nur ihren eigenen Logger über
`logging.getLogger(__name__)` und überlassen die Konfiguration dem
aufrufenden Einstiegspunkt.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_is_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> None:
    """Konfiguriert den Root-Logger für das gesamte Projekt.

    Args:
        level: Mindest-Log-Level, das ausgegeben wird (Standard: INFO).
        log_file: Optionaler Pfad zu einer Log-Datei. Wenn angegeben,
            werden Log-Nachrichten zusätzlich zur Konsole in diese Datei
            geschrieben. Das übergeordnete Verzeichnis wird bei Bedarf
            angelegt.

    Der Aufruf ist idempotent: Mehrfaches Aufrufen innerhalb desselben
    Prozesses fügt keine doppelten Handler hinzu.
    """
    global _is_configured

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if _is_configured:
        return

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _is_configured = True
"""Kommandozeilen-Einstiegspunkt für die KFZ-Schadensprognose-Pipeline.
 
Dritter Eingabekanal neben dem Streamlit-Dashboard (app/dashboard.py)
und dem E-Mail-Worker (src/automation/email_worker.py). Ermöglicht
das direkte Ausführen der Pipeline aus dem Terminal — nützlich für
manuelle Tests und Entwicklung ohne laufendes Dashboard.
 
Verwendung:
    # Nur Text:
    python main.py --text "Frontschaden nach Auffahrunfall, Airbag ausgelöst"
 
    # Text und Foto:
    python main.py --text "Frontschaden nach Auffahrunfall" --photo pfad/zum/foto.jpg
 
    # Nur Foto:
    python main.py --photo pfad/zum/foto.jpg
"""

import argparse
import logging
import sys
 
from dotenv import load_dotenv
 
from src.db.pipeline_repository import PipelineResultRepository
from src.exceptions import ProjectError
from src.logging_config import setup_logging
from src.services.pipeline import Pipeline
 
logger = logging.getLogger(__name__)

def main() -> None:
    """Initialisiert die Pipeline und führt eine Vorhersage durch."""

    load_dotenv()
    setup_logging()

    args = parse_args()

    if not args.text and not args.photo:
        logger.error("Mindestens --text oder --photo muss angegeben werden.")
        sys.exit(1)

    if args.photo:
        import os
        if not os.path.exists(args.photo):
            logger.error("Foto nicht gefunden: %s", args.photo)
            sys.exit(1)
    
    try:
        repo = PipelineResultRepository()
        pipeline = Pipeline(repository=repo)

        result = pipeline.run(
            text=args.text,
            photo_path=args.photo,
            source="cli",
        )

        print("\n" + "=" * 60)
        print("  Ergebnis der Schadensprognose")
        print("=" * 60)
        print(f"  Anfrage-ID:          {result.request_id}")
        print(f"  Schweregrad:         {result.final_severity or '–'}")
        print(f"  Quelle:              {result.severity_source}")

        if result.text_confidence is not None:
            print(f"  Text-Konfidenz:      {result.text_confidence:.1%}")
        if result.photo_confidence is not None:
            print(f"  Foto-Konfidenz:      {result.photo_confidence:.1%}")
        if result.predicted_amount is not None:
            print(f"  Geschätzte Schaden:  {result.predicted_amount:,.2f} EUR")
        else:
            print("  Geschätzte Schaden:  nicht verfügbar")
        if result.missing_fields:
            print(f"  Fehlende Felder:     {', '.join(result.missing_fields)}")
        print("=" * 60)

    except ProjectError as exc:
        logger.error("Pipeline fehlgeschlagen: %s", exc)
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    """Liest Kommandozeilenargumente aus."""
    parser = argparse.ArgumentParser(
        description="KFZ-Schadensprognose — Pipeline direkt ausführen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Beispiele:\n"
            "  python main.py --text \"Frontschaden nach Auffahrunfall\"\n"
            "  python main.py --text \"Seitenschaden\" --photo foto.jpg\n"
            "  python main.py --photo foto.jpg\n"
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Unfallbeschreibung als Freitext",
    )
    parser.add_argument(
        "--photo",
        type=str,
        default=None,
        help="Pfad zu einem Foto des Schadens",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()